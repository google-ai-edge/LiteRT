/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/gemma4/litert/model_builder.h"

#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/litert/active_graph_context.h"
#include "tensor/examples/gemma4/litert/bundle_loader.h"
#include "tensor/examples/gemma4/litert/gemma4_graph.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4::cpu {
namespace {
using T = Tensor<TfLiteMixinTag>;

absl::StatusOr<std::vector<TensorHandle>> BuildSignature(
    const Config& config, const LoadedTensors& loaded,
    const std::vector<KvOwnerSpec>& specs, int rows, bool decode,
    ModelFactory& factory, const ModelAuthoringOptions& options) {
  Gemma4Inputs<TfLiteMixinTag> inputs;
  inputs.bundle_dynamic_qd8_head = true;
  inputs.embedded_input = T({.name = "embedded_input",
                             .type = Type::kFP32,
                             .shape = {1, rows, config.embed_dim}});
  for (const auto& [name, weight] : loaded.weights_handle) {
    inputs.weights.emplace(name, weight);
  }
  for (int layer = 0; layer < config.num_layers; ++layer) {
    inputs.per_layer_token_embeddings.emplace_back(
        TensorInit{.name = absl::StrCat("per_layer_token_embedding_", layer),
                   .type = Type::kFP32,
                   .shape = {1, rows, config.per_layer_input_dim}});
  }
  T positions(
      {.name = "positions", .type = Type::kFP32, .shape = {1, 1, rows, 1}});
  const auto make_rope = [&](int dimension, float base, float proportion) {
    const int half = dimension / 2;
    std::vector<float> inverse(half, 0.0f);
    for (int i = 0; i < static_cast<int>(proportion * half); ++i) {
      inverse[i] = 1.0f / std::pow(base, 2.0f * i / dimension);
    }
    T frequency({.type = Type::kFP32,
                 .shape = {1, 1, 1, half},
                 .buffer = OwningCpuBuffer::Copy<Type::kFP32>(inverse)});
    T angles = Mul(positions, frequency);
    T cosine = Cos(angles), sine = Sin(angles);
    return std::make_pair(Concatenation({cosine, cosine}, 3),
                          Concatenation({sine, sine}, 3));
  };
  std::tie(inputs.rope_global_cos, inputs.rope_global_sin) =
      make_rope(config.global_key_size, config.global_base_frequency,
                config.global_rope_proportion);
  std::tie(inputs.rope_local_cos, inputs.rope_local_sin) =
      make_rope(config.head_dim, config.local_base_frequency,
                config.local_rope_proportion);
  ActiveGraphContext context;
  context.owners = GetKvCacheSharingPatterns(config);
  context.broadcast_mask = !decode;
  context.dynamic_prefill_rows = !decode && options.dynamic_prefill_rows;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    if (layer >= context.owners.size() || context.owners[layer] < 0 ||
        context.owners[layer] >= specs.size()) {
      return absl::InvalidArgumentError("Invalid KV owner index");
    }
    const KvOwnerSpec& spec = specs[context.owners[layer]];
    inputs.key_caches.push_back(MakeInt8KeyCache<TfLiteMixinTag>(
        "key_metadata", 1, spec.head_dim, spec.key_scale));
    inputs.value_caches.push_back(MakeInt8KeyCache<TfLiteMixinTag>(
        "value_metadata", 1, spec.head_dim, spec.value_scale));
  }
  Gemma4Outputs<TfLiteMixinTag> outputs =
      BuildGemma4Graph(inputs, config, context);
  std::vector<TensorHandle> roots;
  if (decode) {
    outputs.logits.SetName("logits");
    roots.push_back(outputs.logits);
  }
  for (const KvOwnerSpec& spec : specs) {
    if (spec.owner < 0 || spec.owner >= context.layers.size()) {
      return absl::InvalidArgumentError("Invalid KV owner layer index");
    }
    ActiveLayerGraph& layer = context.layers[spec.owner];
    layer.new_key.SetName(absl::StrCat("new_key_", spec.owner));
    layer.new_value.SetName(absl::StrCat("new_value_", spec.owner));
    roots.push_back(layer.new_key);
    roots.push_back(layer.new_value);
  }
  for (const TensorHandle& root : roots) {
    LRT_TENSOR_RETURN_IF_ERROR(root.GetStatus());
  }
  if (options.explicit_dynamic_shapes) {
    // Keep the concrete authoring shapes and graph arithmetic unchanged.
    // These boundary contracts declare which dimensions the host may resize;
    // relationships between inputs are enforced by Runner's chunk/KV logic.
    const auto dynamic_axis = [&](const TensorHandle& tensor,
                                  int axis) -> absl::Status {
      Shape signature = tensor.GetShape();
      if (axis < 0 || axis >= signature.size()) {
        return absl::InvalidArgumentError("Dynamic axis out of bounds");
      }
      signature[axis] = -1;
      return factory.SetShapeSignature(tensor, std::move(signature));
    };
    if (context.dynamic_prefill_rows) {
      LRT_TENSOR_RETURN_IF_ERROR(dynamic_axis(inputs.embedded_input, 1));
      LRT_TENSOR_RETURN_IF_ERROR(dynamic_axis(positions, 2));
      for (const T& ple : inputs.per_layer_token_embeddings) {
        LRT_TENSOR_RETURN_IF_ERROR(dynamic_axis(ple, 1));
      }
      for (const TensorHandle& root : roots) {
        LRT_TENSOR_RETURN_IF_ERROR(dynamic_axis(root, 2));
      }
    }
    for (const KvOwnerSpec& spec : specs) {
      if (spec.owner < 0 || spec.owner >= context.layers.size()) {
        return absl::InvalidArgumentError("Invalid KV owner layer index");
      }
      const ActiveLayerGraph& layer = context.layers[spec.owner];
      // The last prefill owner produces KV without consuming attention history.
      if (!decode && !specs.empty() && spec.owner == specs.back().owner) {
        continue;
      }
      for (const TensorHandle& input :
           {layer.past_key, layer.past_value, layer.pad_key, layer.pad_value}) {
        LRT_TENSOR_RETURN_IF_ERROR(dynamic_axis(input, 2));
      }
    }
    for (const TensorHandle& mask : {context.local_mask, context.global_mask}) {
      Shape signature = mask.GetShape();
      if (signature.size() <= 3) {
        return absl::InvalidArgumentError("Mask shape rank must be at least 4");
      }
      signature[3] =
          -1;  // Active history plus this chunk and alignment pad.
      if (context.dynamic_prefill_rows) {
        signature[2] = -1;
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          factory.SetShapeSignature(mask, std::move(signature)));
    }
  }
  return roots;
}

}  // namespace

absl::Status AddGemma4Signatures(ModelFactory& factory,
                                 const LoadedTensors& weights,
                                 const std::vector<KvOwnerSpec>& specs,
                                 const ModelAuthoringOptions& options) {
  if (options.prefill_rows < 32 || options.prefill_rows > 1024 ||
      (options.prefill_rows & (options.prefill_rows - 1)) != 0) {
    return absl::InvalidArgumentError(
        "Require power-of-two prefill rows in [32,1024]");
  }
  const Config config = Config::E2B();
  const std::vector<int> owners = GetKvCacheSharingPatterns(config);
  if (owners.empty()) {
    return absl::InvalidArgumentError("No KV cache sharing patterns");
  }
  const int owner_count = owners.back() + 1;
  if (specs.size() != owner_count) {
    return absl::InvalidArgumentError("Expected 15 E2B KV owners");
  }
  for (int i = 0; i < owner_count; ++i) {
    const KvOwnerSpec& spec = specs[i];
    const int dimension = config.GetLayerType(i) == Config::LayerType::kGlobal
                              ? config.global_key_size
                              : config.head_dim;
    if (spec.owner != i || spec.num_heads != 1 || spec.head_dim != dimension ||
        !std::isfinite(spec.key_scale) || spec.key_scale <= 0 ||
        !std::isfinite(spec.value_scale) || spec.value_scale <= 0) {
      return absl::InvalidArgumentError("Invalid E2B KV owner metadata");
    }
  }
  for (const auto& [name, expected] :
       published_bundle_detail::ExpectedTensors(config)) {
    if (!weights.weights_handle.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing model weight: ", name));
    }
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<TensorHandle> prefill,
      BuildSignature(config, weights, specs, options.prefill_rows, false,
                     factory, options));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<TensorHandle> decode,
      BuildSignature(config, weights, specs, 1, true, factory, options));
  LRT_TENSOR_RETURN_IF_ERROR(
      factory.AddSignature(std::move(prefill), "prefill"));
  return factory.AddSignature(std::move(decode), "decode");
}

}  // namespace litert::tensor::examples::gemma4::cpu
