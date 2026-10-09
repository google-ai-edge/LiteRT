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

#include "tensor/examples/experimental/gemma4/model_builder.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/active_graph_context.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/gemma4_graph.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {
using ::litert::tensor::examples::gemma4::Config;
using TfLiteTensor = Tensor<TfLiteMixinTag>;

struct RopeTables {
  TfLiteTensor cosine = TensorHandle::Invalid();
  TfLiteTensor sine = TensorHandle::Invalid();
};

enum class SignatureMode {
  kPrefill,
  kDecode,
};

// Computes split-half RoPE cosine and sine tables from `positions` for a
// head dimension with `proportion` of the pairs rotated.
absl::StatusOr<RopeTables> BuildRopeCosSin(const TfLiteTensor& positions,
                                           int dimension, float base,
                                           float proportion) {
  if (dimension <= 0 || dimension % 2 != 0 || !std::isfinite(base) ||
      base <= 0.0f || !std::isfinite(proportion) || proportion < 0.0f ||
      proportion > 1.0f) {
    return absl::InvalidArgumentError("Invalid RoPE frequency parameters");
  }
  const int half = dimension / 2;
  std::vector<float> inverse(half, 0.0f);
  const int rotated_pairs = static_cast<int>(proportion * half);
  for (int i = 0; i < rotated_pairs; ++i) {
    inverse[i] = 1.0f / std::pow(base, 2.0f * i / dimension);
  }
  const TfLiteTensor frequency(
      {.type = Type::kFP32,
       .shape = {1, 1, 1, half},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(inverse)});
  const TfLiteTensor angles = Mul(positions, frequency);
  const TfLiteTensor cosine = Cos(angles);
  const TfLiteTensor sine = Sin(angles);
  return RopeTables{.cosine = Concatenation({cosine, cosine}, 3),
                    .sine = Concatenation({sine, sine}, 3)};
}

// Declares each axis in `axes` as dynamic (`-1`) on `tensor`'s TFLite shape
// signature without changing its concrete authoring shape.
absl::Status SetDynamicAxes(ModelFactory& factory, const TensorHandle& tensor,
                            absl::Span<const int> axes) {
  LRT_TENSOR_RETURN_IF_ERROR(tensor.GetStatus());
  Shape signature = tensor.GetShape();
  for (int axis : axes) {
    if (axis < 0 || axis >= static_cast<int>(signature.size())) {
      return absl::InvalidArgumentError("Dynamic axis out of bounds");
    }
    signature[axis] = -1;
  }
  return factory.SetShapeSignature(tensor, std::move(signature));
}

// Authors one Gemma4 signature (`prefill` or `decode`) into `factory`, naming
// outputs and declaring optional explicit dynamic shape signatures. Returns the
// ordered output root tensors for `ModelFactory::AddSignature`.
absl::StatusOr<std::vector<TensorHandle>> BuildSignature(
    const Config& config, const LoadedTensors& loaded,
    absl::Span<const KvOwnerSpec> specs, int rows, SignatureMode mode,
    ModelFactory& factory, const ModelAuthoringOptions& options) {
  const bool is_decode = mode == SignatureMode::kDecode;
  Gemma4Inputs<TfLiteMixinTag> inputs;
  inputs.bundle_dynamic_qd8_head = true;
  inputs.embedded_input = TfLiteTensor({.name = "embedded_input",
                                        .type = Type::kFP32,
                                        .shape = {1, rows, config.embed_dim}});
  inputs.weights.reserve(loaded.weights_handle.size());
  for (const auto& [name, weight] : loaded.weights_handle) {
    inputs.weights.emplace(name, weight);
  }
  inputs.per_layer_token_embeddings.reserve(config.num_layers);
  for (int layer = 0; layer < config.num_layers; ++layer) {
    inputs.per_layer_token_embeddings.emplace_back(
        TensorInit{.name = absl::StrCat("per_layer_token_embedding_", layer),
                   .type = Type::kFP32,
                   .shape = {1, rows, config.per_layer_input_dim}});
  }
  const TfLiteTensor positions(
      {.name = "positions", .type = Type::kFP32, .shape = {1, 1, rows, 1}});
  LRT_TENSOR_ASSIGN_OR_RETURN(RopeTables global_rope,
                              BuildRopeCosSin(positions, config.global_key_size,
                                              config.global_base_frequency,
                                              config.global_rope_proportion));
  inputs.rope_global_cos = std::move(global_rope.cosine);
  inputs.rope_global_sin = std::move(global_rope.sine);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      RopeTables local_rope,
      BuildRopeCosSin(positions, config.head_dim, config.local_base_frequency,
                      config.local_rope_proportion));
  inputs.rope_local_cos = std::move(local_rope.cosine);
  inputs.rope_local_sin = std::move(local_rope.sine);

  ActiveGraphContext context;
  context.owners = examples::gemma4::GetKvCacheSharingPatterns(config);
  context.broadcast_mask = !is_decode;
  context.dynamic_prefill_rows = !is_decode && options.dynamic_prefill_rows;
  inputs.key_caches.reserve(config.num_layers);
  inputs.value_caches.reserve(config.num_layers);
  for (int layer = 0; layer < config.num_layers; ++layer) {
    if (layer >= static_cast<int>(context.owners.size()) ||
        context.owners[layer] < 0 ||
        context.owners[layer] >= static_cast<int>(specs.size())) {
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
  LRT_TENSOR_RETURN_IF_ERROR(outputs.logits.GetStatus());
  std::vector<TensorHandle> roots;
  roots.reserve((is_decode ? 1 : 0) + specs.size() * 2);
  if (is_decode) {
    outputs.logits.SetName("logits");
    roots.push_back(outputs.logits);
  }
  for (const KvOwnerSpec& spec : specs) {
    if (spec.owner < 0 ||
        spec.owner >= static_cast<int>(context.layers.size())) {
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
    if (context.dynamic_prefill_rows) {
      LRT_TENSOR_RETURN_IF_ERROR(
          SetDynamicAxes(factory, inputs.embedded_input, {1}));
      LRT_TENSOR_RETURN_IF_ERROR(SetDynamicAxes(factory, positions, {2}));
      for (const TfLiteTensor& ple : inputs.per_layer_token_embeddings) {
        LRT_TENSOR_RETURN_IF_ERROR(SetDynamicAxes(factory, ple, {1}));
      }
      for (const TensorHandle& root : roots) {
        LRT_TENSOR_RETURN_IF_ERROR(SetDynamicAxes(factory, root, {2}));
      }
    }
    for (const KvOwnerSpec& spec : specs) {
      const ActiveLayerGraph& layer = context.layers[spec.owner];
      // The last prefill owner produces KV without consuming attention history.
      if (!is_decode && spec.owner == specs.back().owner) {
        continue;
      }
      for (const TensorHandle& input :
           {layer.past_key, layer.past_value, layer.pad_key, layer.pad_value}) {
        LRT_TENSOR_RETURN_IF_ERROR(SetDynamicAxes(factory, input, {2}));
      }
    }
    const std::vector<int> mask_axes = context.dynamic_prefill_rows
                                           ? std::vector<int>{2, 3}
                                           : std::vector<int>{3};
    for (const TensorHandle& mask : {context.local_mask, context.global_mask}) {
      if (mask == TensorHandle::Invalid()) {
        continue;
      }
      LRT_TENSOR_RETURN_IF_ERROR(SetDynamicAxes(factory, mask, mask_axes));
    }
  }
  return roots;
}

}  // namespace

absl::Status AddGemma4Signatures(ModelFactory& factory,
                                 const LoadedTensors& weights,
                                 absl::Span<const KvOwnerSpec> specs,
                                 const ModelAuthoringOptions& options) {
  if (options.prefill_rows < 32 || options.prefill_rows > 1024 ||
      (options.prefill_rows & (options.prefill_rows - 1)) != 0) {
    return absl::InvalidArgumentError(
        "Require power-of-two prefill rows in [32,1024]");
  }
  const Config config = Config::E2B();
  const std::vector<int> owners =
      examples::gemma4::GetKvCacheSharingPatterns(config);
  if (owners.empty()) {
    return absl::InvalidArgumentError("No KV cache sharing patterns");
  }
  const int owner_count = *absl::c_max_element(owners) + 1;
  if (static_cast<int>(specs.size()) != owner_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", owner_count, " E2B KV owners"));
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
    const auto it = weights.weights_handle.find(name);
    if (it == weights.weights_handle.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing model weight: ", name));
    }
    const TensorHandle& handle = it->second;
    if (!handle.GetStatus().ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid model weight ", name, ": ", handle.GetStatus().message()));
    }
    if (handle.GetShape() != expected.shape) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unexpected shape for model weight: ", name));
    }
    const Type type = handle.GetType();
    const bool dtype_ok =
        (expected.dtype == "float32" && type == Type::kFP32) ||
        (expected.dtype == "int8" && type == Type::kI8) ||
        // Published INT4 tensors with INT2 provenance are repacked to kI2.
        (expected.dtype == "int4" &&
         (type == Type::kI4 || type == Type::kI2)) ||
        (expected.dtype == "int2" && type == Type::kI2);
    if (!dtype_ok) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unexpected dtype for model weight: ", name));
    }
  }
  const auto softcap_it =
      weights.weights_handle.find("matched.logits_softcap_reciprocal");
  if (softcap_it == weights.weights_handle.end()) {
    return absl::InvalidArgumentError(
        "Missing model weight: matched.logits_softcap_reciprocal");
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      ReadMobileActivationScale(softcap_it->second).status());
  return AddGemma4Signatures(factory, config, weights, specs, options);
}

absl::Status AddGemma4Signatures(ModelFactory& factory, const Config& config,
                                 const LoadedTensors& weights,
                                 absl::Span<const KvOwnerSpec> specs,
                                 const ModelAuthoringOptions& options) {
  if (options.prefill_rows <= 0) {
    return absl::InvalidArgumentError("Require positive prefill rows");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<TensorHandle> prefill,
      BuildSignature(config, weights, specs, options.prefill_rows,
                     SignatureMode::kPrefill, factory, options));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<TensorHandle> decode,
      BuildSignature(config, weights, specs, 1, SignatureMode::kDecode, factory,
                     options));
  LRT_TENSOR_RETURN_IF_ERROR(
      factory.AddSignature(std::move(prefill), "prefill"));
  return factory.AddSignature(std::move(decode), "decode");
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu
