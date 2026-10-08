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

#include <cmath>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/initialize.h"  // from @com_google_absl
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

ABSL_FLAG(std::string, bundle_dir, "", "Verified Gemma4 E2B weight bundle.");
ABSL_FLAG(std::string, model_path, "", "New output .tflite model path.");
ABSL_FLAG(int, prefill_chunk_rows, 128, "Fixed prefill chunk size (32..1024).");

namespace litert::tensor::examples::gemma4::cpu {
namespace {
using T = Tensor<TfLiteMixinTag>;

absl::StatusOr<std::vector<TensorHandle>> BuildSignature(
    const Config& config, const LoadedTensors& loaded,
    const std::vector<KvOwnerSpec>& specs, int rows, bool decode) {
  Gemma4Inputs<TfLiteMixinTag> inputs;
  inputs.bundle_dynamic_qd8_head = true;
  inputs.embedded_input = T({.name = "embedded_input",
                             .type = Type::kFP32,
                             .shape = {1, rows, config.embed_dim}});
  for (const auto& [name, weight] : loaded.weights_handle)
    inputs.weights.emplace(name, weight);
  for (int layer = 0; layer < config.num_layers; ++layer) {
    inputs.per_layer_token_embeddings.emplace_back(
        TensorInit{.name = absl::StrCat("per_layer_token_embedding_", layer),
                   .type = Type::kFP32,
                   .shape = {1, rows, config.per_layer_input_dim}});
  }
  T positions(
      {.name = "positions", .type = Type::kFP32, .shape = {1, 1, rows, 1}});
  auto make_rope = [&](int dimension, float base, float proportion) {
    const int half = dimension / 2;
    std::vector<float> inverse(half, 0.0f);
    for (int i = 0; i < static_cast<int>(proportion * half); ++i)
      inverse[i] = 1.0f / std::pow(base, 2.0f * i / dimension);
    T frequency({.type = Type::kFP32,
                 .shape = {1, 1, 1, half},
                 .buffer = OwningCpuBuffer::Copy<Type::kFP32>(inverse)});
    auto angles = Mul(positions, frequency);
    auto cosine = Cos(angles), sine = Sin(angles);
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
  for (int layer = 0; layer < config.num_layers; ++layer) {
    const auto& spec = specs.at(context.owners[layer]);
    inputs.key_caches.push_back(MakeInt8KeyCache<TfLiteMixinTag>(
        "key_metadata", 1, spec.head_dim, spec.key_scale));
    inputs.value_caches.push_back(MakeInt8KeyCache<TfLiteMixinTag>(
        "value_metadata", 1, spec.head_dim, spec.value_scale));
  }
  auto outputs = BuildGemma4Graph(inputs, config, context);
  std::vector<TensorHandle> roots;
  if (decode) {
    outputs.logits.SetName("logits");
    roots.push_back(outputs.logits);
  }
  for (const auto& spec : specs) {
    auto& layer = context.layers[spec.owner];
    layer.new_key.SetName(absl::StrCat("new_key_", spec.owner));
    layer.new_value.SetName(absl::StrCat("new_value_", spec.owner));
    roots.push_back(layer.new_key);
    roots.push_back(layer.new_value);
  }
  for (const auto& root : roots) LRT_TENSOR_RETURN_IF_ERROR(root.GetStatus());
  return roots;
}

absl::Status Export() {
  const auto config = Config::E2B();
  const auto path = absl::GetFlag(FLAGS_model_path);
  const int rows = absl::GetFlag(FLAGS_prefill_chunk_rows);
  if (path.empty() || std::filesystem::exists(path) || rows < 32 ||
      rows > 1024 || (rows & (rows - 1)) != 0)
    return absl::InvalidArgumentError(
        "Require new model_path and power-of-two chunk rows in [32,1024]");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto loaded,
      LoadPublishedBundle(absl::GetFlag(FLAGS_bundle_dir), config));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto specs,
      LoadPublishedKvSpecs(absl::GetFlag(FLAGS_bundle_dir), config));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto prefill, BuildSignature(config, loaded, specs, rows, false));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto decode,
                              BuildSignature(config, loaded, specs, 1, true));
  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature(prefill, "prefill"));
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature(decode, "decode"));
  return factory.Save(path);
}
}  // namespace
}  // namespace litert::tensor::examples::gemma4::cpu

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  auto status = litert::tensor::examples::gemma4::cpu::Export();
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
