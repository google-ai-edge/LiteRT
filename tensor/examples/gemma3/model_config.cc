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

#include "tensor/examples/gemma3/model_config.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/examples/gemma3/gemma3_graph.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/examples/gemma3/tflite_loader.h"

namespace litert::tensor::examples {

namespace {

constexpr absl::string_view kLayerPrefix = "model.layers.";

bool ParseLayerIndex(absl::string_view tensor_name, int* layer_index) {
  if (!absl::StartsWith(tensor_name, kLayerPrefix)) {
    return false;
  }
  absl::string_view suffix = tensor_name.substr(kLayerPrefix.size());
  size_t pos = 0;
  while (pos < suffix.size() &&
         std::isdigit(static_cast<unsigned char>(suffix[pos])) != 0) {
    ++pos;
  }
  if (pos == 0 || pos >= suffix.size() || suffix[pos] != '.') {
    return false;
  }
  if (!absl::SimpleAtoi(suffix.substr(0, pos), layer_index)) {
    return false;
  }
  return true;
}

template <typename TensorInfoType>
absl::StatusOr<std::vector<int>> ToShape(const TensorInfoType& tensor_info,
                                         absl::string_view tensor_name) {
  std::vector<int> shape;
  shape.reserve(tensor_info.shape.size());
  for (int64_t dim : tensor_info.shape) {
    if (dim < 0 || dim > std::numeric_limits<int>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid shape dimension for ", tensor_name, ": ", dim));
    }
    shape.push_back(static_cast<int>(dim));
  }
  return shape;
}

template <typename LoaderType>
absl::StatusOr<Gemma3Config> InferGemma3ConfigFromLoaderImpl(
    const LoaderType& loader, const Gemma3Config& fallback) {
  Gemma3Config config = fallback;

  int max_layer_index = -1;
  for (const auto& name : loader.GetTensorNames()) {
    int layer_index = -1;
    if (ParseLayerIndex(name, &layer_index)) {
      max_layer_index = std::max(max_layer_index, layer_index);
    }
  }
  if (max_layer_index >= 0) {
    config.n_layers = max_layer_index + 1;
  }

  const auto embed_info_or = loader.GetTensorInfo("model.embed_tokens.weight");
  if (!embed_info_or.ok()) {
    return embed_info_or.status();
  }
  const auto embed_shape_or =
      ToShape(*embed_info_or, "model.embed_tokens.weight");
  if (!embed_shape_or.ok()) {
    return embed_shape_or.status();
  }
  const auto& embed_shape = *embed_shape_or;
  if (embed_shape.size() != 2) {
    return absl::InvalidArgumentError(
        "model.embed_tokens.weight must be rank-2");
  }
  config.vocab_size = embed_shape[0];
  config.emb_dim = embed_shape[1];

  const auto gate_info_or =
      loader.GetTensorInfo("model.layers.0.mlp.gate_proj.weight");
  if (!gate_info_or.ok()) {
    return gate_info_or.status();
  }
  const auto gate_shape_or =
      ToShape(*gate_info_or, "model.layers.0.mlp.gate_proj.weight");
  if (!gate_shape_or.ok()) {
    return gate_shape_or.status();
  }
  const auto& gate_shape = *gate_shape_or;
  if (gate_shape.size() != 2) {
    return absl::InvalidArgumentError(
        "model.layers.0.mlp.gate_proj.weight must be rank-2");
  }
  config.hidden_dim = gate_shape[0];

  const auto q_proj_info_or =
      loader.GetTensorInfo("model.layers.0.self_attn.q_proj.weight");
  if (!q_proj_info_or.ok()) {
    return q_proj_info_or.status();
  }
  const auto q_proj_shape_or =
      ToShape(*q_proj_info_or, "model.layers.0.self_attn.q_proj.weight");
  if (!q_proj_shape_or.ok()) {
    return q_proj_shape_or.status();
  }
  const auto& q_proj_shape = *q_proj_shape_or;
  if (q_proj_shape.size() != 2) {
    return absl::InvalidArgumentError(
        "model.layers.0.self_attn.q_proj.weight must be rank-2");
  }

  const auto k_proj_info_or =
      loader.GetTensorInfo("model.layers.0.self_attn.k_proj.weight");
  if (!k_proj_info_or.ok()) {
    return k_proj_info_or.status();
  }
  const auto k_proj_shape_or =
      ToShape(*k_proj_info_or, "model.layers.0.self_attn.k_proj.weight");
  if (!k_proj_shape_or.ok()) {
    return k_proj_shape_or.status();
  }
  const auto& k_proj_shape = *k_proj_shape_or;
  if (k_proj_shape.size() != 2) {
    return absl::InvalidArgumentError(
        "model.layers.0.self_attn.k_proj.weight must be rank-2");
  }

  const auto q_norm_info_or =
      loader.GetTensorInfo("model.layers.0.self_attn.q_norm.weight");
  if (!q_norm_info_or.ok()) {
    return q_norm_info_or.status();
  }
  const auto q_norm_shape_or =
      ToShape(*q_norm_info_or, "model.layers.0.self_attn.q_norm.weight");
  if (!q_norm_shape_or.ok()) {
    return q_norm_shape_or.status();
  }
  const auto& q_norm_shape = *q_norm_shape_or;
  if (q_norm_shape.size() != 1) {
    return absl::InvalidArgumentError(
        "model.layers.0.self_attn.q_norm.weight must be rank-1");
  }

  config.head_dim = q_norm_shape[0];
  if (config.head_dim <= 0) {
    return absl::InvalidArgumentError("Invalid head_dim inferred from q_norm");
  }
  if (q_proj_shape[0] % config.head_dim != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("q_proj output dim ", q_proj_shape[0],
                     " is not divisible by head_dim ", config.head_dim));
  }
  if (k_proj_shape[0] % config.head_dim != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("k_proj output dim ", k_proj_shape[0],
                     " is not divisible by head_dim ", config.head_dim));
  }
  config.n_heads = q_proj_shape[0] / config.head_dim;
  config.n_kv_groups = k_proj_shape[0] / config.head_dim;
  config.query_pre_attn_scalar = static_cast<float>(config.head_dim);

  return config;
}

}  // namespace

absl::StatusOr<Gemma3ModelVariant> ParseGemma3ModelVariant(
    absl::string_view variant) {
  std::string normalized = std::string(variant);
  absl::AsciiStrToLower(&normalized);
  if (normalized == "auto" || normalized == "infer") {
    return Gemma3ModelVariant::kAuto;
  }
  if (normalized == "270m" || normalized == "gemma3_270m" ||
      normalized == "gemma-3-270m" || normalized == "gemma-3-270m-it") {
    return Gemma3ModelVariant::k270M;
  }
  if (normalized == "1b" || normalized == "gemma3_1b" ||
      normalized == "gemma-3-1b" || normalized == "gemma-3-1b-it") {
    return Gemma3ModelVariant::k1B;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported model variant: ", variant,
                   ". Expected one of: auto, 270m, 1b"));
}

absl::string_view Gemma3ModelVariantToString(Gemma3ModelVariant variant) {
  switch (variant) {
    case Gemma3ModelVariant::kAuto:
      return "auto";
    case Gemma3ModelVariant::k270M:
      return "270m";
    case Gemma3ModelVariant::k1B:
      return "1b";
  }
  return "unknown";
}

Gemma3Config GetGemma3BaseConfig(Gemma3ModelVariant variant) {
  switch (variant) {
    case Gemma3ModelVariant::k1B:
      return GetGemma3_1B_Config();
    case Gemma3ModelVariant::k270M:
    case Gemma3ModelVariant::kAuto:
      return GetGemma3_270M_Config();
  }
  return GetGemma3_270M_Config();
}

absl::StatusOr<Gemma3Config> InferGemma3ConfigFromLoader(
    const SafetensorLoader& loader, const Gemma3Config& fallback) {
  return InferGemma3ConfigFromLoaderImpl(loader, fallback);
}

absl::StatusOr<Gemma3Config> ResolveGemma3Config(const SafetensorLoader& loader,
                                                 Gemma3ModelVariant variant) {
  Gemma3Config base = GetGemma3BaseConfig(variant);
  auto inferred_or = InferGemma3ConfigFromLoader(loader, base);
  if (!inferred_or.ok()) {
    return inferred_or.status();
  }
  Gemma3Config resolved = *inferred_or;

  if (variant != Gemma3ModelVariant::kAuto) {
    const Gemma3Config expected = GetGemma3BaseConfig(variant);
    if (resolved.n_layers != expected.n_layers ||
        resolved.emb_dim != expected.emb_dim) {
      ABSL_LOG(WARNING)
          << "Resolved weights do not match requested model variant "
          << Gemma3ModelVariantToString(variant)
          << ". Using inferred configuration from weights instead.";
    }
  }

  return resolved;
}

absl::StatusOr<Gemma3Config> ResolveGemma3Config(const TfliteLoader& loader,
                                                 Gemma3ModelVariant variant) {
  if (variant == Gemma3ModelVariant::kAuto) {
    return absl::InvalidArgumentError(
        "Automatic model variant inference is not supported for TFLite models. "
        "Please specify --model_variant explicitly (e.g., 1b or 270m).");
  }
  return GetGemma3BaseConfig(variant);
}

}  // namespace litert::tensor::examples
