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

#include "tensor/examples/gemma4/gemma4_weights.h"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

namespace litert::tensor::examples::gemma4 {

absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
    int n_layers) {
  absl::flat_hash_map<std::string, std::string> mapping;

  // Embedding
  mapping["model.language_model.embed_tokens.weight"] =
      "model.embed_tokens.weight";

  // LM head.
  mapping["lm_head.weight"] = "lm_head.weight";

  // Final norm
  mapping["model.language_model.norm.weight"] = "model.norm.weight";

  // Per-layer input weights
  mapping["model.language_model.embed_tokens_per_layer.weight"] =
      "model.embed_tokens_per_layer.weight";
  mapping["model.language_model.per_layer_model_projection.weight"] =
      "model.per_layer_model_projection.weight";
  mapping["model.language_model.per_layer_projection_norm.weight"] =
      "model.per_layer_projection_norm.weight";

  // Per-layer weights
  for (int i = 0; i < n_layers; ++i) {
    std::string hf_prefix = absl::StrCat("model.language_model.layers.", i);
    std::string model_prefix = absl::StrCat("model.layers.", i);

    // Attention weights
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.v_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.v_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.o_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.o_proj.weight");

    // QK normalization
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_norm.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_norm.weight");

    // MLP weights
    mapping[absl::StrCat(hf_prefix, ".mlp.gate_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.gate_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.up_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.up_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.down_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.down_proj.weight");

    // Layer norms
    mapping[absl::StrCat(hf_prefix, ".input_layernorm.weight")] =
        absl::StrCat(model_prefix, ".input_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_attention_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_attention_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".pre_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".pre_feedforward_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_feedforward_layernorm.weight");

    // Per-layer input integration weights
    mapping[absl::StrCat(hf_prefix, ".per_layer_input_gate.weight")] =
        absl::StrCat(model_prefix, ".per_layer_input_gate.weight");
    mapping[absl::StrCat(hf_prefix, ".per_layer_projection.weight")] =
        absl::StrCat(model_prefix, ".per_layer_projection.weight");
    mapping[absl::StrCat(hf_prefix, ".post_per_layer_input_norm.weight")] =
        absl::StrCat(model_prefix, ".post_per_layer_input_norm.weight");

    // Layer Scalar (replaces Gemma3 skip_scale)
    mapping[absl::StrCat(hf_prefix, ".layer_scalar")] =
        absl::StrCat(model_prefix, ".layer_scalar");
  }

  return mapping;
}

absl::StatusOr<TensorHandle> Gemma4WeightHooks::OnNotFound(
    TensorMapping& mapping, absl::string_view model_name) {
  static constexpr absl::string_view kPerLayerModelProjectionPrefix =
      "model.layers.";
  static constexpr absl::string_view kPerLayerModelProjectionSuffix =
      ".per_layer_model_projection.weight";

  if (model_name == "lm_head.weight") {
    return mapping.Get("model.embed_tokens.weight");
  }
  if (auto [layer, str] = std::tuple(0, model_name);
      absl::ConsumePrefix(&str, kPerLayerModelProjectionPrefix) &&
      absl::ConsumeSuffix(&str, kPerLayerModelProjectionSuffix) &&
      absl::SimpleAtoi(str, &layer)) {
    return SlicePerLayerModelProjection(mapping, model_name, layer);
  }
  return TensorMappingHooks::OnNotFound(mapping, model_name);
}

absl::StatusOr<TensorHandle> Gemma4WeightHooks::SlicePerLayerModelProjection(
    TensorMapping& mapping, absl::string_view model_name, int layer) {
  static constexpr absl::string_view kPerLayerModelProjection =
      "model.per_layer_model_projection.weight";

  if (layer < 0 || layer >= config_.num_layers) {
    return absl::NotFoundError(absl::StrCat("No weight maps to ", model_name,
                                            ", the model has ",
                                            config_.num_layers, " layers."));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle proj_w,
                              mapping.Get(kPerLayerModelProjection));
  // Slicing a quantized weight would also require slicing its quantization
  // parameters.
  if (proj_w.GetQuantization() != nullptr) {
    return absl::UnimplementedError(absl::StrCat(
        "Slicing quantized ", kPerLayerModelProjection, " isn't supported."));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & proj_w_buf, proj_w.GetBuffer());
  auto proj_locked = proj_w_buf.Lock();
  const std::byte* proj_w_bytes = proj_locked.data();
  if (proj_w_bytes == nullptr) {
    return absl::InternalError(
        absl::StrCat("Null buffer data for ", kPerLayerModelProjection));
  }

  const Type type = proj_w.GetType();
  const size_t layer_w_elements =
      static_cast<size_t>(config_.per_layer_input_dim) * config_.embed_dim;
  if (layer_w_elements * BitSize(type) % 8 != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPerLayerModelProjection, " layers of type ", type,
                     " don't start on a byte boundary."));
  }
  const size_t layer_w_bytes = BufferSize(type, layer_w_elements);
  if (proj_locked.size() < (layer + 1) * layer_w_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPerLayerModelProjection, " holds ", proj_locked.size(),
                     " bytes, which is too small to slice layer ", layer));
  }

  const std::byte* layer_bytes = proj_w_bytes + layer * layer_w_bytes;
  return TensorHandle({
      .name = std::string(model_name),
      .type = type,
      .shape = {config_.per_layer_input_dim, config_.embed_dim},
      // The combined weight is stored in the checkpoint mapping, which keeps
      // its data alive for the slices that don't own their data.
      .buffer = std::make_shared<SpanCpuBuffer>(layer_bytes, layer_w_bytes),
  });
}

absl::Status FallbackBF16ToFp32Hooks::OnLoaded(absl::string_view model_name,
                                               TensorHandle& weight) {
  if (weight.GetType() != Type::kBF16) {
    return absl::OkStatus();
  }
  TRACE_EVENT(kTensorApiCategory, "FallbackBF16ToFp32");
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, weight.GetBuffer());
  std::shared_ptr<OwningCpuBuffer> fp32_buf =
      OwningCpuBuffer::Copy<Type::kFP32>(buffer.Lock().As<const bf16_t>());
  if (fp32_buf == nullptr) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Failed to allocate FP32 buffer for weight ", weight.GetName()));
  }
  weight.SetType(Type::kFP32);
  weight.SetBuffer(fp32_buf);
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4
