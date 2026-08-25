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

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::tensor::examples::gemma4 {

absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
    int n_layers) {
  absl::flat_hash_map<std::string, std::string> mapping;

  // Embedding
  mapping["model.language_model.embed_tokens.weight"] =
      "model.embed_tokens.weight";

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

}  // namespace litert::tensor::examples::gemma4
