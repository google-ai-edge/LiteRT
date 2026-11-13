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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_GRAPH_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/ops/transformer/transformer_block.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4 {

template <class... Mixins>
struct Gemma4Inputs {
  Tensor<Mixins...> embedded_input = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> per_layer_token_embeddings;
  Tensor<Mixins...> global_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> sliding_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_sin = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_sin = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;
};

template <class... Mixins>
struct Gemma4Outputs {
  Tensor<Mixins...> logits;
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;

  // Returns all valid output handles in a single vector, skipping
  // `TensorHandle::Invalid()` entries (such as shared-KV layers that do not
  // produce updated caches).
  std::vector<TensorHandle> GetAllHandles() {
    std::vector<TensorHandle> output_handles;
    output_handles.reserve(1 + key_caches.size() + value_caches.size());
    if (logits != TensorHandle::Invalid()) {
      output_handles.push_back(logits);
    }
    for (const auto& k : key_caches) {
      if (k != TensorHandle::Invalid()) {
        output_handles.push_back(k);
      }
    }
    for (const auto& v : value_caches) {
      if (v != TensorHandle::Invalid()) {
        output_handles.push_back(v);
      }
    }
    return output_handles;
  }
};

inline std::vector<int> GetKvCacheSharingPatterns(const Config& config) {
  if (config.num_layers <= 0) {
    return {};
  }
  std::vector<int> patterns(config.num_layers);
  std::iota(patterns.begin(), patterns.end(), 0);

  const int num_shared_layers = std::clamp(
      static_cast<int>(std::lround(
          static_cast<double>(config.frac_shared_layers) * config.num_layers)),
      0, config.num_layers);
  const int num_unshared_layers = config.num_layers - num_shared_layers;
  for (int i = num_unshared_layers; i < config.num_layers; ++i) {
    const Config::LayerType layer_type = config.GetLayerType(i);
    const bool should_share =
        (layer_type == Config::LayerType::kLocalSliding &&
         config.share_local) ||
        (layer_type == Config::LayerType::kGlobal && config.share_global);
    if (!should_share) {
      continue;
    }
    for (int j = num_unshared_layers - 1; j >= 0; --j) {
      if (config.GetLayerType(j) == layer_type) {
        patterns[i] = j;
        break;
      }
    }
  }
  return patterns;
}

namespace internal {

// Resolves the `TransformerBlockWeights` for `layer_prefix` from `weights`.
// K/V projection and normalization weights are omitted when `is_shared` is
// true; per-layer input weights are loaded iff `has_per_layer_input` is true.
template <class... Mixins>
absl::StatusOr<TransformerBlockWeights<Mixins...>> BuildGemma4BlockWeights(
    TensorMapping& weights, absl::string_view layer_prefix,
    const Config& config, bool is_shared, bool has_per_layer_input) {
  const std::string attn_prefix = absl::StrCat(layer_prefix, ".self_attn.");
  const std::string mlp_prefix = absl::StrCat(layer_prefix, ".mlp.");

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> pre_attn_norm_weight,
      weights.Get(absl::StrCat(layer_prefix, ".input_layernorm.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> q_weight,
      weights.Get(absl::StrCat(attn_prefix, "q_proj.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> o_weight,
      weights.Get(absl::StrCat(attn_prefix, "o_proj.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> q_norm_weight,
      weights.Get(absl::StrCat(attn_prefix, "q_norm.weight")));

  Tensor<Mixins...> k_weight = TensorHandle::Invalid();
  Tensor<Mixins...> v_weight = TensorHandle::Invalid();
  std::optional<Tensor<Mixins...>> k_norm_weight = std::nullopt;
  if (!is_shared) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        k_weight, weights.Get(absl::StrCat(attn_prefix, "k_proj.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        v_weight, weights.Get(absl::StrCat(attn_prefix, "v_proj.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> k_norm,
        weights.Get(absl::StrCat(attn_prefix, "k_norm.weight")));
    k_norm_weight = std::move(k_norm);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> pre_ffn_norm_weight,
      weights.Get(
          absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> ffn_gate_weight,
      weights.Get(absl::StrCat(mlp_prefix, "gate_proj.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> ffn_up_weight,
      weights.Get(absl::StrCat(mlp_prefix, "up_proj.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> ffn_down_weight,
      weights.Get(absl::StrCat(mlp_prefix, "down_proj.weight")));

  std::optional<Tensor<Mixins...>> post_attn_norm_weight = std::nullopt;
  if (config.use_post_attn_norm) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> post_attn_norm,
        weights.Get(
            absl::StrCat(layer_prefix, ".post_attention_layernorm.weight")));
    post_attn_norm_weight = std::move(post_attn_norm);
  }

  std::optional<Tensor<Mixins...>> post_ffn_norm_weight = std::nullopt;
  if (config.use_post_ffw_norm) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> post_ffn_norm,
        weights.Get(
            absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight")));
    post_ffn_norm_weight = std::move(post_ffn_norm);
  }

  std::optional<Tensor<Mixins...>> per_layer_input_gate_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> per_layer_projection_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_per_layer_input_norm_weight =
      std::nullopt;
  if (has_per_layer_input) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> ple_gate,
        weights.Get(
            absl::StrCat(layer_prefix, ".per_layer_input_gate.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> ple_proj,
        weights.Get(
            absl::StrCat(layer_prefix, ".per_layer_projection.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> post_ple_norm,
        weights.Get(
            absl::StrCat(layer_prefix, ".post_per_layer_input_norm.weight")));
    per_layer_input_gate_weight = std::move(ple_gate);
    per_layer_projection_weight = std::move(ple_proj);
    post_per_layer_input_norm_weight = std::move(post_ple_norm);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> layer_scalar,
      weights.Get(absl::StrCat(layer_prefix, ".layer_scalar")));

  return TransformerBlockWeights<Mixins...>{
      .pre_attn_norm_weight = std::move(pre_attn_norm_weight),
      .q_weight = std::move(q_weight),
      .k_weight = std::move(k_weight),
      .v_weight = std::move(v_weight),
      .o_weight = std::move(o_weight),
      .pre_ffn_norm_weight = std::move(pre_ffn_norm_weight),
      .ffn_gate_weight = std::move(ffn_gate_weight),
      .ffn_up_weight = std::move(ffn_up_weight),
      .ffn_down_weight = std::move(ffn_down_weight),
      .q_norm_weight = std::move(q_norm_weight),
      .k_norm_weight = std::move(k_norm_weight),
      .post_attn_norm_weight = std::move(post_attn_norm_weight),
      .post_ffn_norm_weight = std::move(post_ffn_norm_weight),
      .per_layer_input_gate_weight = std::move(per_layer_input_gate_weight),
      .per_layer_projection_weight = std::move(per_layer_projection_weight),
      .post_per_layer_input_norm_weight =
          std::move(post_per_layer_input_norm_weight),
      .layer_scalar = std::move(layer_scalar),
  };
}

}  // namespace internal

template <class... Mixins>
absl::StatusOr<Gemma4Outputs<Mixins...>> BuildGemma4Graph(
    const Gemma4Inputs<Mixins...>& inputs, TensorMapping& weights,
    const Config& config) {
  if (config.num_layers <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BuildGemma4Graph requires config.num_layers > 0, got ",
        config.num_layers, "."));
  }
  LRT_TENSOR_RETURN_IF_ERROR(inputs.embedded_input);
  const Shape& emb_shape = inputs.embedded_input.GetShape();
  if (emb_shape.size() != 3 || emb_shape[2] != config.embed_dim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BuildGemma4Graph expects rank 3 embedded_input [batch, seq_len, "
        "embed_dim=",
        config.embed_dim, "], got [", absl::StrJoin(emb_shape, ", "), "]."));
  }
  const Type datatype = inputs.embedded_input.GetType();
  if (datatype == Type::kUnknown) {
    return absl::InvalidArgumentError(
        "BuildGemma4Graph expects a typed embedded_input, got Type::kUnknown.");
  }
  const bool has_per_layer_input = config.per_layer_input_dim > 0;
  if (has_per_layer_input) {
    if (inputs.per_layer_token_embeddings.size() !=
        static_cast<size_t>(config.num_layers)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "BuildGemma4Graph requires per_layer_token_embeddings of size "
          "num_layers=",
          config.num_layers, " when per_layer_input_dim=",
          config.per_layer_input_dim, ", got ",
          inputs.per_layer_token_embeddings.size(), "."));
    }
    const Shape expected_tok_emb_shape = {emb_shape[0], emb_shape[1],
                                          config.per_layer_input_dim};
    for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
      const Tensor<Mixins...>& layer_tok_emb =
          inputs.per_layer_token_embeddings[layer_idx];
      LRT_TENSOR_RETURN_IF_ERROR(layer_tok_emb);
      const Shape& tok_emb_shape = layer_tok_emb.GetShape();
      if (tok_emb_shape != expected_tok_emb_shape) {
        return absl::InvalidArgumentError(absl::StrCat(
            "BuildGemma4Graph expects per_layer_token_embeddings[", layer_idx,
            "] to have shape [", emb_shape[0], ", ", emb_shape[1], ", ",
            config.per_layer_input_dim, "], got [",
            absl::StrJoin(tok_emb_shape, ", "), "]."));
      }
    }
  }
  ABSL_VLOG(4) << "Building Gemma4 Graph, seq_len=" << emb_shape[1];

  Tensor<Mixins...> emb_scale_tensor(
      {.type = datatype,
       .shape = {1},
       .buffer = std::sqrt(static_cast<float>(config.embed_dim))});
  LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> hidden_states,
                              Mul(inputs.embedded_input, emb_scale_tensor));
  Tensor<Mixins...> eps_tensor(
      {.type = datatype, .shape = {1}, .buffer = config.rms_norm_eps});
  std::optional<Tensor<Mixins...>> sqrt_per_layer_dim_tensor;
  std::optional<Tensor<Mixins...>> rsqrt_2_tensor;
  std::optional<Tensor<Mixins...>> per_layer_projection_norm_weight;
  if (has_per_layer_input) {
    sqrt_per_layer_dim_tensor = Tensor<Mixins...>(
        {.type = datatype,
         .shape = {1},
         .buffer = std::sqrt(static_cast<float>(config.per_layer_input_dim))});
    rsqrt_2_tensor = Tensor<Mixins...>(
        {.type = datatype, .shape = {1}, .buffer = 1.0f / std::sqrt(2.0f)});
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor<Mixins...> norm_w,
        weights.Get("model.per_layer_projection_norm.weight"));
    if (norm_w.GetShape() != Shape({config.per_layer_input_dim})) {
      return absl::InvalidArgumentError(absl::StrCat(
          "BuildGemma4Graph expects model.per_layer_projection_norm.weight "
          "to have shape [",
          config.per_layer_input_dim, "], got [",
          absl::StrJoin(norm_w.GetShape(), ", "), "]."));
    }
    per_layer_projection_norm_weight = std::move(norm_w);
  }

  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  updated_key_caches.reserve(config.num_layers);
  updated_value_caches.reserve(config.num_layers);

  const std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config);
  std::vector<SharedKv<Mixins...>> computed_kv(
      config.num_layers,
      SharedKv<Mixins...>{
          .k = Tensor<Mixins...>(TensorHandle::Invalid()),
          .v = Tensor<Mixins...>(TensorHandle::Invalid()),
          .v_transposed = false,
      });

  const RopeCosSinParams<Mixins...> global_rope{
      .cos = inputs.rope_global_cos,
      .sin = inputs.rope_global_sin,
  };
  const RopeCosSinParams<Mixins...> local_rope{
      .cos = inputs.rope_local_cos,
      .sin = inputs.rope_local_sin,
  };

  for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
    ABSL_VLOG(4) << "  Layer " << layer_idx << " ("
                 << config.GetLayerType(layer_idx) << ")";

    const std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    graph::ScopeGuard layer_scope("transformer_layer",
                                  absl::StrCat("layer_", layer_idx));
    layer_scope.Attr("layer_idx", layer_idx);

    const int shared_idx = sharing_patterns[layer_idx];
    const bool is_shared = (shared_idx != layer_idx);
    const bool is_global =
        config.GetLayerType(layer_idx) == Config::LayerType::kGlobal;
    const int head_dim = is_global ? config.global_key_size : config.head_dim;

    Tensor<Mixins...> attention_mask = is_global
                                           ? inputs.global_attention_mask
                                           : inputs.sliding_attention_mask;

    std::optional<KvCache<Mixins...>> kv_cache = std::nullopt;
    std::optional<SharedKv<Mixins...>> shared_kv = std::nullopt;
    if (is_shared) {
      shared_kv = computed_kv[shared_idx];
    } else {
      kv_cache =
          LayerKvCache(inputs.key_caches, inputs.value_caches, layer_idx);
    }

    std::optional<Tensor<Mixins...>> per_layer_input = std::nullopt;
    if (has_per_layer_input) {
      const Tensor<Mixins...>& layer_tok_emb =
          inputs.per_layer_token_embeddings[layer_idx];

      LRT_TENSOR_ASSIGN_OR_RETURN(
          Tensor<Mixins...> proj_w,
          weights.Get(absl::StrCat(layer_prefix,
                                   ".per_layer_model_projection.weight")));
      if (proj_w.GetShape() !=
          Shape({config.per_layer_input_dim, config.embed_dim})) {
        return absl::InvalidArgumentError(absl::StrCat(
            "BuildGemma4Graph expects ", layer_prefix,
            ".per_layer_model_projection.weight to have shape [",
            config.per_layer_input_dim, ", ", config.embed_dim, "], got [",
            absl::StrJoin(proj_w.GetShape(), ", "), "]."));
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(
          Tensor<Mixins...> proj_out,
          FullyConnected(inputs.embedded_input, proj_w));

      LRT_TENSOR_ASSIGN_OR_RETURN(
          Tensor<Mixins...> normed_proj,
          RmsNorm(proj_out, *per_layer_projection_norm_weight, eps_tensor));

      LRT_TENSOR_ASSIGN_OR_RETURN(
          Tensor<Mixins...> scaled_tok_emb,
          Mul(layer_tok_emb, *sqrt_per_layer_dim_tensor));

      LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> sum_emb,
                                  Add(normed_proj, scaled_tok_emb));

      LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> ple,
                                  Mul(sum_emb, *rsqrt_2_tensor));
      per_layer_input = std::move(ple);
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(
        TransformerBlockWeights<Mixins...> block_weights,
        internal::BuildGemma4BlockWeights<Mixins...>(
            weights, layer_prefix, config, is_shared, has_per_layer_input));

    // Gemma 4 replaces 1/sqrt(head_dim) scaling with QK-norm (`scale = 1.0f`)
    // and normalizes V per head without a learned scale (`v_norm = true`).
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TransformerBlockOutputs<Mixins...> block_out,
        TransformerBlock(hidden_states, block_weights,
                         is_global ? global_rope : local_rope, config.num_heads,
                         config.num_kv_heads, head_dim, std::move(kv_cache),
                         attention_mask, config.rms_norm_eps,
                         FFNActivation::kGeluApproximate, /*scale=*/1.0f,
                         config.attn_logits_soft_cap, /*v_norm=*/true,
                         std::move(shared_kv), std::move(per_layer_input)));

    hidden_states = block_out.output;
    computed_kv[layer_idx] = SharedKv<Mixins...>{
        .k = block_out.k_for_attn,
        .v = block_out.v_for_attn,
        .v_transposed = block_out.v_transposed,
    };
    updated_key_caches.push_back(block_out.updated_kv_cache_k);
    updated_value_caches.push_back(block_out.updated_kv_cache_v);
  }

  // Final RMSNorm
  LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> final_norm_scale,
                              weights.Get("model.norm.weight"));
  if (final_norm_scale.GetShape() != Shape({config.embed_dim})) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BuildGemma4Graph expects model.norm.weight to have shape [embed_dim=",
        config.embed_dim, "], got [",
        absl::StrJoin(final_norm_scale.GetShape(), ", "), "]."));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor<Mixins...> final_output,
      RmsNorm(hidden_states, final_norm_scale, eps_tensor));

  // LM Head.
  LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> lm_head,
                              weights.Get("lm_head.weight"));
  if (lm_head.GetShape() != Shape({config.vocab_size, config.embed_dim})) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BuildGemma4Graph expects lm_head.weight to have shape [vocab_size=",
        config.vocab_size, ", embed_dim=", config.embed_dim, "], got [",
        absl::StrJoin(lm_head.GetShape(), ", "), "]."));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> logits,
                              FullyConnected(final_output, lm_head));

  // Logits Soft Capping
  if (config.final_logit_softcap > 0.0f) {
    Tensor<Mixins...> soft_cap_tensor(
        {.type = logits.GetType(),
         .shape = {1},
         .buffer = config.final_logit_softcap});
    LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> scaled_logits,
                                Div(logits, soft_cap_tensor));
    LRT_TENSOR_ASSIGN_OR_RETURN(Tensor<Mixins...> capped_logits,
                                Tanh(scaled_logits));
    LRT_TENSOR_ASSIGN_OR_RETURN(logits, Mul(capped_logits, soft_cap_tensor));
  }

  return Gemma4Outputs<Mixins...>{logits, updated_key_caches,
                                  updated_value_caches};
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_GRAPH_H_
