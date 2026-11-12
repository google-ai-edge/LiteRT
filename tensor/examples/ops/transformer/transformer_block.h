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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_BLOCK_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_BLOCK_H_

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/meta/type_traits.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

template <class... Mixins>
struct TransformerBlockWeights {
  Tensor<Mixins...> pre_attn_norm_weight;
  Tensor<Mixins...> q_weight;
  Tensor<Mixins...> k_weight;
  Tensor<Mixins...> v_weight;
  Tensor<Mixins...> o_weight;
  Tensor<Mixins...> pre_ffn_norm_weight;
  Tensor<Mixins...> ffn_gate_weight;
  Tensor<Mixins...> ffn_up_weight;
  Tensor<Mixins...> ffn_down_weight;
  std::optional<Tensor<Mixins...>> q_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> k_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_attn_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_ffn_norm_weight = std::nullopt;
};

template <class... Mixins>
struct KvCache {
  Tensor<Mixins...> k;
  Tensor<Mixins...> v;
  std::optional<Tensor<Mixins...>> params = std::nullopt;
  // Active cache window size extracted for single-token decode when `params`
  // is set; defaults to the full cache size (`k.GetShape()[2]`).
  std::optional<int> local_cache_size = std::nullopt;
};

// Returns the `KvCache` for `layer_idx`, or `std::nullopt` if `layer_idx` is
// out of bounds or either cache tensor is invalid or not rank 4 with a
// positive sequence length (`shape[2] > 0`).
template <class... Mixins>
std::optional<KvCache<Mixins...>> LayerKvCache(
    const std::vector<Tensor<Mixins...>>& key_caches,
    const std::vector<Tensor<Mixins...>>& value_caches, size_t layer_idx,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> cache_params =
        std::nullopt,
    std::optional<int> local_cache_size = std::nullopt) {
  auto valid = [](const std::vector<Tensor<Mixins...>>& caches, size_t idx) {
    if (idx >= caches.size() || !caches[idx].GetStatus().ok()) {
      return false;
    }
    const Shape& shape = caches[idx].GetShape();
    return shape.size() == 4 && shape[2] > 0;
  };
  if (!valid(key_caches, layer_idx) || !valid(value_caches, layer_idx)) {
    return std::nullopt;
  }
  return KvCache<Mixins...>{
      .k = key_caches[layer_idx],
      .v = value_caches[layer_idx],
      .params = std::move(cache_params),
      .local_cache_size = local_cache_size,
  };
}

template <class... Mixins>
struct SelfAttentionOutputs {
  Tensor<Mixins...> output;
  Tensor<Mixins...> updated_kv_cache_k;
  Tensor<Mixins...> updated_kv_cache_v;
};

template <class... Mixins>
struct TransformerBlockOutputs {
  Tensor<Mixins...> output;
  Tensor<Mixins...> updated_kv_cache_k;
  Tensor<Mixins...> updated_kv_cache_v;
};

namespace internal {

template <class RopeParams, class... Mixins>
absl::StatusOr<SelfAttentionOutputs<Mixins...>> SelfAttentionImpl(
    const Tensor<Mixins...>& x, const Tensor<Mixins...>& q_weight,
    const Tensor<Mixins...>& k_weight, const Tensor<Mixins...>& v_weight,
    const Tensor<Mixins...>& o_weight,
    const std::optional<Tensor<Mixins...>>& q_norm_weight,
    const std::optional<Tensor<Mixins...>>& k_norm_weight,
    const RopeParams& rope, int num_heads, int num_kv_heads, int head_dim,
    std::optional<KvCache<Mixins...>> kv_cache,
    const Tensor<Mixins...>& attention_mask, float rms_norm_epsilon,
    std::optional<float> scale, std::optional<float> soft_cap) {
  using TensorType = Tensor<Mixins...>;

  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight));
  if constexpr (std::is_same_v<RopeParams, RopeCosSinParams<Mixins...>>) {
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(rope.cos, rope.sin));
  } else {
    LRT_TENSOR_RETURN_IF_ERROR(rope.position_ids);
  }
  if (x.GetShape().size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelfAttention expects rank 3 input [batch, seq_len, hidden_dim], "
        "got shape [",
        absl::StrJoin(x.GetShape(), ", "), "]."));
  }
  const Type datatype = x.GetType();
  if (datatype == Type::kUnknown) {
    return absl::InvalidArgumentError(
        "SelfAttention expects a typed input `x`, got Type::kUnknown.");
  }
  if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelfAttention needs num_heads to be a positive multiple of a "
        "positive num_kv_heads, got num_heads=",
        num_heads, " and num_kv_heads=", num_kv_heads, "."));
  }
  if (head_dim <= 0 || head_dim % 2 != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelfAttention RoPE needs an even head_dim, got ", head_dim, "."));
  }
  const int batch_size = x.GetShape()[0];
  const int seq_len = x.GetShape()[1];

  if (kv_cache.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(
        CheckTensors(kv_cache->k, kv_cache->v, kv_cache->params));
    const Shape& k_shape = kv_cache->k.GetShape();
    const Shape& v_shape = kv_cache->v.GetShape();
    if (k_shape.size() != 4 || k_shape != v_shape || k_shape[0] != batch_size ||
        k_shape[1] != num_kv_heads || k_shape[3] != head_dim) {
      return absl::InvalidArgumentError(absl::StrCat(
          "SelfAttention KV cache must have matching rank 4 shapes [batch=",
          batch_size, ", num_kv_heads=", num_kv_heads, ", seq_len, head_dim=",
          head_dim, "], got k shape [", absl::StrJoin(k_shape, ", "),
          "] and v shape [", absl::StrJoin(v_shape, ", "), "]."));
    }
    if (kv_cache->params.has_value()) {
      const Shape& param_shape = kv_cache->params->GetShape();
      if (param_shape.size() != 1 || param_shape[0] < 3) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention cache_params must have shape {3+} (packed cache "
            "params), got [",
            absl::StrJoin(param_shape, ", "), "]."));
      }
      if (seq_len > k_shape[2]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention with cache_params requires seq_len <= cache_size (",
            k_shape[2], "), got seq_len=", seq_len, "."));
      }
      if (seq_len == 1 && batch_size != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention decode with cache_params requires batch_size == 1, "
            "got ",
            batch_size, "."));
      }
    }
    if (kv_cache->local_cache_size.has_value()) {
      if (!kv_cache->params.has_value()) {
        return absl::InvalidArgumentError(
            "SelfAttention local_cache_size requires cache_params to be set.");
      }
      if (seq_len != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention local_cache_size is only supported for single-token "
            "decode (seq_len == 1), got seq_len=",
            seq_len, "."));
      }
      const int cache_size = k_shape[2];
      if (*kv_cache->local_cache_size <= 0 ||
          *kv_cache->local_cache_size > cache_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention local_cache_size must be in [1, cache_size=",
            cache_size, "], got ", *kv_cache->local_cache_size, "."));
      }
    }
  }

  graph::ScopeGuard scope("self_attention");
  scope.Bind("input", x)
      .Bind("q_weight", q_weight)
      .Bind("k_weight", k_weight)
      .Bind("v_weight", v_weight)
      .Bind("o_weight", o_weight)
      .Bind("q_norm_weight", q_norm_weight)
      .Bind("k_norm_weight", k_norm_weight)
      .Attr("num_heads", num_heads)
      .Attr("num_kv_heads", num_kv_heads)
      .Attr("head_dim", head_dim)
      .Attr("rms_norm_epsilon", static_cast<double>(rms_norm_epsilon));
  if constexpr (std::is_same_v<RopeParams, RopeCosSinParams<Mixins...>>) {
    scope.Bind("rope_cos", rope.cos).Bind("rope_sin", rope.sin);
  } else {
    scope.Bind("position_ids", rope.position_ids);
  }
  if (kv_cache.has_value()) {
    scope.Bind("kv_cache_k", kv_cache->k)
        .Bind("kv_cache_v", kv_cache->v)
        .Bind("cache_params", kv_cache->params);
    if (kv_cache->local_cache_size.has_value()) {
      scope.Attr("local_cache_size", *kv_cache->local_cache_size);
    }
  }
  if (attention_mask != TensorHandle::Invalid()) {
    LRT_TENSOR_RETURN_IF_ERROR(attention_mask);
    scope.Bind("attention_mask", attention_mask);
  }
  if (scale.has_value()) {
    scope.Attr("scale", static_cast<double>(*scale));
  }
  if (soft_cap.has_value()) {
    scope.Attr("soft_cap", static_cast<double>(*soft_cap));
  }

  TensorType q = FullyConnected(x, q_weight);
  TensorType k = FullyConnected(x, k_weight);
  TensorType v = FullyConnected(x, v_weight);
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(q, k, v));
  scope.Bind("q_raw", q).Bind("k_raw", k).Bind("v_raw", v);

  q = Reshape(q, {batch_size, seq_len, num_heads, head_dim});
  k = Reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
  v = Reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(q, k, v));

  q = Transpose(q, {0, 2, 1, 3});
  k = Transpose(k, {0, 2, 1, 3});
  v = Transpose(v, {0, 2, 1, 3});
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(q, k, v));
  scope.Bind("v_cache_src", v);

  if (q_norm_weight.has_value() || k_norm_weight.has_value()) {
    TensorType eps_tensor(
        {.type = datatype, .shape = {1}, .buffer = rms_norm_epsilon});
    if (q_norm_weight.has_value()) {
      q = RmsNorm(q, *q_norm_weight, eps_tensor);
    }
    if (k_norm_weight.has_value()) {
      k = RmsNorm(k, *k_norm_weight, eps_tensor);
    }
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(q, k));
  }

  q = RoPE(q, rope);
  k = RoPE(k, rope);
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(q, k));
  scope.Bind("q_rope", q).Bind("k_rope", k);

  TensorType k_for_attn = k;
  TensorType v_for_attn = v;
  TensorType updated_k = k;
  TensorType updated_v = v;
  bool adj_v = false;

  if (kv_cache.has_value() && !kv_cache->params.has_value()) {
    k_for_attn = Concatenation({kv_cache->k, k}, /*axis=*/2);
    v_for_attn = Concatenation({kv_cache->v, v}, /*axis=*/2);
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k_for_attn, v_for_attn));
  } else if (kv_cache.has_value() && kv_cache->params.has_value()) {
    const int cache_size = kv_cache->k.GetShape()[2];
    // `token_index_offset=0` and `active_tokens=1` are indices into the
    // `params` buffer (`params[0]` = write offset, `params[1]` = active token
    // limit), not literal token counts.
    std::vector<TensorType> updated_caches = AddValuesToKvCache(
        kv_cache->k, kv_cache->v, k, v, *kv_cache->params,
        /*num_of_kv_heads=*/num_kv_heads,
        /*kv_cache_batch_size=*/batch_size, /*cache_size=*/cache_size,
        /*head_dimension=*/head_dim, /*token_index_offset=*/0,
        /*active_tokens=*/1);
    updated_k = updated_caches[0];
    updated_v = updated_caches[1];

    if (seq_len == 1) {
      const int local_cache_size =
          kv_cache->local_cache_size.value_or(cache_size);
      auto [k_local, v_local] = ExtractLocalCache(
          updated_k, updated_v, *kv_cache->params, local_cache_size,
          num_kv_heads, head_dim, local_cache_size, cache_size);
      k_for_attn = k_local;
      v_for_attn = v_local;
      adj_v = true;
    }
  }
  scope.Bind("k_for_attn", k_for_attn).Bind("v_for_attn", v_for_attn);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorType attention_output,
      ScaledDotProductAttention(q, k_for_attn, v_for_attn, attention_mask,
                                scale, soft_cap, datatype, adj_v));

  TensorType context = Transpose(attention_output, {0, 2, 1, 3});
  LRT_TENSOR_RETURN_IF_ERROR(context);
  context = Reshape(context, {batch_size, seq_len, num_heads * head_dim});
  LRT_TENSOR_RETURN_IF_ERROR(context);
  scope.Bind("attn_output_flat", context);

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType attn_output,
                              FullyConnected(context, o_weight));
  scope.Bind("output", attn_output)
      .Bind("updated_kv_cache_k", updated_k)
      .Bind("updated_kv_cache_v", updated_v);

  return SelfAttentionOutputs<Mixins...>{.output = attn_output,
                                         .updated_kv_cache_k = updated_k,
                                         .updated_kv_cache_v = updated_v};
}

template <class RopeParams, class... Mixins>
absl::StatusOr<TransformerBlockOutputs<Mixins...>> TransformerBlockImpl(
    const Tensor<Mixins...>& x,
    const TransformerBlockWeights<Mixins...>& weights, const RopeParams& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<KvCache<Mixins...>> kv_cache,
    const Tensor<Mixins...>& attention_mask, float rms_norm_epsilon,
    FFNActivation activation, std::optional<float> scale,
    std::optional<float> soft_cap) {
  using TensorType = Tensor<Mixins...>;
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(
      x, weights.pre_attn_norm_weight, weights.q_weight, weights.k_weight,
      weights.v_weight, weights.o_weight, weights.pre_ffn_norm_weight,
      weights.ffn_gate_weight, weights.ffn_up_weight, weights.ffn_down_weight,
      weights.q_norm_weight, weights.k_norm_weight,
      weights.post_attn_norm_weight, weights.post_ffn_norm_weight));
  const Type datatype = x.GetType();
  if (datatype == Type::kUnknown) {
    return absl::InvalidArgumentError(
        "TransformerBlock expects a typed input `x`, got Type::kUnknown.");
  }

  graph::ScopeGuard scope("transformer_block");
  scope.Bind("input", x)
      .Bind("pre_attn_norm_weight", weights.pre_attn_norm_weight)
      .Bind("q_weight", weights.q_weight)
      .Bind("k_weight", weights.k_weight)
      .Bind("v_weight", weights.v_weight)
      .Bind("o_weight", weights.o_weight)
      .Bind("pre_ffn_norm_weight", weights.pre_ffn_norm_weight)
      .Bind("ffn_gate_weight", weights.ffn_gate_weight)
      .Bind("ffn_up_weight", weights.ffn_up_weight)
      .Bind("ffn_down_weight", weights.ffn_down_weight)
      .Bind("q_norm_weight", weights.q_norm_weight)
      .Bind("k_norm_weight", weights.k_norm_weight)
      .Bind("post_attn_norm_weight", weights.post_attn_norm_weight)
      .Bind("post_ffn_norm_weight", weights.post_ffn_norm_weight)
      .Attr("num_heads", num_heads)
      .Attr("num_kv_heads", num_kv_heads)
      .Attr("head_dim", head_dim)
      .Attr("rms_norm_epsilon", static_cast<double>(rms_norm_epsilon))
      .Attr("activation", activation == FFNActivation::kGeluApproximate
                              ? "gelu_approximate"
                              : "swish");
  if (kv_cache.has_value()) {
    scope.Bind("kv_cache_k", kv_cache->k)
        .Bind("kv_cache_v", kv_cache->v)
        .Bind("cache_params", kv_cache->params);
    if (kv_cache->local_cache_size.has_value()) {
      scope.Attr("local_cache_size", *kv_cache->local_cache_size);
    }
  }
  if (attention_mask != TensorHandle::Invalid()) {
    LRT_TENSOR_RETURN_IF_ERROR(attention_mask);
    scope.Bind("attention_mask", attention_mask);
  }
  if (scale.has_value()) {
    scope.Attr("scale", static_cast<double>(*scale));
  }
  if (soft_cap.has_value()) {
    scope.Attr("soft_cap", static_cast<double>(*soft_cap));
  }

  TensorType eps_tensor(
      {.type = datatype, .shape = {1}, .buffer = rms_norm_epsilon});

  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorType attn_input,
      RmsNorm(x, weights.pre_attn_norm_weight, eps_tensor));
  scope.Bind("normed_input", attn_input);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      SelfAttentionOutputs<Mixins...> sa_outputs,
      SelfAttentionImpl(attn_input, weights.q_weight, weights.k_weight,
                        weights.v_weight, weights.o_weight,
                        weights.q_norm_weight, weights.k_norm_weight, rope,
                        num_heads, num_kv_heads, head_dim, kv_cache,
                        attention_mask, rms_norm_epsilon, scale, soft_cap));
  scope.Bind("attn_output", sa_outputs.output);

  TensorType attn_output = sa_outputs.output;
  if (weights.post_attn_norm_weight.has_value()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        attn_output,
        RmsNorm(attn_output, *weights.post_attn_norm_weight, eps_tensor));
    scope.Bind("post_attn_norm_output", attn_output);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType x_after_attn, Add(x, attn_output));
  scope.Bind("post_attn", x_after_attn);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorType ffn_input,
      RmsNorm(x_after_attn, weights.pre_ffn_norm_weight, eps_tensor));
  scope.Bind("ffn_input", ffn_input);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorType ffn_output,
      FeedForward(ffn_input, weights.ffn_gate_weight, weights.ffn_up_weight,
                  weights.ffn_down_weight, activation));

  if (weights.post_ffn_norm_weight.has_value()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        ffn_output,
        RmsNorm(ffn_output, *weights.post_ffn_norm_weight, eps_tensor));
    scope.Bind("post_ffn_norm_output", ffn_output);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType output, Add(x_after_attn, ffn_output));
  scope.Bind("output", output)
      .Bind("updated_kv_cache_k", sa_outputs.updated_kv_cache_k)
      .Bind("updated_kv_cache_v", sa_outputs.updated_kv_cache_v);

  return TransformerBlockOutputs<Mixins...>{
      .output = output,
      .updated_kv_cache_k = sa_outputs.updated_kv_cache_k,
      .updated_kv_cache_v = sa_outputs.updated_kv_cache_v};
}

}  // namespace internal

// Multi-head self-attention with optional QK-norm, RoPE, and optional KV cache.
//
// x: [batch_size, seq_len, hidden_dim]; `output` has the same shape.
// q_norm_weight, k_norm_weight: optional per-head RMSNorm weights
//   ([head_dim]); if unset, Q/K normalization is skipped.
// num_heads: must be a positive multiple of `num_kv_heads`.
// head_dim: must be even and positive.
// kv_cache:
//   - unset: attends over this step's k and v; returns them as
//     `updated_kv_cache_*` ([batch, num_kv_heads, seq_len, head_dim]).
//   - set without `params`: prepends `k` and `v` along axis 2 before
//     attending; returns this step's k and v.
//   - set with `params` (shape {3+}): writes this step's `k` and `v` into the
//     cache via `params` (index 0 = write offset, index 1 = active token
//     limit). When `seq_len == 1`, extracts a window of size
//     `local_cache_size.value_or(k.GetShape()[2])` and attends over it; when
//     `seq_len > 1` (initial prefill into an empty cache), attends only over
//     this step's `k` and `v` and ignores prior cache contents (chunked
//     prefill with `seq_len > 1` is not supported). Returns the updated caches.
// rope: RoPE parameters (`RopeCosSinParams` for decomposed cos/sin RoPE, or
//   `RotaryEmbeddingParams` for the fused `RotaryEmbedding` op).
template <class... Mixins>
absl::StatusOr<SelfAttentionOutputs<Mixins...>> SelfAttention(
    const Tensor<Mixins...>& x, const Tensor<Mixins...>& q_weight,
    const Tensor<Mixins...>& k_weight, const Tensor<Mixins...>& v_weight,
    const Tensor<Mixins...>& o_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> q_norm_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> k_norm_weight,
    const absl::type_identity_t<RopeCosSinParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f, std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt) {
  return internal::SelfAttentionImpl(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
      rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, scale, soft_cap);
}

template <class... Mixins>
absl::StatusOr<SelfAttentionOutputs<Mixins...>> SelfAttention(
    const Tensor<Mixins...>& x, const Tensor<Mixins...>& q_weight,
    const Tensor<Mixins...>& k_weight, const Tensor<Mixins...>& v_weight,
    const Tensor<Mixins...>& o_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> q_norm_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> k_norm_weight,
    const absl::type_identity_t<RotaryEmbeddingParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f, std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt) {
  return internal::SelfAttentionImpl(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
      rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, scale, soft_cap);
}

// Pre-norm transformer block (RMSNorm -> SelfAttention -> [RMSNorm] -> residual
// -> RMSNorm -> FeedForward -> [RMSNorm] -> residual) with optional QK,
// post-attention, and post-FFN norms.
template <class... Mixins>
absl::StatusOr<TransformerBlockOutputs<Mixins...>> TransformerBlock(
    const Tensor<Mixins...>& x,
    const TransformerBlockWeights<Mixins...>& weights,
    const absl::type_identity_t<RopeCosSinParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f,
    FFNActivation activation = FFNActivation::kGeluApproximate,
    std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt) {
  return internal::TransformerBlockImpl(
      x, weights, rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, activation, scale, soft_cap);
}

template <class... Mixins>
absl::StatusOr<TransformerBlockOutputs<Mixins...>> TransformerBlock(
    const Tensor<Mixins...>& x,
    const TransformerBlockWeights<Mixins...>& weights,
    const absl::type_identity_t<RotaryEmbeddingParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f,
    FFNActivation activation = FFNActivation::kGeluApproximate,
    std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt) {
  return internal::TransformerBlockImpl(
      x, weights, rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, activation, scale, soft_cap);
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_BLOCK_H_
