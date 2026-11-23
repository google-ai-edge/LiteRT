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
#include "absl/strings/string_view.h"  // from @com_google_absl
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
  Tensor<Mixins...> k_weight = TensorHandle::Invalid();
  Tensor<Mixins...> v_weight = TensorHandle::Invalid();
  Tensor<Mixins...> o_weight;
  Tensor<Mixins...> pre_ffn_norm_weight;
  Tensor<Mixins...> ffn_gate_weight;
  Tensor<Mixins...> ffn_up_weight;
  Tensor<Mixins...> ffn_down_weight;
  std::optional<Tensor<Mixins...>> q_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> k_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_attn_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_ffn_norm_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> per_layer_input_gate_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> per_layer_projection_weight = std::nullopt;
  std::optional<Tensor<Mixins...>> post_per_layer_input_norm_weight =
      std::nullopt;
  std::optional<Tensor<Mixins...>> layer_scalar = std::nullopt;
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
struct SharedKv {
  // Post-norm, post-RoPE, post-cache keys with shape
  // [batch, num_kv_heads, seq_k, head_dim].
  Tensor<Mixins...> k;
  // Post-norm, post-cache values with shape:
  // - !v_transposed: [batch, num_kv_heads, seq_k, head_dim]
  // - v_transposed:  [batch, num_kv_heads, head_dim, seq_k]
  Tensor<Mixins...> v;
  // True when `v` has transposed `[batch, num_kv_heads, head_dim, seq_k]`
  // layout (e.g. from single-token decode `ExtractLocalCache`).
  bool v_transposed = false;
};

template <class... Mixins>
struct SelfAttentionOutputs {
  Tensor<Mixins...> output;
  Tensor<Mixins...> updated_kv_cache_k;
  Tensor<Mixins...> updated_kv_cache_v;
  Tensor<Mixins...> k_for_attn;
  Tensor<Mixins...> v_for_attn;
  bool v_transposed = false;
};

template <class... Mixins>
struct TransformerBlockOutputs {
  Tensor<Mixins...> output;
  Tensor<Mixins...> updated_kv_cache_k;
  Tensor<Mixins...> updated_kv_cache_v;
  Tensor<Mixins...> k_for_attn;
  Tensor<Mixins...> v_for_attn;
  bool v_transposed = false;
};

namespace internal {

// Validates tensor statuses, ranks, datatypes, head counts, KV cache, and
// shared-KV shapes before `SelfAttentionImpl` constructs graph ops.
template <class RopeParams, class... Mixins>
absl::Status ValidateSelfAttentionInputs(
    const Tensor<Mixins...>& x, const Tensor<Mixins...>& q_weight,
    const Tensor<Mixins...>& k_weight, const Tensor<Mixins...>& v_weight,
    const Tensor<Mixins...>& o_weight,
    const std::optional<Tensor<Mixins...>>& q_norm_weight,
    const std::optional<Tensor<Mixins...>>& k_norm_weight,
    const RopeParams& rope, int num_heads, int num_kv_heads, int head_dim,
    const std::optional<KvCache<Mixins...>>& kv_cache,
    const std::optional<SharedKv<Mixins...>>& shared_kv) {
  LRT_TENSOR_RETURN_IF_ERROR(
      CheckTensors(x, q_weight, o_weight, q_norm_weight));
  if (!shared_kv.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k_weight, v_weight, k_norm_weight));
  }
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
  if (x.GetType() == Type::kUnknown) {
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
  if (q_norm_weight.has_value() &&
      q_norm_weight->GetShape() != Shape({head_dim})) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelfAttention q_norm_weight must have shape [head_dim=", head_dim,
        "], got [", absl::StrJoin(q_norm_weight->GetShape(), ", "), "]."));
  }
  if (!shared_kv.has_value() && k_norm_weight.has_value() &&
      k_norm_weight->GetShape() != Shape({head_dim})) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SelfAttention k_norm_weight must have shape [head_dim=", head_dim,
        "], got [", absl::StrJoin(k_norm_weight->GetShape(), ", "), "]."));
  }
  const int batch_size = x.GetShape()[0];
  const int seq_len = x.GetShape()[1];

  if (kv_cache.has_value() && shared_kv.has_value()) {
    return absl::InvalidArgumentError(
        "SelfAttention cannot use both kv_cache and shared_kv.");
  }

  if (shared_kv.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(shared_kv->k, shared_kv->v));
    const Shape& k_shape = shared_kv->k.GetShape();
    const Shape& v_shape = shared_kv->v.GetShape();
    const bool k_valid = k_shape.size() == 4 && k_shape[0] == batch_size &&
                         k_shape[1] == num_kv_heads && k_shape[2] > 0 &&
                         k_shape[3] == head_dim;
    const Shape expected_v_shape =
        shared_kv->v_transposed
            ? Shape({batch_size, num_kv_heads, head_dim,
                     k_valid ? k_shape[2] : 0})
            : k_shape;
    if (!k_valid || v_shape != expected_v_shape) {
      if (!shared_kv->v_transposed) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SelfAttention shared_kv must have matching rank 4 shapes [batch=",
            batch_size, ", num_kv_heads=", num_kv_heads, ", seq_len, head_dim=",
            head_dim, "], got k shape [", absl::StrJoin(k_shape, ", "),
            "] and v shape [", absl::StrJoin(v_shape, ", "), "]."));
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "SelfAttention shared_kv with v_transposed must have rank 4 k shape "
          "[batch=",
          batch_size, ", num_kv_heads=", num_kv_heads, ", seq_len, head_dim=",
          head_dim,
          "] and transposed v shape [batch, num_kv_heads, head_dim, seq_len], "
          "got k shape [",
          absl::StrJoin(k_shape, ", "), "] and v shape [",
          absl::StrJoin(v_shape, ", "), "]."));
    }
  }

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
      if (kv_cache->params->GetType() != Type::kI32) {
        return absl::InvalidArgumentError(
            "SelfAttention cache_params must have element type Type::kI32.");
      }
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
  return absl::OkStatus();
}

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
    std::optional<float> scale, std::optional<float> soft_cap, bool v_norm,
    std::optional<SharedKv<Mixins...>> shared_kv) {
  using TensorType = Tensor<Mixins...>;

  LRT_TENSOR_RETURN_IF_ERROR(ValidateSelfAttentionInputs(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
      rope, num_heads, num_kv_heads, head_dim, kv_cache, shared_kv));
  const Type datatype = x.GetType();
  const int batch_size = x.GetShape()[0];
  const int seq_len = x.GetShape()[1];

  graph::ScopeGuard scope("self_attention");
  scope.Bind("input", x)
      .Bind("q_weight", q_weight)
      .Bind("o_weight", o_weight)
      .Bind("q_norm_weight", q_norm_weight)
      .Attr("num_heads", num_heads)
      .Attr("num_kv_heads", num_kv_heads)
      .Attr("head_dim", head_dim)
      .Attr("rms_norm_epsilon", static_cast<double>(rms_norm_epsilon))
      .Attr("v_norm", v_norm);
  if (!shared_kv.has_value()) {
    scope.Bind("k_weight", k_weight)
        .Bind("v_weight", v_weight)
        .Bind("k_norm_weight", k_norm_weight);
  } else {
    scope.Bind("shared_kv_k", shared_kv->k)
        .Bind("shared_kv_v", shared_kv->v)
        .Attr("shared_v_transposed", shared_kv->v_transposed);
  }
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

  std::optional<TensorType> eps_tensor;
  if (q_norm_weight.has_value() ||
      (!shared_kv.has_value() && (k_norm_weight.has_value() || v_norm))) {
    eps_tensor = TensorType(
        {.type = datatype, .shape = {1}, .buffer = rms_norm_epsilon});
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType q, FullyConnected(x, q_weight));
  scope.Bind("q_raw", q);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      q, Reshape(q, {batch_size, seq_len, num_heads, head_dim}));
  LRT_TENSOR_ASSIGN_OR_RETURN(q, Transpose(q, {0, 2, 1, 3}));
  if (q_norm_weight.has_value()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(q, RmsNorm(q, *q_norm_weight, *eps_tensor));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(q, RoPE(q, rope));
  scope.Bind("q_rope", q);

  TensorType k_for_attn;
  TensorType v_for_attn;
  TensorType updated_k;
  TensorType updated_v;
  bool adj_v = false;

  if (shared_kv.has_value()) {
    k_for_attn = shared_kv->k;
    v_for_attn = shared_kv->v;
    updated_k = TensorType(TensorHandle::Invalid());
    updated_v = TensorType(TensorHandle::Invalid());
    adj_v = shared_kv->v_transposed;
  } else {
    TensorType k = FullyConnected(x, k_weight);
    TensorType v = FullyConnected(x, v_weight);
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k, v));
    scope.Bind("k_raw", k).Bind("v_raw", v);

    k = Reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
    v = Reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k, v));

    k = Transpose(k, {0, 2, 1, 3});
    v = Transpose(v, {0, 2, 1, 3});
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k, v));

    if (k_norm_weight.has_value()) {
      LRT_TENSOR_ASSIGN_OR_RETURN(k, RmsNorm(k, *k_norm_weight, *eps_tensor));
    }
    if (v_norm) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          v, RmsNorm(v, TensorType(TensorHandle::Invalid()), *eps_tensor));
    }
    scope.Bind("v_cache_src", v);

    LRT_TENSOR_ASSIGN_OR_RETURN(k, RoPE(k, rope));
    scope.Bind("k_rope", k);

    k_for_attn = k;
    v_for_attn = v;
    updated_k = k;
    updated_v = v;

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
      if (updated_caches.size() != 2) {
        return absl::InternalError(absl::StrCat(
            "AddValuesToKvCache expected 2 outputs, got ",
            updated_caches.size(), "."));
      }
      updated_k = updated_caches[0];
      updated_v = updated_caches[1];
      LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(updated_k, updated_v));

      if (seq_len == 1) {
        const int local_cache_size =
            kv_cache->local_cache_size.value_or(cache_size);
        auto [k_local, v_local] = ExtractLocalCache(
            updated_k, updated_v, *kv_cache->params, local_cache_size,
            num_kv_heads, head_dim, local_cache_size, cache_size);
        LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(k_local, v_local));
        k_for_attn = k_local;
        v_for_attn = v_local;
        adj_v = true;
      }
    }
  }
  scope.Bind("k_for_attn", k_for_attn).Bind("v_for_attn", v_for_attn);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorType attention_output,
      ScaledDotProductAttention(q, k_for_attn, v_for_attn, attention_mask,
                                scale, soft_cap, datatype, adj_v));

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType context,
                              Transpose(attention_output, {0, 2, 1, 3}));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      context, Reshape(context, {batch_size, seq_len, num_heads * head_dim}));
  scope.Bind("attn_output_flat", context);

  LRT_TENSOR_ASSIGN_OR_RETURN(TensorType attn_output,
                              FullyConnected(context, o_weight));
  scope.Bind("output", attn_output);
  if (updated_k != TensorHandle::Invalid()) {
    scope.Bind("updated_kv_cache_k", updated_k)
        .Bind("updated_kv_cache_v", updated_v);
  }

  return SelfAttentionOutputs<Mixins...>{.output = attn_output,
                                         .updated_kv_cache_k = updated_k,
                                         .updated_kv_cache_v = updated_v,
                                         .k_for_attn = k_for_attn,
                                         .v_for_attn = v_for_attn,
                                         .v_transposed = adj_v};
}

// Validates input, block norm, scalar, and PLE tensors/shapes before
// `TransformerBlockImpl` builds graph ops.
template <class... Mixins>
absl::Status ValidateTransformerBlockInputs(
    const Tensor<Mixins...>& x,
    const TransformerBlockWeights<Mixins...>& weights,
    const std::optional<SharedKv<Mixins...>>& shared_kv,
    const std::optional<Tensor<Mixins...>>& per_layer_input) {
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(
      x, weights.pre_attn_norm_weight, weights.q_weight, weights.o_weight,
      weights.pre_ffn_norm_weight, weights.ffn_gate_weight,
      weights.ffn_up_weight, weights.ffn_down_weight, weights.q_norm_weight,
      weights.post_attn_norm_weight, weights.post_ffn_norm_weight,
      weights.layer_scalar));
  if (!shared_kv.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(weights.k_weight, weights.v_weight,
                                            weights.k_norm_weight));
  }
  if (x.GetType() == Type::kUnknown) {
    return absl::InvalidArgumentError(
        "TransformerBlock expects a typed input `x`, got Type::kUnknown.");
  }
  if (x.GetShape().size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TransformerBlock expects rank 3 input [batch, seq_len, hidden_dim], "
        "got shape [",
        absl::StrJoin(x.GetShape(), ", "), "]."));
  }
  const int hidden_dim = x.GetShape()[2];
  auto check_norm_shape = [&](const Tensor<Mixins...>& norm_w,
                              absl::string_view name) -> absl::Status {
    if (norm_w.GetShape() != Shape({hidden_dim})) {
      return absl::InvalidArgumentError(
          absl::StrCat("TransformerBlock ", name,
                       " must have shape [hidden_dim=", hidden_dim, "], got [",
                       absl::StrJoin(norm_w.GetShape(), ", "), "]."));
    }
    return absl::OkStatus();
  };
  LRT_TENSOR_RETURN_IF_ERROR(
      check_norm_shape(weights.pre_attn_norm_weight, "pre_attn_norm_weight"));
  LRT_TENSOR_RETURN_IF_ERROR(
      check_norm_shape(weights.pre_ffn_norm_weight, "pre_ffn_norm_weight"));
  if (weights.post_attn_norm_weight.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(check_norm_shape(*weights.post_attn_norm_weight,
                                                "post_attn_norm_weight"));
  }
  if (weights.post_ffn_norm_weight.has_value()) {
    LRT_TENSOR_RETURN_IF_ERROR(check_norm_shape(*weights.post_ffn_norm_weight,
                                                "post_ffn_norm_weight"));
  }
  if (weights.layer_scalar.has_value()) {
    const Shape& scalar_shape = weights.layer_scalar->GetShape();
    if (!scalar_shape.empty() && scalar_shape != Shape({1})) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TransformerBlock layer_scalar must have shape [] or [1], got [",
          absl::StrJoin(scalar_shape, ", "), "]."));
    }
  }

  const bool has_ple_input = per_layer_input.has_value();
  const bool has_any_ple_weight =
      weights.per_layer_input_gate_weight.has_value() ||
      weights.per_layer_projection_weight.has_value() ||
      weights.post_per_layer_input_norm_weight.has_value();
  const bool has_all_ple_weights =
      weights.per_layer_input_gate_weight.has_value() &&
      weights.per_layer_projection_weight.has_value() &&
      weights.post_per_layer_input_norm_weight.has_value();
  if (has_ple_input != has_any_ple_weight ||
      has_any_ple_weight != has_all_ple_weights) {
    return absl::InvalidArgumentError(
        "TransformerBlock per-layer input requires `per_layer_input`, "
        "`per_layer_input_gate_weight`, `per_layer_projection_weight`, and "
        "`post_per_layer_input_norm_weight` to all be set together.");
  }
  if (has_ple_input) {
    LRT_TENSOR_RETURN_IF_ERROR(
        CheckTensors(*per_layer_input, *weights.per_layer_input_gate_weight,
                     *weights.per_layer_projection_weight,
                     *weights.post_per_layer_input_norm_weight));
    const Shape& ple_shape = per_layer_input->GetShape();
    if (ple_shape.size() != 3 || ple_shape[0] != x.GetShape()[0] ||
        ple_shape[1] != x.GetShape()[1] || ple_shape[2] <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TransformerBlock per_layer_input must have rank 3 shape [batch=",
          x.GetShape()[0], ", seq_len=", x.GetShape()[1],
          ", per_layer_input_dim > 0], got [", absl::StrJoin(ple_shape, ", "),
          "]."));
    }
    const int ple_dim = ple_shape[2];
    const Shape& gate_shape = weights.per_layer_input_gate_weight->GetShape();
    if (gate_shape != Shape({ple_dim, hidden_dim})) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TransformerBlock per_layer_input_gate_weight must have shape "
          "[per_layer_input_dim=",
          ple_dim, ", hidden_dim=", hidden_dim, "], got [",
          absl::StrJoin(gate_shape, ", "), "]."));
    }
    const Shape& proj_shape = weights.per_layer_projection_weight->GetShape();
    if (proj_shape != Shape({hidden_dim, ple_dim})) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TransformerBlock per_layer_projection_weight must have shape "
          "[hidden_dim=",
          hidden_dim, ", per_layer_input_dim=", ple_dim, "], got [",
          absl::StrJoin(proj_shape, ", "), "]."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(
        check_norm_shape(*weights.post_per_layer_input_norm_weight,
                         "post_per_layer_input_norm_weight"));
  }
  return absl::OkStatus();
}

template <class RopeParams, class... Mixins>
absl::StatusOr<TransformerBlockOutputs<Mixins...>> TransformerBlockImpl(
    const Tensor<Mixins...>& x,
    const TransformerBlockWeights<Mixins...>& weights, const RopeParams& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<KvCache<Mixins...>> kv_cache,
    const Tensor<Mixins...>& attention_mask, float rms_norm_epsilon,
    FFNActivation activation, std::optional<float> scale,
    std::optional<float> soft_cap, bool v_norm,
    std::optional<SharedKv<Mixins...>> shared_kv,
    std::optional<Tensor<Mixins...>> per_layer_input) {
  using TensorType = Tensor<Mixins...>;
  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateTransformerBlockInputs(x, weights, shared_kv, per_layer_input));
  const Type datatype = x.GetType();
  const bool has_ple_input = per_layer_input.has_value();

  graph::ScopeGuard scope("transformer_block");
  scope.Bind("input", x)
      .Bind("pre_attn_norm_weight", weights.pre_attn_norm_weight)
      .Bind("q_weight", weights.q_weight)
      .Bind("o_weight", weights.o_weight)
      .Bind("pre_ffn_norm_weight", weights.pre_ffn_norm_weight)
      .Bind("ffn_gate_weight", weights.ffn_gate_weight)
      .Bind("ffn_up_weight", weights.ffn_up_weight)
      .Bind("ffn_down_weight", weights.ffn_down_weight)
      .Bind("q_norm_weight", weights.q_norm_weight)
      .Bind("post_attn_norm_weight", weights.post_attn_norm_weight)
      .Bind("post_ffn_norm_weight", weights.post_ffn_norm_weight)
      .Bind("per_layer_input", per_layer_input)
      .Bind("per_layer_input_gate_weight", weights.per_layer_input_gate_weight)
      .Bind("per_layer_projection_weight", weights.per_layer_projection_weight)
      .Bind("post_per_layer_input_norm_weight",
            weights.post_per_layer_input_norm_weight)
      .Bind("layer_scalar", weights.layer_scalar)
      .Attr("num_heads", num_heads)
      .Attr("num_kv_heads", num_kv_heads)
      .Attr("head_dim", head_dim)
      .Attr("rms_norm_epsilon", static_cast<double>(rms_norm_epsilon))
      .Attr("v_norm", v_norm)
      .Attr("activation", activation == FFNActivation::kGeluApproximate
                              ? "gelu_approximate"
                              : "swish");
  if (!shared_kv.has_value()) {
    scope.Bind("k_weight", weights.k_weight)
        .Bind("v_weight", weights.v_weight)
        .Bind("k_norm_weight", weights.k_norm_weight);
  } else {
    scope.Bind("shared_kv_k", shared_kv->k)
        .Bind("shared_kv_v", shared_kv->v)
        .Attr("shared_v_transposed", shared_kv->v_transposed);
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
                        num_heads, num_kv_heads, head_dim, std::move(kv_cache),
                        attention_mask, rms_norm_epsilon, scale, soft_cap,
                        v_norm, std::move(shared_kv)));
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
  scope.Bind("post_ffn", output);

  if (has_ple_input) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TensorType ple_gate,
        FullyConnected(output, *weights.per_layer_input_gate_weight));
    LRT_TENSOR_ASSIGN_OR_RETURN(ple_gate, Gelu(ple_gate, /*approximate=*/true));
    LRT_TENSOR_ASSIGN_OR_RETURN(TensorType ple_gated,
                                Mul(ple_gate, *per_layer_input));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TensorType ple_projected,
        FullyConnected(ple_gated, *weights.per_layer_projection_weight));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        TensorType ple_normed,
        RmsNorm(ple_projected, *weights.post_per_layer_input_norm_weight,
                eps_tensor));
    scope.Bind("per_layer_input_output", ple_normed);
    LRT_TENSOR_ASSIGN_OR_RETURN(output, Add(output, ple_normed));
  }

  if (weights.layer_scalar.has_value()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(output, Mul(output, *weights.layer_scalar));
  }

  scope.Bind("output", output)
      .Bind("k_for_attn", sa_outputs.k_for_attn)
      .Bind("v_for_attn", sa_outputs.v_for_attn);
  if (sa_outputs.updated_kv_cache_k != TensorHandle::Invalid()) {
    scope.Bind("updated_kv_cache_k", sa_outputs.updated_kv_cache_k)
        .Bind("updated_kv_cache_v", sa_outputs.updated_kv_cache_v);
  }

  return TransformerBlockOutputs<Mixins...>{
      .output = output,
      .updated_kv_cache_k = sa_outputs.updated_kv_cache_k,
      .updated_kv_cache_v = sa_outputs.updated_kv_cache_v,
      .k_for_attn = sa_outputs.k_for_attn,
      .v_for_attn = sa_outputs.v_for_attn,
      .v_transposed = sa_outputs.v_transposed};
}

}  // namespace internal

// Multi-head self-attention with optional QK/V-norm, RoPE, KV cache, and
// cross-layer shared KV.
//
// x: [batch_size, seq_len, hidden_dim]; `output` has the same shape.
// q_norm_weight, k_norm_weight: optional per-head RMSNorm weights
//   ([head_dim]); if unset, Q/K normalization is skipped.
// v_norm: when true, normalizes V per head with scale-less RMSNorm before
//   updating the KV cache.
// shared_kv: optional pre-computed post-norm, post-RoPE, post-cache K and V
//   from an earlier layer (`k` has shape [batch, num_kv_heads, seq_k,
//   head_dim]; `v` has the same shape when `!v_transposed`, or
//   [batch, num_kv_heads, head_dim, seq_k] when `v_transposed`). Forward
//   `v_transposed` from the anchor layer's `SelfAttentionOutputs::v_transposed`
//   (or `TransformerBlockOutputs::v_transposed`). When set, `k_weight`,
//   `v_weight`, `k_norm_weight`, `v_norm`, `RoPE(k)`, and `kv_cache` are
//   bypassed; `shared_kv->k` / `shared_kv->v` are returned as `*_for_attn`,
//   while `updated_kv_cache_*` are `TensorHandle::Invalid()`. Cannot be
//   combined with `kv_cache`.
// num_heads: must be a positive multiple of `num_kv_heads`.
// head_dim: must be even and positive.
// kv_cache:
//   - unset: attends over this step's k and v; returns them as
//     `updated_kv_cache_*` and `*_for_attn`
//     ([batch, num_kv_heads, seq_len, head_dim]).
//   - set without `params`: prepends `k` and `v` along axis 2 before
//     attending (`*_for_attn`); returns this step's k and v in
//     `updated_kv_cache_*`.
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
    const Tensor<Mixins...>& x,
    const absl::type_identity_t<Tensor<Mixins...>>& q_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& k_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& v_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& o_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> q_norm_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> k_norm_weight,
    const absl::type_identity_t<RopeCosSinParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f, std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt, bool v_norm = false,
    std::optional<absl::type_identity_t<SharedKv<Mixins...>>> shared_kv =
        std::nullopt) {
  return internal::SelfAttentionImpl(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
      rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, scale, soft_cap, v_norm,
      std::move(shared_kv));
}

template <class... Mixins>
absl::StatusOr<SelfAttentionOutputs<Mixins...>> SelfAttention(
    const Tensor<Mixins...>& x,
    const absl::type_identity_t<Tensor<Mixins...>>& q_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& k_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& v_weight,
    const absl::type_identity_t<Tensor<Mixins...>>& o_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> q_norm_weight,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> k_norm_weight,
    const absl::type_identity_t<RotaryEmbeddingParams<Mixins...>>& rope,
    int num_heads, int num_kv_heads, int head_dim,
    std::optional<absl::type_identity_t<KvCache<Mixins...>>> kv_cache =
        std::nullopt,
    const absl::type_identity_t<Tensor<Mixins...>>& attention_mask =
        TensorHandle::Invalid(),
    float rms_norm_epsilon = 1e-6f, std::optional<float> scale = std::nullopt,
    std::optional<float> soft_cap = std::nullopt, bool v_norm = false,
    std::optional<absl::type_identity_t<SharedKv<Mixins...>>> shared_kv =
        std::nullopt) {
  return internal::SelfAttentionImpl(
      x, q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
      rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, scale, soft_cap, v_norm,
      std::move(shared_kv));
}

// Pre-norm transformer block (RMSNorm -> SelfAttention -> [RMSNorm] -> residual
// -> RMSNorm -> FeedForward -> [RMSNorm] -> residual -> [PerLayerInput] ->
// [LayerScalar]) with optional QK/V, post-attention, and post-FFN norms,
// cross-layer shared KV, per-layer input gating, and layer output scaling.
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
    std::optional<float> soft_cap = std::nullopt, bool v_norm = false,
    std::optional<absl::type_identity_t<SharedKv<Mixins...>>> shared_kv =
        std::nullopt,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> per_layer_input =
        std::nullopt) {
  return internal::TransformerBlockImpl(
      x, weights, rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, activation, scale, soft_cap, v_norm,
      std::move(shared_kv), std::move(per_layer_input));
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
    std::optional<float> soft_cap = std::nullopt, bool v_norm = false,
    std::optional<absl::type_identity_t<SharedKv<Mixins...>>> shared_kv =
        std::nullopt,
    std::optional<absl::type_identity_t<Tensor<Mixins...>>> per_layer_input =
        std::nullopt) {
  return internal::TransformerBlockImpl(
      x, weights, rope, num_heads, num_kv_heads, head_dim, std::move(kv_cache),
      attention_mask, rms_norm_epsilon, activation, scale, soft_cap, v_norm,
      std::move(shared_kv), std::move(per_layer_input));
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_OPS_TRANSFORMER_TRANSFORMER_BLOCK_H_
