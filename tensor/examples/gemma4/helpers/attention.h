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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ATTENTION_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4 {

template <class... Mixins>
absl::StatusOr<Tensor<Mixins...>> GetWeight(
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view name) {
  auto it = weights.find(name);
  if (it != weights.end()) {
    return it->second;
  }
  return absl::NotFoundError(absl::StrCat("Tensor ", name, " was not loaded."));
}

template <class... Mixins>
struct AttentionOutput {
  Tensor<Mixins...> output;
  Tensor<Mixins...> key_cache;
  Tensor<Mixins...> value_cache;
  Tensor<Mixins...> key_for_attn;
  Tensor<Mixins...> value_for_attn;
};

template <class... Mixins>
absl::StatusOr<AttentionOutput<Mixins...>> Attention(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& attention_mask,
    const Tensor<Mixins...>& cos, const Tensor<Mixins...>& sin,
    const Tensor<Mixins...>& key_cache, const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& shared_key, const Tensor<Mixins...>& shared_value,
    const Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view name, Config::LayerType layer_type,
    const Tensor<Mixins...>& rmsnorm_eps) {
  int head_dim = layer_type == Config::LayerType::kGlobal
                     ? config.global_key_size
                     : config.head_dim;
  int q_out_dim = config.num_heads * head_dim;

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor q_proj, GetWeight(weights, absl::StrCat(name, ".q_proj.weight")));

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor q_norm_scale,
      GetWeight(weights, absl::StrCat(name, ".q_norm.weight")));

  Tensor q = FullyConnected(input, q_proj);

  const Shape& input_shape = input.GetShape();
  int batch_size = input_shape[0];
  int seq_len = input_shape[1];

  q = Reshape(q, {batch_size, seq_len, config.num_heads, head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  q = RmsNorm(q, q_norm_scale, rmsnorm_eps);
  q = RoPE(q, cos, sin);

  Tensor<Mixins...> k_for_attn;
  Tensor<Mixins...> v_for_attn;
  Tensor<Mixins...> updated_key_cache;
  Tensor<Mixins...> updated_value_cache;

  bool has_valid_shared_kv =
      shared_key.GetStatus().ok() && shared_value.GetStatus().ok() &&
      shared_key.GetShape().size() == 4 &&
      shared_value.GetShape().size() == 4 &&
      shared_key.GetShape()[0] == batch_size &&
      shared_key.GetShape()[1] == config.num_kv_heads &&
      shared_key.GetShape()[3] == head_dim &&
      shared_value.GetShape()[0] == batch_size &&
      shared_value.GetShape()[1] == config.num_kv_heads &&
      shared_value.GetShape()[3] == head_dim &&
      shared_key.GetShape()[2] == shared_value.GetShape()[2];

  if (has_valid_shared_kv) {
    k_for_attn = shared_key;
    v_for_attn = shared_value;
    updated_key_cache = shared_key;
    updated_value_cache = shared_value;
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor k_proj,
        GetWeight(weights, absl::StrCat(name, ".k_proj.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor v_proj,
        GetWeight(weights, absl::StrCat(name, ".v_proj.weight")));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        Tensor k_norm_scale,
        GetWeight(weights, absl::StrCat(name, ".k_norm.weight")));

    Tensor k = FullyConnected(input, k_proj);
    Tensor v = FullyConnected(input, v_proj);

    k = Reshape(k, {batch_size, seq_len, config.num_kv_heads, head_dim});
    k = Transpose(k, {0, 2, 1, 3});
    k = RmsNorm(k, k_norm_scale, rmsnorm_eps);
    k = RoPE(k, cos, sin);
    k_for_attn = k;

    v = Reshape(v, {batch_size, seq_len, config.num_kv_heads, head_dim});
    v = Transpose(v, {0, 2, 1, 3});
    v = RmsNorm(v, /*scale=*/Tensor<Mixins...>(TensorHandle::Invalid()),
                rmsnorm_eps);
    v_for_attn = v;

    if (key_cache.GetStatus().ok() && value_cache.GetStatus().ok()) {
      const Shape& key_cache_shape = key_cache.GetShape();
      if (key_cache_shape.size() == 4 && key_cache_shape[2] > 0) {
        k_for_attn = Concatenation({key_cache, k}, /*axis=*/2);
        v_for_attn = Concatenation({value_cache, v}, /*axis=*/2);
      }
    }
    updated_key_cache = k;
    updated_value_cache = v;
  }

  // Keep the original KV heads for cache updates and cross-layer sharing.
  Tensor<Mixins...> k_for_attn_untiled = k_for_attn;
  Tensor<Mixins...> v_for_attn_untiled = v_for_attn;

  int num_groups = config.num_heads / config.num_kv_heads;
  // BatchMatMul broadcasts a single KV head to all query heads. Avoid an
  // explicit Tile, which can leave an unsupported broadcast when XNNPACK's
  // consistent arithmetic mode prevents its optimizer rewrite.
  if (num_groups > 1 && config.num_kv_heads > 1) {
    k_for_attn = RepeatKVHeads(k_for_attn, num_groups);
    v_for_attn = RepeatKVHeads(v_for_attn, num_groups);
  }

  Tensor context =
      ScaledDotProductAttention(q, k_for_attn, v_for_attn, attention_mask,
                                /*scale=*/1.0f, config.attn_logits_soft_cap);

  // Reshape context back to [B, L, N * H]
  context = Transpose(context, {0, 2, 1, 3});
  context = Reshape(context, {batch_size, seq_len, q_out_dim});

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Tensor o_proj, GetWeight(weights, absl::StrCat(name, ".o_proj.weight")));

  Tensor output = FullyConnected(context, o_proj);
  return AttentionOutput<Mixins...>{output, updated_key_cache,
                                    updated_value_cache, k_for_attn_untiled,
                                    v_for_attn_untiled};
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ATTENTION_H_
