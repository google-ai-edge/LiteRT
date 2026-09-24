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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_ATTENTION_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_ATTENTION_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/litert/active_graph_context.h"
#include "tensor/examples/gemma4/litert/kv_cache.h"
#include "tensor/examples/gemma4/litert/mobile_fully_connected.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4::cpu {

template <class... Mixins>
Tensor<Mixins...> GetWeight(
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    std::string name, Type type, const std::vector<int>& shape) {
  auto it = weights.find(name);
  if (it != weights.end()) {
    return it->second;
  }
  return Tensor<Mixins...>(
      {.name = std::move(name), .type = type, .shape = shape});
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
AttentionOutput<Mixins...> Attention(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& attention_mask,
    const Tensor<Mixins...>& cos, const Tensor<Mixins...>& sin,
    const Tensor<Mixins...>& key_cache, const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& shared_key, const Tensor<Mixins...>& shared_value,
    const Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view name, bool is_global,
    const Tensor<Mixins...>& rmsnorm_eps,
    ActiveGraphContext* active_graph_context) {
  int head_dim = is_global ? config.global_key_size : config.head_dim;
  int q_out_dim = config.num_heads * head_dim;
  int kv_out_dim = config.num_kv_heads * head_dim;

  Tensor q_proj = GetWeight(weights, absl::StrCat(name, ".q_proj.weight"),
                            Type::kFP32, {q_out_dim, config.embed_dim});
  Tensor k_proj = GetWeight(weights, absl::StrCat(name, ".k_proj.weight"),
                            Type::kFP32, {kv_out_dim, config.embed_dim});
  Tensor v_proj = GetWeight(weights, absl::StrCat(name, ".v_proj.weight"),
                            Type::kFP32, {kv_out_dim, config.embed_dim});
  Tensor o_proj = GetWeight(weights, absl::StrCat(name, ".o_proj.weight"),
                            Type::kFP32, {config.embed_dim, q_out_dim});

  Tensor q_norm_scale = GetWeight(weights, absl::StrCat(name, ".q_norm.weight"),
                                  Type::kFP32, {head_dim});
  Tensor k_norm_scale = GetWeight(weights, absl::StrCat(name, ".k_norm.weight"),
                                  Type::kFP32, {head_dim});

  Tensor q = MobileFullyConnected(input, q_proj, &weights);

  const Shape& input_shape = input.GetShape();
  int batch_size = input_shape[0];
  int seq_len = input_shape[1];

  q = Reshape(q, {batch_size, seq_len, config.num_heads, head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  q = PrimitiveRmsNorm(q, q_norm_scale, rmsnorm_eps);
  q = RoPE(q, cos, sin);

  auto& ctx = *active_graph_context;
  auto& cut = ctx.layers[ctx.layer];
  cut.owner = ctx.owners[ctx.layer];
  cut.dim = head_dim;
  cut.global = is_global;
  cut.query = Reshape(q, {batch_size, 1, config.num_heads * seq_len, head_dim});
  Tensor<Mixins...> nk = TensorHandle::Invalid(), nv = TensorHandle::Invalid();
  if (cut.owner == ctx.layer) {
    auto k = MobileFullyConnected(input, k_proj, &weights);
    k = Transpose(
        Reshape(k, {batch_size, seq_len, config.num_kv_heads, head_dim}),
        {0, 2, 1, 3});
    k = RoPE(PrimitiveRmsNorm(k, k_norm_scale, rmsnorm_eps), cos, sin);
    auto v = MobileFullyConnected(input, v_proj, &weights);
    v = Transpose(
        Reshape(v, {batch_size, seq_len, config.num_kv_heads, head_dim}),
        {0, 2, 1, 3});
    v = PrimitiveRmsNorm(v, Tensor<Mixins...>(TensorHandle::Invalid()),
                         rmsnorm_eps);
    nk = QuantizeInt8Kv(k, key_cache);
    nv = QuantizeInt8Kv(v, value_cache);
    cut.new_key = nk;
    cut.new_value = nv;
  }
  if (cut.owner == ctx.layer) {
    auto make_cache_input = [&](const char* suffix, const TensorHandle& cache) {
      return Tensor<Mixins...>({.name = absl::StrCat(name, suffix),
                                .type = Type::kI8,
                                .shape = {1, 1, 1, head_dim},
                                .quantization = CloneKvQuantization(cache)});
    };
    Tensor past_k = make_cache_input(".past_k", key_cache);
    Tensor past_v = make_cache_input(".past_v", value_cache);
    Tensor pad_k = make_cache_input(".pad_k", key_cache);
    Tensor pad_v = make_cache_input(".pad_v", value_cache);
    cut.past_key = past_k;
    cut.past_value = past_v;
    cut.pad_key = pad_k;
    cut.pad_value = pad_v;
    auto keys = Concatenation({past_k, nk, pad_k}, 2);
    auto values = Concatenation({past_v, nv, pad_v}, 2);
    keys.SetQuantization(CloneKvQuantization(key_cache));
    values.SetQuantization(CloneKvQuantization(value_cache));
    cut.active_key = keys;
    cut.active_value = values;
  } else {
    cut.active_key = ctx.layers[cut.owner].active_key;
    cut.active_value = ctx.layers[cut.owner].active_value;
  }
  auto& shared_mask = cut.global ? ctx.global_mask : ctx.local_mask;
  if (!shared_mask.GetStatus().ok()) {
    shared_mask = Tensor<Mixins...>(
        {.name = cut.global ? "joined_global_mask" : "joined_local_mask",
         .type = Type::kFP32,
         .shape = {1, 1, (ctx.broadcast_mask ? 1 : config.num_heads) * seq_len,
                   seq_len + 2}});
  }
  Tensor<Mixins...> mask(shared_mask);
  cut.mask = mask;
  // Keep heads as a batch dimension to broadcast one causal mask across
  // them. The KV operand has a singleton batch and is shared by all heads.
  auto scores = BatchMatMul(
      ctx.broadcast_mask ? q : Tensor<Mixins...>(cut.query),
      Cast(Tensor<Mixins...>(cut.active_key), Type::kFP32), false, true);
  if (config.attn_logits_soft_cap) {
    float cap = *config.attn_logits_soft_cap;
    scores = Mul(Tanh(Mul(scores, 1.0f / cap)), cap);
  }
  auto context = BatchMatMul(
      Softmax(Add(scores, mask)),
      Cast(Tensor<Mixins...>(cut.active_value), Type::kFP32), false, false);
  cut.context_input = context;
  context = Reshape(context, {batch_size, config.num_heads, seq_len, head_dim});
  context = Transpose(context, {0, 2, 1, 3});
  context = Reshape(context, {batch_size, seq_len, q_out_dim});
  return {MobileFullyConnected(context, o_proj, &weights), nk, nv, nk, nv};
}
}  // namespace litert::tensor::examples::gemma4::cpu
#endif
