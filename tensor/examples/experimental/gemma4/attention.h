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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ATTENTION_H_

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/active_graph_context.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/experimental/gemma4/rope.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Output tensors produced by a single Gemma4 self-attention block.
template <class... Mixins>
struct AttentionOutput {
  Tensor<Mixins...> output = TensorHandle::Invalid();
  Tensor<Mixins...> key_cache = TensorHandle::Invalid();
  Tensor<Mixins...> value_cache = TensorHandle::Invalid();
};

// Authors one Gemma4 CPU self-attention block with INT8 external KV cache
// inputs (`past_k`/`past_v`/`pad_k`/`pad_v`) on KV-owner layers and shared
// active KV reuse on non-owner layers. Requires `batch_size == 1`,
// `config.num_kv_heads == 1`, and `context.layer` within `context.layers` and
// `context.owners`.
template <class... Mixins>
AttentionOutput<Mixins...> Attention(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& cos,
    const Tensor<Mixins...>& sin, const Tensor<Mixins...>& key_cache,
    const Tensor<Mixins...>& value_cache,
    const examples::gemma4::Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view name, examples::gemma4::Config::LayerType layer_type,
    const Tensor<Mixins...>& rmsnorm_eps, ActiveGraphContext& context) {
  const auto fail = [](absl::Status status) -> AttentionOutput<Mixins...> {
    Tensor<Mixins...> err(std::move(status));
    return {err, err, err};
  };
  if (!input.GetStatus().ok()) {
    return fail(input.GetStatus());
  }
  const Shape& input_shape = input.GetShape();
  if (input_shape.size() < 2) {
    return fail(absl::InvalidArgumentError(
        "Attention expects input with rank >= 2 ([batch, seq_len, ...])"));
  }
  const int batch_size = input_shape[0];
  const int seq_len = input_shape[1];
  if (batch_size != 1 || config.num_kv_heads != 1) {
    return fail(absl::InvalidArgumentError(
        "Experimental CPU Gemma4 attention requires batch_size == 1 and "
        "num_kv_heads == 1"));
  }
  if (context.layer < 0 ||
      context.layer >= static_cast<int>(context.layers.size()) ||
      context.layer >= static_cast<int>(context.owners.size())) {
    return fail(absl::InvalidArgumentError(
        "ActiveGraphContext layer index out of bounds"));
  }
  const int owner = context.owners[context.layer];
  if (owner < 0 || owner > context.layer) {
    return fail(absl::InvalidArgumentError(
        "ActiveGraphContext KV owner must satisfy 0 <= owner <= layer"));
  }

  const bool is_global =
      layer_type == examples::gemma4::Config::LayerType::kGlobal;
  const int head_dim = is_global ? config.global_key_size : config.head_dim;
  const int q_out_dim = config.num_heads * head_dim;
  const int kv_out_dim = config.num_kv_heads * head_dim;

  const Tensor<Mixins...> q_proj =
      GetWeightOrPlaceholder(weights, absl::StrCat(name, ".q_proj.weight"),
                             Type::kFP32, {q_out_dim, config.embed_dim});
  const Tensor<Mixins...> o_proj =
      GetWeightOrPlaceholder(weights, absl::StrCat(name, ".o_proj.weight"),
                             Type::kFP32, {config.embed_dim, q_out_dim});
  const Tensor<Mixins...> q_norm_scale = GetWeightOrPlaceholder(
      weights, absl::StrCat(name, ".q_norm.weight"), Type::kFP32, {head_dim});

  const int reshape_rows =
      context.dynamic_prefill_rows ? kInferredDim : seq_len;
  const RopeLeadingDims rope_dims = context.dynamic_prefill_rows
                                        ? RopeLeadingDims::kDynamic
                                        : RopeLeadingDims::kStatic;

  Tensor<Mixins...> q = MobileFullyConnected(input, q_proj, &weights);
  if (!q.GetStatus().ok()) {
    return fail(q.GetStatus());
  }
  q = Reshape(q, {batch_size, reshape_rows, config.num_heads, head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  q = PrimitiveRmsNorm(q, q_norm_scale, rmsnorm_eps);
  q = SplitHalfRoPE(q, cos, sin, rope_dims);
  if (!q.GetStatus().ok()) {
    return fail(q.GetStatus());
  }

  ActiveLayerGraph& layer_graph = context.layers[context.layer];
  layer_graph.owner = owner;
  layer_graph.global = is_global;
  if (!context.broadcast_mask) {
    layer_graph.query =
        Reshape(q, {batch_size, 1,
                    context.dynamic_prefill_rows ? kInferredDim
                                                 : config.num_heads * seq_len,
                    head_dim});
  }

  Tensor<Mixins...> new_key = TensorHandle::Invalid();
  Tensor<Mixins...> new_value = TensorHandle::Invalid();
  if (layer_graph.owner == context.layer) {
    if (!key_cache.GetStatus().ok() || !value_cache.GetStatus().ok() ||
        !key_cache.GetQuantization() || !value_cache.GetQuantization()) {
      return fail(absl::InvalidArgumentError(
          "Owner KV cache metadata tensors must carry quantization params"));
    }
    const Tensor<Mixins...> k_proj =
        GetWeightOrPlaceholder(weights, absl::StrCat(name, ".k_proj.weight"),
                               Type::kFP32, {kv_out_dim, config.embed_dim});
    const Tensor<Mixins...> v_proj =
        GetWeightOrPlaceholder(weights, absl::StrCat(name, ".v_proj.weight"),
                               Type::kFP32, {kv_out_dim, config.embed_dim});
    const Tensor<Mixins...> k_norm_scale = GetWeightOrPlaceholder(
        weights, absl::StrCat(name, ".k_norm.weight"), Type::kFP32, {head_dim});

    Tensor<Mixins...> k = MobileFullyConnected(input, k_proj, &weights);
    if (!k.GetStatus().ok()) {
      return fail(k.GetStatus());
    }
    k = Transpose(
        Reshape(k, {batch_size, reshape_rows, config.num_kv_heads, head_dim}),
        {0, 2, 1, 3});
    k = SplitHalfRoPE(PrimitiveRmsNorm(k, k_norm_scale, rmsnorm_eps), cos, sin,
                      rope_dims);
    if (!k.GetStatus().ok()) {
      return fail(k.GetStatus());
    }

    Tensor<Mixins...> v = MobileFullyConnected(input, v_proj, &weights);
    if (!v.GetStatus().ok()) {
      return fail(v.GetStatus());
    }
    v = Transpose(
        Reshape(v, {batch_size, reshape_rows, config.num_kv_heads, head_dim}),
        {0, 2, 1, 3});
    v = PrimitiveRmsNorm(v, Tensor<Mixins...>(TensorHandle::Invalid()),
                         rmsnorm_eps);
    if (!v.GetStatus().ok()) {
      return fail(v.GetStatus());
    }

    new_key = QuantizeInt8Kv(k, key_cache);
    new_value = QuantizeInt8Kv(v, value_cache);
    if (!new_key.GetStatus().ok()) {
      return fail(new_key.GetStatus());
    }
    if (!new_value.GetStatus().ok()) {
      return fail(new_value.GetStatus());
    }
    layer_graph.new_key = new_key;
    layer_graph.new_value = new_value;

    absl::StatusOr<std::shared_ptr<PerChannelAffineQuantization>> key_quant =
        CloneKvQuantization(key_cache);
    if (!key_quant.ok()) {
      return fail(key_quant.status());
    }
    absl::StatusOr<std::shared_ptr<PerChannelAffineQuantization>> value_quant =
        CloneKvQuantization(value_cache);
    if (!value_quant.ok()) {
      return fail(value_quant.status());
    }
    const auto make_cache_input =
        [&](absl::string_view suffix,
            const PerChannelAffineQuantization& quantization) {
          return Tensor<Mixins...>(
              {.name = absl::StrCat(name, suffix),
               .type = Type::kI8,
               .shape = {1, 1, 1, head_dim},
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   quantization)});
        };
    const Tensor<Mixins...> past_k = make_cache_input(".past_k", **key_quant);
    const Tensor<Mixins...> past_v = make_cache_input(".past_v", **value_quant);
    const Tensor<Mixins...> pad_k = make_cache_input(".pad_k", **key_quant);
    const Tensor<Mixins...> pad_v = make_cache_input(".pad_v", **value_quant);
    layer_graph.past_key = past_k;
    layer_graph.past_value = past_v;
    layer_graph.pad_key = pad_k;
    layer_graph.pad_value = pad_v;

    Tensor<Mixins...> keys = Concatenation({past_k, new_key, pad_k}, 2);
    Tensor<Mixins...> values = Concatenation({past_v, new_value, pad_v}, 2);
    keys.SetQuantization(std::move(*key_quant));
    values.SetQuantization(std::move(*value_quant));
    layer_graph.active_key = keys;
    layer_graph.active_value = values;
  } else {
    layer_graph.active_key = context.layers[layer_graph.owner].active_key;
    layer_graph.active_value = context.layers[layer_graph.owner].active_value;
    if (!layer_graph.active_key.GetStatus().ok() ||
        !layer_graph.active_value.GetStatus().ok()) {
      return fail(absl::FailedPreconditionError(
          "Shared KV owner layer has not been authored yet"));
    }
  }

  TensorHandle& shared_mask =
      layer_graph.global ? context.global_mask : context.local_mask;
  if (!shared_mask.GetStatus().ok()) {
    shared_mask = Tensor<Mixins...>(
        {.name =
             layer_graph.global ? "joined_global_mask" : "joined_local_mask",
         .type = Type::kFP32,
         .shape = {1, 1,
                   (context.broadcast_mask ? 1 : config.num_heads) * seq_len,
                   seq_len + 2}});
  }
  const Tensor<Mixins...> mask(shared_mask);

  // Keep heads as a batch dimension to broadcast one causal mask across
  // them. The KV operand has a singleton batch and is shared by all heads.
  Tensor<Mixins...> scores = BatchMatMul(
      context.broadcast_mask ? q : Tensor<Mixins...>(layer_graph.query),
      Dequantize(Tensor<Mixins...>(layer_graph.active_key)),
      /*adj_x=*/false, /*adj_y=*/true);
  if (config.attn_logits_soft_cap.has_value()) {
    const float cap = *config.attn_logits_soft_cap;
    if (!std::isfinite(cap) || cap <= 0.0f) {
      return fail(absl::InvalidArgumentError(
          "Attention logits soft cap must be finite and positive"));
    }
    scores = Mul(Tanh(Mul(scores, 1.0f / cap)), cap);
  }
  Tensor<Mixins...> attn_context =
      BatchMatMul(Softmax(Add(scores, mask)),
                  Dequantize(Tensor<Mixins...>(layer_graph.active_value)),
                  /*adj_x=*/false, /*adj_y=*/false);
  attn_context = Reshape(
      attn_context, {batch_size, config.num_heads, reshape_rows, head_dim});
  attn_context = Transpose(attn_context, {0, 2, 1, 3});
  attn_context = Reshape(attn_context, {batch_size, reshape_rows, q_out_dim});
  return {MobileFullyConnected(attn_context, o_proj, &weights), new_key,
          new_value};
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ATTENTION_H_
