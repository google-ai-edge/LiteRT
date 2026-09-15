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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_ATTENTION_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_ATTENTION_H_

#include <string>
#include "tensor/examples/gemma4/native/model/helpers/active_graph_context.h"
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h" // from @com_google_absl
#include "absl/strings/str_cat.h"         // from @com_google_absl
#include "absl/strings/string_view.h"     // from @com_google_absl
#include "absl/types/span.h"              // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"
#include "tensor/examples/gemma4/native/model/helpers/mobile_fully_connected.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4::native {

template <class... Mixins>
Tensor<Mixins...>
GetWeight(const absl::flat_hash_map<std::string, Tensor<Mixins...>> &weights,
          std::string name, Type type, const std::vector<int> &shape) {
  auto it = weights.find(name);
  if (it != weights.end()) {
    return it->second;
  }
  return Tensor<Mixins...>(
      {.name = std::move(name), .type = type, .shape = shape});
}

template <class... Mixins> struct AttentionOutput {
  Tensor<Mixins...> output;
  Tensor<Mixins...> key_cache;
  Tensor<Mixins...> value_cache;
  Tensor<Mixins...> key_for_attn;
  Tensor<Mixins...> value_for_attn;
};

template <class... Mixins>
AttentionOutput<Mixins...>
Attention(const Tensor<Mixins...> &input,
          const Tensor<Mixins...> &attention_mask, const Tensor<Mixins...> &cos,
          const Tensor<Mixins...> &sin, const Tensor<Mixins...> &key_cache,
          const Tensor<Mixins...> &value_cache,
          const Tensor<Mixins...> &shared_key,
          const Tensor<Mixins...> &shared_value, const Config &config,
          const absl::flat_hash_map<std::string, Tensor<Mixins...>> &weights,
          absl::string_view name, bool is_global,
          const Tensor<Mixins...> &rmsnorm_eps) {
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

  const Shape &input_shape = input.GetShape();
  int batch_size = input_shape[0];
  int seq_len = input_shape[1];

  q = Reshape(q, {batch_size, seq_len, config.num_heads, head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  q = RmsNorm(q, q_norm_scale, rmsnorm_eps);
  q = RoPE(q, cos, sin);

  if (active_graph_context) {
    auto& ctx = *active_graph_context;
    auto& cut = ctx.layers[ctx.layer];
    cut.owner = ctx.owners[ctx.layer];
    cut.dim = head_dim;
    cut.global = is_global;
    cut.query = Reshape(q, {batch_size, 1, config.num_heads * seq_len, head_dim});
    Tensor<Mixins...> nk = TensorHandle::Invalid(), nv = TensorHandle::Invalid();
    if (cut.owner == ctx.layer) {
      auto k = MobileFullyConnected(input, k_proj, &weights);
      k = Transpose(Reshape(k, {batch_size, seq_len, config.num_kv_heads, head_dim}), {0,2,1,3});
      k = RoPE(RmsNorm(k, k_norm_scale, rmsnorm_eps), cos, sin);
      auto v = MobileFullyConnected(input, v_proj, &weights);
      v = Transpose(Reshape(v, {batch_size, seq_len, config.num_kv_heads, head_dim}), {0,2,1,3});
      v = RmsNorm(v, Tensor<Mixins...>(TensorHandle::Invalid()), rmsnorm_eps);
      nk = QuantizeInt8Kv(k, key_cache);
      nv = QuantizeInt8Kv(v, value_cache);
      cut.new_key = nk;
      cut.new_value = nv;
    }
    Tensor<Mixins...> context({.name=absl::StrCat(name,".active_context"), .type=Type::kFP32,
                              .shape={batch_size,1,config.num_heads*seq_len,head_dim}});
    cut.context_input = context;
    context = Reshape(context, {batch_size,config.num_heads,seq_len,head_dim});
    context = Transpose(context, {0,2,1,3});
    context = Reshape(context, {batch_size,seq_len,q_out_dim});
    auto output = MobileFullyConnected(context, o_proj, &weights);
    return {output,nk,nv,nk,nv};
  }

  Tensor<Mixins...> k_for_attn;
  Tensor<Mixins...> v_for_attn;
  Tensor<Mixins...> updated_key_cache;
  Tensor<Mixins...> updated_value_cache;

  // Artifact-only matched-bundle mode. The cache is persistently INT8, K is
  // [1,1,C,D], and V is [1,1,D,C]. Dequantization is local to the two BMMs.
  const bool shared_int8 =
      (shared_key.GetStatus().ok() && shared_key.GetType() == Type::kI8) ||
      (shared_value.GetStatus().ok() && shared_value.GetType() == Type::kI8);
  const bool int8_cache =
      shared_int8 ||
      (key_cache.GetStatus().ok() && key_cache.GetType() == Type::kI8) ||
      (value_cache.GetStatus().ok() && value_cache.GetType() == Type::kI8);
  const auto &layout_key = shared_int8 ? shared_key : key_cache;
  const auto &layout_value = shared_int8 ? shared_value : value_cache;
  int cache_capacity = 0;
  auto error = [](absl::Status status) -> AttentionOutput<Mixins...> {
    Tensor<Mixins...> failed(graph::ErrorTensor(std::move(status)));
    return {failed, failed, failed, failed, failed};
  };
  if (int8_cache) {
    if (batch_size != 1 || config.num_kv_heads != 1 ||
        layout_key.GetType() != Type::kI8 ||
        layout_value.GetType() != Type::kI8 ||
        layout_key.GetShape().size() != 4 || layout_key.GetShape()[2] < 1 ||
        layout_key.GetShape() !=
            Shape{1, 1, layout_key.GetShape()[2], head_dim} ||
        layout_value.GetShape() !=
            Shape{1, 1, head_dim, layout_key.GetShape()[2]}) {
      return error(
          absl::InvalidArgumentError("Matched INT8 cache requires E2B "
                                     "singleton-KV full-capacity K/V layouts"));
    }
    cache_capacity = layout_key.GetShape()[2];
    for (const auto *cache : {&layout_key, &layout_value}) {
      if (!cache->GetQuantization())
        return error(
            absl::InvalidArgumentError("INT8 cache has no quantization"));
      auto quant =
          cache->GetQuantization()->template As<PerChannelAffineQuantization>();
      if (!quant.ok() || quant->scales.size() != 1 ||
          !std::isfinite(quant->scales[0]) || quant->scales[0] <= 0 ||
          quant->zero_points != std::vector<int64_t>{0}) {
        return error(absl::InvalidArgumentError(
            "INT8 cache requires positive per-tensor scale and zero point0"));
      }
    }
  }
  bool has_valid_shared_kv =
      shared_key.GetStatus().ok() && shared_value.GetStatus().ok() &&
      shared_key.GetShape().size() == 4 &&
      shared_value.GetShape().size() == 4 &&
      shared_key.GetShape()[0] == batch_size &&
      shared_key.GetShape()[1] == config.num_kv_heads &&
      shared_key.GetShape()[3] == head_dim &&
      shared_value.GetShape()[0] == batch_size &&
      shared_value.GetShape()[1] == config.num_kv_heads &&
      shared_value.GetShape()[int8_cache ? 2 : 3] == head_dim &&
      shared_key.GetShape()[2] == shared_value.GetShape()[int8_cache ? 3 : 2] &&
      (!int8_cache || (shared_key.GetType() == Type::kI8 &&
                       shared_value.GetType() == Type::kI8 &&
                       shared_key.GetShape()[2] == cache_capacity));

  if (has_valid_shared_kv) {
    k_for_attn = shared_key;
    v_for_attn = shared_value;
    updated_key_cache = shared_key;
    updated_value_cache = shared_value;
  } else {
    Tensor k = MobileFullyConnected(input, k_proj, &weights);
    Tensor v = MobileFullyConnected(input, v_proj, &weights);

    k = Reshape(k, {batch_size, seq_len, config.num_kv_heads, head_dim});
    k = Transpose(k, {0, 2, 1, 3});
    k = RmsNorm(k, k_norm_scale, rmsnorm_eps);
    k = RoPE(k, cos, sin);
    k_for_attn = k;

    v = Reshape(v, {batch_size, seq_len, config.num_kv_heads, head_dim});
    v = Transpose(v, {0, 2, 1, 3});
    v = RmsNorm(v, Tensor<Mixins...>(TensorHandle::Invalid()), rmsnorm_eps);
    v_for_attn = v;

    if (int8_cache) {
      auto new_key = QuantizeInt8Kv(k, key_cache);
      auto new_value = QuantizeInt8Kv(v, value_cache);
      new_value = Transpose(new_value, {0, 1, 3, 2});
      new_value.SetQuantization(CloneKvQuantization(value_cache));
      const auto keep = weights.find(kKvKeepMaskName);
      const auto write = weights.find(kKvWriteMaskName);
      if ((keep == weights.end()) != (write == weights.end())) {
        return error(
            absl::InvalidArgumentError("Both INT8 cache masks are required"));
      }
      if (keep != weights.end()) {
        if (seq_len != 1 ||
            keep->second.GetShape() != Shape{1, 1, cache_capacity, 1} ||
            write->second.GetShape() != Shape{1, 1, cache_capacity, 1}) {
          return error(absl::InvalidArgumentError(
              "INT8 masked update requires one-token decode and capacity-sized "
              "masks"));
        }
        for (const auto *mask : {&keep->second, &write->second}) {
          const auto quant = CloneKvQuantization(*mask);
          if (mask->GetType() != Type::kI8 || !quant ||
              quant->scales != std::vector<float>{1.0f} ||
              quant->zero_points != std::vector<int64_t>{0}) {
            return error(absl::InvalidArgumentError(
                "INT8 update masks require scale1 and zero point0"));
          }
        }
        auto value_keep = Transpose(keep->second, {0, 1, 3, 2});
        auto value_write = Transpose(write->second, {0, 1, 3, 2});
        value_keep.SetQuantization(CloneKvQuantization(keep->second));
        value_write.SetQuantization(CloneKvQuantization(write->second));
        updated_key_cache =
            UpdateInt8Kv(key_cache, new_key, keep->second, write->second);
        updated_value_cache =
            UpdateInt8Kv(value_cache, new_value, value_keep, value_write);
      } else {
        updated_key_cache = PadInitialInt8Kv(new_key, 2, cache_capacity);
        updated_value_cache = PadInitialInt8Kv(new_value, 3, cache_capacity);
      }
      k_for_attn = updated_key_cache;
      v_for_attn = updated_value_cache;
      if (!is_global) {
        const int window = config.sliding_window_size;
        if (window < 1 || window > cache_capacity ||
            seq_len > cache_capacity - window + 1)
          return error(absl::InvalidArgumentError(
              "Compact local extent exceeds full cache"));
        if (keep != weights.end()) {
          auto local_key = weights.find(LocalKvKeyName(std::string(name)));
          auto local_value = weights.find(LocalKvValueName(std::string(name)));
          auto local_keep = weights.find(kLocalKvKeepMaskName);
          auto local_write = weights.find(kLocalKvWriteMaskName);
          if (local_key == weights.end() || local_value == weights.end() ||
              local_keep == weights.end() || local_write == weights.end())
            return error(absl::InvalidArgumentError(
                "Missing compact local cache inputs/masks"));
          for (int kind = 0; kind < 2; ++kind) {
            const auto &t = kind ? local_value->second : local_key->second;
            auto quant = CloneKvQuantization(t);
            auto full_quant =
                CloneKvQuantization(kind ? value_cache : key_cache);
            auto expected = kind ? Shape{1, 1, head_dim, window}
                                 : Shape{1, 1, window, head_dim};
            if (t.GetType() != Type::kI8 || t.GetShape() != expected ||
                !quant || quant->scales != full_quant->scales ||
                quant->zero_points != full_quant->zero_points)
              return error(absl::InvalidArgumentError(
                  "Compact local input layout or quantization mismatch"));
          }
          for (const auto *t : {&local_keep->second, &local_write->second}) {
            auto quant = CloneKvQuantization(*t);
            if (t->GetType() != Type::kI8 ||
                t->GetShape() != Shape{1, 1, window, 1} || !quant ||
                quant->scales != std::vector<float>{1.0f} ||
                quant->zero_points != std::vector<int64_t>{0})
              return error(absl::InvalidArgumentError(
                  "Invalid compact local update mask"));
          }
          auto value_keep = Transpose(local_keep->second, {0, 1, 3, 2});
          auto value_write = Transpose(local_write->second, {0, 1, 3, 2});
          value_keep.SetQuantization(CloneKvQuantization(local_keep->second));
          value_write.SetQuantization(CloneKvQuantization(local_write->second));
          k_for_attn = UpdateInt8Kv(local_key->second, new_key,
                                    local_keep->second, local_write->second);
          v_for_attn = UpdateInt8Kv(local_value->second, new_value, value_keep,
                                    value_write);
        } else {
          const int extent = window + seq_len - 1;
          k_for_attn =
              Slice(updated_key_cache, {0, 0, 0, 0}, {1, 1, extent, head_dim});
          v_for_attn = Slice(updated_value_cache, {0, 0, 0, 0},
                             {1, 1, head_dim, extent});
          k_for_attn.SetQuantization(CloneKvQuantization(key_cache));
          v_for_attn.SetQuantization(CloneKvQuantization(value_cache));
        }
      }
    } else {
      if (key_cache.GetStatus().ok() && value_cache.GetStatus().ok()) {
        const Shape &key_cache_shape = key_cache.GetShape();
        if (key_cache_shape.size() == 4 && key_cache_shape[2] > 0) {
          k_for_attn = Concatenation({key_cache, k}, /*axis=*/2);
          v_for_attn = Concatenation({value_cache, v}, /*axis=*/2);
        }
      }
      updated_key_cache = k;
      updated_value_cache = v;
    }
  }

  // BatchMatMul broadcasts a single KV head across the query heads. Keep that
  // dimension singleton so this path also works when XNNPACK's arithmetic
  // optimizations are disabled. Multiple KV heads still need explicit grouping.
  Tensor<Mixins...> k_for_attn_untiled = k_for_attn;
  Tensor<Mixins...> v_for_attn_untiled = v_for_attn;

  if (int8_cache) {
    k_for_attn = Cast(k_for_attn, Type::kFP32);
    v_for_attn = Cast(v_for_attn, Type::kFP32);
  }

  int num_groups = config.num_heads / config.num_kv_heads;
  if (num_groups > 1 && config.num_kv_heads > 1) {
    std::vector<Tensor<Mixins...>> k_sliced;
    k_sliced.reserve(config.num_kv_heads);
    std::vector<Tensor<Mixins...>> v_sliced;
    v_sliced.reserve(config.num_kv_heads);
    const Shape &shape = k_for_attn.GetShape();
    for (int h = 0; h < config.num_kv_heads; ++h) {
      Tensor<Mixins...> k_h =
          Slice(k_for_attn, {0, h, 0, 0}, {shape[0], 1, -1, shape[3]});
      k_sliced.push_back(Tile(k_h, {1, num_groups, 1, 1}));

      Tensor<Mixins...> v_h =
          Slice(v_for_attn, {0, h, 0, 0}, {shape[0], 1, -1, shape[3]});
      v_sliced.push_back(Tile(v_h, {1, num_groups, 1, 1}));
    }
    k_for_attn =
        Concatenation<Mixins...>(absl::Span<Tensor<Mixins...>>(k_sliced), 1);
    v_for_attn =
        Concatenation<Mixins...>(absl::Span<Tensor<Mixins...>>(v_sliced), 1);
  }

  const int attention_extent = k_for_attn.GetShape()[2];
  if (int8_cache) {
    if (attention_mask.GetShape() != Shape{1, 1, seq_len, attention_extent})
      return error(absl::InvalidArgumentError(
          "Attention mask must match compact/full cache extent"));
    // Bundle head-major fold: no transpose between [1,H,S,D] and [1,1,H*S,D].
    q = Reshape(q, {batch_size, 1, config.num_heads * seq_len, head_dim});
  }
  Tensor scores = BatchMatMul(q, k_for_attn, /*adj_x=*/false, /*adj_y=*/true);
  if (int8_cache)
    scores = Reshape(scores,
                     {batch_size, config.num_heads, seq_len, attention_extent});

  if (config.attn_logits_soft_cap.has_value()) {
    float cap = config.attn_logits_soft_cap.value();
    Tensor cap_tensor = Tensor<Mixins...>(
        {.type = Type::kFP32,
         .shape = {1},
         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({cap})});
    Tensor inv_cap_tensor = Tensor<Mixins...>(
        {.type = Type::kFP32,
         .shape = {1},
         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1.0f / cap})});
    Tensor scaled_scores = Mul(scores, inv_cap_tensor);
    Tensor tanh_scores = Tanh(scaled_scores);
    scores = Mul(tanh_scores, cap_tensor);
  }

  scores = Add(scores, attention_mask);
  Tensor probs = Softmax(scores);
  if (int8_cache)
    probs = Reshape(
        probs, {batch_size, 1, config.num_heads * seq_len, attention_extent});

  Tensor context =
      BatchMatMul(probs, v_for_attn, /*adj_x=*/false, /*adj_y=*/int8_cache);

  if (int8_cache)
    context =
        Reshape(context, {batch_size, config.num_heads, seq_len, head_dim});
  // Reshape context back to [B, L, N * H]
  context = Transpose(context, {0, 2, 1, 3});
  context = Reshape(context, {batch_size, seq_len, q_out_dim});

  Tensor output = MobileFullyConnected(context, o_proj, &weights);
  return {output, updated_key_cache, updated_value_cache, k_for_attn_untiled,
          v_for_attn_untiled};
}

} // namespace litert::tensor::examples::gemma4::native

#endif // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_ATTENTION_H_
