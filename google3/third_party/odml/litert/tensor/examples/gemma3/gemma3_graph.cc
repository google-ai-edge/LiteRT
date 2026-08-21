/* Copyright 2025 Google LLC.

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

#include "third_party/odml/litert/tensor/examples/gemma3/gemma3_graph.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/backends/ml_drift/arithmetic_ml_drift.h"  // IWYU pragma: keep
#include "third_party/odml/litert/tensor/backends/xnnpack/arithmetic.h"  // IWYU pragma: keep
#include "third_party/odml/litert/tensor/examples/ops/transformer/transformer_ops.h"
#include "third_party/odml/litert/tensor/examples/ops/transformer/transformer_ops_ml_drift.h"  // IWYU pragma: keep
#include "third_party/odml/litert/tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor::examples {

namespace {

// Gets the tensor named `name` in the `weights` map.
//
// If no tensor matches, returns a placeholder tensor built using the given
// `type` and `shape`.
template <class... Mixins>
Tensor<Mixins...> GetWeight(
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    std::string name, Type type, const std::vector<int>& shape) {
  auto it = weights.find(name);
  if (it != weights.end()) {
    Tensor<Mixins...> weight = it->second;
    if (weight.GetType() == Type::kI8 && weight.GetQuantization() == nullptr) {
      // Try to find real scales in the weights map.
      std::vector<float> scales;

      auto scales_it = weights.find(name + ".scales");
      if (scales_it == weights.end()) {
        scales_it = weights.find(name + ".weight_scales");
      }
      if (scales_it == weights.end()) {
        scales_it = weights.find(name + "_scales");
      }

      if (scales_it != weights.end()) {
        auto buffer_or = scales_it->second.GetBuffer();
        if (buffer_or.ok()) {
          auto lock = buffer_or->Lock();
          auto float_span = std::move(lock).template As<const float>();
          scales.assign(float_span.begin(), float_span.end());
        }
      }

      auto q_params = std::make_shared<PerChannelAffineQuantization>();
      q_params->quantized_dimension = 0;
      int output_channels = weight.GetShape()[0];

      if (!scales.empty()) {
        q_params->scales = std::move(scales);
      } else {
        // Fallback to dummy scales if not found.
        q_params->scales = std::vector<float>(output_channels, 1.0f);
      }

      q_params->zero_points = std::vector<int64_t>(output_channels, 0);
      weight.SetQuantization(q_params);
    }
    return weight;
  }
  // Create placeholder weight tensor if not provided.
  return Tensor<Mixins...>(
      {.name = std::move(name), .type = type, .shape = shape});
}

// RMS Normalization with Gemma's zero-centered weights.
template <class... Mixins>
Tensor<Mixins...> Gemma3RmsNorm(const Tensor<Mixins...>& input,
                                const Tensor<Mixins...>& scale,
                                float eps = 1e-6f) {
  // Compute: sqrt(mean(x^2) + eps).
  Tensor x_squared = Mul(input, input);

  // Mean over last dimension (emb_dim).
  const auto& input_shape = input.GetShape();
  int last_axis = static_cast<int>(input_shape.size()) - 1;
  Tensor mean_squared = Mean(x_squared, {last_axis}, /*keep_dims=*/true);

  // Add epsilon and compute rsqrt.
  Tensor eps_tensor =
      Tensor<Mixins...>({.type = Type::kFP32,
                         .shape = {1},
                         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({eps})});
  Tensor variance_plus_eps = Add(mean_squared, eps_tensor);
  Tensor inv_rms = Rsqrt(variance_plus_eps);

  // Normalize
  Tensor x_norm = Mul(input, inv_rms);

  // Apply Gemma's zero-centered scale: $out = x_norm * (1 + scale)$.
  //
  // Note: Gemma uses: $out = x_norm * (1 + scale)$ instead of $out = x_norm *
  // scale$.
  Tensor one_tensor =
      Tensor<Mixins...>({.type = Type::kFP32,
                         .shape = {1},
                         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1.0f})});
  Tensor scale_plus_one = Add(scale, one_tensor);

  return Mul(x_norm, scale_plus_one);
}

// GELU activation with tanh approximation.
template <class... Mixins>
Tensor<Mixins...> GeluTanh(const Tensor<Mixins...>& input) {
  return Gelu(input, /*approximate=*/true);
}

// Feed-forward network layer.
template <class... Mixins>
Tensor<Mixins...> MakeFeedForwardLayer(
    const Tensor<Mixins...>& input, const std::string& name,
    const Gemma3Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights) {
  Tensor gate_proj =
      GetWeight(weights, absl::StrCat(name, ".gate_proj.weight"), Type::kFP32,
                {config.hidden_dim, config.emb_dim});
  Tensor up_proj = GetWeight(weights, absl::StrCat(name, ".up_proj.weight"),
                             Type::kFP32, {config.hidden_dim, config.emb_dim});
  Tensor down_proj =
      GetWeight(weights, absl::StrCat(name, ".down_proj.weight"), Type::kFP32,
                {config.emb_dim, config.hidden_dim});

  // SwiGLU: out = down_proj(gelu(gate_proj(x)) * up_proj(x)).
  Tensor gate = GeluTanh(FullyConnected(input, gate_proj));
  Tensor up = FullyConnected(input, up_proj);
  return FullyConnected(Mul(gate, up), down_proj);
}

// Rotary positional embedding.
template <class... Mixins>
Tensor<Mixins...> ApplyRotaryEmbedding(const Tensor<Mixins...>& x,
                                       const Tensor<Mixins...>& cos,
                                       const Tensor<Mixins...>& sin) {
  // x shape: [batch, n_heads, seq_len, head_dim].
  const auto& x_shape = x.GetShape();
  int head_dim = x_shape[3];
  int half_dim = head_dim / 2;

  // Split x into first and second half along head_dim.
  Tensor x1 =
      Slice(x, {0, 0, 0, 0}, {x_shape[0], x_shape[1], x_shape[2], half_dim});
  Tensor x2 = Slice(x, {0, 0, 0, half_dim},
                    {x_shape[0], x_shape[1], x_shape[2], half_dim});

  // rotated = cat(-x2, x1).
  Tensor neg_one = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({-1.0f})});
  Tensor neg_x2 = Mul(x2, neg_one);
  Tensor rotated = Concatenation({neg_x2, x1}, /*axis=*/3);

  // Apply rotation: x * cos + rotated * sin.
  Tensor x_cos = Mul(x, cos);
  Tensor rotated_sin = Mul(rotated, sin);
  return Add(x_cos, rotated_sin);
}

// Self-attention layer output structure.
template <class... Mixins>
struct SelfAttentionOutput {
  Tensor<Mixins...> output;
  Tensor<Mixins...> key_cache;
  Tensor<Mixins...> value_cache;
};

// Grouped Query Attention layer.
template <class... Mixins>
SelfAttentionOutput<Mixins...> MakeSelfAttentionLayer(
    const Tensor<Mixins...>& input, const std::string& name,
    const Gemma3Config& config, bool is_sliding_attention,
    const Tensor<Mixins...>& attention_mask, const Tensor<Mixins...>& cos,
    const Tensor<Mixins...>& sin, const Tensor<Mixins...>& key_cache,
    const Tensor<Mixins...>& value_cache,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    const Tensor<Mixins...>* cache_params = nullptr) {
  // Get weight tensors.
  int qkv_out_dim = config.n_heads * config.head_dim;
  int kv_out_dim = config.n_kv_groups * config.head_dim;

  Tensor q_proj = GetWeight(weights, absl::StrCat(name, ".q_proj.weight"),
                            Type::kFP32, {qkv_out_dim, config.emb_dim});
  Tensor k_proj = GetWeight(weights, absl::StrCat(name, ".k_proj.weight"),
                            Type::kFP32, {kv_out_dim, config.emb_dim});
  Tensor v_proj = GetWeight(weights, absl::StrCat(name, ".v_proj.weight"),
                            Type::kFP32, {kv_out_dim, config.emb_dim});
  Tensor o_proj = GetWeight(weights, absl::StrCat(name, ".o_proj.weight"),
                            Type::kFP32, {config.emb_dim, qkv_out_dim});

  // QK normalization weights (Gemma3 specific).
  Tensor q_norm = GetWeight(weights, absl::StrCat(name, ".q_norm.weight"),
                            Type::kFP32, {config.head_dim});
  Tensor k_norm = GetWeight(weights, absl::StrCat(name, ".k_norm.weight"),
                            Type::kFP32, {config.head_dim});

  // Project Q, K, V.
  Tensor q = FullyConnected(input, q_proj);
  Tensor k = FullyConnected(input, k_proj);
  Tensor v = FullyConnected(input, v_proj);

  // Get input shape for reshaping.
  const Shape& input_shape = input.GetShape();
  int batch_size = input_shape[0];
  int seq_len = input_shape[1];

  // Reshape Q: [batch, seq, n_heads * head_dim] -> [batch, seq, n_heads,
  // head_dim].
  q = Reshape(q, {batch_size, seq_len, config.n_heads, config.head_dim});
  q = Transpose(q, {0, 2, 1, 3});  // [batch, n_heads, seq, head_dim]

  // Reshape K, V: [batch, seq, n_kv_groups * head_dim] -> [batch, seq,
  // n_kv_groups, head_dim].
  k = Reshape(k, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  k = Transpose(k, {0, 2, 1, 3});  // [batch, n_kv_groups, seq, head_dim]
  v = Reshape(v, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  v = Transpose(v, {0, 2, 1, 3});  // [batch, n_kv_groups, seq, head_dim]

  // Apply QK normalization (Gemma3 specific).
  q = Gemma3RmsNorm(q, q_norm, config.rms_norm_eps);
  k = Gemma3RmsNorm(k, k_norm, config.rms_norm_eps);

  // Apply rotary positional embedding.
  q = ApplyRotaryEmbedding(q, cos, sin);
  k = ApplyRotaryEmbedding(k, cos, sin);

  Tensor updated_key_cache = k;
  Tensor updated_value_cache = v;

  Tensor k_for_attn = k;
  Tensor v_for_attn = v;
  bool used_mld_cache_update = false;

  constexpr bool kIsMlDrift = (std::is_same_v<MlDriftMixinTag, Mixins> || ...);
  if constexpr (kIsMlDrift) {
    if (cache_params != nullptr && key_cache.GetStatus().ok() &&
        value_cache.GetStatus().ok()) {
      const Shape& key_cache_shape = key_cache.GetShape();
      if (key_cache_shape.size() == 4 && key_cache_shape[2] > 0) {
        const int cache_heads = key_cache_shape[1];
        const int cache_size = key_cache_shape[2];
        Tensor<Mixins...> k_cache_src = k;
        Tensor<Mixins...> v_cache_src = v;
        if (config.n_kv_groups > 0 && cache_heads % config.n_kv_groups == 0) {
          std::vector<Tensor<Mixins...>> updated_caches = AddValuesToKvCache(
              key_cache, value_cache, k_cache_src, v_cache_src, *cache_params,
              /*num_of_kv_heads=*/cache_heads,
              /*kv_cache_batch_size=*/batch_size,
              /*cache_size=*/cache_size,
              /*head_dimension=*/config.head_dim,
              /*token_index_offset=*/0, /*active_tokens=*/1);
          updated_key_cache = updated_caches[0];
          updated_value_cache = updated_caches[1];
          k_for_attn = updated_key_cache;
          v_for_attn = updated_value_cache;
          used_mld_cache_update = true;
        }
      }
    }
  }

  if (!used_mld_cache_update) {
    if (key_cache.GetStatus().ok() && value_cache.GetStatus().ok()) {
      const auto& key_cache_shape = key_cache.GetShape();
      if (key_cache_shape.size() == 4 && key_cache_shape[2] > 0) {
        k_for_attn = Concatenation({key_cache, k}, /*axis=*/2);
        v_for_attn = Concatenation({value_cache, v}, /*axis=*/2);
      }
    }
  }

  float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  Tensor scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({scale})});
  int kv_heads_for_attn = config.n_kv_groups;
  const auto& k_for_attn_shape = k_for_attn.GetShape();
  if (k_for_attn_shape.size() == 4 && k_for_attn_shape[1] > 0) {
    kv_heads_for_attn = k_for_attn_shape[1];
  }

  Tensor<Mixins...> scores;
  Tensor<Mixins...> attn_weights;
  Tensor<Mixins...> attn_output;

  const auto& q_shape = q.GetShape();
  bool is_decode = (q_shape.size() == 4 && q_shape[2] == 1);

  if (used_mld_cache_update && is_decode) {
    Tensor<Mixins...> tiled_k = updated_key_cache;
    Tensor<Mixins...> tiled_v = updated_value_cache;
    scores = MatMulWithCache(q, tiled_k, *cache_params, /*is_v=*/false);
    scores = Mul(scores, scale_tensor);
    scores = Add(scores, attention_mask);
    attn_weights =
        SoftmaxWithRuntimeCheck(scores, *cache_params, std::nullopt, 2);
    attn_output =
        MatMulWithCache(attn_weights, tiled_v, *cache_params, /*is_v=*/true);
  } else {
    if (kv_heads_for_attn > 0 && config.n_heads % kv_heads_for_attn == 0) {
      int num_groups = config.n_heads / kv_heads_for_attn;
      if (num_groups > 1) {
        k_for_attn = Tile(k_for_attn, {1, num_groups, 1, 1});
        v_for_attn = Tile(v_for_attn, {1, num_groups, 1, 1});
      }
    }

    scores = BatchMatMul(q, k_for_attn, /*adj_x=*/false, /*adj_y=*/true);
    scores = Mul(scores, scale_tensor);
    scores = Add(scores, attention_mask);
    attn_weights = Softmax(scores);
    attn_output = BatchMatMul(attn_weights, v_for_attn);
  }

  // Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads *
  // head_dim].
  attn_output = Transpose(attn_output, {0, 2, 1, 3});
  attn_output = Reshape(attn_output, {batch_size, seq_len, qkv_out_dim});

  // Output projection.
  Tensor output = FullyConnected(attn_output, o_proj);

  return {output, updated_key_cache, updated_value_cache};
}

}  // namespace

template <class... Mixins>
Gemma3_Outputs<Mixins...> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<Mixins...> embedded_input,
    Tensor<Mixins...> position_ids,
    const std::vector<Tensor<Mixins...>>& key_caches,
    const std::vector<Tensor<Mixins...>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights) {
  // Get input shape (embedded_input is already [batch, seq, emb_dim]).
  const Shape& input_shape = embedded_input.GetShape();
  // int batch_size = input_shape[0];
  int seq_len = input_shape[1];
  int key_len = seq_len;
  if (!key_caches.empty()) {
    const auto& key_cache_shape = graph::GetInfo(key_caches[0].GetRaw())->shape;
    if (key_cache_shape.size() >= 3 && key_cache_shape[2] > 0) {
      key_len = key_cache_shape[2];
    }
  }

  // Scale embeddings by sqrt(emb_dim) - Gemma3 specific.
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor scaled_embedded = Mul(embedded_input, emb_scale_tensor);

  // Get layer types.
  std::vector<std::string> layer_types = config.GetLayerTypes();

  // Prepare updated KV caches.
  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  updated_key_caches.reserve(config.n_layers);
  updated_value_caches.reserve(config.n_layers);

  // Process transformer layers.
  Tensor hidden_states = scaled_embedded;

  Tensor mask_params = Tensor<Mixins...>(
      {.name = "mask_params",
       .type = Type::kI32,
       .shape = {4},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 0, 0, 0})});

  Tensor global_attention_mask =
      FillAttentionMask(mask_params, {1, 1, seq_len, key_len},
                        /*is_local=*/false, config.sliding_window);
  Tensor sliding_attention_mask =
      FillAttentionMask(mask_params, {1, 1, seq_len, key_len},
                        /*is_local=*/true, config.sliding_window);

  auto [global_cos, global_sin] = FillRopeCosSin<Mixins...>(
      seq_len, config.head_dim, config.rope_global_base);
  auto [sliding_cos, sliding_sin] = FillRopeCosSin<Mixins...>(
      seq_len, config.head_dim, config.rope_local_base);

  const Tensor<Mixins...>* cache_params = nullptr;
  Tensor<Mixins...> prefill_cache_params;
  constexpr bool kIsMlDrift = (std::is_same_v<MlDriftMixinTag, Mixins> || ...);
  if constexpr (kIsMlDrift) {
    if (!key_caches.empty() && key_len > 0) {
      prefill_cache_params = Tensor<Mixins...>(
          {.name = "cache_params",
           .type = Type::kI32,
           .shape = {3},
           .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, seq_len, seq_len})});
      cache_params = &prefill_cache_params;
    }
  }

  for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
    std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    bool is_sliding = (layer_types[layer_idx] == "sliding_attention");

    // Create attention mask.
    Tensor attention_mask =
        is_sliding ? sliding_attention_mask : global_attention_mask;
    Tensor cos = is_sliding ? sliding_cos : global_cos;
    Tensor sin = is_sliding ? sliding_sin : global_sin;

    // Pre-attention RMS normalization.
    Tensor input_norm_scale = GetWeight(
        weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_input =
        Gemma3RmsNorm(hidden_states, input_norm_scale, config.rms_norm_eps);

    // Self-attention.
    SelfAttentionOutput attn_output = MakeSelfAttentionLayer(
        normed_input, absl::StrCat(layer_prefix, ".self_attn"), config,
        is_sliding, attention_mask, cos, sin,
        layer_idx < key_caches.size() ? key_caches[layer_idx]
                                      : Tensor<Mixins...>(),
        layer_idx < value_caches.size() ? value_caches[layer_idx]
                                        : Tensor<Mixins...>(),
        weights, cache_params);

    updated_key_caches.push_back(attn_output.key_cache);
    updated_value_caches.push_back(attn_output.value_cache);

    // Post-attention RMS normalization.
    Tensor post_attn_norm_scale = GetWeight(
        weights, absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_attn_output = Gemma3RmsNorm(
        attn_output.output, post_attn_norm_scale, config.rms_norm_eps);

    // Residual connection after attention.
    hidden_states = Add(hidden_states, normed_attn_output);

    // Pre-FFN RMS normalization.
    Tensor pre_ffn_norm_scale = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_for_ffn =
        Gemma3RmsNorm(hidden_states, pre_ffn_norm_scale, config.rms_norm_eps);

    // Feed-forward network.
    Tensor ffn_output = MakeFeedForwardLayer(
        normed_for_ffn, absl::StrCat(layer_prefix, ".mlp"), config, weights);

    // Post-FFN RMS normalization.
    Tensor post_ffn_norm_scale = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_ffn_output =
        Gemma3RmsNorm(ffn_output, post_ffn_norm_scale, config.rms_norm_eps);

    // Residual connection after FFN.
    hidden_states = Add(hidden_states, normed_ffn_output);
  }

  // Final RMS normalization.
  Tensor final_norm_scale =
      GetWeight(weights, "model.norm.weight", Type::kFP32, {config.emb_dim});
  Tensor final_output =
      Gemma3RmsNorm(hidden_states, final_norm_scale, config.rms_norm_eps);

  // Output head - need embedding table for tied weights.
  Tensor embedding_table =
      GetWeight(weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});
  Tensor logits =
      FullyConnected(final_output, embedding_table, kActNone, /*adj_y=*/true);
  if constexpr (kIsMlDrift) {
    Tensor output_ids = ArgMax(logits, -1, Type::kI32);
    output_ids.SetName("output");

    return {output_ids, updated_key_caches, updated_value_caches};
  } else {
    logits.SetName("output");
    return {logits, updated_key_caches, updated_value_caches};
  }
}

// Explicit template instantiation for XnnpackMixinTag.
template Gemma3_Outputs<XnnpackMixinTag> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<XnnpackMixinTag> embedded_input,
    Tensor<XnnpackMixinTag> position_ids,
    const std::vector<Tensor<XnnpackMixinTag>>& key_caches,
    const std::vector<Tensor<XnnpackMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<XnnpackMixinTag>>& weights);

// Explicit template instantiation for MlDriftMixinTag.
template Gemma3_Outputs<MlDriftMixinTag> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<MlDriftMixinTag> embedded_input,
    Tensor<MlDriftMixinTag> position_ids,
    const std::vector<Tensor<MlDriftMixinTag>>& key_caches,
    const std::vector<Tensor<MlDriftMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>>& weights);

template <class... Mixins>
Gemma3_Outputs<Mixins...> BuildGemma3_FromEmbeddings_Decode(
    const Gemma3Config& config, Tensor<Mixins...> embedded_input,
    const Tensor<Mixins...>& rope_global_cos,
    const Tensor<Mixins...>& rope_global_sin,
    const Tensor<Mixins...>& rope_local_cos,
    const Tensor<Mixins...>& rope_local_sin,
    const Tensor<Mixins...>& sliding_attention_mask,
    const std::vector<Tensor<Mixins...>>& key_caches,
    const std::vector<Tensor<Mixins...>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    const Tensor<Mixins...>* global_attention_mask,
    const Tensor<Mixins...>* cache_params) {
  // embedded_input is [batch, 1, emb_dim].
  const Shape& input_shape = embedded_input.GetShape();
  // int batch_size = input_shape[0];
  int seq_len = input_shape[1];

  // Scale embeddings by sqrt(emb_dim) - Gemma3 specific.
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor hidden_states = Mul(embedded_input, emb_scale_tensor);

  std::vector<std::string> layer_types = config.GetLayerTypes();

  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  updated_key_caches.reserve(config.n_layers);
  updated_value_caches.reserve(config.n_layers);

  Tensor mask_params =
      Tensor<Mixins...>({.name = "mask_params",
                         .type = Type::kFP32,
                         .shape = {1},
                         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({0.0f})});
  Tensor xnnpack_global_attention_mask =
      FillAttentionMask(mask_params, {1, 1, seq_len, seq_len},
                        /*is_local=*/false, config.sliding_window);

  const Tensor<Mixins...>* global_mask = global_attention_mask != nullptr
                                             ? global_attention_mask
                                             : &xnnpack_global_attention_mask;
  const Tensor<Mixins...>* sliding_mask = &sliding_attention_mask;
  Tensor<Mixins...> mldrift_sliding_mask;
  Tensor<Mixins...> mldrift_global_mask;

  int key_len = seq_len;
  if (!key_caches.empty()) {
    const auto& key_cache_shape = graph::GetInfo(key_caches[0].GetRaw())->shape;
    if (key_cache_shape.size() >= 3 && key_cache_shape[2] > 0) {
      key_len = key_cache_shape[2];
    }
  }
  if (cache_params != nullptr && key_len > 0) {
    mldrift_sliding_mask =
        FillAttentionMask(*cache_params, {1, 1, seq_len, key_len},
                          /*is_local=*/true, config.sliding_window);
    mldrift_global_mask =
        FillAttentionMask(*cache_params, {1, 1, seq_len, key_len},
                          /*is_local=*/false, config.sliding_window);
    sliding_mask = &mldrift_sliding_mask;
    global_mask = &mldrift_global_mask;
  }

  for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
    std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    bool is_sliding = (layer_types[layer_idx] == "sliding_attention");

    Tensor attention_mask = is_sliding ? *sliding_mask : *global_mask;
    Tensor cos = is_sliding ? rope_local_cos : rope_global_cos;
    Tensor sin = is_sliding ? rope_local_sin : rope_global_sin;

    Tensor input_norm_scale = GetWeight(
        weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_input =
        Gemma3RmsNorm(hidden_states, input_norm_scale, config.rms_norm_eps);

    SelfAttentionOutput attn_output = MakeSelfAttentionLayer(
        normed_input, absl::StrCat(layer_prefix, ".self_attn"), config,
        is_sliding, attention_mask, cos, sin,
        layer_idx < key_caches.size() ? key_caches[layer_idx]
                                      : Tensor<Mixins...>(),
        layer_idx < value_caches.size() ? value_caches[layer_idx]
                                        : Tensor<Mixins...>(),
        weights, cache_params);

    updated_key_caches.push_back(attn_output.key_cache);
    updated_value_caches.push_back(attn_output.value_cache);

    Tensor post_attn_norm_scale = GetWeight(
        weights, absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_attn_output = Gemma3RmsNorm(
        attn_output.output, post_attn_norm_scale, config.rms_norm_eps);

    hidden_states = Add(hidden_states, normed_attn_output);

    Tensor pre_ffn_norm_scale = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_for_ffn =
        Gemma3RmsNorm(hidden_states, pre_ffn_norm_scale, config.rms_norm_eps);

    Tensor ffn_output = MakeFeedForwardLayer(
        normed_for_ffn, absl::StrCat(layer_prefix, ".mlp"), config, weights);

    Tensor post_ffn_norm_scale = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_ffn_output =
        Gemma3RmsNorm(ffn_output, post_ffn_norm_scale, config.rms_norm_eps);

    hidden_states = Add(hidden_states, normed_ffn_output);
  }

  Tensor final_norm_scale =
      GetWeight(weights, "model.norm.weight", Type::kFP32, {config.emb_dim});
  Tensor final_output =
      Gemma3RmsNorm(hidden_states, final_norm_scale, config.rms_norm_eps);

  Tensor embedding_table =
      GetWeight(weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});
  Tensor logits = FullyConnected(final_output, embedding_table, kActNone,
                                 /*keep_num_dims=*/true);
  constexpr bool kIsMlDrift = (std::is_same_v<MlDriftMixinTag, Mixins> || ...);
  if constexpr (kIsMlDrift) {
    Tensor output_ids = ArgMax(logits, -1, Type::kI32);
    output_ids.SetName("output");

    return {output_ids, updated_key_caches, updated_value_caches};
  } else {
    logits.SetName("output");
    return {logits, updated_key_caches, updated_value_caches};
  }
}

template Gemma3_Outputs<XnnpackMixinTag> BuildGemma3_FromEmbeddings_Decode(
    const Gemma3Config& config, Tensor<XnnpackMixinTag> embedded_input,
    const Tensor<XnnpackMixinTag>& rope_global_cos,
    const Tensor<XnnpackMixinTag>& rope_global_sin,
    const Tensor<XnnpackMixinTag>& rope_local_cos,
    const Tensor<XnnpackMixinTag>& rope_local_sin,
    const Tensor<XnnpackMixinTag>& sliding_attention_mask,
    const std::vector<Tensor<XnnpackMixinTag>>& key_caches,
    const std::vector<Tensor<XnnpackMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<XnnpackMixinTag>>& weights,
    const Tensor<XnnpackMixinTag>* global_attention_mask,
    const Tensor<XnnpackMixinTag>* cache_params);

template Gemma3_Outputs<MlDriftMixinTag> BuildGemma3_FromEmbeddings_Decode(
    const Gemma3Config& config, Tensor<MlDriftMixinTag> embedded_input,
    const Tensor<MlDriftMixinTag>& rope_global_cos,
    const Tensor<MlDriftMixinTag>& rope_global_sin,
    const Tensor<MlDriftMixinTag>& rope_local_cos,
    const Tensor<MlDriftMixinTag>& rope_local_sin,
    const Tensor<MlDriftMixinTag>& sliding_attention_mask,
    const std::vector<Tensor<MlDriftMixinTag>>& key_caches,
    const std::vector<Tensor<MlDriftMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>>& weights,
    const Tensor<MlDriftMixinTag>* global_attention_mask,
    const Tensor<MlDriftMixinTag>* cache_params);

template <>
Gemma3_Xnnpack_Outputs Gemma3_Xnnpack_Model::operator()(
    Gemma3_Xnnpack_Inputs& inputs) {
  auto model_outputs = BuildGemma3_FromEmbeddings(
      config, inputs.embedded_input, inputs.position_ids, inputs.key_caches,
      inputs.value_caches, inputs.weights);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches};
}

template <>
Gemma3_MlDrift_Outputs Gemma3_MlDrift_Model::operator()(
    Gemma3_MlDrift_Inputs& inputs) {
  auto model_outputs = BuildGemma3_FromEmbeddings(
      config, inputs.embedded_input, inputs.position_ids, inputs.key_caches,
      inputs.value_caches, inputs.weights);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches};
}

template <>
Gemma3_Xnnpack_Decode_Outputs Gemma3_Xnnpack_Decode_Model::operator()(
    Gemma3_Xnnpack_Decode_Inputs& inputs) {
  auto model_outputs = BuildGemma3_FromEmbeddings_Decode(
      config, inputs.embedded_input, inputs.rope_global_cos,
      inputs.rope_global_sin, inputs.rope_local_cos, inputs.rope_local_sin,
      inputs.sliding_attention_mask, inputs.key_caches, inputs.value_caches,
      inputs.weights);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches};
}

template <>
Gemma3_MlDrift_Decode_Outputs Gemma3_MlDrift_Decode_Model::operator()(
    Gemma3_MlDrift_Decode_Inputs& inputs) {
  auto model_outputs = BuildGemma3_FromEmbeddings_Decode(
      config, inputs.embedded_input, inputs.rope_global_cos,
      inputs.rope_global_sin, inputs.rope_local_cos, inputs.rope_local_sin,
      inputs.sliding_attention_mask, inputs.key_caches, inputs.value_caches,
      inputs.weights, &inputs.global_attention_mask, &inputs.cache_params);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches};
}

}  // namespace litert::tensor::examples
