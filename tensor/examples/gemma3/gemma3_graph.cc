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

#include "tensor/examples/gemma3/gemma3_graph.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#ifndef GEMMA3_XNNPACK_ONLY
#include "absl/log/absl_check.h"  // from @com_google_absl
#endif
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/examples/gemma3/gemma3_config.h"
#ifndef GEMMA3_XNNPACK_ONLY
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"  // IWYU pragma: keep
#endif
#ifndef GEMMA3_ML_DRIFT_ONLY
#include "tensor/backends/xnnpack/arithmetic.h"  // IWYU pragma: keep
#endif
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_block.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#ifndef GEMMA3_XNNPACK_ONLY
#include "tensor/examples/ops/transformer/transformer_ops_ml_drift.h"  // IWYU pragma: keep
#endif
#ifndef GEMMA3_ML_DRIFT_ONLY
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#endif
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

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

// Returns the `KvCache` for `layer_idx`, or `std::nullopt` if either cache is
// uninitialized or zero-length.
template <class... Mixins>
std::optional<KvCache<Mixins...>> LayerKvCache(
    const std::vector<Tensor<Mixins...>>& key_caches,
    const std::vector<Tensor<Mixins...>>& value_caches, size_t layer_idx,
    std::optional<Tensor<Mixins...>> cache_params = std::nullopt,
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
TransformerBlockWeights<Mixins...> BuildGemma3BlockWeights(
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    absl::string_view layer_prefix, const Gemma3Config& config) {
  const int qkv_out_dim = config.n_heads * config.head_dim;
  const int kv_out_dim = config.n_kv_groups * config.head_dim;
  const std::string attn_prefix = absl::StrCat(layer_prefix, ".self_attn.");
  const std::string mlp_prefix = absl::StrCat(layer_prefix, ".mlp.");
  return TransformerBlockWeights<Mixins...>{
      .pre_attn_norm_weight = GetWeight(
          weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
          Type::kFP32, {config.emb_dim}),
      .q_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "q_proj.weight"),
                    Type::kFP32, {qkv_out_dim, config.emb_dim}),
      .k_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "k_proj.weight"),
                    Type::kFP32, {kv_out_dim, config.emb_dim}),
      .v_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "v_proj.weight"),
                    Type::kFP32, {kv_out_dim, config.emb_dim}),
      .o_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "o_proj.weight"),
                    Type::kFP32, {config.emb_dim, qkv_out_dim}),
      .pre_ffn_norm_weight = GetWeight(
          weights,
          absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
          Type::kFP32, {config.emb_dim}),
      .ffn_gate_weight =
          GetWeight(weights, absl::StrCat(mlp_prefix, "gate_proj.weight"),
                    Type::kFP32, {config.hidden_dim, config.emb_dim}),
      .ffn_up_weight =
          GetWeight(weights, absl::StrCat(mlp_prefix, "up_proj.weight"),
                    Type::kFP32, {config.hidden_dim, config.emb_dim}),
      .ffn_down_weight =
          GetWeight(weights, absl::StrCat(mlp_prefix, "down_proj.weight"),
                    Type::kFP32, {config.emb_dim, config.hidden_dim}),
      .q_norm_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "q_norm.weight"),
                    Type::kFP32, {config.head_dim}),
      .k_norm_weight =
          GetWeight(weights, absl::StrCat(attn_prefix, "k_norm.weight"),
                    Type::kFP32, {config.head_dim}),
      .post_attn_norm_weight = GetWeight(
          weights,
          absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
          Type::kFP32, {config.emb_dim}),
      .post_ffn_norm_weight = GetWeight(
          weights,
          absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
          Type::kFP32, {config.emb_dim}),
  };
}

}  // namespace

template <class... Mixins>
Gemma3_Outputs<Mixins...> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<Mixins...> embedded_input,
    Tensor<Mixins...> position_ids, Tensor<Mixins...> slice_index,
    const std::vector<Tensor<Mixins...>>& key_caches,
    const std::vector<Tensor<Mixins...>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights) {
  // Get input shape (embedded_input is already [batch, seq, emb_dim]).
  const Shape& input_shape = embedded_input.GetShape();
  int seq_len = input_shape[1];
  int key_len = seq_len;

  // Scale embeddings by sqrt(emb_dim) - Gemma3 specific.
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor eps_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({config.rms_norm_eps})});
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

  absl::flat_hash_map<std::string, Tensor<Mixins...>> intermediate_tensors;

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
  global_cos.SetName("global_cos");
  global_sin.SetName("global_sin");
  intermediate_tensors["global_cos"] = global_cos;
  intermediate_tensors["global_sin"] = global_sin;
  auto [sliding_cos, sliding_sin] = FillRopeCosSin<Mixins...>(
      seq_len, config.head_dim, config.rope_local_base);

  constexpr bool kIsMlDrift = (std::is_same_v<MlDriftMixinTag, Mixins> || ...);
  std::optional<Tensor<Mixins...>> cache_params;
  if constexpr (kIsMlDrift) {
    if (LayerKvCache(key_caches, value_caches, 0).has_value()) {
      cache_params = Tensor<Mixins...>(
          {.name = "cache_params",
           .type = Type::kI32,
           .shape = {3},
           .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, seq_len, seq_len})});
    }
  }

  RopeCosSinParams<Mixins...> global_cos_sin{
      .cos = global_cos,
      .sin = global_sin,
  };
  RopeCosSinParams<Mixins...> sliding_cos_sin{
      .cos = sliding_cos,
      .sin = sliding_sin,
  };
  RotaryEmbeddingParams<Mixins...> global_rotary{
      .position_ids = position_ids,
      .max_timescale = config.rope_global_base,
  };
  RotaryEmbeddingParams<Mixins...> sliding_rotary{
      .position_ids = position_ids,
      .max_timescale = config.rope_local_base,
  };

  const float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
    std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    graph::ScopeGuard layer_scope("transformer_layer",
                                  absl::StrCat("layer_", layer_idx));
    layer_scope.Attr("layer_idx", layer_idx);
    bool is_sliding = (layer_types[layer_idx] == "sliding_attention");

    Tensor attention_mask =
        is_sliding ? sliding_attention_mask : global_attention_mask;
    TransformerBlockWeights<Mixins...> block_weights =
        BuildGemma3BlockWeights(weights, layer_prefix, config);
    std::optional<KvCache<Mixins...>> kv_cache =
        LayerKvCache(key_caches, value_caches, layer_idx, cache_params);

    absl::StatusOr<TransformerBlockOutputs<Mixins...>> block_output;
    if constexpr (kIsMlDrift) {
      block_output = TransformerBlock(
          hidden_states, block_weights,
          is_sliding ? sliding_rotary : global_rotary, config.n_heads,
          config.n_kv_groups, config.head_dim, std::move(kv_cache),
          attention_mask, config.rms_norm_eps, FFNActivation::kGeluApproximate,
          scale);
    } else {
      block_output = TransformerBlock(
          hidden_states, block_weights,
          is_sliding ? sliding_cos_sin : global_cos_sin, config.n_heads,
          config.n_kv_groups, config.head_dim, std::move(kv_cache),
          attention_mask, config.rms_norm_eps, FFNActivation::kGeluApproximate,
          scale);
    }
    if (!block_output.ok()) {
      Tensor<Mixins...> error(block_output.status());
      return {error, updated_key_caches, updated_value_caches,
              intermediate_tensors};
    }

    hidden_states = block_output->output;
    updated_key_caches.push_back(block_output->updated_kv_cache_k);
    updated_value_caches.push_back(block_output->updated_kv_cache_v);
    intermediate_tensors[absl::StrCat("layer_", layer_idx, "_post_ffn")] =
        hidden_states;
  }

  // Final RMS normalization.
  Tensor final_norm_scale =
      GetWeight(weights, "model.norm.weight", Type::kFP32, {config.emb_dim});
  Tensor final_output = RmsNorm(hidden_states, final_norm_scale, eps_tensor);
  final_output.SetName("model.norm.output");

  // Output head - need embedding table for tied weights.
  Tensor embedding_table =
      GetWeight(weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});
  intermediate_tensors["final_output"] = final_output;

  if constexpr (kIsMlDrift) {
    int axis = (final_output.GetShape().size() == 3) ? 1 : 0;
    Tensor<Mixins...> gathered_output = Gather(final_output, slice_index, axis);
    gathered_output.SetName("gathered_output");
    intermediate_tensors["gathered_output"] = gathered_output;
    std::string gathered_shape_str = "";
    for (int d : gathered_output.GetShape())
      absl::StrAppend(&gathered_shape_str, d, " ");
    ABSL_LOG(WARNING) << "Gathered output shape: [ " << gathered_shape_str
                      << "]";

    if (config.bypass_lm_head) {
      gathered_output.SetName("output");
      return {gathered_output, updated_key_caches, updated_value_caches,
              intermediate_tensors};
    }

    Tensor logits = FullyConnected(gathered_output, embedding_table, kActNone,
                                   /*adj_y=*/true);
    std::string logits_shape_str = "";
    for (int d : logits.GetShape()) absl::StrAppend(&logits_shape_str, d, " ");
    ABSL_LOG(WARNING) << "Logits shape: [ " << logits_shape_str << "]";
    logits.SetName("logits");
    intermediate_tensors["logits"] = logits;

    Tensor output_ids = ArgMax(logits, -1, Type::kI32);
    output_ids.SetName("output");

    return {output_ids, updated_key_caches, updated_value_caches,
            intermediate_tensors};
  } else {
    Tensor logits =
        FullyConnected(final_output, embedding_table, kActNone, /*adj_y=*/true);
    logits.SetName("output");
    intermediate_tensors["logits"] = logits;
    return {logits, updated_key_caches, updated_value_caches,
            intermediate_tensors};
  }
}

// Explicit template instantiation for XnnpackMixinTag.
#ifndef GEMMA3_ML_DRIFT_ONLY
template Gemma3_Outputs<XnnpackMixinTag> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<XnnpackMixinTag> embedded_input,
    Tensor<XnnpackMixinTag> position_ids, Tensor<XnnpackMixinTag> slice_index,
    const std::vector<Tensor<XnnpackMixinTag>>& key_caches,
    const std::vector<Tensor<XnnpackMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<XnnpackMixinTag>>& weights);
#endif

// Explicit template instantiation for MlDriftMixinTag.
#ifndef GEMMA3_XNNPACK_ONLY
template Gemma3_Outputs<MlDriftMixinTag> BuildGemma3_FromEmbeddings(
    const Gemma3Config& config, Tensor<MlDriftMixinTag> embedded_input,
    Tensor<MlDriftMixinTag> position_ids, Tensor<MlDriftMixinTag> slice_index,
    const std::vector<Tensor<MlDriftMixinTag>>& key_caches,
    const std::vector<Tensor<MlDriftMixinTag>>& value_caches,
    const absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>>& weights);
#endif

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
  int seq_len = input_shape[1];

  // Scale embeddings by sqrt(emb_dim) - Gemma3 specific.
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor hidden_states = Mul(embedded_input, emb_scale_tensor);
  Tensor eps_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({config.rms_norm_eps})});

  absl::flat_hash_map<std::string, Tensor<Mixins...>> intermediate_tensors;

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
  const int local_window = config.sliding_window > 0
                               ? std::min(config.sliding_window, key_len)
                               : key_len;
  if (cache_params != nullptr && key_len > 0) {
    mldrift_sliding_mask =
        FillAttentionMask(*cache_params, {1, 1, seq_len, local_window},
                          /*is_local=*/false, local_window,
                          /*is_decode=*/true);
    mldrift_global_mask =
        FillAttentionMask(*cache_params, {1, 1, seq_len, key_len},
                          /*is_local=*/false, config.sliding_window,
                          /*is_decode=*/true);
    sliding_mask = &mldrift_sliding_mask;
    global_mask = &mldrift_global_mask;
  }

  constexpr bool kIsMlDrift = (std::is_same_v<MlDriftMixinTag, Mixins> || ...);
  Tensor<Mixins...> position_ids = TensorHandle::Invalid();
  if constexpr (kIsMlDrift) {
    if (cache_params != nullptr) {
      position_ids = FillSegmentPos(*cache_params, {1, 1, seq_len, 1}, 0);
    }
  }

  RopeCosSinParams<Mixins...> global_cos_sin{
      .cos = rope_global_cos,
      .sin = rope_global_sin,
  };
  RopeCosSinParams<Mixins...> sliding_cos_sin{
      .cos = rope_local_cos,
      .sin = rope_local_sin,
  };
  RotaryEmbeddingParams<Mixins...> global_rotary{
      .position_ids = position_ids,
      .max_timescale = config.rope_global_base,
  };
  RotaryEmbeddingParams<Mixins...> sliding_rotary{
      .position_ids = position_ids,
      .max_timescale = config.rope_local_base,
  };

  const std::optional<Tensor<Mixins...>> decode_cache_params =
      cache_params != nullptr ? std::optional(*cache_params) : std::nullopt;
  const float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
    std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    graph::ScopeGuard layer_scope("transformer_layer",
                                  absl::StrCat("layer_", layer_idx));
    layer_scope.Attr("layer_idx", layer_idx);
    bool is_sliding = (layer_types[layer_idx] == "sliding_attention");

    Tensor attention_mask = is_sliding ? *sliding_mask : *global_mask;
    TransformerBlockWeights<Mixins...> block_weights =
        BuildGemma3BlockWeights(weights, layer_prefix, config);

    std::optional<int> local_cache_size;
    if (cache_params != nullptr && is_sliding) {
      local_cache_size = local_window;
    }
    std::optional<KvCache<Mixins...>> kv_cache =
        LayerKvCache(key_caches, value_caches, layer_idx, decode_cache_params,
                     local_cache_size);

    absl::StatusOr<TransformerBlockOutputs<Mixins...>> block_output;
    if (position_ids.GetStatus().ok()) {
      block_output = TransformerBlock(
          hidden_states, block_weights,
          is_sliding ? sliding_rotary : global_rotary, config.n_heads,
          config.n_kv_groups, config.head_dim, std::move(kv_cache),
          attention_mask, config.rms_norm_eps, FFNActivation::kGeluApproximate,
          scale);
    } else {
      block_output = TransformerBlock(
          hidden_states, block_weights,
          is_sliding ? sliding_cos_sin : global_cos_sin, config.n_heads,
          config.n_kv_groups, config.head_dim, std::move(kv_cache),
          attention_mask, config.rms_norm_eps, FFNActivation::kGeluApproximate,
          scale);
    }
    if (!block_output.ok()) {
      Tensor<Mixins...> error(block_output.status());
      return {error, updated_key_caches, updated_value_caches,
              intermediate_tensors};
    }

    hidden_states = block_output->output;
    updated_key_caches.push_back(block_output->updated_kv_cache_k);
    updated_value_caches.push_back(block_output->updated_kv_cache_v);
    intermediate_tensors[absl::StrCat("layer_", layer_idx, "_post_ffn")] =
        hidden_states;
  }

  Tensor final_norm_scale =
      GetWeight(weights, "model.norm.weight", Type::kFP32, {config.emb_dim});
  Tensor final_output = RmsNorm(hidden_states, final_norm_scale, eps_tensor);

  Tensor embedding_table =
      GetWeight(weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});
  intermediate_tensors["final_output"] = final_output;

  if constexpr (kIsMlDrift) {
    if (config.bypass_lm_head) {
      final_output.SetName("output");
      return {final_output, updated_key_caches, updated_value_caches,
              intermediate_tensors};
    }

    Tensor logits = FullyConnected(final_output, embedding_table, kActNone,
                                   /*keep_num_dims=*/true);
    intermediate_tensors["logits"] = logits;

    Tensor output_ids = ArgMax(logits, -1, Type::kI32);
    output_ids.SetName("output");

    return {output_ids, updated_key_caches, updated_value_caches,
            intermediate_tensors};
  } else {
    Tensor logits = FullyConnected(final_output, embedding_table, kActNone,
                                   /*keep_num_dims=*/true);
    logits.SetName("output");
    intermediate_tensors["logits"] = logits;
    return {logits, updated_key_caches, updated_value_caches,
            intermediate_tensors};
  }
}

#ifndef GEMMA3_ML_DRIFT_ONLY
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
#endif

#ifndef GEMMA3_XNNPACK_ONLY
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
#endif

#ifndef GEMMA3_ML_DRIFT_ONLY
template <>
Gemma3_Xnnpack_Outputs Gemma3_Xnnpack_Model::operator()(
    Gemma3_Xnnpack_Inputs& inputs) {
  auto model_outputs = BuildGemma3_FromEmbeddings(
      config, inputs.embedded_input, inputs.position_ids, inputs.slice_index,
      inputs.key_caches, inputs.value_caches, inputs.weights);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches};
}
#endif

#ifndef GEMMA3_XNNPACK_ONLY
template <>
Gemma3_MlDrift_Outputs Gemma3_MlDrift_Model::operator()(
    Gemma3_MlDrift_Inputs& inputs) {
  Tensor<MlDriftMixinTag> embedded_input;
  if (inputs.embedded_input.GetStatus().ok()) {
    ABSL_LOG(INFO) << "Bypassing EmbeddingLookup in Gemma3_MlDrift_Model";
    embedded_input = inputs.embedded_input;
  } else {
    auto emb_it = inputs.weights.find("model.embed_tokens.weight");
    ABSL_CHECK(emb_it != inputs.weights.end())
        << "Embedding table weight not found";
    Tensor<MlDriftMixinTag> embedding_table = emb_it->second;
    embedded_input = EmbeddingLookup(inputs.input_ids, embedding_table);
    embedded_input.SetName("embedded_input");
  }

  auto model_outputs = BuildGemma3_FromEmbeddings(
      config, embedded_input, inputs.position_ids, inputs.slice_index,
      inputs.key_caches, inputs.value_caches, inputs.weights);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches, embedded_input,
          model_outputs.intermediate_tensors};
}
#endif

#ifndef GEMMA3_ML_DRIFT_ONLY
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
#endif

#ifndef GEMMA3_XNNPACK_ONLY
template <>
Gemma3_MlDrift_Decode_Outputs Gemma3_MlDrift_Decode_Model::operator()(
    Gemma3_MlDrift_Decode_Inputs& inputs) {
  Tensor<MlDriftMixinTag> embedded_input;
  if (inputs.embedded_input.GetStatus().ok()) {
    ABSL_LOG(INFO)
        << "Bypassing EmbeddingLookup in Gemma3_MlDrift_Decode_Model";
    embedded_input = inputs.embedded_input;
  } else {
    auto emb_it = inputs.weights.find("model.embed_tokens.weight");
    ABSL_CHECK(emb_it != inputs.weights.end())
        << "Embedding table weight not found";
    Tensor<MlDriftMixinTag> embedding_table = emb_it->second;
    embedded_input = EmbeddingLookup(inputs.input_ids, embedding_table);
    embedded_input.SetName("embedded_input");
  }

  auto model_outputs = BuildGemma3_FromEmbeddings_Decode(
      config, embedded_input, inputs.rope_global_cos, inputs.rope_global_sin,
      inputs.rope_local_cos, inputs.rope_local_sin,
      inputs.sliding_attention_mask, inputs.key_caches, inputs.value_caches,
      inputs.weights, &inputs.global_attention_mask, &inputs.cache_params);
  return {model_outputs.output, model_outputs.key_caches,
          model_outputs.value_caches, model_outputs.intermediate_tensors};
}
#endif

}  // namespace litert::tensor::examples
