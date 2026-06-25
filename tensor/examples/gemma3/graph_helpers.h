/* Copyright 2025-2026 Google LLC.

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
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GRAPH_HELPERS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GRAPH_HELPERS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/config.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma3 {

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
    return it->second;
  }
  ABSL_LOG(WARNING) << "!!! WEIGHT NOT FOUND !!!: " << name;
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
  int last_axis = static_cast<int>(input.GetShape().size()) - 1;
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

  return Mul(x_norm, scale);
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
    const Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights) {
  Tensor gate_proj =
      GetWeight(weights, absl::StrCat(name, ".gate_proj.weight"), Type::kFP32,
                {config.hidden_dim, config.emb_dim});
  Tensor up_proj = GetWeight(weights, absl::StrCat(name, ".up_proj.weight"),
                             Type::kFP32, {config.hidden_dim, config.emb_dim});
  Tensor down_proj =
      GetWeight(weights, absl::StrCat(name, ".down_proj.weight"), Type::kFP32,
                {config.emb_dim, config.hidden_dim});

  // SwiGLU: out = down_proj(up_proj(x) * gelu(gate_proj(x))).
  Tensor up = FullyConnected(input, up_proj);
  Tensor gate = GeluTanh(FullyConnected(input, gate_proj));
  Tensor ffn_out = FullyConnected(Mul(up, gate), down_proj);
  ffn_out.SetName(absl::StrCat(name, ".output"));
  return ffn_out;
}

// Rotary positional embedding (statically sliced for CPU/standard runners).
template <class... Mixins>
Tensor<Mixins...> ApplyRotaryEmbedding(const Tensor<Mixins...>& x,
                                       const Tensor<Mixins...>& cos,
                                       const Tensor<Mixins...>& sin) {
  const Shape& x_shape = x.GetShape();
  int num_dims = static_cast<int>(x_shape.size());

  std::vector<int> begin1(num_dims, 0);
  std::vector<int> size1(num_dims, -1);
  size1[num_dims - 1] = x_shape[num_dims - 1] / 2;

  std::vector<int> begin2(num_dims, 0);
  begin2[num_dims - 1] = x_shape[num_dims - 1] / 2;
  std::vector<int> size2(num_dims, -1);
  size2[num_dims - 1] = x_shape[num_dims - 1] / 2;

  Tensor x1 = Slice(x, begin1, size1);
  Tensor x2 = Slice(x, begin2, size2);

  // rotated = cat(-x2, x1).
  Tensor neg_one = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({-1.0f})});
  Tensor neg_x2 = Mul(x2, neg_one);
  Tensor rotated = Concatenation({neg_x2, x1}, /*axis=*/num_dims - 1);

  // Apply rotation: x * cos + rotated * sin.
  Tensor x_cos = Mul(x, cos);
  Tensor rotated_sin = Mul(rotated, sin);

  return Add(x_cos, rotated_sin);
}

// Rotary positional embedding.
template <class... Mixins>
Tensor<Mixins...> ApplyRotaryEmbedding(const Tensor<Mixins...>& x,
                                       const Tensor<Mixins...>& cos,
                                       const Tensor<Mixins...>& sin,
                                       const Tensor<Mixins...>& slice_offset_1,
                                       const Tensor<Mixins...>& slice_size_1,
                                       const Tensor<Mixins...>& slice_offset_2,
                                       const Tensor<Mixins...>& slice_size_2) {
  // Split x into first and second half along head_dim using dynamic CPU-staged
  // slices!
  Tensor x1 = Slice(x, slice_offset_1, slice_size_1);
  Tensor x2 = Slice(x, slice_offset_2, slice_size_2);

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
//
// `cache_params` is for MLDrift.
template <class... Mixins>
SelfAttentionOutput<Mixins...> MakeSelfAttentionLayer(
    const Tensor<Mixins...>& input, const std::string& name,
    const Config& config, bool is_sliding_attention,
    const Tensor<Mixins...>& attention_mask, const Tensor<Mixins...>& cos,
    const Tensor<Mixins...>& sin, const Tensor<Mixins...>& slice_offset_1,
    const Tensor<Mixins...>& slice_size_1,
    const Tensor<Mixins...>& slice_offset_2,
    const Tensor<Mixins...>& slice_size_2, const Tensor<Mixins...>& key_cache,
    const Tensor<Mixins...>& value_cache,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    const std::optional<Tensor<Mixins...>> cache_params = std::nullopt) {
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
  q = ApplyRotaryEmbedding(q, cos, sin, slice_offset_1, slice_size_1,
                           slice_offset_2, slice_size_2);
  k = ApplyRotaryEmbedding(k, cos, sin, slice_offset_1, slice_size_1,
                           slice_offset_2, slice_size_2);

  Tensor updated_key_cache = k;
  Tensor updated_value_cache = v;

  Tensor k_for_attn = k;
  Tensor v_for_attn = v;

  if (key_cache.GetStatus().ok() && value_cache.GetStatus().ok()) {
    const Shape& key_cache_shape = key_cache.GetShape();
    if (key_cache_shape.size() == 4 && key_cache_shape[2] > 0) {
      if (cache_params.has_value()) {
        std::vector<Tensor<Mixins...>> updated_caches =
            AddValuesToCache(key_cache, value_cache, k, v, *cache_params,
                             /*num_of_kv_heads=*/config.n_kv_groups,
                             /*kv_cache_batch_size=*/batch_size,
                             /*cache_size=*/key_cache_shape[2],
                             /*head_dimension=*/config.head_dim,
                             /*token_index_offset=*/0, /*active_tokens=*/1);
        updated_key_cache = updated_caches[0];
        updated_value_cache = updated_caches[1];
        k_for_attn = updated_key_cache;
        v_for_attn = updated_value_cache;
      } else {
        k_for_attn = Concatenation({key_cache, k}, /*axis=*/2);
        v_for_attn = Concatenation({value_cache, v}, /*axis=*/2);
      }
    }
  }

  int num_groups = config.n_heads / config.n_kv_groups;
  if (num_groups > 1) {
    k_for_attn = Tile(k_for_attn, {1, num_groups, 1, 1});
    v_for_attn = Tile(v_for_attn, {1, num_groups, 1, 1});
  }

  // Scaled dot-product attention.
  // scores = Q @ K^T / sqrt(query_pre_attn_scalar)
  Tensor scores = BatchMatMul(q, k_for_attn, /*adj_x=*/false, /*adj_y=*/true);

  // Scale by query_pre_attn_scalar^(-0.5).
  float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  Tensor scale_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({scale})});
  scores = Mul(scores, scale_tensor);

  // Apply attention mask (add -inf for masked positions).
  scores = Add(scores, attention_mask);

  // Softmax.
  Tensor attn_weights = Softmax(scores);

  // Attention output = weights @ V.
  Tensor attn_output = BatchMatMul(attn_weights, v_for_attn);

  // Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads *
  // head_dim].
  attn_output = Transpose(attn_output, {0, 2, 1, 3});
  attn_output = Reshape(attn_output, {batch_size, seq_len, qkv_out_dim});

  // Output projection.
  Tensor output = FullyConnected(attn_output, o_proj).SetName("attn_output");

  updated_key_cache.SetName(absl::StrCat(name, ".updated_key_cache"));
  updated_value_cache.SetName(absl::StrCat(name, ".updated_value_cache"));

  return {output, updated_key_cache, updated_value_cache};
}

// Creates two constant tensors that contain fixed Rotary Positional Embeddings
// for the given sequence length.
std::pair<TensorHandle, TensorHandle> RopeCosSin(int seq_len, int head_dim,
                                                 float rope_base);

// Computes the Rotary Positional Embeddings for the given sequence length.
//
// - The embeddings are computed from the `start` position.
// - `cos.size()` and `sin.size()` must be equal to `seq_len * head_dim`.
void RopeCosSin(int start, int seq_len, float rope_base, absl::Span<float> cos,
                absl::Span<float> sin);

// Perform embedding lookup.
//
// Updates the `embeddings` vector in place.
//
// Input: token indices [batch, seq_len]
// Output: embeddings [batch, seq_len, emb_dim]
void EmbeddingLookupCpu(const std::vector<int32_t>& tokens,
                        const float* embedding_table, int vocab_size,
                        int emb_dim, std::vector<float>& embeddings);

// Perform embedding lookup.
//
// Input: token indices [batch, seq_len]
// Output: embeddings [batch, seq_len, emb_dim]
std::vector<float> EmbeddingLookupCpu(const std::vector<int32_t>& tokens,
                                      const float* embedding_table,
                                      int vocab_size, int emb_dim);

// Computes the attention mask data.
absl::Status FillAttentionMask(const Shape& shape, absl::Span<float> mask,
                               bool is_local, int sliding_window_size = 0);

// Create a constant tensor holding an attention mask.
absl::StatusOr<TensorHandle> AttentionMask(Shape shape, bool is_local,
                                           int sliding_window_size = 0);

}  // namespace litert::tensor::examples::gemma3

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GRAPH_HELPERS_H_
