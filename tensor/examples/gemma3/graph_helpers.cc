// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensor/examples/gemma3/graph_helpers.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma3 {

std::pair<TensorHandle, TensorHandle> RopeCosSin(const int seq_len,
                                                 const int head_dim,
                                                 const float rope_base) {
  const Shape shape({1, 1, seq_len, head_dim});
  auto cos_data = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  auto sin_data = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  TensorHandle cos(
      {.name = "cos", .type = Type::kFP32, .shape = shape, .buffer = cos_data});
  TensorHandle sin(
      {.name = "sin", .type = Type::kFP32, .shape = shape, .buffer = sin_data});
  RopeCosSin(/*start=*/0, seq_len, rope_base, cos_data->Span<float>(),
             sin_data->Span<float>());
  return {cos, sin};
}

void RopeCosSin(const int start, const int seq_len, const float rope_base,
                const absl::Span<float> cos, const absl::Span<float> sin) {
  const int head_dim = cos.size() / seq_len;
  const int half_dim = head_dim / 2;
  std::vector<float> inv_freq(half_dim);
  for (int i = 0; i < half_dim; ++i) {
    inv_freq[i] = 1.0f / std::pow(rope_base, 2.0f * i / head_dim);
  }
  for (int s = 0, position = start; s < seq_len; ++s, ++position) {
    const size_t embedding_offset = s * head_dim;
    for (int i = 0; i < half_dim; ++i) {
      const float angle = static_cast<float>(position) * inv_freq[i];
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      // Duplicate both halves to match ApplyRotaryEmbedding implementation.
      cos[embedding_offset + i] = cos_val;
      cos[embedding_offset + half_dim + i] = cos_val;
      sin[embedding_offset + i] = sin_val;
      sin[embedding_offset + half_dim + i] = sin_val;
    }
  }
}

void EmbeddingLookupCpu(const std::vector<int32_t>& tokens,
                        const float* embedding_table, int vocab_size,
                        int emb_dim, std::vector<float>& embeddings) {
  const size_t seq_len = tokens.size();
  embeddings.resize(seq_len * emb_dim);

  for (size_t i = 0; i < seq_len; ++i) {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= vocab_size) {
      ABSL_LOG(WARNING) << "Token ID " << token_id << " out of range [0, "
                        << vocab_size << "), using 0";
      token_id = 0;
    }
    // Copy embedding for this token (no scaling)
    const float* src = embedding_table + token_id * emb_dim;
    float* dst = embeddings.data() + i * emb_dim;
    std::copy(src, src + emb_dim, dst);
  }
}

std::vector<float> EmbeddingLookupCpu(const std::vector<int32_t>& tokens,
                                      const float* embedding_table,
                                      int vocab_size, int emb_dim) {
  std::vector<float> embeddings;
  EmbeddingLookupCpu(tokens, embedding_table, vocab_size, emb_dim, embeddings);
  return embeddings;
}

absl::Status FillAttentionMask(const Shape& shape, const absl::Span<float> mask,
                               const bool is_local,
                               const int sliding_window_size) {
  if (shape.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask output shape must have at least 2 dims, got %zu",
        shape.size()));
  }

  const int64_t seq_q = shape[shape.size() - 2];
  const int64_t seq_k = shape[shape.size() - 1];
  if (seq_q <= 0 || seq_k <= 0 || seq_q != seq_k) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask expects output shape [..., S, S] with S>0; got [%s]",
        absl::StrJoin(shape, ", ")));
  }
  if (std::any_of(shape.begin(), shape.end() - 2,
                  [](auto d) { return d <= 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask does not support non-positive "
                        "leading dims. Got shape [%s]",
                        absl::StrJoin(shape, ", ")));
  }
  const int64_t leading =
      std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<>());
  const int64_t matrix_size = seq_q * seq_q;
  const int64_t tensor_size = leading * matrix_size;

  if (mask.size() != tensor_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask output mask should hold %" PRIi64
                        " elements but holds %zu",
                        tensor_size, mask.size()));
  }

  const float neg_inf = std::numeric_limits<float>::lowest();

  for (int64_t b = 0; b < leading; ++b) {
    const int64_t base = b * matrix_size;
    for (int64_t i = 0; i < seq_q; ++i) {
      const int64_t I = i * seq_q;
      for (int64_t j = 0; j < seq_q; ++j) {
        const bool is_causal_masked = (j > i);
        const bool is_sliding_masked = is_local && (sliding_window_size > 0) &&
                                       (i - j >= sliding_window_size);
        mask[static_cast<size_t>(base + I + j)] =
            (is_causal_masked || is_sliding_masked) ? neg_inf : 0.0f;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorHandle> AttentionMask(Shape shape, const bool is_local,
                                           const int sliding_window_size) {
  auto buffer = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  TensorHandle mask({.name = "attention_mask",
                     .type = Type::kFP32,
                     .shape = std::move(shape),
                     .buffer = buffer});
  LRT_TENSOR_RETURN_IF_ERROR(FillAttentionMask(
      mask.GetShape(), buffer->Span<float>(), is_local, sliding_window_size));
  return mask;
}

}  // namespace litert::tensor::examples::gemma3
