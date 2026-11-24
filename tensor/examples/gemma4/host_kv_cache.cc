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

#include "tensor/examples/gemma4/host_kv_cache.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/examples/utils/ring_buffer.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::gemma4 {

void AppendTokenToKvCache(absl::Span<float> cache_buf,
                          absl::Span<const float> new_kv,
                          const int num_kv_heads, const int cache_len,
                          const int head_dim) {
  float* const cache = cache_buf.data();
  if (num_kv_heads == 1) {
    std::copy_n(new_kv.data(), head_dim, cache + cache_len * head_dim);
    return;
  }
  // Head `h` moves from offset `h * cache_len` to `h * (cache_len + 1)`
  // (in tokens). Going from the last head to the first one never overwrites
  // data that hasn't been moved yet.
  for (int h = num_kv_heads - 1; h >= 0; --h) {
    if (h > 0) {
      std::copy_backward(
          cache + h * cache_len * head_dim,
          cache + (h + 1) * cache_len * head_dim,
          cache + h * (cache_len + 1) * head_dim + cache_len * head_dim);
    }
    std::copy_n(new_kv.data() + h * head_dim, head_dim,
                cache + h * (cache_len + 1) * head_dim + cache_len * head_dim);
  }
}

absl::StatusOr<HostKvCache> HostKvCache::Create(const int num_kv_heads,
                                                const int head_dim,
                                                const int max_cache_len,
                                                const int window_len,
                                                const RingMode ring_mode) {
  if (num_kv_heads <= 0 || head_dim <= 0 || max_cache_len <= 0 ||
      window_len < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid HostKvCache dimensions: num_kv_heads=%d, head_dim=%d, "
        "max_cache_len=%d, window_len=%d.",
        num_kv_heads, head_dim, max_cache_len, window_len));
  }
  if (window_len == 0) {
    std::vector<float> storage(static_cast<size_t>(num_kv_heads) *
                               max_cache_len * head_dim);
    return HostKvCache(num_kv_heads, head_dim, max_cache_len,
                       std::move(storage));
  }
  const int capacity = std::min(max_cache_len, window_len);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      RingBuffer<float> storage,
      RingBuffer<float>::Create(
          static_cast<size_t>(num_kv_heads) * capacity * head_dim, ring_mode));
  return HostKvCache(num_kv_heads, head_dim, capacity, std::move(storage));
}

absl::Status HostKvCache::InitFromPrefill(absl::Span<const float> prefill_kv,
                                          const int seq_len) {
  const size_t token_size = static_cast<size_t>(num_kv_heads_) * head_dim_;
  if (seq_len < 0 || prefill_kv.size() < token_size * seq_len) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The prefill K/V should hold %d tokens of %d values but holds %d "
        "values.",
        seq_len, token_size, prefill_kv.size()));
  }

  if (auto* const vector = std::get_if<std::vector<float>>(&storage_)) {
    if (seq_len > capacity_) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "The prefill holds %d tokens but the KV cache only has room for %d.",
          seq_len, capacity_));
    }
    std::copy_n(prefill_kv.data(), token_size * seq_len, vector->data());
    len_ = seq_len;
    return absl::OkStatus();
  }

  RingBuffer<float>& ring = std::get<RingBuffer<float>>(storage_);
  ring.clear();
  if (seq_len <= capacity_) {
    ring.append_range(prefill_kv.first(token_size * seq_len));
    len_ = seq_len;
    return absl::OkStatus();
  }
  // Keep the last `capacity_` tokens of each head.
  const size_t head_window_size = static_cast<size_t>(capacity_) * head_dim_;
  LRT_TENSOR_RETURN_IF_ERROR(
      ring.resize(num_kv_heads_ * head_window_size, kNoInit));
  float* const window = ring.data();
  for (int h = 0; h < num_kv_heads_; ++h) {
    const float* const head_tail =
        prefill_kv.data() +
        (static_cast<size_t>(h) * seq_len + seq_len - capacity_) * head_dim_;
    std::copy_n(head_tail, head_window_size, window + h * head_window_size);
  }
  len_ = capacity_;
  return absl::OkStatus();
}

absl::Status HostKvCache::Push(absl::Span<const float> new_kv) {
  const size_t token_size = static_cast<size_t>(num_kv_heads_) * head_dim_;
  if (new_kv.size() < token_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The new K/V should hold %d values but holds %d.",
                        token_size, new_kv.size()));
  }

  if (auto* const vector = std::get_if<std::vector<float>>(&storage_)) {
    if (len_ == capacity_) {
      return absl::FailedPreconditionError(
          absl::StrFormat("The KV cache is full (%d tokens).", capacity_));
    }
    AppendTokenToKvCache(absl::MakeSpan(*vector), new_kv, num_kv_heads_, len_,
                         head_dim_);
    ++len_;
    return absl::OkStatus();
  }

  RingBuffer<float>& ring = std::get<RingBuffer<float>>(storage_);
  if (len_ < capacity_) {
    LRT_TENSOR_RETURN_IF_ERROR(ring.resize(ring.size() + token_size, kNoInit));
    AppendTokenToKvCache(absl::MakeSpan(ring), new_kv, num_kv_heads_, len_,
                         head_dim_);
    ++len_;
    return absl::OkStatus();
  }
  // The window is full: move it by one token. In the [num_kv_heads, capacity_,
  // head_dim] window, slot `capacity_ - 1` of head `h` now holds the dropped
  // token of head `h + 1` (or the new tail for the last head). Write the new
  // token of head `h` there.
  LRT_TENSOR_RETURN_IF_ERROR(ring.pop_front(head_dim_));
  LRT_TENSOR_RETURN_IF_ERROR(ring.resize(ring.size() + head_dim_, kNoInit));
  float* const window = ring.data();
  for (int h = 0; h < num_kv_heads_; ++h) {
    std::copy_n(new_kv.data() + h * head_dim_, head_dim_,
                window + (static_cast<size_t>(h) * capacity_ + capacity_ - 1) *
                             head_dim_);
  }
  return absl::OkStatus();
}

absl::Span<const float> HostKvCache::data() const {
  const size_t size = static_cast<size_t>(num_kv_heads_) * len_ * head_dim_;
  if (const auto* const vector = std::get_if<std::vector<float>>(&storage_)) {
    return absl::MakeConstSpan(vector->data(), size);
  }
  return absl::MakeConstSpan(std::get<RingBuffer<float>>(storage_));
}

}  // namespace litert::tensor::examples::gemma4
