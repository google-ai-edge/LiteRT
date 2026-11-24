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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HOST_KV_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HOST_KV_CACHE_H_

#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/examples/utils/ring_buffer.h"

namespace litert::tensor::examples::gemma4 {

// Appends one token worth of K or V data (`new_kv`, [num_kv_heads, head_dim])
// to a cache laid out as [num_kv_heads, cache_len, head_dim], growing it in
// place to [num_kv_heads, cache_len + 1, head_dim].
//
// `cache_buf` must hold at least `num_kv_heads * (cache_len + 1) * head_dim`
// values.
void AppendTokenToKvCache(absl::Span<float> cache_buf,
                          absl::Span<const float> new_kv, int num_kv_heads,
                          int cache_len, int head_dim);

// Host copy of the K or V cache of one layer, laid out as
// [num_kv_heads, len(), head_dim]. `data()` is meant to be bound to the decode
// graph without a copy.
//
// - Without a window (`window_len == 0`), the cache keeps every token, up to
//   `max_cache_len`.
// - With a window (`window_len > 0`), it only keeps the last `window_len`
//   tokens, in token order, in a `RingBuffer`. Once the window is
//   full, pushing a token writes it in place: nothing is moved or allocated.
class HostKvCache {
 public:
  using RingMode = RingBuffer<float>::Mode;

  // Creates an empty cache.
  //
  // - `window_len`: maximum number of tokens kept, or 0 to keep all of them.
  // - `ring_mode`: storage mode of the window (only used when `window_len >
  // 0`).
  static absl::StatusOr<HostKvCache> Create(
      int num_kv_heads, int head_dim, int max_cache_len, int window_len,
      RingMode ring_mode = RingMode::kAuto);

  HostKvCache(HostKvCache&&) = default;
  HostKvCache& operator=(HostKvCache&&) = default;

  // Replaces the contents of the cache with a prefill output
  // ([num_kv_heads, seq_len, head_dim]). With a window, only the last
  // `window_len` tokens are kept.
  absl::Status InitFromPrefill(absl::Span<const float> prefill_kv, int seq_len);

  // Pushes the K or V of one token ([num_kv_heads, head_dim]). With a full
  // window, the oldest token is dropped.
  absl::Status Push(absl::Span<const float> new_kv);

  // Number of tokens in the cache.
  int len() const { return len_; }

  // Maximum number of tokens in the cache.
  int capacity() const { return capacity_; }

  // Returns the cache contents, [num_kv_heads, len(), head_dim]. Invalidated
  // by `InitFromPrefill` and `Push`.
  absl::Span<const float> data() const;

 private:
  HostKvCache(int num_kv_heads, int head_dim, int capacity,
              std::variant<std::vector<float>, RingBuffer<float>> storage)
      : num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        capacity_(capacity),
        storage_(std::move(storage)) {}

  int num_kv_heads_;
  int head_dim_;
  int capacity_;
  int len_ = 0;
  // `std::vector`: every token, with room for `capacity_` of them.
  // `RingBuffer`: the last `capacity_` tokens.
  std::variant<std::vector<float>, RingBuffer<float>> storage_;
};

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HOST_KV_CACHE_H_
