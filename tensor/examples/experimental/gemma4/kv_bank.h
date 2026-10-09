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

// Token-major INT8 KV storage with transactional appends and active views.
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_BANK_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_BANK_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/examples/experimental/gemma4/host_buffer.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Static-sized, contiguous token-major INT8 KV cache bank with transactional
// appends and active-prefix views (not a paged KV cache).
//
// Motivation & Design:
// Baseline implementations compute attention over the entire pre-allocated KV
// cache capacity on every step. Instead, `ActiveKvBank` pre-allocates fixed
// `[capacity, head_dim]` contiguous INT8 buffers once at startup (`Create`),
// and exposes zero-copy `absl::Span` slices covering only the active token
// range (e.g. `[0, length)` for global attention or the active sliding window
// for local attention). The runner binds only the active slice to the graph's
// `past_k` / `past_v` inputs and appends the newly produced KV rows back into
// the bank after execution. Although slicing and concatenating inside the
// TFLite graph may still copy the active prefix (a current limitation without
// custom TFLite in-place mutation ops), restricting attention computation to
// the active slice rather than the full `capacity` yields a major speedup.
//
// Storage Layout & KV Sharing:
// Each KV owner layer and head stores K and V separately as contiguous
// `[capacity, head_dim]` INT8 codes, copied verbatim by `Append` without
// runtime transposition or requantization. Owner IDs identify the earlier
// layers that produce KV caches; later layers in Gemma4 that share an earlier
// layer's KV cache read the same owner's views and do not append again.
//
// Buffer Row Layout (for a given owner and head):
//   |<------ [0, L) ------>|<--- [L, L + N) --->|<--- [L + N, capacity) --->|
//   |  Committed history   |   Pending append   |     Unused / padding      |
//   | (always readable via | (readable for this | (only accessible via      |
//   |  Keys() / Values())  |  owner/head after  |  PaddedKeys/PaddedValues; |
//   |                      |  its Append() call)|  must be masked out)      |
//
// Transaction State Machine (L = length(), N = pending_count):
//
//   Create() / Reset()
//          |
//          v
//       [ Idle ] <-------------------------------------------+
//       (length = L, pending = 0)                            |
//          |                                                 |
//          | BeginAppend(N)                                  | Abort()
//          v                                                 | (discards N,
//     [ Appending ] -----------------------------------------+  keeps L)
//     (length = L, pending = N)
//          |
//          | Append(owner, keys, values, head)
//          | (called once per owner/head; writes to [L, L + N))
//          v
//   [ All owners/heads appended ]
//          |
//          | Commit() (publishes new length = L + N)
//          v
//       [ Idle ]
//       (length = L + N, pending = 0)
//
//   Note: `Reset()` transitions from any state to `Idle (length = 0)`.
//
// Lifetime & Safety:
// Buffers never reallocate after `Create`. Each buffer allocation includes 64
// bytes of accessible tail padding beyond `capacity * head_dim` for SIMD loads;
// padding and stale rows are never included in `Keys()` / `Values()` spans.
// `Abort()` and `Reset()` roll back logical row counts without zeroing memory,
// so callers must request fresh spans after either call. Not thread-safe.
class ActiveKvBank {
 public:
  // Allocates contiguous `[capacity, head_dim]` INT8 K and V buffers (plus 64
  // bytes of tail padding) for every owner and head described by `specs`.
  static absl::StatusOr<ActiveKvBank> Create(std::vector<KvOwnerSpec> specs,
                                             int capacity) {
    if (capacity <= 0 || specs.empty()) {
      return absl::InvalidArgumentError(
          "Positive capacity and owners required");
    }
    for (size_t i = 0; i < specs.size(); ++i) {
      const KvOwnerSpec& spec = specs[i];
      if (spec.owner < 0 || spec.head_dim <= 0 || spec.num_heads <= 0 ||
          spec.num_heads > 8 || !std::isfinite(spec.key_scale) ||
          spec.key_scale <= 0 || !std::isfinite(spec.value_scale) ||
          spec.value_scale <= 0) {
        return absl::InvalidArgumentError("Invalid KV owner metadata");
      }
      for (size_t j = 0; j < i; ++j) {
        if (specs[j].owner == spec.owner) {
          return absl::InvalidArgumentError("Duplicate KV owner");
        }
      }
      if (static_cast<size_t>(capacity) >
          (std::numeric_limits<size_t>::max() - 64) /
              static_cast<size_t>(spec.head_dim)) {
        return absl::InvalidArgumentError("KV allocation size overflows");
      }
    }
    ActiveKvBank bank;
    bank.capacity_ = capacity;
    bank.specs_ = std::move(specs);
    bank.buffers_.resize(bank.specs_.size());
    for (size_t i = 0; i < bank.specs_.size(); ++i) {
      const size_t bytes =
          static_cast<size_t>(capacity) * bank.specs_[i].head_dim + 64;
      bank.buffers_[i].heads.resize(bank.specs_[i].num_heads);
      for (HeadBuffers& head : bank.buffers_[i].heads) {
        head.keys.Resize(bytes);
        head.values.Resize(bytes);
      }
    }
    return bank;
  }

  ActiveKvBank(ActiveKvBank&&) = default;
  ActiveKvBank& operator=(ActiveKvBank&&) = default;
  ActiveKvBank(const ActiveKvBank&) = delete;
  ActiveKvBank& operator=(const ActiveKvBank&) = delete;

  const std::vector<KvOwnerSpec>& specs() const { return specs_; }
  int capacity() const { return capacity_; }
  int length() const { return length_; }
  int pending_end() const { return length_ + pending_count_; }
  bool append_in_flight() const { return pending_count_ != 0; }

  // Starts an append transaction for `count` new token rows at `[length(),
  // length() + count)`. Transitions from `Idle` to `Appending`.
  absl::Status BeginAppend(int count) {
    if (append_in_flight()) {
      return absl::FailedPreconditionError("KV append already in flight");
    }
    if (count <= 0) {
      return absl::InvalidArgumentError("Append count must be positive");
    }
    if (count > capacity_ - length_) {
      return absl::OutOfRangeError("KV append exceeds allocation capacity");
    }
    pending_count_ = count;
    for (OwnerBuffers& owner : buffers_) {
      for (HeadBuffers& head : owner.heads) {
        head.appended = false;
      }
    }
    return absl::OkStatus();
  }

  // Copies `[pending_count, head_dim]` INT8 `keys` and `values` into `[length_,
  // length_ + pending_count_)` for `(owner, head)` using the owner's original
  // quantization scales. Copies integer codes verbatim without transpose,
  // requantization, prefix copying, or writes to any other owner's storage.
  absl::Status Append(int owner, absl::Span<const int8_t> keys,
                      absl::Span<const int8_t> values, int head = 0) {
    if (!append_in_flight()) {
      return absl::FailedPreconditionError("BeginAppend must precede Append");
    }
    const size_t index = Index(owner);
    if (index == specs_.size()) {
      return absl::InvalidArgumentError("Unknown KV owner");
    }
    if (head < 0 || head >= specs_[index].num_heads) {
      return absl::InvalidArgumentError("Invalid KV head");
    }
    HeadBuffers& buffer = buffers_[index].heads[head];
    if (buffer.appended) {
      return absl::FailedPreconditionError("KV owner already appended");
    }
    const size_t dim = specs_[index].head_dim;
    const size_t bytes = static_cast<size_t>(pending_count_) * dim;
    if (keys.size() != bytes || values.size() != bytes) {
      return absl::InvalidArgumentError("KV block size does not match append");
    }
    const size_t offset = static_cast<size_t>(length_) * dim;
    std::memmove(buffer.keys.data() + offset, keys.data(), bytes);
    std::memmove(buffer.values.data() + offset, values.data(), bytes);
    buffer.appended = true;
    return absl::OkStatus();
  }

  // Verifies that every owner and head has appended its `pending_count_` rows,
  // advances `length_` by `pending_count_`, and returns to `Idle`.
  absl::Status Commit() {
    if (!append_in_flight()) {
      return absl::FailedPreconditionError("No KV append to commit");
    }
    for (const OwnerBuffers& owner : buffers_) {
      for (const HeadBuffers& head : owner.heads) {
        if (!head.appended) {
          return absl::FailedPreconditionError(
              "KV owner/head has not appended");
        }
      }
    }
    length_ += pending_count_;
    Abort();
    return absl::OkStatus();
  }

  // Rolls back an in-flight append transaction and returns to `Idle` at the
  // prior committed `length_`. Uncommitted writes are exclusively beyond
  // `length_`, so all committed bytes in `[0, length_)` remain intact.
  void Abort() {
    pending_count_ = 0;
    for (OwnerBuffers& owner : buffers_) {
      for (HeadBuffers& head : owner.heads) {
        head.appended = false;
      }
    }
  }

  // Aborts any in-flight transaction and resets the committed `length_` to 0
  // without zeroing the underlying buffer memory.
  void Reset() {
    Abort();
    length_ = 0;
  }

  // Returns a read-only view of key rows `[begin, end)` for `(owner, head)`.
  absl::StatusOr<absl::Span<const int8_t>> Keys(int owner, int begin, int end,
                                                int head = 0) const {
    return View(owner, begin, end, false, head);
  }

  // Returns a read-only view of value rows `[begin, end)` for `(owner, head)`.
  absl::StatusOr<absl::Span<const int8_t>> Values(int owner, int begin, int end,
                                                  int head = 0) const {
    return View(owner, begin, end, true, head);
  }

  // Borrows `[begin, padded_end)` from the token-major allocation while
  // validating `[begin, valid_end)` using the ordinary visibility rules.
  // Padding never extends logical history: `[valid_end, padded_end)` may
  // contain stale INT8 codes (including aborted writes or rows from an earlier
  // session), so attention MUST mask every padded position. Storage is
  // zero-initialized at `Create`, so every padded byte is safely readable.
  absl::StatusOr<absl::Span<const int8_t>> PaddedKeys(int owner, int begin,
                                                      int valid_end,
                                                      int padded_end,
                                                      int head = 0) const {
    return PaddedView(owner, begin, valid_end, padded_end, false, head);
  }

  // Borrows `[begin, padded_end)` of value rows with the same validity and
  // masking requirements as `PaddedKeys`.
  absl::StatusOr<absl::Span<const int8_t>> PaddedValues(int owner, int begin,
                                                        int valid_end,
                                                        int padded_end,
                                                        int head = 0) const {
    return PaddedView(owner, begin, valid_end, padded_end, true, head);
  }

 private:
  struct HeadBuffers {
    HostBuffer keys;
    HostBuffer values;
    bool appended = false;
  };

  struct OwnerBuffers {
    std::vector<HeadBuffers> heads;
  };

  ActiveKvBank() = default;

  size_t Index(int owner) const {
    for (size_t i = 0; i < specs_.size(); ++i) {
      if (specs_[i].owner == owner) {
        return i;
      }
    }
    return specs_.size();
  }

  absl::StatusOr<absl::Span<const int8_t>> View(int owner, int begin, int end,
                                                bool value, int head) const {
    const size_t index = Index(owner);
    if (index == specs_.size()) {
      return absl::InvalidArgumentError("Unknown KV owner");
    }
    if (begin < 0 || end < begin || end > pending_end()) {
      return absl::OutOfRangeError("KV view exceeds visible token interval");
    }
    if (head < 0 || head >= specs_[index].num_heads) {
      return absl::InvalidArgumentError("Invalid KV head");
    }
    const HeadBuffers& buffer = buffers_[index].heads[head];
    if (end > length_ && !buffer.appended) {
      return absl::FailedPreconditionError("KV owner has not appended yet");
    }
    const HostBuffer& data = value ? buffer.values : buffer.keys;
    const size_t dim = specs_[index].head_dim;
    return absl::Span<const int8_t>(
        reinterpret_cast<const int8_t*>(data.data()) +
            static_cast<size_t>(begin) * dim,
        static_cast<size_t>(end - begin) * dim);
  }

  absl::StatusOr<absl::Span<const int8_t>> PaddedView(int owner, int begin,
                                                      int valid_end,
                                                      int padded_end,
                                                      bool value,
                                                      int head) const {
    const absl::StatusOr<absl::Span<const int8_t>> valid =
        View(owner, begin, valid_end, value, head);
    if (!valid.ok()) {
      return valid.status();
    }
    if (padded_end < valid_end || padded_end > capacity_) {
      return absl::OutOfRangeError("KV padding exceeds allocation capacity");
    }
    const size_t dim = specs_[Index(owner)].head_dim;
    return absl::Span<const int8_t>(
        valid->data(), static_cast<size_t>(padded_end - begin) * dim);
  }

  std::vector<KvOwnerSpec> specs_;
  std::vector<OwnerBuffers> buffers_;
  int capacity_ = 0;
  int length_ = 0;
  int pending_count_ = 0;
};

}  // namespace litert::tensor::examples::experimental::gemma4::cpu
#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_KV_BANK_H_
