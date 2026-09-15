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

// Artifact-only token-major INT8 KV storage for the active-extent experiment.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_KV_BANK_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_KV_BANK_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xnnpack.h"
#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"

namespace litert::tensor::examples::gemma4::native {

// Each owner stores K and V separately, both as contiguous [capacity, head_dim]
// INT8 codes. Allocation capacity never determines an attention view's extent.
// Owner IDs identify unique KV producers; shared consumer layers use the same
// owner's views and must not append another copy of its rows.
//
// BeginAppend establishes a transaction at the committed length. After an
// owner's Append, that owner's pending prefix is readable for attention, even
// while other owners have not run yet. Commit publishes the new length only
// after every owner has appended. Abort and Reset invalidate pending/logical
// rows without clearing their bytes; callers must request fresh views rather
// than continuing to use stale views after either operation.
//
// Buffers never resize after Create. A view's pointer remains allocated until
// this object is destroyed or move-assigned, but its logical validity depends
// on the current transaction. Moving this object transfers pointer ownership.
// XNN_EXTRA_BYTES of accessible padding follow each full-capacity allocation;
// padding and stale rows are never included in the returned logical spans.
// The object is not thread-safe; appends and graph reads must be sequenced.
class ActiveKvBank {
 public:
  static absl::StatusOr<ActiveKvBank> Create(std::vector<KvOwnerSpec> specs,
                                            int capacity) {
    if (capacity <= 0 || specs.empty()) {
      return absl::InvalidArgumentError("Positive capacity and owners required");
    }
    for (size_t i = 0; i < specs.size(); ++i) {
      const auto& spec = specs[i];
      if (spec.owner < 0 || spec.head_dim <= 0 ||
          !std::isfinite(spec.key_scale) || spec.key_scale <= 0 ||
          !std::isfinite(spec.value_scale) || spec.value_scale <= 0) {
        return absl::InvalidArgumentError("Invalid KV owner metadata");
      }
      for (size_t j = 0; j < i; ++j) {
        if (specs[j].owner == spec.owner) {
          return absl::InvalidArgumentError("Duplicate KV owner");
        }
      }
      if (static_cast<size_t>(capacity) >
          (std::numeric_limits<size_t>::max() - XNN_EXTRA_BYTES) /
              static_cast<size_t>(spec.head_dim)) {
        return absl::InvalidArgumentError("KV allocation size overflows");
      }
    }
    ActiveKvBank bank;
    bank.capacity_ = capacity;
    bank.specs_ = std::move(specs);
    bank.buffers_.resize(bank.specs_.size());
    for (size_t i = 0; i < bank.specs_.size(); ++i) {
      const size_t bytes = static_cast<size_t>(capacity) *
                               bank.specs_[i].head_dim +
                           XNN_EXTRA_BYTES;
      bank.buffers_[i].keys.resize(bytes, 0);
      bank.buffers_[i].values.resize(bytes, 0);
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
    for (auto& buffer : buffers_) buffer.appended = false;
    return absl::OkStatus();
  }

  // Both input blocks are [pending_count, head_dim], with the owner's original
  // quantization scales. This copies integer codes verbatim, without transpose,
  // requantization, prefix copying, or writes to any other owner's storage.
  absl::Status Append(int owner, absl::Span<const int8_t> keys,
                      absl::Span<const int8_t> values) {
    if (!append_in_flight()) {
      return absl::FailedPreconditionError("BeginAppend must precede Append");
    }
    const size_t index = Index(owner);
    if (index == specs_.size()) {
      return absl::InvalidArgumentError("Unknown KV owner");
    }
    auto& buffer = buffers_[index];
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

  absl::Status Commit() {
    if (!append_in_flight()) {
      return absl::FailedPreconditionError("No KV append to commit");
    }
    for (const auto& buffer : buffers_) {
      if (!buffer.appended) {
        return absl::FailedPreconditionError("KV owner has not appended");
      }
    }
    length_ += pending_count_;
    Abort();
    return absl::OkStatus();
  }

  // Uncommitted writes are exclusively beyond length_, so invalidating the
  // transaction preserves all committed bytes and views of the old prefix.
  void Abort() {
    pending_count_ = 0;
    for (auto& buffer : buffers_) buffer.appended = false;
  }

  void Reset() {
    Abort();
    length_ = 0;
  }

  absl::StatusOr<absl::Span<const int8_t>> Keys(int owner, int begin,
                                              int end) const {
    return View(owner, begin, end, false);
  }

  absl::StatusOr<absl::Span<const int8_t>> Values(int owner, int begin,
                                                int end) const {
    return View(owner, begin, end, true);
  }

  // Borrow [begin,padded_end) from the same token-major allocation while
  // validating [begin,valid_end) using the ordinary visibility rules. Padding
  // never extends logical history: [valid_end,padded_end) may contain stale
  // INT8 codes, including aborted writes or rows from an earlier session.
  // Attention MUST mask every padded position. A returned padded span must
  // never be treated as a Keys/Values view whose entire extent is valid.
  // Storage is initialized at Create, so every padded byte is accessible;
  // this does not promise zero values or zero attention without that mask.
  absl::StatusOr<absl::Span<const int8_t>> PaddedKeys(
      int owner, int begin, int valid_end, int padded_end) const {
    return PaddedView(owner, begin, valid_end, padded_end, false);
  }

  absl::StatusOr<absl::Span<const int8_t>> PaddedValues(
      int owner, int begin, int valid_end, int padded_end) const {
    return PaddedView(owner, begin, valid_end, padded_end, true);
  }

 private:
  struct OwnerBuffers {
    std::vector<int8_t> keys;
    std::vector<int8_t> values;
    bool appended = false;
  };

  ActiveKvBank() = default;

  size_t Index(int owner) const {
    for (size_t i = 0; i < specs_.size(); ++i) {
      if (specs_[i].owner == owner) return i;
    }
    return specs_.size();
  }

  absl::StatusOr<absl::Span<const int8_t>> View(int owner, int begin, int end,
                                              bool value) const {
    const size_t index = Index(owner);
    if (index == specs_.size()) {
      return absl::InvalidArgumentError("Unknown KV owner");
    }
    if (begin < 0 || end < begin || end > pending_end()) {
      return absl::OutOfRangeError("KV view exceeds visible token interval");
    }
    const auto& buffer = buffers_[index];
    if (end > length_ && !buffer.appended) {
      return absl::FailedPreconditionError("KV owner has not appended yet");
    }
    const auto& data = value ? buffer.values : buffer.keys;
    const size_t dim = specs_[index].head_dim;
    return absl::Span<const int8_t>(
        data.data() + static_cast<size_t>(begin) * dim,
        static_cast<size_t>(end - begin) * dim);
  }

  absl::StatusOr<absl::Span<const int8_t>> PaddedView(
      int owner, int begin, int valid_end, int padded_end, bool value) const {
    const auto valid = View(owner, begin, valid_end, value);
    if (!valid.ok()) return valid.status();
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

}  // namespace litert::tensor::examples::gemma4::native
#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_KV_BANK_H_
