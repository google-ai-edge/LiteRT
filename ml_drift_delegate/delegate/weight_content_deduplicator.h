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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEIGHT_CONTENT_DEDUPLICATOR_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEIGHT_CONTENT_DEDUPLICATOR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "tflite/core/c/common.h"

namespace litert::ml_drift {

// Deduplicates constant weight buffers across subgraphs and within a single
// subgraph (e.g., looped transformer layers whose unrolled passes reference
// distinct FlatBuffer buffer indices with identical weight bytes).
//
// By mapping duplicate constant buffers to a single canonical buffer ID before
// graph/IR construction, SharedMemoryManager packs, uploads, and serializes
// each unique weight tensor once and binds all consumers (across loop passes
// and subgraphs) to the same GPU SpatialTensor.
class WeightContentDeduplicator {
 public:
  using BufferIdMap = std::unordered_map<size_t, size_t>;

  WeightContentDeduplicator() = default;

  // Returns a buffer-id map where each constant tensor whose shape, dtype,
  // raw bytes, and quantization parameters match a previously observed buffer
  // is remapped to that buffer's canonical ID. Non-constant tensors or tensors
  // without inline host data (e.g., externalized/streamed weights) retain
  // their original buffer ID. Thread-safe.
  BufferIdMap CanonicalizeBufferIdMap(
      const TfLiteContext& context, const BufferIdMap& tensor_to_buffer_id_map);

  // Returns the number of distinct input buffer IDs that have been remapped
  // to an earlier canonical buffer ID. Thread-safe.
  size_t NumDeduplicatedBuffers() const;

 private:
  struct ContentKey {
    int32_t type = 0;
    uint64_t bytes = 0;
    uint64_t data_lo = 0;
    uint64_t data_hi = 0;
    uint64_t meta_lo = 0;
    uint64_t meta_hi = 0;

    bool operator==(const ContentKey& other) const = default;

    template <typename H>
    friend H AbslHashValue(H h, const ContentKey& k) {
      return H::combine(std::move(h), k.type, k.bytes, k.data_lo, k.data_hi,
                        k.meta_lo, k.meta_hi);
    }
  };

  // Computes a fingerprint key over the tensor's dtype, dimensions, raw bytes,
  // and quantization metadata. Returns std::nullopt when the tensor is not an
  // inline constant with valid host data.
  static std::optional<ContentKey> ComputeContentKey(
      const TfLiteContext& context, const TfLiteTensor& tensor);

  mutable absl::Mutex mutex_;
  // Maps original FlatBuffer buffer_id -> canonical buffer_id.
  absl::flat_hash_map<size_t, size_t> buffer_id_to_canonical_
      ABSL_GUARDED_BY(mutex_);
  // Maps content fingerprint key -> canonical buffer_id.
  absl::flat_hash_map<ContentKey, size_t> content_to_canonical_
      ABSL_GUARDED_BY(mutex_);
  size_t num_deduplicated_buffers_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_WEIGHT_CONTENT_DEDUPLICATOR_H_
