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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_WEBGPU_PIPELINE_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_WEBGPU_PIPELINE_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/cache/simple_cache.h"

namespace litert::ml_drift {

// Persistent cache for WebGPU pipeline objects.
//
// Clients can update the cache entries on any time. It's NOT thread safe.
// Clients are responsible to do the synchronization if needed.
//
// Updated/inserted entries are written to the file when the cache is destroyed.
//
// For the sake of simplicity, it's not a LRU cache. It can have at most
// |max_num_entries_| entries. Once the number of entries equals to
// |max_num_entries_|, it doesn't insert new entries any more. Clients can still
// update existing entries.
class WebGpuPipelineCache {
 public:
  WebGpuPipelineCache(SimpleCache cache, size_t max_num_entries);
  ~WebGpuPipelineCache();

  // Not copyable or movable.
  WebGpuPipelineCache(const WebGpuPipelineCache&) = delete;
  WebGpuPipelineCache(WebGpuPipelineCache&&) = delete;
  WebGpuPipelineCache& operator=(const WebGpuPipelineCache&) = delete;
  WebGpuPipelineCache& operator=(WebGpuPipelineCache&&) = delete;

  // Loads the data from the cache.
  // If the key is not found, it returns 0.
  // If the data buffer is nullptr or its size is 0, it returns the size of the
  // data without copying, which can be used to allocate the buffer.
  // If the data buffer is valid, its size must be equal or larger than the
  // size of the cached data.
  size_t Load(uint64_t key, absl::Span<uint8_t> data);

  // Stores the data in the cache.
  // If the key is already in the cache, it will be overwritten.
  // If the cache is full, it will return false.
  // Otherwise, it will store the data and return true.
  bool Store(uint64_t key, absl::Span<const uint8_t> data);

 private:
  SimpleCache cache_;
  const size_t max_num_entries_;
  ::ml_drift::MMapHandle mmap_handle_;
  absl::flat_hash_map<uint64_t, absl::Span<const uint8_t>> old_cache_;
  absl::flat_hash_map<uint64_t, std::vector<uint8_t>> new_cache_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_WEBGPU_PIPELINE_CACHE_H_
