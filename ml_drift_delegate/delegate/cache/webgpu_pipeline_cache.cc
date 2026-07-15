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

#include "ml_drift_delegate/delegate/cache/webgpu_pipeline_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/cache/simple_cache.h"
#include "ml_drift_delegate/delegate/cache/webgpu_pipeline_cache_generated.h"

namespace litert::ml_drift {

WebGpuPipelineCache::WebGpuPipelineCache(SimpleCache cache,
                                         size_t max_num_entries)
    : cache_(std::move(cache)), max_num_entries_(max_num_entries) {
  auto status = cache_.Load([this](absl::Span<const uint8_t> data,
                                   ::ml_drift::MMapHandle& mmap_handle) {
    mmap_handle_ = std::move(mmap_handle);
    const auto* cache_fb = schema::GetWebGpuPipelineCache(mmap_handle_.data());
    for (const auto& entry : *cache_fb->entries()) {
      old_cache_[entry->key()] =
          absl::MakeSpan(entry->data()->data(), entry->data()->size());
    }
    return absl::OkStatus();
  });
  if (!status.ok()) {
    ABSL_LOG(WARNING) << "Failed to load cache file: " << cache_.ToString()
                      << ", status=" << status;
    ABSL_LOG(WARNING) << "Note that the cache will be created from scratch if "
                      << "any new entries are added.";
  } else {
    ABSL_LOG(INFO) << "Loaded cache file: " << cache_.ToString()
                   << ", num_entries=" << old_cache_.size();
  }
}

WebGpuPipelineCache::~WebGpuPipelineCache() {
  if (new_cache_.empty()) {
    ABSL_LOG(INFO) << "No new cache to write.";
    return;
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<schema::WebGpuPipelineCacheEntry>>
      entry_offsets;
  entry_offsets.reserve(old_cache_.size() + new_cache_.size());
  for (const auto& [key, data] : old_cache_) {
    entry_offsets.push_back(schema::CreateWebGpuPipelineCacheEntry(
        builder, key, builder.CreateVector(data.data(), data.size())));
  }
  for (const auto& [key, data] : new_cache_) {
    entry_offsets.push_back(schema::CreateWebGpuPipelineCacheEntry(
        builder, key, builder.CreateVector(data)));
  }
  auto entries_vec = builder.CreateVector(entry_offsets);
  schema::WebGpuPipelineCacheBuilder cache_builder(builder);
  cache_builder.add_entries(entries_vec);
  auto cache_fb = cache_builder.Finish();
  builder.Finish(cache_fb);

  auto status = cache_.Store(
      absl::MakeConstSpan(builder.GetBufferPointer(), builder.GetSize()));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to write cache file: " << cache_.ToString()
                    << ", status=" << status;
  } else {
    ABSL_LOG(INFO) << "Wrote cache file: " << cache_.ToString()
                   << ", new=" << new_cache_.size()
                   << ", total=" << old_cache_.size() + new_cache_.size();
  }
}

size_t WebGpuPipelineCache::Load(uint64_t key, absl::Span<uint8_t> data) {
  auto old_it = old_cache_.find(key);
  if (old_it != old_cache_.end()) {
    if (data.data() == nullptr || data.empty()) {
      return old_it->second.size();
    }
    memcpy(data.data(), old_it->second.data(), old_it->second.size());
    return old_it->second.size();
  }

  auto new_it = new_cache_.find(key);
  if (new_it != new_cache_.end()) {
    if (data.data() == nullptr || data.empty()) {
      return new_it->second.size();
    }
    memcpy(data.data(), new_it->second.data(), new_it->second.size());
    return new_it->second.size();
  }

  return 0;
}

bool WebGpuPipelineCache::Store(uint64_t key, absl::Span<const uint8_t> data) {
  size_t num_entries = old_cache_.size() + new_cache_.size();
  if (num_entries >= max_num_entries_) {
    if (!old_cache_.contains(key) && !new_cache_.contains(key)) {
      ABSL_LOG(WARNING) << "Cache is full, cannot store new entries.";
      return false;
    }
  }

  old_cache_.erase(key);
  new_cache_[key] =
      std::vector<uint8_t>(data.data(), data.data() + data.size());
  return true;
}

}  // namespace litert::ml_drift
