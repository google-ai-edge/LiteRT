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

#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cerrno>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/serialization_base.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/build_identifier.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"

namespace ml_drift {

namespace {

// Returns the next offset value that is aligned to `alignement`.
size_t Align(size_t offset, const size_t alignment) {
  const size_t misalign = offset % alignment;
  return offset + (misalign ? alignment - misalign : 0);
}

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(const char* path) {
  // Use strncmp to check for the prefix.
  return !strncmp(path, ml_drift::kInMemoryCachePath,
                  sizeof(ml_drift::kInMemoryCachePath) - 1);
}

}  // namespace

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(absl::string_view path) {
  return IsInMemoryCachePath(path.data());
}

CacheBuilder::CacheBuilder(CacheBuilder&& other)
    : data_(std::move(other.data_)),
      header_base_offset_(std::move(other.header_base_offset_)),
      capacity_(std::move(other.capacity_)),
      build_segment_size_(std::move(other.build_segment_size_)),
      build_segment_start_(std::move(other.build_segment_start_)),
      first_write_done_(std::move(other.first_write_done_)),
      fd_(std::move(other.fd_)),
      file_path_(std::move(other.file_path_)) {}

CacheBuilder& CacheBuilder::operator=(CacheBuilder&& other) {
  data_ = std::move(other.data_);
  header_base_offset_ = std::move(other.header_base_offset_);
  capacity_ = std::move(other.capacity_);
  build_segment_size_ = std::move(other.build_segment_size_);
  build_segment_start_ = std::move(other.build_segment_start_);
  first_write_done_ = std::move(other.first_write_done_);
  fd_ = std::move(other.fd_);
  file_path_ = std::move(other.file_path_);
  return *this;
}

absl::Status CacheBuilder::Start(absl::string_view path,
                                 uint64_t unique_model_identifier,
                                 bool append) {
  if (IsStarted()) {
    return absl::InvalidArgumentError(
        "Cannot start a new build step when the cache is already started.");
  }
  file_path_ = path;

  if (IsInMemoryCachePath(file_path_)) {
    fd_ = CreateInMemoryFileDescriptor("MLDrift in-memory weight cache");
  } else {
    int flags = O_CREAT | O_RDWR;  // NOLINT: b/332641196
    if (!append) {
      flags |= O_TRUNC;  // NOLINT: b/332641196
    }
    fd_ = FileDescriptor::Open(file_path_.c_str(), flags, 0644);
  }
  if (!fd_.IsValid()) {
    return absl::InternalError(absl::StrCat(
        "Could not open file ('", file_path_, "'): ", strerror(errno)));
  }

  return StartInternal(unique_model_identifier, append);
}

absl::Status CacheBuilder::StartInternal(uint64_t unique_model_identifier,
                                         bool append) {
  header_base_offset_ = Align(sizeof(MLDriftCacheHeader), kMinAlignment);

  unique_model_identifier_ = unique_model_identifier;

  if (append) {
    // If we are appending, we don't want to overwrite the existing header yet.
    // The existing data will be loaded when StartBuildStep is called
    // explicitly.
    return absl::OkStatus();
  }

  // Write data in the header, this will be overwritten in the `Finalize` call.
  // We explicitly set the header as invalid. If any error happens during
  // the build, reloading the cache file will fail.
  MLDriftCacheHeader header{MLDriftCacheHeader::kInvalidHeader};
  header.buffer_list_offset = sizeof(header);

  if (!fd_.Write(&header, sizeof(header))) {
    return absl::InternalError(
        absl::StrCat("could not write initial cache header in ", file_path_));
  }

  ABSL_RETURN_IF_ERROR(StartBuildStep(unique_model_identifier));
  ABSL_RETURN_IF_ERROR(StopBuildStep());
  return absl::OkStatus();
}

absl::Status CacheBuilder::Start(FileDescriptor&& fd,
                                 uint64_t unique_model_identifier,
                                 bool append) {
  if (IsStarted()) {
    return absl::InvalidArgumentError(
        "Cannot start a new build step when the cache is already started.");
  }
  fd_ = std::move(fd);
  if (!fd_.IsValid()) {
    return absl::InvalidArgumentError("Invalid file descriptor.");
  }

  file_path_ = "FileDescriptor";

  return StartInternal(unique_model_identifier, append);
}

absl::Status CacheBuilder::StartBuildStep(uint64_t unique_model_identifier) {
  if (!IsStarted()) {
    return absl::InvalidArgumentError(
        "Cannot start a new build step when the cache is not started.");
  }
  if (is_build_step_.exchange(true)) {
    return absl::InvalidArgumentError(
        "Failed to start build step: already started. This may be a "
        "concurrency issue.");
  }
  unique_model_identifier_ = unique_model_identifier;

  // Clear the builder and lists since we are going to reload from disk.
  builder_.Clear();
  buffer_fb_list_.clear();
  other_subgraphs_offsets_.clear();

  // Reload flatbuffer data.
  MLDriftCacheHeader header;
  fd_.SetPos(0);
  if (!fd_.Read(&header, sizeof(header))) {
    return absl::InternalError("could not read cache file header.");
  }
  if (header.buffer_list_size) {
    MMapHandle buffer_list_handle;
    ABSL_RETURN_IF_ERROR(buffer_list_handle.Map(fd_, header.buffer_list_offset,
                                                header.buffer_list_size,
                                                file_path_.c_str()));

    if (buffer_list_handle.size() < header.buffer_list_size) {
      return absl::InternalError("Invalid buffer list size");
    }

    // Verify the flatbuffer part of the file.
    flatbuffers::Verifier verifier(buffer_list_handle.data(),
                                   header.buffer_list_size);
    if (!ml_drift::cache::schema::VerifyModelCacheBuffer(verifier)) {
      return absl::InternalError("Model cache validation failed.");
    }

    // Load flatbuffer.
    const ml_drift::cache::schema::ModelCache* model_cache =
        ml_drift::cache::schema::GetModelCache(buffer_list_handle.data());
    if (!model_cache) {
      return absl::InternalError("Could not load model cache from cache file.");
    }

    const ml_drift::cache::schema::SubgraphBufferList* current_subgraph =
        nullptr;
    if (model_cache->subgraphs()) {
      if (model_cache->subgraphs()->size() > kMaxSupportedSubgraphs) {
        return absl::InternalError("Corrupted cache: Too many subgraphs.");
      }
      for (const auto* subgraph : *model_cache->subgraphs()) {
        if (subgraph->unique_model_identifier() == unique_model_identifier_) {
          current_subgraph = subgraph;
        } else {
          // Re-encode this subgraph into the new builder.
          std::vector<flatbuffers::Offset<ml_drift::cache::schema::Buffer>>
              existing_buffers;
          if (subgraph->buffers()) {
            for (const auto* buffer : *subgraph->buffers()) {
              TensorDescriptor tensor_desc;
              ABSL_RETURN_IF_ERROR(
                  Decode(buffer->tensor_descriptor(), &tensor_desc));
              BufferLocation loc = {static_cast<size_t>(buffer->offset()),
                                    static_cast<size_t>(buffer->size())};
              auto buffer_fb =
                  EncodeBuffer(buffer->global_tensor_id(),
                               buffer->is_quantization_param_tensor(),
                               tensor_desc, loc, &builder_);
              existing_buffers.push_back(buffer_fb);
            }
          }
          auto existing_buffers_vec = builder_.CreateVector(existing_buffers);
          ml_drift::cache::schema::SubgraphBufferListBuilder
              existing_subgraph_builder(builder_);
          existing_subgraph_builder.add_unique_model_identifier(
              subgraph->unique_model_identifier());
          existing_subgraph_builder.add_buffers(existing_buffers_vec);
          existing_subgraph_builder.add_base_offset(subgraph->base_offset());
          other_subgraphs_offsets_.push_back(
              existing_subgraph_builder.Finish());
        }
      }
    }

    if (current_subgraph && current_subgraph->buffers()) {
      for (const ml_drift::cache::schema::Buffer* buffer :
           *current_subgraph->buffers()) {
        if (!buffer->tensor_descriptor()) {
          return absl::InternalError("Buffer has no tensor descriptor.");
        }
        TensorDescriptor tensor_desc;
        ABSL_RETURN_IF_ERROR(Decode(buffer->tensor_descriptor(), &tensor_desc));
        BufferLocation loc = {static_cast<size_t>(buffer->offset()),
                              static_cast<size_t>(buffer->size())};
        auto buffer_fb = EncodeBuffer(buffer->global_tensor_id(),
                                      buffer->is_quantization_param_tensor(),
                                      tensor_desc, loc, &builder_);
        buffer_fb_list_.push_back(buffer_fb);
      }
    }
  }
  buffer_fb_list_changed_ = false;

  build_segment_size_ = 0;
  build_segment_start_ = fd_.SetPos(header.buffer_list_offset);
  if (build_segment_start_ < 0) {
    return absl::InternalError("Could not move in the file.");
  }

  return absl::OkStatus();
}

void* CacheBuilder::Reserve(size_t size) {
  if (size > capacity_) {
    // We don't care about the data when we are reserving space. We save memory
    // by deleting the existing buffer first.
    data_.reset(nullptr);
    uint8_t* ptr = new (std::nothrow) uint8_t[size + kMinAlignment];
    if (ptr == nullptr) {
      return nullptr;
    }
    data_.reset(ptr);
    capacity_ = size;
  }
  return reinterpret_cast<void*>(
      Align(reinterpret_cast<size_t>(data_.get()), kMinAlignment));
}

absl::Status CacheBuilder::Append(uint32_t global_tensor_id,
                                  bool is_quantization_param_tensor,
                                  const TensorDescriptor& tensor_desc,
                                  const void* data, uint64_t size,
                                  BufferLocation& loc) {
  if (!IsStarted()) {
    return absl::InvalidArgumentError(
        "Cannot append data to an unstarted builder.");
  }
  if (!is_build_step_) {
    return absl::InvalidArgumentError(
        "Cannot append data to a non build step.");
  }

  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t offset = Align(fd_.GetPos(), kMinAlignment);
  if (fd_.SetPos(offset) == -1) {
    loc = BufferLocation::Invalid();
    return absl::InternalError(
        "Could not move cursor to the buffer list offset");
  }

  loc.offset = offset - header_base_offset_;
  loc.size = size;

  auto buffer_fb = EncodeBuffer(global_tensor_id, is_quantization_param_tensor,
                                tensor_desc, loc, &builder_);

  if (!fd_.Write(data, size)) {
    loc = BufferLocation::Invalid();
    return absl::InternalError(
        "MLDrift weight cache: cannot append buffer to cache file");
  }

  buffer_fb_list_.push_back(buffer_fb);
  buffer_fb_list_changed_ = true;

  return absl::OkStatus();
}

absl::Status CacheBuilder::StopBuildStep() {
  if (!is_build_step_.exchange(false)) {
    return absl::InvalidArgumentError(
        "Attempting to stop a non existing build step. This may "
        "be a concurrency issue.");
  }
  if (!fd_.IsValid()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cache file ", file_path_, " is not open for writing: .",
                     strerror(errno)));
  }

  // Use other subgraphs loaded in StartBuildStep.
  std::vector<flatbuffers::Offset<ml_drift::cache::schema::SubgraphBufferList>>
      subgraph_fb_list = other_subgraphs_offsets_;

  // Create SubgraphBufferList for current subgraph.
  auto buffer_fb_vec = builder_.CreateVector(buffer_fb_list_);
  ml_drift::cache::schema::SubgraphBufferListBuilder subgraph_builder(builder_);
  subgraph_builder.add_unique_model_identifier(unique_model_identifier_);
  subgraph_builder.add_buffers(buffer_fb_vec);
  subgraph_builder.add_base_offset(header_base_offset_);
  auto current_subgraph_fb = subgraph_builder.Finish();

  subgraph_fb_list.push_back(current_subgraph_fb);

  // Create ModelCache.
  auto subgraphs_vec = builder_.CreateVector(subgraph_fb_list);
  ml_drift::cache::schema::ModelCacheBuilder model_cache_builder(builder_);
  model_cache_builder.add_subgraphs(subgraphs_vec);
  auto model_cache_fb = model_cache_builder.Finish();

  ml_drift::cache::schema::FinishModelCacheBuffer(builder_, model_cache_fb);

  // Add some padding so that the cache file can be mmaped and the buffer
  // stays aligned correctly.
  const size_t layout_offset = Align(fd_.GetPos(), kMinAlignment);
  if (fd_.SetPos(layout_offset) == -1) {
    return absl::InternalError(
        absl::StrCat("Could not move in the file: ", strerror(errno)));
  }

  absl::Span<const uint8_t> build_identifier = GetBuildIdentifier();
  if (sizeof(MLDriftCacheHeader::mldrift_build_identifier) !=
      build_identifier.size()) {
    return absl::InternalError(absl::StrCat(
        "Cache file (", file_path_,
        ") header cannot hold MLDrift's build identifier: ", strerror(errno)));
  }

  MLDriftCacheHeader header{MLDriftCacheHeader::kVersion};
  memcpy(header.mldrift_build_identifier, build_identifier.data(),
         build_identifier.size());
  header.buffer_list_offset = fd_.GetPos();
  header.buffer_list_size = builder_.GetSize();

  // Write the flatbuffer which serves as a header to index the buffer data.
  if (!fd_.Write(builder_.GetBufferPointer(), builder_.GetSize())) {
    return absl::InternalError(absl::StrCat(
        "Cannot write buffer list to '", file_path_, "'. ", strerror(errno)));
  }

  // Save the segment size for that it can be individually mapped.
  build_segment_size_ = fd_.GetPos() - build_segment_start_;

  // Write the header at the beginning of the file.
  if (fd_.SetPos(0) == -1) {
    return absl::InternalError(absl::StrCat(
        "Could not move in the file to write header: ", strerror(errno)));
  }
  if (!fd_.Write(&header, sizeof(header))) {
    return absl::InternalError(
        absl::StrCat("Cannot write cache header to '", file_path_, "'."));
  }

  first_write_done_ = true;
  buffer_fb_list_changed_ = false;
  // Free the temporary data buffer to avoid holding large weights in memory
  // during execution. This may cause re-allocations if another build step
  // is started, but it's a good trade-off to save memory during execution.
  data_.reset(nullptr);
  capacity_ = 0;

  return absl::OkStatus();
}

flatbuffers::Offset<ml_drift::cache::schema::Buffer> CacheBuilder::EncodeBuffer(
    uint32_t global_tensor_id, bool is_quantization_param_tensor,
    const TensorDescriptor& tensor_desc, const BufferLocation& loc,
    flatbuffers::FlatBufferBuilder* builder) {
  auto tensor_desc_fb = Encode(tensor_desc, builder);
  ml_drift::cache::schema::BufferBuilder buffer_builder(*builder);
  buffer_builder.add_tensor_descriptor(tensor_desc_fb);
  buffer_builder.add_global_tensor_id(global_tensor_id);
  buffer_builder.add_is_quantization_param_tensor(is_quantization_param_tensor);
  buffer_builder.add_offset(loc.offset);
  buffer_builder.add_size(loc.size);
  return buffer_builder.Finish();
}

}  // namespace ml_drift
