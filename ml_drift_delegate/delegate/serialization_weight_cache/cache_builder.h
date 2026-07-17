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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_CACHE_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_CACHE_BUILDER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"

namespace ml_drift {

// Reserved value to request the delegate to use an in-memory cache instead of
// saving it to disk.
//
// This is useful when disk space is not available or when having to manage the
// cache file freshness is too complicated and still provides the deduplication
// mechanism for constant buffers that are reused across graph signatures.
inline constexpr char kInMemoryCachePath[] = ":memory";

// The minimum alignment required between elements in the cache file.
#if defined(__APPLE__)
inline constexpr uint64_t kMinAlignment = 4096;
#else
inline constexpr uint64_t kMinAlignment = 128;
#endif

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(absl::string_view path);

// The maximum number of subgraphs allowed in a single model cache file.
inline constexpr size_t kMaxSupportedSubgraphs = 100;

// This structure is written at the start of every cache file.
//
// When changing this structure or anything in the cache file layout,
// `kVersion` should be incremented by one.
//
// When creating a new cache file, `version` should be set to `kVersion`.
//
// When reading a cache file, the cache should be rejected if `version`
// doesn't match `kVersion`.
struct MLDriftCacheHeader {
  enum : uint64_t { kInvalidHeader = 0, kVersion = 2 };
  // The version of the cache file header. If this doesn't match
  // `kVersion` then the cache file needs to be rebuilt.
  uint64_t version;
  // Used to identify when MLDrift code has change enough that the cache file
  // needs to be rebuilt.
  uint8_t mldrift_build_identifier[32];
  // Points to the ModelCache flatbuffer.
  uint64_t buffer_list_offset;
  uint64_t buffer_list_size;
};

// The location of a buffer in the cache file.
struct BufferLocation {
  uint64_t offset;
  uint64_t size;

  static constexpr BufferLocation Invalid() { return {SIZE_MAX, SIZE_MAX}; }

  constexpr bool IsInvalid() const {
    constexpr BufferLocation invalid = Invalid();
    return offset == invalid.offset && size == invalid.size;
  }
};

// Provides storage to write the packed buffers to and saves those to disk.
class CacheBuilder {
 public:
  CacheBuilder() = default;
  ~CacheBuilder() = default;

  // Non-copyable.
  CacheBuilder(const CacheBuilder&) = delete;
  CacheBuilder& operator=(const CacheBuilder&) = delete;

  // Moveable.
  CacheBuilder(CacheBuilder&&);
  CacheBuilder& operator=(CacheBuilder&&);

  // Starts the builder.
  // If `append` is true, it will open the file without truncation.
  // Appending should be used when loading an existing cache and adding more
  // tensors to it.
  absl::Status Start(absl::string_view path, uint64_t unique_model_identifier,
                     bool append = false);

  // Starts the builder with a file descriptor.
  // If `append` is true, it will assume the file already has data.
  absl::Status Start(FileDescriptor&& fd, uint64_t unique_model_identifier,
                     bool append = false);

  bool IsStarted() const { return fd_.IsValid(); }

  // Reopens the given file to add data to it.
  absl::Status StartBuildStep(uint64_t unique_model_identifier);

  // Reserves space in the data buffer for the required size in bytes and
  // returns the address of that space.
  //
  // A call to `Reserve` should alway be followed by a call to `Append`.
  void* Reserve(size_t size);

  // Adds a buffer to the cache.
  //
  // The buffer space must have been reserved before using `Reserve`. If not, a
  // new call to `Reserve` will be done and the data will be copied over.
  absl::Status Append(uint32_t global_tensor_id,
                      bool is_quantization_param_tensor,
                      const TensorDescriptor& tensor_desc, const void* data,
                      uint64_t size, BufferLocation& loc);

  // Writes the flatbuffer to disk.
  absl::Status StopBuildStep();

  // Returns the file descriptor.
  const FileDescriptor& GetFileDescriptor() const { return fd_; }

  // Returns the capacity of the underlying reserved buffer.
  //
  // WARNING: this exposes class implementation details for testing purposes and
  // may be removed at any time.
  size_t capacity() const { return capacity_; }

 private:
  absl::Status StartInternal(uint64_t unique_model_identifier, bool append);

  // Encode the TensorDescriptor and BufferLocation into a flatbuffer.
  flatbuffers::Offset<ml_drift::cache::schema::Buffer> EncodeBuffer(
      uint32_t global_tensor_id, bool is_quantization_param_tensor,
      const TensorDescriptor& tensor_desc, const BufferLocation& loc,
      flatbuffers::FlatBufferBuilder* builder);

  std::unique_ptr<uint8_t[]> data_ = nullptr;
  // FlatBufferBuilder used to build the cache.
  flatbuffers::FlatBufferBuilder builder_;
  // List of encoded Buffers that will be used to generate the BufferList.
  std::vector<flatbuffers::Offset<ml_drift::cache::schema::Buffer>>
      buffer_fb_list_;
  // Offsets of other subgraphs loaded from existing cache file.
  std::vector<flatbuffers::Offset<ml_drift::cache::schema::SubgraphBufferList>>
      other_subgraphs_offsets_;
  // Whether the buffer_fb_list_ has changed this is used to determine if the
  // buffer_fb_list needs to be updated on disk. If we load an existing cache
  // file and append to it, we need to update the buffer_fb_list_ on disk.
  bool buffer_fb_list_changed_ = false;
  // Points to the start of the memory after the header.
  uint64_t header_base_offset_ = 0;
  // The underlying capacity of the reserved buffer.
  size_t capacity_ = 0;
  // Size of the data written between Start and Stop.
  size_t build_segment_size_ = 0;
  // Offset in the cache file when Start was called.
  size_t build_segment_start_ = 0;
  // The call to Stop may short circuit when nothing was written to the
  // cache. To ensure a smooth reloading, we need to ensure that the file header
  // is correct. This flag lets us know if that has happened.
  bool first_write_done_ = false;
  // Whether a build step is in progress.
  std::atomic<bool> is_build_step_ = false;
  // Unique model identifier that should be stored in the cache header.
  uint64_t unique_model_identifier_ = 0;
  // Temporary file descriptor to write the weights to disk immediately.
  FileDescriptor fd_;
  std::string file_path_;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_SERIALIZATION_WEIGHT_CACHE_CACHE_BUILDER_H_
