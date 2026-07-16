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

#include "ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache_schema_generated.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"

namespace ml_drift {

namespace {

// The alignment of each entry in the program cache file.
//
// 64KB was chosen because when the file is mapped on Windows machines, it will
// use MapViewOfFile to map the file. MapViewOfFile uses dwAllocationGranularity
// (default 64KB) as the allocation granularity. By using the same alignment,
// we can avoid allocating extra memory when mapping the file.
// Other operating systems have lower alignment requirements.
constexpr size_t kAlignment = 65536;

size_t Align(size_t offset, size_t alignment) {
  const size_t misalign = offset % alignment;
  return offset + (misalign ? alignment - misalign : 0);
}

std::string JoinPath(absl::string_view path1, absl::string_view path2) {
#if defined(_WIN32)
  char slash = '\\';
#else   // defined(_WIN32)
  char slash = '/';
#endif  // defined(_WIN32)
  if (path1.empty()) return std::string(path2);
  return (path1.back() == slash)
             ? absl::StrCat(path1, path2)
             : absl::StrCat(path1, std::string(1, slash), path2);
}

}  // namespace

SerializationProgramCache::SerializationProgramCache(int fd) : fd_(fd) {}

SerializationProgramCache::SerializationProgramCache(
    absl::string_view file_path)
    : fd_(FileDescriptor::Open(std::string(file_path).c_str(),
                               O_RDWR | O_CREAT,  // NOLINT: b/332641196
                               0644)) {}

SerializationProgramCache::SerializationProgramCache(
    absl::string_view directory_path, absl::string_view model_token)
    : SerializationProgramCache(
          JoinPath(directory_path,
                   absl::StrCat(model_token, "_mldrift_program_cache.bin"))) {}

absl::Status SerializationProgramCache::Insert(uint64_t key,
                                               absl::string_view value) {
  if (!fd_.IsValid()) {
    return absl::InvalidArgumentError("Invalid file descriptor.");
  }

  // 1. Read existing metadata if present.
  std::vector<ml_drift::program_cache::schema::CacheEntryT> entries;
  size_t file_size = fd_.SetPosFromEnd(0);

  size_t header_read_size = std::min(file_size, kAlignment);
  if (header_read_size > 0) {
    if (fd_.SetPos(0) == -1) {
      return absl::InternalError("Failed to seek to beginning of file.");
    }
    std::vector<uint8_t> header_data(header_read_size);
    if (!fd_.Read(header_data.data(), header_read_size)) {
      return absl::InternalError("Failed to read header.");
    }

    flatbuffers::Verifier verifier(header_data.data(), header_data.size());
    if (ml_drift::program_cache::schema::VerifyCacheMetadataBuffer(verifier)) {
      auto metadata =
          ml_drift::program_cache::schema::GetCacheMetadata(header_data.data());
      if (metadata->entries()) {
        for (const auto* entry : *metadata->entries()) {
          std::unique_ptr<ml_drift::program_cache::schema::CacheEntryT> entry_t(
              entry->UnPack());
          entries.push_back(*entry_t);
        }
      }
    }
  }

  // 2. Check if key exists.
  bool found = false;
  for (auto& entry : entries) {
    if (entry.key == key) {
      found = true;
      break;
    }
  }

  // 3. Determine write location.
  // Align file size to 64KB for the new buffer.
  size_t write_offset = Align(file_size, kAlignment);
  // Ensure we start at least at 64KB.
  if (write_offset < kAlignment) write_offset = kAlignment;

  // 4. Write data.
  if (fd_.SetPos(write_offset) == -1) {
    return absl::InternalError("Failed to seek to write offset.");
  }
  if (!fd_.Write(value.data(), value.size())) {
    return absl::InternalError("Failed to write value.");
  }

  // 5. Update metadata structure.
  if (!found) {
    ml_drift::program_cache::schema::CacheEntryT new_entry;
    new_entry.key = key;
    entries.push_back(new_entry);
  }

  for (auto& entry : entries) {
    if (entry.key == key) {
      entry.offset = write_offset;
      entry.size = value.size();
      break;
    }
  }

  // 6. Serialize metadata.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<ml_drift::program_cache::schema::CacheEntry>>
      entry_offsets;
  entry_offsets.reserve(entries.size());
  for (const auto& entry : entries) {
    entry_offsets.push_back(ml_drift::program_cache::schema::CreateCacheEntry(
        builder, entry.key, entry.offset, entry.size));
  }
  auto entries_vec = builder.CreateVector(entry_offsets);
  auto metadata_offset = ml_drift::program_cache::schema::CreateCacheMetadata(
      builder, entries_vec);
  builder.Finish(metadata_offset);

  if (builder.GetSize() > kAlignment) {
    return absl::ResourceExhaustedError("Metadata size exceeds 64KB limit.");
  }

  // 7. Write metadata.
  if (fd_.SetPos(0) == -1) {
    return absl::InternalError("Failed to seek to beginning.");
  }
  if (!fd_.Write(builder.GetBufferPointer(), builder.GetSize())) {
    return absl::InternalError("Failed to write metadata.");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> SerializationProgramCache::LookUp(uint64_t key) {
  if (!fd_.IsValid()) {
    return absl::InvalidArgumentError("Invalid file descriptor.");
  }

  size_t file_size = fd_.SetPosFromEnd(0);

  MMapHandle mmap_handle;
  // Map up to kAlignment, or file_size if smaller.
  size_t map_size = std::min(file_size, kAlignment);
  if (map_size == 0) {
    return absl::NotFoundError("File is empty.");
  }

  if (auto status = mmap_handle.Map(fd_, 0, map_size); !status.ok()) {
    return status;
  }

  flatbuffers::Verifier verifier(mmap_handle.data(), mmap_handle.size());
  if (!ml_drift::program_cache::schema::VerifyCacheMetadataBuffer(verifier)) {
    return absl::InternalError("Corrupt metadata.");
  }

  auto metadata =
      ml_drift::program_cache::schema::GetCacheMetadata(mmap_handle.data());
  uint64_t found_offset = 0;
  uint64_t found_size = 0;
  bool found = false;

  if (metadata->entries()) {
    for (const auto* entry : *metadata->entries()) {
      if (entry->key() == key) {
        found_offset = entry->offset();
        found_size = entry->size();
        found = true;
        break;
      }
    }
  }

  if (!found) {
    return absl::NotFoundError(absl::StrCat("Key ", key, " not found."));
  }

  if (found_offset + found_size > file_size) {
    return absl::InternalError("Data offset/size out of bounds.");
  }

  std::string result;
  result.resize(found_size);
  if (fd_.SetPos(found_offset) == -1) {
    return absl::InternalError("Failed to seek to data.");
  }
  if (!fd_.Read(result.data(), found_size)) {
    return absl::InternalError("Failed to read data.");
  }

  return result;
}

}  // namespace ml_drift
