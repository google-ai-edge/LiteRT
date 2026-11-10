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

#include "litert/vendors/nvidia/compiler/weight_store_builder.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/nvidia/bytecode.h"

namespace litert::nvidia {
namespace {

uint64_t RoundUp(uint64_t value, uint64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

template <typename T>
void AppendKey(std::string& key, T value) {
  key.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

Expected<std::unique_ptr<TensorRtWeightStoreBuilder>>
TensorRtWeightStoreBuilder::Create(const std::string& model_path,
                                   uint64_t granule, uint64_t segment_bytes) {
  if (granule == 0 || (granule & (granule - 1)) != 0 ||
      granule % kEntryAlignment != 0 || segment_bytes == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid TensorRT weight store granule or segment size");
  }
  char resolved[PATH_MAX];
  if (realpath(model_path.c_str(), resolved) == nullptr) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to resolve TensorRT weight source " + model_path +
                     ": " + std::strerror(errno));
  }
  struct stat stat_buffer{};
  if (stat(resolved, &stat_buffer) != 0 || !S_ISREG(stat_buffer.st_mode) ||
      stat_buffer.st_size <= 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "TensorRT weight source is not a readable regular file: " +
                     std::string(resolved));
  }
  std::unique_ptr<TensorRtWeightStoreBuilder> builder(
      new TensorRtWeightStoreBuilder());
  builder->source_path_ = resolved;
  builder->source_fd_ = open(resolved, O_RDONLY | O_CLOEXEC);
  if (builder->source_fd_ < 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to open TensorRT weight source " +
                     std::string(resolved) + ": " + std::strerror(errno));
  }
  builder->source_size_ = static_cast<uint64_t>(stat_buffer.st_size);
  builder->source_identity_ = {
      static_cast<uint64_t>(stat_buffer.st_dev),
      static_cast<uint64_t>(stat_buffer.st_ino),
      static_cast<int64_t>(stat_buffer.st_mtim.tv_sec),
      static_cast<int64_t>(stat_buffer.st_mtim.tv_nsec),
      static_cast<int64_t>(stat_buffer.st_ctim.tv_sec),
      static_cast<int64_t>(stat_buffer.st_ctim.tv_nsec)};
  builder->granule_ = granule;
  builder->segment_bytes_ = RoundUp(segment_bytes, granule);

  // The mappings of the file in this process: "begin-end perms offset
  // major:minor inode path". The device numbers of the listing can differ
  // from st_dev (overlay and btrfs mounts), so the mappings are matched by
  // path and inode.
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (maps == nullptr) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The memory mappings of this process are not readable");
  }
  char line[PATH_MAX + 256];
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long long begin = 0;
    unsigned long long end = 0;
    unsigned long long offset = 0;
    unsigned long long inode = 0;
    int path_start = 0;
    if (std::sscanf(line, "%llx-%llx %*4s %llx %*x:%*x %llu %n", &begin, &end,
                    &offset, &inode, &path_start) != 4 ||
        path_start == 0 ||
        inode != static_cast<unsigned long long>(stat_buffer.st_ino)) {
      continue;
    }
    std::string path(line + path_start);
    while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) {
      path.pop_back();
    }
    if (path != builder->source_path_) {
      continue;
    }
    builder->mappings_.push_back(
        {static_cast<uintptr_t>(begin), static_cast<uintptr_t>(end), offset});
  }
  std::fclose(maps);
  if (builder->mappings_.empty()) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The model file is not memory-mapped in this process: " +
                     builder->source_path_);
  }
  return builder;
}

TensorRtWeightStoreBuilder::~TensorRtWeightStoreBuilder() {
  if (source_fd_ >= 0) {
    close(source_fd_);
  }
}

bool TensorRtWeightStoreBuilder::EqualsSource(absl::Span<const uint8_t> bytes,
                                              uint64_t offset) const {
  std::vector<uint8_t> file(bytes.size());
  for (size_t read = 0; read < file.size();) {
    const ssize_t result =
        pread(source_fd_, file.data() + read, file.size() - read,
              static_cast<off_t>(offset + read));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      return false;
    }
    read += static_cast<size_t>(result);
  }
  return std::memcmp(file.data(), bytes.data(), bytes.size()) == 0;
}

Expected<uint64_t> TensorRtWeightStoreBuilder::SourceOffset(
    absl::Span<const uint8_t> bytes) const {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(bytes.data());
  const uintptr_t end = begin + bytes.size();
  const Mapping* mapping = nullptr;
  for (const Mapping& candidate : mappings_) {
    if (begin >= candidate.begin && end <= candidate.end) {
      mapping = &candidate;
      break;
    }
  }
  if (bytes.empty() || mapping == nullptr) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The weights are not a view of the model file");
  }
  const uint64_t offset = mapping->file_offset + (begin - mapping->begin);
  if (bytes.size() > source_size_ || offset > source_size_ - bytes.size()) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The weights lie past the end of the model file");
  }
  // A private mapping may hold pages the process wrote to. The page map
  // tells which without touching the weights: a present page of the file is
  // flagged as a file page, a written one is anonymous memory, possibly
  // swapped out. A written page can still hold what the file holds, when
  // other bytes of the page were written or the same bytes were written
  // back, so its part of the weights is compared with the file.
  const int pagemap = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (pagemap < 0) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The page map of this process is not readable");
  }
  const uint64_t page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
  constexpr uint64_t kPresent = 1ull << 63;
  constexpr uint64_t kSwapped = 1ull << 62;
  constexpr uint64_t kFilePage = 1ull << 61;
  const uint64_t first_page = begin / page_size;
  const uint64_t last_page = (end - 1) / page_size;
  std::vector<uint64_t> entries(
      std::min<uint64_t>(last_page - first_page + 1, 1 << 15));
  bool unmodified = true;
  for (uint64_t page = first_page; page <= last_page && unmodified;) {
    const uint64_t count =
        std::min<uint64_t>(last_page - page + 1, entries.size());
    const ssize_t wanted = static_cast<ssize_t>(count * sizeof(uint64_t));
    if (pread(pagemap, entries.data(), wanted,
              static_cast<off_t>(page * sizeof(uint64_t))) != wanted) {
      unmodified = false;
      break;
    }
    for (uint64_t i = 0; i < count; ++i) {
      if ((entries[i] & kSwapped) == 0 &&
          ((entries[i] & kPresent) == 0 || (entries[i] & kFilePage) != 0)) {
        continue;
      }
      const uintptr_t part_begin =
          std::max<uintptr_t>((page + i) * page_size, begin);
      const uintptr_t part_end =
          std::min<uintptr_t>((page + i + 1) * page_size, end);
      if (!EqualsSource({reinterpret_cast<const uint8_t*>(part_begin),
                         part_end - part_begin},
                        offset + (part_begin - begin))) {
        unmodified = false;
        break;
      }
    }
    page += count;
  }
  close(pagemap);
  if (!unmodified) {
    return Error(kLiteRtStatusErrorNotFound,
                 "The weights were modified after the model file was mapped");
  }
  return offset;
}

Expected<TensorRtWeightLocation> TensorRtWeightStoreBuilder::Add(
    absl::Span<const absl::Span<const uint8_t>> members) {
  if (members.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "A launch needs at least one weight buffer");
  }
  std::vector<std::pair<uint64_t, uint64_t>> key;
  key.reserve(members.size());
  uint64_t bytes = 0;
  for (const auto& member : members) {
    LITERT_ASSIGN_OR_RETURN(const uint64_t offset, SourceOffset(member));
    key.emplace_back(offset, member.size());
    bytes += member.size();
  }
  if (const auto found = locations_.find(key); found != locations_.end()) {
    return found->second;
  }
  const uint64_t aligned = RoundUp(bytes, kEntryAlignment);
  if (has_open_segment_ && fill_ + aligned > segment_bytes_) {
    CloseOpenSegment();
  }
  if (!has_open_segment_) {
    segments_.emplace_back();
    has_open_segment_ = true;
    fill_ = 0;
  }
  Segment& segment = segments_.back();
  const TensorRtWeightLocation location = {
      static_cast<uint32_t>(segments_.size() - 1), fill_};
  uint64_t offset = fill_;
  for (const auto& [source_offset, size] : key) {
    // Pieces that continue each other in the file and the segment are one.
    if (!segment.pieces.empty() &&
        segment.pieces.back().source_offset + segment.pieces.back().size ==
            source_offset &&
        segment.pieces.back().segment_offset + segment.pieces.back().size ==
            offset) {
      segment.pieces.back().size += size;
    } else {
      segment.pieces.push_back({source_offset, size, offset});
    }
    offset += size;
  }
  fill_ += aligned;
  if (fill_ >= segment_bytes_) {
    CloseOpenSegment();
  }
  locations_.emplace(std::move(key), location);
  return location;
}

void TensorRtWeightStoreBuilder::EndPartition() { CloseOpenSegment(); }

void TensorRtWeightStoreBuilder::CloseOpenSegment() {
  if (!has_open_segment_) {
    return;
  }
  Segment& segment = segments_.back();
  segment.size = RoundUp(fill_, granule_);
  segment.closed = true;
  std::string description;
  description.reserve(source_path_.size() + 64 + segment.pieces.size() * 24);
  description += source_path_;
  AppendKey(description, source_size_);
  AppendKey(description, source_identity_.device);
  AppendKey(description, source_identity_.inode);
  AppendKey(description, source_identity_.mtime_seconds);
  AppendKey(description, source_identity_.mtime_nanoseconds);
  AppendKey(description, source_identity_.ctime_seconds);
  AppendKey(description, source_identity_.ctime_nanoseconds);
  AppendKey(description, segment.size);
  for (const auto& piece : segment.pieces) {
    AppendKey(description, piece.source_offset);
    AppendKey(description, piece.size);
    AppendKey(description, piece.segment_offset);
  }
  segment.key =
      FingerprintTensorRtArtifact(description.data(), description.size());
  has_open_segment_ = false;
  fill_ = 0;
}

}  // namespace litert::nvidia
