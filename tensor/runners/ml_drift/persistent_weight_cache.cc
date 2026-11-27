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

#include "tensor/runners/ml_drift/persistent_weight_cache.h"

#include <fcntl.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#if defined(__APPLE__)
#include <sys/clonefile.h>
#endif
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/crc/crc32c.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "farmhash.h"

namespace litert::tensor {
namespace {

// Entry id of the base index (name -> entry id, CRC32Cs and location of every
// weight). Incremental appends add delta indexes under `kIndexId - gen`
// (`gen = 1..kMaxDeltaIndexes`) so existing bytes stay untouched.
constexpr uint32_t kIndexId = 0xFFFFFFFFu;
constexpr uint32_t kMaxDeltaIndexes = 1024;
// First entry id of the padding entries (in the id space of the index).
constexpr uint32_t kFirstPaddingId = 0x80000000u;
// Padding id space reserved for incremental appends so each delta uses distinct
// ids below `kIndexId - kMaxDeltaIndexes`.
constexpr uint32_t kFirstDeltaPaddingId = 0xC0000000u;
constexpr uint32_t kMaxPaddingPerDelta = 100000u;
// The minimum alignment of `SerializationWeightCache` entries in the file
// (page-aligned on Apple platforms).
constexpr uint64_t kBuilderAlignment = ::ml_drift::kMinAlignment;

#if defined(_WIN32)
constexpr int kOpenFlags = O_BINARY;
int64_t Seek(int fd, int64_t offset, int whence) {
  return _lseeki64(fd, offset, whence);
}
#else
// `O_NONBLOCK` keeps `open()` from blocking on a FIFO at the cache path (which
// is then rejected as not a regular file); it doesn't affect regular files.
constexpr int kOpenFlags = O_CLOEXEC | O_NONBLOCK;
int64_t Seek(int fd, int64_t offset, int whence) {
  return lseek(fd, offset, whence);
}
#endif

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

uint32_t Crc32c(absl::Span<const uint8_t> data) {
  return static_cast<uint32_t>(absl::ComputeCrc32c(absl::string_view(
      reinterpret_cast<const char*>(data.data()), data.size())));
}

// CRC32C of the first and last `kSampleBytes` of `data` (all of it if small).
uint32_t SampleCrc32c(absl::Span<const uint8_t> data) {
  constexpr size_t kSample = PersistentWeightCache::kSampleBytes;
  if (data.size() <= 2 * kSample) return Crc32c(data);
  const absl::crc32c_t head = absl::ComputeCrc32c(
      absl::string_view(reinterpret_cast<const char*>(data.data()), kSample));
  return static_cast<uint32_t>(absl::ExtendCrc32c(
      head, absl::string_view(reinterpret_cast<const char*>(data.data()) +
                                  data.size() - kSample,
                              kSample)));
}

template <typename T>
void Append(std::vector<uint8_t>& out, T v) {
  const size_t pos = out.size();
  out.resize(pos + sizeof(v));
  std::memcpy(out.data() + pos, &v, sizeof(v));
}

struct IndexEntry {
  std::string name;
  PersistentWeightCache::IndexValue value;
};

struct Index {
  // End of the last weight's data in the file.
  uint64_t data_end = 0;
  std::vector<IndexEntry> entries;
};

std::vector<uint8_t> SerializeIndex(const Index& index) {
  std::vector<uint8_t> out;
  Append<uint32_t>(out, PersistentWeightCache::kFormatVersion);
  Append<uint32_t>(out, static_cast<uint32_t>(index.entries.size()));
  Append<uint64_t>(out, index.data_end);
  for (const IndexEntry& e : index.entries) {
    Append<uint32_t>(out, e.value.id);
    Append<uint32_t>(out, e.value.crc);
    Append<uint32_t>(out, e.value.sample_crc);
    Append<uint64_t>(out, e.value.offset);
    Append<uint64_t>(out, e.value.size);
    Append<uint32_t>(out, static_cast<uint32_t>(e.name.size()));
    out.insert(out.end(), e.name.begin(), e.name.end());
  }
  // The index checks itself too.
  Append<uint32_t>(out, Crc32c(out));
  return out;
}

absl::StatusOr<Index> ParseIndex(absl::Span<const uint8_t> data) {
  // Version, entry count, `data_end` and CRC of an empty index.
  constexpr size_t kMinIndexBytes = 3 * sizeof(uint32_t) + sizeof(uint64_t);
  if (data.size() < kMinIndexBytes) {
    return absl::DataLossError("Truncated weight cache index");
  }
  uint32_t crc = 0;
  std::memcpy(&crc, data.data() + data.size() - sizeof(crc), sizeof(crc));
  data.remove_suffix(sizeof(crc));
  if (Crc32c(data) != crc) {
    return absl::DataLossError("Corrupt weight cache index");
  }
  size_t pos = 0;
  auto read = [&](auto* v) {
    if (data.size() - pos < sizeof(*v)) return false;
    std::memcpy(v, data.data() + pos, sizeof(*v));
    pos += sizeof(*v);
    return true;
  };
  uint32_t version = 0;
  if (!read(&version)) {
    return absl::DataLossError("Truncated weight cache index");
  }
  if (version != PersistentWeightCache::kFormatVersion) {
    return absl::FailedPreconditionError(
        absl::StrCat("Weight cache format version ", version, ", expected ",
                     PersistentWeightCache::kFormatVersion));
  }
  uint32_t count = 0;
  Index index;
  if (!read(&count) || !read(&index.data_end)) {
    return absl::DataLossError("Truncated weight cache index");
  }
  using IndexValue = PersistentWeightCache::IndexValue;
  constexpr size_t kMinEntryBytes =
      sizeof(IndexValue::id) + sizeof(IndexValue::crc) +
      sizeof(IndexValue::sample_crc) + sizeof(IndexValue::offset) +
      sizeof(IndexValue::size) + sizeof(uint32_t);
  if (count > (data.size() - pos) / kMinEntryBytes) {
    return absl::DataLossError("Truncated weight cache index");
  }
  index.entries.resize(count);
  for (IndexEntry& e : index.entries) {
    uint32_t len = 0;
    if (!read(&e.value.id) || !read(&e.value.crc) ||
        !read(&e.value.sample_crc) || !read(&e.value.offset) ||
        !read(&e.value.size) || !read(&len) || data.size() - pos < len) {
      return absl::DataLossError("Truncated weight cache index");
    }
    e.name.assign(reinterpret_cast<const char*>(data.data() + pos), len);
    pos += len;
  }
  if (pos != data.size()) {
    return absl::DataLossError("Trailing bytes in weight cache index");
  }
  return index;
}

// Checks that the descriptor of a cached weight is well-formed and that its
// `data_size` bytes of data cover the memory it describes: the descriptor
// comes from the cache file and isn't covered by the index CRCs, so a corrupt
// one must make the cache rebuild instead of failing the GPU upload later.
absl::Status ValidateWeightDescriptor(absl::string_view name,
                                      const ::ml_drift::TensorDescriptor& desc,
                                      size_t data_size) {
  const absl::Status invalid = absl::DataLossError(
      absl::StrCat("Invalid descriptor of weight cache entry '", name, "'"));
  const ::ml_drift::BHWDC& shape = desc.GetBHWDCShape();
  if (desc.GetStorageType() == ::ml_drift::TensorStorageType::kUnknown ||
      shape.b <= 0 || shape.h <= 0 || shape.w <= 0 || shape.d <= 0 ||
      shape.c <= 0) {
    return invalid;
  }
  uint64_t bytes = static_cast<uint64_t>(desc.GetElementSize()) *
                   ::ml_drift::SizeOf(desc.GetDataType());
  const std::vector<uint64_t> dims = desc.GetStorageDims();
  if (bytes == 0 || dims.empty()) return invalid;
  for (const uint64_t dim : dims) {
    if (dim == 0 || bytes > std::numeric_limits<uint64_t>::max() / dim) {
      return invalid;
    }
    bytes *= dim;
  }
  if (data_size < bytes) return invalid;
  return absl::OkStatus();
}

// Unmaps a weight mapped by a successful `SerializationWeightCache::LookUp`
// (which always sets `release`).
struct Mapping {
  explicit Mapping(::ml_drift::ReleaseDataCallback release)
      : release(std::move(release)) {}
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  ~Mapping() { (*release)(); }
  ::ml_drift::ReleaseDataCallback release;
};

std::string MakeUniqueTempPath(absl::string_view path) {
  static std::atomic<uint64_t> counter{0};
#if defined(_WIN32)
  const int pid = _getpid();
#else
  const int pid = static_cast<int>(getpid());
#endif
  return absl::StrCat(path, ".tmp.", pid, ".",
                      counter.fetch_add(1, std::memory_order_relaxed));
}

absl::Status ReplaceFileAtomically(const std::string& tmp_path,
                                   const std::string& dst_path) {
#if defined(_WIN32)
  if (!MoveFileExA(tmp_path.c_str(), dst_path.c_str(),
                   MOVEFILE_REPLACE_EXISTING)) {
    return absl::UnavailableError(absl::StrCat(
        "Can't move ", tmp_path, " to ", dst_path, ": error ", GetLastError()));
  }
#else
  if (std::rename(tmp_path.c_str(), dst_path.c_str()) != 0) {
    return absl::UnavailableError(absl::StrCat("Can't rename ", tmp_path,
                                               " to ", dst_path, ": ",
                                               std::strerror(errno)));
  }
#endif
  return absl::OkStatus();
}

bool IsLookupMiss(const absl::Status& status) {
  return absl::IsInvalidArgument(status) &&
         absl::StartsWith(status.message(), "Failed to look up ");
}

absl::Status OpenValidatedCache(
    const std::string& path, uint64_t identifier,
    std::unique_ptr<::ml_drift::SerializationWeightCache>* out_cache,
    Index* out_index, uint32_t* out_next_delta_gen = nullptr,
    uint32_t* out_next_entry_id = nullptr) {
  const int fd = open(path.c_str(), O_RDONLY | kOpenFlags);
  if (fd < 0) {
    // Only a missing file is `NotFound` (built without a warning).
    if (errno == ENOENT) {
      return absl::NotFoundError(absl::StrCat("No weight cache at ", path));
    }
    return absl::UnavailableError(absl::StrCat(
        "Can't open the weight cache ", path, ": ", std::strerror(errno)));
  }
#if !defined(_WIN32)
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(
        absl::StrCat("Weight cache ", path, " is not a regular file"));
  }
#endif
  const int64_t file_size = Seek(fd, 0, SEEK_END);
  Seek(fd, 0, SEEK_SET);
  auto cache = std::make_unique<::ml_drift::SerializationWeightCache>();
  // Owns `fd` from here.
  if (absl::Status status = cache->Load(fd, identifier); !status.ok()) {
    // `NotFound` here means the file is for another identifier.
    if (absl::IsNotFound(status)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Stale weight cache ", path, ": ", status.message()));
    }
    return status;
  }
  ::ml_drift::TensorDescriptor index_desc;
  absl::Status status = cache->LookUp(
      kIndexId, /*is_quantization_param_tensor=*/true, index_desc);
  absl::StatusOr<Index> index =
      status.ok() ? ParseIndex(index_desc.GetData()) : status;
  if (!index.ok()) return index.status();

  uint32_t last_delta_gen = 0;
  for (uint32_t gen = 1; gen <= kMaxDeltaIndexes; ++gen) {
    ::ml_drift::TensorDescriptor delta_desc;
    const absl::Status delta_status = cache->LookUp(
        kIndexId - gen, /*is_quantization_param_tensor=*/true, delta_desc);
    if (!delta_status.ok()) {
      if (IsLookupMiss(delta_status)) break;
      return delta_status;
    }
    absl::StatusOr<Index> delta = ParseIndex(delta_desc.GetData());
    if (!delta.ok()) return delta.status();
    index->data_end = std::max(index->data_end, delta->data_end);
    index->entries.insert(index->entries.end(),
                          std::make_move_iterator(delta->entries.begin()),
                          std::make_move_iterator(delta->entries.end()));
    last_delta_gen = gen;
  }
  if (file_size < 0 || index->data_end > static_cast<uint64_t>(file_size)) {
    return absl::DataLossError("Truncated weight cache");
  }
  uint32_t next_entry_id = 0;
  for (const IndexEntry& e : index->entries) {
    if (e.value.id >= next_entry_id) {
      next_entry_id = e.value.id + 1;
    }
  }
  if (out_next_delta_gen != nullptr) *out_next_delta_gen = last_delta_gen + 1;
  if (out_next_entry_id != nullptr) *out_next_entry_id = next_entry_id;
  *out_cache = std::move(cache);
  *out_index = *std::move(index);
  return absl::OkStatus();
}

using WeightEntryRef =
    const std::pair<std::string, ::ml_drift::TensorDescriptor>*;

// Appends `to_add` and a delta index (`kIndexId - delta_gen`) to an existing
// cloned/copied cache file at `tmp_path` without rewriting prior weight bytes.
absl::Status AppendIncrementalToFile(
    const std::string& tmp_path, uint64_t identifier,
    absl::Span<const WeightEntryRef> to_add, uint32_t next_entry_id,
    uint32_t delta_gen) {
  ::ml_drift::MLDriftCacheHeader header{};
  {
    const int hdr_fd = open(tmp_path.c_str(), O_RDONLY | kOpenFlags);
    if (hdr_fd < 0) {
      return absl::UnavailableError(
          absl::StrCat("Can't open cloned weight cache ", tmp_path, ": ",
                       std::strerror(errno)));
    }
    const auto n = read(hdr_fd, &header, sizeof(header));
    close(hdr_fd);
    if (n != static_cast<decltype(n)>(sizeof(header))) {
      return absl::DataLossError(
          absl::StrCat("Truncated weight cache header in ", tmp_path));
    }
  }
  ::ml_drift::CacheBuilder builder;
  if (absl::Status status = builder.Start(tmp_path, identifier,
                                          /*append=*/true);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = builder.StartBuildStep(identifier); !status.ok()) {
    return status;
  }
  const uint64_t header_base_offset =
      AlignUp(sizeof(::ml_drift::MLDriftCacheHeader), kBuilderAlignment);
  uint64_t cur_pos = header.buffer_list_offset;
  const uint32_t pad_id_base =
      kFirstDeltaPaddingId + (delta_gen - 1) * kMaxPaddingPerDelta;
  uint32_t pad_idx = 0;
  Index delta_index;
  delta_index.entries.reserve(to_add.size());
  static constexpr uint8_t kEmptyByte = 0;

  for (const WeightEntryRef entry : to_add) {
    const uint64_t next = AlignUp(cur_pos, kBuilderAlignment);
    if (next % PersistentWeightCache::kEntryAlignment != 0) {
      if (pad_idx >= kMaxPaddingPerDelta) {
        return absl::ResourceExhaustedError(
            "Exceeded padding id budget for incremental weight cache append");
      }
      const size_t padding = PersistentWeightCache::kEntryAlignment -
                             next % PersistentWeightCache::kEntryAlignment;
      ::ml_drift::TensorDescriptor padding_desc(
          ::ml_drift::DataType::kUint8,
          ::ml_drift::TensorStorageType::kBuffer,
          ::ml_drift::Layout::kLinear);
      padding_desc.SetBHWCShape(
          ::ml_drift::BHWC(1, 1, 1, static_cast<int>(padding)));
      std::vector<uint8_t> zeros(padding, 0);
      ::ml_drift::BufferLocation pad_loc;
      if (absl::Status status = builder.Append(
              pad_id_base + pad_idx++, /*is_quantization_param_tensor=*/true,
              padding_desc, zeros.data(), zeros.size(), pad_loc);
          !status.ok()) {
        return status;
      }
      cur_pos = header_base_offset + pad_loc.offset + pad_loc.size;
    }
    ::ml_drift::TensorDescriptor meta_desc;
    entry->second.CopyWithoutData(&meta_desc);
    const absl::Span<const uint8_t> data = entry->second.GetData();
    ::ml_drift::BufferLocation loc;
    const uint32_t id = next_entry_id++;
    if (absl::Status status = builder.Append(
            id, /*is_quantization_param_tensor=*/false, meta_desc,
            data.empty() ? &kEmptyByte : data.data(), data.size(), loc);
        !status.ok()) {
      return status;
    }
    const uint64_t file_offset = header_base_offset + loc.offset;
    cur_pos = file_offset + loc.size;
    delta_index.data_end =
        std::max(delta_index.data_end, file_offset + data.size());
    delta_index.entries.push_back(IndexEntry{
        .name = entry->first,
        .value = PersistentWeightCache::IndexValue{
            .id = id,
            .crc = Crc32c(data),
            .sample_crc = SampleCrc32c(data),
            .offset = file_offset,
            .size = data.size()}});
  }

  std::vector<uint8_t> serialized = SerializeIndex(delta_index);
  ::ml_drift::TensorDescriptor index_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kLinear);
  index_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, 1, static_cast<int>(serialized.size())));
  ::ml_drift::BufferLocation index_loc;
  if (absl::Status status = builder.Append(
          kIndexId - delta_gen, /*is_quantization_param_tensor=*/true,
          index_desc, serialized.data(), serialized.size(), index_loc);
      !status.ok()) {
    return status;
  }
  return builder.StopBuildStep();
}

absl::Status CopyFileToNewTemp(const std::string& src_path,
                               std::string* out_tmp_path) {
  const int src_fd = open(src_path.c_str(), O_RDONLY | kOpenFlags);
  if (src_fd < 0) {
    return absl::UnavailableError(absl::StrCat("Can't open ", src_path, ": ",
                                               std::strerror(errno)));
  }
  int dst_fd = -1;
  std::string tmp_path;
  for (int attempt = 0; attempt < 16 && dst_fd < 0; ++attempt) {
    tmp_path = MakeUniqueTempPath(src_path);
    dst_fd =
        open(tmp_path.c_str(), O_RDWR | O_CREAT | O_EXCL | kOpenFlags, 0644);
    if (dst_fd < 0 && errno != EEXIST) break;
  }
  if (dst_fd < 0) {
    close(src_fd);
    return absl::UnavailableError(absl::StrCat(
        "Can't create temporary copy for ", src_path, ": ",
        std::strerror(errno)));
  }
  std::vector<uint8_t> buf(65536);
  absl::Status status = absl::OkStatus();
  while (true) {
    const auto n = read(src_fd, buf.data(), buf.size());
    if (n < 0) {
      if (errno == EINTR) continue;
      status = absl::UnavailableError(absl::StrCat(
          "Read failed while copying ", src_path, ": ", std::strerror(errno)));
      break;
    }
    if (n == 0) break;
    size_t written = 0;
    while (written < static_cast<size_t>(n)) {
      const auto w = write(dst_fd, buf.data() + written,
                           static_cast<size_t>(n) - written);
      if (w < 0) {
        if (errno == EINTR) continue;
        status = absl::UnavailableError(absl::StrCat(
            "Write failed while copying ", src_path, ": ",
            std::strerror(errno)));
        break;
      }
      written += static_cast<size_t>(w);
    }
    if (!status.ok()) break;
  }
  close(src_fd);
  close(dst_fd);
  if (!status.ok()) {
    std::remove(tmp_path.c_str());
    return status;
  }
  *out_tmp_path = std::move(tmp_path);
  return absl::OkStatus();
}

}  // namespace

PersistentWeightCache::PersistentWeightCache() = default;
PersistentWeightCache::~PersistentWeightCache() {
  if (building_) {
    cache_.reset();
    if (!tmp_path_.empty()) std::remove(tmp_path_.c_str());
  }
}

absl::StatusOr<std::unique_ptr<PersistentWeightCache>>
PersistentWeightCache::Open(absl::string_view path, uint64_t identifier,
                            const Options& options) {
  std::unique_ptr<PersistentWeightCache> cache(new PersistentWeightCache());
  cache->path_ = std::string(path);
  const absl::Status load_status = cache->Load(identifier, options);
  if (load_status.ok()) return cache;
  if (!absl::IsNotFound(load_status)) {
    ABSL_LOG(WARNING) << "Rebuilding the GPU weight cache '" << path
                      << "': " << load_status.message();
  }
  cache->descriptors_.clear();
  cache->mapped_.clear();
  cache->num_entries_ = 0;
  cache->data_bytes_ = 0;
  if (absl::Status status = cache->StartBuild(identifier); !status.ok()) {
    return status;
  }
  return cache;
}

absl::Status PersistentWeightCache::Load(uint64_t identifier,
                                         const Options& options) {
  Index index;
  if (absl::Status status =
          OpenValidatedCache(path_, identifier, &cache_, &index);
      !status.ok()) {
    return status;
  }
  absl::Status status = absl::OkStatus();
  for (const IndexEntry& e : index.entries) {
    if (e.value.offset > index.data_end ||
        e.value.size > index.data_end - e.value.offset) {
      status = absl::DataLossError(
          absl::StrCat("Weight cache entry '", e.name, "' out of bounds"));
      break;
    }
    absl::Span<const uint8_t> data;
    ::ml_drift::TensorDescriptor desc;
    std::shared_ptr<Mapping> mapping;
    if (options.map_data) {
      ::ml_drift::UnownedDataTensorDescriptor unowned;
      size_t page_offset = 0;
      ::ml_drift::ReleaseDataCallback release;
      status =
          cache_->LookUp(e.value.id, /*is_quantization_param_tensor=*/false,
                         unowned, page_offset, release);
      if (!status.ok()) break;
      mapping = std::make_shared<Mapping>(std::move(release));
      data = unowned.GetData();
      unowned.CopyWithoutData(&desc);
    } else {
      status = cache_->LookUp(e.value.id,
                              /*is_quantization_param_tensor=*/false, desc);
      if (!status.ok()) break;
      data = desc.GetData();
    }
    if (data.size() != e.value.size) {
      status = absl::DataLossError(
          absl::StrCat("Weight cache entry '", e.name, "' has ", data.size(),
                       " bytes, expected ", e.value.size));
      break;
    }
    if (SampleCrc32c(data) != e.value.sample_crc ||
        (options.verify_all_bytes && Crc32c(data) != e.value.crc)) {
      status = absl::DataLossError(
          absl::StrCat("Corrupt weight cache entry '", e.name, "'"));
      break;
    }
    status = ValidateWeightDescriptor(e.name, desc, data.size());
    if (!status.ok()) break;
    data_bytes_ += data.size();
    if (options.map_data) {
      mapped_.insert_or_assign(e.name,
                               MappedWeight{.desc = std::move(desc),
                                            .data = data,
                                            .mapping = std::move(mapping)});
    } else {
      descriptors_.insert_or_assign(e.name, std::move(desc));
    }
  }
  // The mappings outlive the file descriptor.
  cache_.reset();
  if (!status.ok()) return status;
  num_entries_ = options.map_data ? mapped_.size() : descriptors_.size();
  loaded_ = true;
  return absl::OkStatus();
}

absl::Status PersistentWeightCache::StartBuild(uint64_t identifier) {
  int fd = -1;
#if defined(_WIN32)
  static std::atomic<uint64_t> counter{0};
  const int pid = _getpid();
  for (int attempt = 0; attempt < 16 && fd < 0; ++attempt) {
    tmp_path_ = absl::StrCat(path_, ".tmp.", pid, ".",
                             counter.fetch_add(1, std::memory_order_relaxed));
    fd = open(tmp_path_.c_str(), O_RDWR | O_CREAT | O_EXCL | kOpenFlags, 0644);
    if (fd < 0 && errno != EEXIST) break;
  }
#else
  std::string tmpl = absl::StrCat(path_, ".tmp.XXXXXX");
  fd = mkstemp(tmpl.data());
  if (fd >= 0) {
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    tmp_path_ = std::move(tmpl);
  }
#endif
  if (fd < 0) {
    tmp_path_.clear();
    return absl::UnavailableError(
        absl::StrCat("Can't create a temporary weight cache file for ", path_,
                     ": ", std::strerror(errno)));
  }
  cache_ = std::make_unique<::ml_drift::SerializationWeightCache>();
  // Owns `fd` from here.
  if (absl::Status status = cache_->StartBuild(fd, identifier); !status.ok()) {
    cache_.reset();
    std::remove(tmp_path_.c_str());
    tmp_path_.clear();
    return status;
  }
  build_fd_ = fd;
  building_ = true;
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, ::ml_drift::TensorDescriptor>
PersistentWeightCache::ReleaseDescriptors() {
  return std::move(descriptors_);
}

absl::flat_hash_map<std::string, PersistentWeightCache::MappedWeight>
PersistentWeightCache::ReleaseMappedWeights() {
  return std::move(mapped_);
}

absl::StatusOr<uint64_t> PersistentWeightCache::AlignNextEntry() {
  // `SerializationWeightCache` writes every entry to the file at the next
  // multiple of `kBuilderAlignment`; a padding entry moves that to the next
  // multiple of `kEntryAlignment`.
  int64_t pos = Seek(build_fd_, 0, SEEK_CUR);
  if (pos < 0) {
    return absl::InternalError("Can't get the weight cache file position");
  }
  const uint64_t next = AlignUp(static_cast<uint64_t>(pos), kBuilderAlignment);
  if (next % kEntryAlignment == 0) return next;
  const size_t padding = kEntryAlignment - next % kEntryAlignment;
  ::ml_drift::TensorDescriptor padding_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kLinear);
  padding_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, 1, static_cast<int>(padding)));
  if (absl::Status status = cache_->InsertWithWriter(
          kFirstPaddingId + num_padding_entries_,
          /*is_quantization_param_tensor=*/true, padding_desc, padding,
          [](absl::Span<uint8_t> staging) {
            std::fill(staging.begin(), staging.end(), 0);
            return absl::OkStatus();
          });
      !status.ok()) {
    return status;
  }
  ++num_padding_entries_;
  pos = Seek(build_fd_, 0, SEEK_CUR);
  if (pos < 0) {
    return absl::InternalError("Can't get the weight cache file position");
  }
  return AlignUp(static_cast<uint64_t>(pos), kBuilderAlignment);
}

absl::Status PersistentWeightCache::InsertRaw(
    const std::string& name, const ::ml_drift::TensorDescriptor& desc,
    absl::Span<const uint8_t> data, std::optional<uint32_t> expected_crc) {
  if (!building_) {
    return absl::FailedPreconditionError("The weight cache is not building");
  }
  if (index_.contains(name)) return absl::OkStatus();
  if (absl::StatusOr<uint64_t> next = AlignNextEntry(); !next.ok()) {
    return next.status();
  }
  const uint32_t id = static_cast<uint32_t>(index_.size());
  if (absl::Status status = cache_->InsertWithWriter(
          id, /*is_quantization_param_tensor=*/false, desc, data.size(),
          [data](absl::Span<uint8_t> staging) {
            if (!data.empty()) {
              absl::c_copy(data, staging.begin());
            }
            return absl::OkStatus();
          });
      !status.ok()) {
    return status;
  }
  // The entry's data ends at the file position.
  const int64_t end = Seek(build_fd_, 0, SEEK_CUR);
  if (end < 0 || static_cast<uint64_t>(end) < data.size()) {
    return absl::InternalError("Can't get the weight cache file position");
  }
  index_[name] =
      IndexValue{.id = id,
                 .crc = expected_crc.has_value() ? *expected_crc : Crc32c(data),
                 .sample_crc = SampleCrc32c(data),
                 .offset = static_cast<uint64_t>(end) - data.size(),
                 .size = data.size()};
  ++num_entries_;
  data_bytes_ += data.size();
  return absl::OkStatus();
}

absl::Status PersistentWeightCache::Insert(
    const std::string& name, const ::ml_drift::TensorDescriptor& desc) {
  return InsertRaw(name, desc, desc.GetData());
}

absl::Status PersistentWeightCache::Append(
    absl::string_view path, uint64_t identifier,
    absl::Span<const std::pair<std::string, ::ml_drift::TensorDescriptor>>
        extra_entries,
    bool verify_all_bytes) {
  if (extra_entries.empty()) return absl::OkStatus();
  const std::string path_str(path);
  std::unique_ptr<::ml_drift::SerializationWeightCache> old_cache;
  Index old_index;
  uint32_t next_delta_gen = 1;
  uint32_t next_entry_id = 0;
  if (absl::Status status =
          OpenValidatedCache(path_str, identifier, &old_cache, &old_index,
                             &next_delta_gen, &next_entry_id);
      !status.ok()) {
    return status;
  }
  absl::flat_hash_set<absl::string_view> known_names;
  known_names.reserve(old_index.entries.size() + extra_entries.size());
  for (const IndexEntry& e : old_index.entries) {
    known_names.insert(e.name);
  }
  std::vector<const std::pair<std::string, ::ml_drift::TensorDescriptor>*>
      to_add;
  to_add.reserve(extra_entries.size());
  for (const auto& entry : extra_entries) {
    if (known_names.insert(entry.first).second) {
      to_add.push_back(&entry);
    }
  }
  if (to_add.empty()) return absl::OkStatus();

#if defined(__APPLE__)
  if (!verify_all_bytes && next_delta_gen <= kMaxDeltaIndexes) {
    std::string cloned_tmp;
    int clone_rc = -1;
    for (int attempt = 0; attempt < 16; ++attempt) {
      cloned_tmp = MakeUniqueTempPath(path_str);
      clone_rc = clonefile(path_str.c_str(), cloned_tmp.c_str(), 0);
      if (clone_rc == 0 || errno != EEXIST) break;
    }
    if (clone_rc == 0) {
      // Release old FD before mutating the clone and renaming over `path_str`.
      old_cache.reset();
      absl::Status status = AppendIncrementalToFile(
          cloned_tmp, identifier, to_add, next_entry_id, next_delta_gen);
      if (status.ok()) {
        status = ReplaceFileAtomically(cloned_tmp, path_str);
      }
      if (status.ok()) return absl::OkStatus();
      std::remove(cloned_tmp.c_str());
      if (!absl::IsResourceExhausted(status)) return status;
      // Fall back to a compacting full rewrite if the padding budget ran out.
      if (absl::Status reopen =
              OpenValidatedCache(path_str, identifier, &old_cache, &old_index);
          !reopen.ok()) {
        return reopen;
      }
    }
  }
#endif

  PersistentWeightCache next;
  next.path_ = path_str;
  if (absl::Status status = next.StartBuild(identifier); !status.ok()) {
    return status;
  }
  for (const IndexEntry& e : old_index.entries) {
    if (e.value.offset > old_index.data_end ||
        e.value.size > old_index.data_end - e.value.offset) {
      return absl::DataLossError(
          absl::StrCat("Weight cache entry '", e.name, "' out of bounds"));
    }
    ::ml_drift::UnownedDataTensorDescriptor unowned;
    size_t page_offset = 0;
    ::ml_drift::ReleaseDataCallback release;
    if (absl::Status status = old_cache->LookUp(
            e.value.id, /*is_quantization_param_tensor=*/false, unowned,
            page_offset, release);
        !status.ok()) {
      return status;
    }
    Mapping mapping(std::move(release));
    const absl::Span<const uint8_t> data = unowned.GetData();
    if (data.size() != e.value.size ||
        SampleCrc32c(data) != e.value.sample_crc ||
        (verify_all_bytes && Crc32c(data) != e.value.crc)) {
      return absl::DataLossError(
          absl::StrCat("Corrupt weight cache entry '", e.name, "'"));
    }
    ::ml_drift::TensorDescriptor meta_desc;
    unowned.CopyWithoutData(&meta_desc);
    if (absl::Status status =
            ValidateWeightDescriptor(e.name, meta_desc, data.size());
        !status.ok()) {
      return status;
    }
    if (absl::Status status =
            next.InsertRaw(e.name, meta_desc, data, e.value.crc);
        !status.ok()) {
      return status;
    }
  }
  // Release all mmap handles on `path` before `Finish()` renames `tmp_path_`.
  old_cache.reset();
  for (const auto* entry : to_add) {
    if (absl::Status status = next.Insert(entry->first, entry->second);
        !status.ok()) {
      return status;
    }
  }
  return next.Finish();
}

absl::Status PersistentWeightCache::AppendIncrementalForTesting(
    absl::string_view path, uint64_t identifier,
    absl::Span<const std::pair<std::string, ::ml_drift::TensorDescriptor>>
        extra_entries) {
  if (extra_entries.empty()) return absl::OkStatus();
  const std::string path_str(path);
  std::unique_ptr<::ml_drift::SerializationWeightCache> old_cache;
  Index old_index;
  uint32_t next_delta_gen = 1;
  uint32_t next_entry_id = 0;
  if (absl::Status status =
          OpenValidatedCache(path_str, identifier, &old_cache, &old_index,
                             &next_delta_gen, &next_entry_id);
      !status.ok()) {
    return status;
  }
  if (next_delta_gen > kMaxDeltaIndexes) {
    return absl::ResourceExhaustedError("Exceeded maximum delta index count");
  }
  absl::flat_hash_set<absl::string_view> known_names;
  known_names.reserve(old_index.entries.size() + extra_entries.size());
  for (const IndexEntry& e : old_index.entries) {
    known_names.insert(e.name);
  }
  std::vector<const std::pair<std::string, ::ml_drift::TensorDescriptor>*>
      to_add;
  to_add.reserve(extra_entries.size());
  for (const auto& entry : extra_entries) {
    if (known_names.insert(entry.first).second) {
      to_add.push_back(&entry);
    }
  }
  if (to_add.empty()) return absl::OkStatus();
  old_cache.reset();
  std::string tmp_path;
  if (absl::Status status = CopyFileToNewTemp(path_str, &tmp_path);
      !status.ok()) {
    return status;
  }
  absl::Status status = AppendIncrementalToFile(
      tmp_path, identifier, to_add, next_entry_id, next_delta_gen);
  if (status.ok()) {
    status = ReplaceFileAtomically(tmp_path, path_str);
  }
  if (!status.ok()) std::remove(tmp_path.c_str());
  return status;
}

absl::Status PersistentWeightCache::Finish() {
  if (!building_) {
    return absl::FailedPreconditionError("The weight cache is not building");
  }
  Index index;
  index.entries.reserve(index_.size());
  for (const auto& [name, value] : index_) {
    index.entries.push_back(IndexEntry{.name = name, .value = value});
    index.data_end = std::max(index.data_end, value.offset + value.size);
  }
  std::sort(index.entries.begin(), index.entries.end(),
            [](const IndexEntry& a, const IndexEntry& b) {
              return a.value.id < b.value.id;
            });
  std::vector<uint8_t> serialized = SerializeIndex(index);
  ::ml_drift::TensorDescriptor index_desc(
      ::ml_drift::DataType::kUint8, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kLinear);
  index_desc.SetBHWCShape(
      ::ml_drift::BHWC(1, 1, 1, static_cast<int>(serialized.size())));
  index_desc.SetData(std::move(serialized));
  absl::Status status = cache_->Insert(
      kIndexId, /*is_quantization_param_tensor=*/true, index_desc);
  if (status.ok()) status = cache_->StopBuild();
  cache_.reset();
  build_fd_ = -1;
  building_ = false;
  if (status.ok()) {
    status = ReplaceFileAtomically(tmp_path_, path_);
  }
  if (!status.ok() && !tmp_path_.empty()) std::remove(tmp_path_.c_str());
  tmp_path_.clear();
  return status;
}

uint64_t WeightCacheIdentifier(absl::string_view model_token,
                               absl::string_view gpu_key,
                               absl::string_view options_key) {
  const std::string key =
      // Length-prefixed, so that different keys can't give the same string.
      absl::StrCat("lrt_weight_cache:", PersistentWeightCache::kFormatVersion,
                   "|", model_token.size(), ":", model_token, "|",
                   gpu_key.size(), ":", gpu_key, "|", options_key.size(), ":",
                   options_key);
  return util::Fingerprint64(key.data(), key.size());
}

}  // namespace litert::tensor
