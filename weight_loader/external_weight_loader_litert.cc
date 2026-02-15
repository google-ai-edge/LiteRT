// Copyright 2025 Google LLC
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

#include "weight_loader/external_weight_loader_litert.h"

#include "litert/cc/internal/scoped_weight_source.h"
#include "litert/core/model/flatbuffer_to_litert.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // !defined(_WIN32)

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#if LITERT_HAS_OPENCL_SUPPORT
#include "litert/runtime/open_cl_sync.h"
#include <CL/cl.h>
#endif  // LITERT_HAS_OPENCL_SUPPORT
#include "tflite/schema/schema_generated.h"

namespace weight_loader {

WeightAccess::WeightAccess() = default;
WeightAccess::WeightAccess(WeightAccess&&) noexcept = default;
WeightAccess& WeightAccess::operator=(WeightAccess&&) noexcept = default;
WeightAccess::~WeightAccess() = default;

void WeightAccess::Reset() {
  host_tensor_buffer = nullptr;
  device_tensor_buffer = nullptr;
}

void WeightAccess::SetHostBuffer(LiteRtTensorBufferPtr buffer) {
  host_tensor_buffer = std::move(buffer);
}

void WeightAccess::SetDeviceBuffer(LiteRtTensorBufferPtr buffer) {
  device_tensor_buffer = std::move(buffer);
}

namespace {

// Information about a single external weight tensor.
struct LiteRtWeightInfo : public WeightInfo {
  // The index of the subgraph that contains the tensor.
  uint32_t subgraph_index;
  // The index of the tensor in the subgraph.
  uint32_t tensor_index;
  // The ID of the group that the tensor belongs to. Tensors in the same group
  // are typically stored together in the same external file.
  uint32_t group_id;
  // The offset of the tensor data in the external file, in bytes.
  uint64_t offset;
  // The length of the tensor data in the external file, in bytes.
  uint64_t length;
  // TODO(b/453768409): Refactor external weight loader to only use cc API.
  // The type of the tensor.
  LiteRtRankedTensorType tensor_type;
  // The GPU buffer type for the tensor data.
  LiteRtTensorBufferType gpu_buffer_type;
};

absl::StatusOr<LiteRtRankedTensorType> BuildRankedTensorType(
    const tflite::Tensor& tensor_fb) {
  LiteRtRankedTensorType type;
  // Use proper mapping instead of direct cast: TFLite has FLOAT32=0,
  // but LiteRT has None=0, Float32=1.
  type.element_type = litert::internal::MapElementType(tensor_fb.type());

  const auto* shape = tensor_fb.shape();
  if (!shape) {
    return absl::InvalidArgumentError("Missing tensor shape metadata");
  }
  const int rank = shape->size();
  if (rank > LITERT_TENSOR_MAX_RANK) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tensor rank %d exceeds max rank %d", rank, LITERT_TENSOR_MAX_RANK));
  }

  type.layout.rank = static_cast<unsigned int>(rank);
  type.layout.has_strides = false;
  for (int i = 0; i < rank; ++i) {
    type.layout.dimensions[i] = shape->Get(i);
    type.layout.strides[i] = 0;
  }
  for (int i = rank; i < LITERT_TENSOR_MAX_RANK; ++i) {
    type.layout.dimensions[i] = 0;
    type.layout.strides[i] = 0;
  }

  return type;
}

LiteRtTensorBufferType SelectGpuBufferType() {
  return kLiteRtTensorBufferTypeOpenClBuffer;
}

bool IsAbsolutePath(absl::string_view path) {
  if (path.empty()) {
    return false;
  }
#if defined(_WIN32)
  if (path.size() >= 2 && path[1] == ':' &&
      ((path[0] >= 'A' && path[0] <= 'Z') ||
       (path[0] >= 'a' && path[0] <= 'z'))) {
    return true;
  }
  if (path.size() >= 2 && path[0] == '\\' && path[1] == '\\') {
    return true;
  }
#endif
  return path[0] == '/';
}

std::string JoinPath(absl::string_view base, absl::string_view relative) {
  if (base.empty()) {
    return std::string(relative);
  }
  if (relative.empty()) {
    return std::string(base);
  }
  std::string result(base);
  char last = result.back();
  if (last != '/' && last != '\\') {
    result.push_back('/');
  }
  result.append(relative.begin(), relative.end());
  return result;
}

struct CpuMapping {
#if !defined(_WIN32)
  void* base = nullptr;
  size_t length = 0;
  size_t page_offset = 0;
#else
  uint8_t* owned_data = nullptr;
#endif
  uint8_t* data = nullptr;
  size_t data_length = 0;
};

struct Entry {
  size_t info_index = 0;
  std::optional<WeightAccess> access;
  std::optional<CpuMapping> cpu_mapping;
};

struct WeightSource {
  enum class Kind { kFilePath, kScopedFile };
  Kind kind = Kind::kFilePath;
  std::string path;
  const litert::ScopedWeightSection* section = nullptr;
  litert::ScopedWeightSource* scoped_source = nullptr;
};

#if defined(_WIN32)
absl::Status ReadScopedRangeWin32(HANDLE handle, uint64_t absolute_offset,
                                  size_t length, uint8_t* destination) {
  LARGE_INTEGER li;
  li.QuadPart = absolute_offset;
  if (!SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
    return absl::UnknownError("Failed to seek scoped weight file");
  }
  size_t remaining = length;
  uint8_t* cursor = destination;
  while (remaining > 0) {
    DWORD chunk = static_cast<DWORD>(
        std::min<uint64_t>(remaining, std::numeric_limits<DWORD>::max()));
    DWORD bytes_read = 0;
    if (!ReadFile(handle, cursor, chunk, &bytes_read, nullptr)) {
      return absl::UnknownError("Failed to read scoped weight file");
    }
    if (bytes_read == 0) {
      return absl::InvalidArgumentError(
          "Unexpected EOF while reading scoped weight file");
    }
    cursor += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}
#endif  // defined(_WIN32)

void ReleaseEntry(Entry& entry) {
  entry.access.reset();
#if !defined(_WIN32)
  if (entry.cpu_mapping) {
    munmap(entry.cpu_mapping->base, entry.cpu_mapping->length);
  }
#else
  if (entry.cpu_mapping && entry.cpu_mapping->owned_data) {
    delete[] entry.cpu_mapping->owned_data;
  }
#endif
  entry.cpu_mapping.reset();
}

absl::StatusOr<std::string> ResolveGroupPath(
    uint32_t group_id,
    const absl::flat_hash_map<uint32_t, std::string>& group_paths,
    const std::optional<std::string>& model_directory) {
  auto it = group_paths.find(group_id);
  if (it == group_paths.end()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Missing external buffer group %u", group_id));
  }
  if (IsAbsolutePath(it->second) || !model_directory ||
      model_directory->empty()) {
    return it->second;
  }
  return JoinPath(*model_directory, it->second);
}

absl::StatusOr<CpuMapping> MapFileSliceFromPath(const LiteRtWeightInfo& info,
                                                const std::string& path) {
#if !defined(_WIN32)
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::InternalError(absl::StrFormat("Failed to open %s: %s",
                                               path.c_str(), strerror(errno)));
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    int saved_errno = errno;
    close(fd);
    return absl::InternalError(absl::StrFormat(
        "Failed to stat %s: %s", path.c_str(), strerror(saved_errno)));
  }

  const uint64_t file_size = static_cast<uint64_t>(st.st_size);
  if (info.offset > file_size || info.length > file_size - info.offset) {
    close(fd);
    return absl::InvalidArgumentError(
        absl::StrFormat("External weight slice out of range for %s", path));
  }

  const int64_t page_size = sysconf(_SC_PAGESIZE);
  const size_t page_offset =
      static_cast<size_t>(info.offset % static_cast<uint64_t>(page_size));
  const off_t map_offset = static_cast<off_t>(info.offset - page_offset);
  const size_t map_length = page_offset + static_cast<size_t>(info.length);

  void* base =
      mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, fd, map_offset);
  int saved_errno = errno;
  close(fd);
  if (base == MAP_FAILED) {
    return absl::InternalError(absl::StrFormat(
        "mmap failed for %s: %s", path.c_str(), strerror(saved_errno)));
  }

  uint8_t* data_ptr = static_cast<uint8_t*>(base) + page_offset;

  CpuMapping mapping;
  mapping.base = base;
  mapping.length = map_length;
  mapping.page_offset = page_offset;
  mapping.data = data_ptr;
  mapping.data_length = static_cast<size_t>(info.length);
  return mapping;
#else
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return absl::InternalError(absl::StrFormat("Failed to open %s", path));
  }
  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to determine size of %s", path));
  }
  if (static_cast<uint64_t>(file_size) < info.offset + info.length) {
    return absl::InvalidArgumentError(
        absl::StrFormat("External weight slice out of range for %s", path));
  }
  file.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  auto* buffer = new uint8_t[info.length];
  file.read(reinterpret_cast<char*>(buffer), info.length);
  if (!file) {
    delete[] buffer;
    return absl::InternalError(
        absl::StrFormat("Failed to read %llu bytes from %s",
                        static_cast<unsigned long long>(info.length), path));
  }

  CpuMapping mapping;
  mapping.owned_data = buffer;
  mapping.data = buffer;
  mapping.data_length = static_cast<size_t>(info.length);
  return mapping;
#endif
}

absl::StatusOr<std::vector<uint8_t>> ReadFileSliceFromPath(
    const LiteRtWeightInfo& info, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return absl::InternalError(absl::StrFormat("Failed to open %s", path));
  }
  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    return absl::InternalError(
        absl::StrFormat("Failed to determine size of %s", path));
  }
  if (static_cast<uint64_t>(file_size) < info.offset + info.length) {
    return absl::InvalidArgumentError(
        absl::StrFormat("External weight slice out of range for %s", path));
  }
  file.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);

  std::vector<uint8_t> data(info.length);
  file.read(reinterpret_cast<char*>(data.data()), info.length);
  if (!file) {
    return absl::InternalError(
        absl::StrFormat("Failed to read %lu bytes from %s", info.length, path));
  }
  return data;
}

absl::Status ValidateScopedSlice(const LiteRtWeightInfo& info,
                                 const litert::ScopedWeightSection& section,
                                 uint64_t* absolute_offset) {
  if (section.length < info.offset) {
    return absl::InvalidArgumentError(
        "External weight offset exceeds scoped section length");
  }
  const uint64_t remaining = section.length - info.offset;
  if (info.length > remaining) {
    return absl::InvalidArgumentError(
        "External weight slice out of range for scoped section");
  }
  if (absolute_offset) {
    *absolute_offset = section.offset + info.offset;
  }
  return absl::OkStatus();
}

absl::StatusOr<CpuMapping> MapScopedFileSlice(
    const LiteRtWeightInfo& info, const litert::ScopedWeightSection& section,
    litert::ScopedWeightSource* source) {
  if (!source || !source->file.IsValid()) {
    return absl::FailedPreconditionError(
        "Scoped weight source is not available");
  }
  uint64_t absolute_offset = 0;
  LITERT_RETURN_IF_ERROR(ValidateScopedSlice(info, section, &absolute_offset));
#if !defined(_WIN32)
  int fd = source->file.file();
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  const size_t page_offset =
      static_cast<size_t>(absolute_offset % static_cast<uint64_t>(page_size));
  const off_t map_offset = static_cast<off_t>(absolute_offset - page_offset);
  const size_t map_length = page_offset + static_cast<size_t>(info.length);
  void* base =
      mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE, fd, map_offset);
  int saved_errno = errno;
  if (base == MAP_FAILED) {
    return absl::InternalError(
        absl::StrFormat("mmap failed: %s", strerror(saved_errno)));
  }
  CpuMapping mapping;
  mapping.base = base;
  mapping.length = map_length;
  mapping.page_offset = page_offset;
  mapping.data = static_cast<uint8_t*>(base) + page_offset;
  mapping.data_length = static_cast<size_t>(info.length);
  return mapping;
#else
  HANDLE handle = source->file.file();
  auto* buffer = new uint8_t[info.length];
  absl::Status read_status = ReadScopedRangeWin32(
      handle, absolute_offset, static_cast<size_t>(info.length), buffer);
  if (!read_status.ok()) {
    delete[] buffer;
    return read_status;
  }
  CpuMapping mapping;
  mapping.owned_data = buffer;
  mapping.data = buffer;
  mapping.data_length = static_cast<size_t>(info.length);
  return mapping;
#endif
}

absl::StatusOr<std::vector<uint8_t>> ReadScopedFileSlice(
    const LiteRtWeightInfo& info, const litert::ScopedWeightSection& section,
    litert::ScopedWeightSource* source) {
  if (!source || !source->file.IsValid()) {
    return absl::FailedPreconditionError(
        "Scoped weight source is not available");
  }
  uint64_t absolute_offset = 0;
  LITERT_RETURN_IF_ERROR(ValidateScopedSlice(info, section, &absolute_offset));
  std::vector<uint8_t> data(info.length);
#if !defined(_WIN32)
  int fd = source->file.file();
  size_t remaining = static_cast<size_t>(info.length);
  uint8_t* cursor = data.data();
  off_t offset = static_cast<off_t>(absolute_offset);
  while (remaining > 0) {
    ssize_t read_bytes = pread(fd, cursor, remaining, offset);
    if (read_bytes < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::ErrnoToStatus(errno, "pread failed for scoped weight");
    }
    if (read_bytes == 0) {
      return absl::InvalidArgumentError(
          "Unexpected EOF while reading scoped weight");
    }
    cursor += read_bytes;
    remaining -= read_bytes;
    offset += read_bytes;
  }
  return data;
#else
  HANDLE handle = source->file.file();
  LITERT_RETURN_IF_ERROR(ReadScopedRangeWin32(
      handle, absolute_offset, static_cast<size_t>(info.length), data.data()));
  return data;
#endif
}

absl::StatusOr<CpuMapping> MapWeightSlice(const LiteRtWeightInfo& info,
                                          const WeightSource& source) {
  if (source.kind == WeightSource::Kind::kScopedFile) {
    return MapScopedFileSlice(info, *source.section, source.scoped_source);
  }
  return MapFileSliceFromPath(info, source.path);
}

absl::StatusOr<std::vector<uint8_t>> ReadWeightSlice(
    const LiteRtWeightInfo& info, const WeightSource& source) {
  if (source.kind == WeightSource::Kind::kScopedFile) {
    return ReadScopedFileSlice(info, *source.section, source.scoped_source);
  }
  return ReadFileSliceFromPath(info, source.path);
}

absl::Status EnsureCpuTensorBuffer(Entry& entry, const LiteRtWeightInfo& info,
                                   const WeightSource& source) {
  if (!entry.access.has_value()) {
    entry.access.emplace();
  }
  if (entry.access->GetHostBuffer() != nullptr) {
    LiteRtTensorBufferType buffer_type;
    if (LiteRtGetTensorBufferType(entry.access->GetHostBuffer(),
                                  &buffer_type) == kLiteRtStatusOk &&
        buffer_type == kLiteRtTensorBufferTypeHostMemory) {
      return absl::OkStatus();
    }
  }

  if (!entry.cpu_mapping) {
    absl::StatusOr<CpuMapping> mapping = MapWeightSlice(info, source);
    if (!mapping.ok()) {
      return mapping.status();
    }
    entry.cpu_mapping = std::move(*mapping);
  }

  LiteRtTensorBuffer host_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromHostMemory(
      &info.tensor_type, static_cast<void*>(entry.cpu_mapping->data),
      entry.cpu_mapping->data_length, /*deallocator=*/nullptr, &host_buffer));
  entry.access->SetHostBuffer(LiteRtTensorBufferPtr(host_buffer));
  return absl::OkStatus();
}

#if LITERT_HAS_OPENCL_SUPPORT
absl::Status EnsureOpenClTensorBuffer(Entry& entry,
                                      const LiteRtWeightInfo& info,
                                      const WeightSource& path,
                                      LiteRtEnvironmentT* env) {
  if (!entry.access.has_value()) {
    entry.access.emplace();
  }
  if (!env) {
    return absl::FailedPreconditionError(
        "LiteRtEnvironment must not be null for OpenCL access");
  }

  LITERT_ASSIGN_OR_RETURN(auto* gpu_env, env->GetGpuEnvironment());

  LITERT_ASSIGN_OR_RETURN(std::vector<uint8_t> data,
                          ReadWeightSlice(info, path));

  LiteRtTensorBuffer device_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      env, info.gpu_buffer_type, &info.tensor_type, info.length,
      &device_buffer));
  entry.access->SetDeviceBuffer(LiteRtTensorBufferPtr(device_buffer));

  cl_mem cl_memory;
  LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferOpenClMemory(
      entry.access->GetDeviceBuffer(), &cl_memory));

  LiteRtRankedTensorType tensor_type_c = info.tensor_type;
  LiteRtStatus upload_status = ::litert::internal::LiteRtGpuMemoryUpload(
      gpu_env, &tensor_type_c, info.gpu_buffer_type, info.length, data.data(),
      cl_memory);
  if (upload_status != kLiteRtStatusOk) {
    return absl::InternalError(
        absl::StrFormat("Failed to upload OpenCL buffer (status=%d)",
                        static_cast<int>(upload_status)));
  }
  return absl::OkStatus();
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

void ParseFlatBuffer(const tflite::Model& model,
                     std::vector<LiteRtWeightInfo>& infos,
                     absl::flat_hash_map<uint32_t, Entry>& entries,
                     absl::flat_hash_map<uint32_t, std::string>& group_paths) {
  const auto* buffers = model.external_buffers();
  const auto* groups = model.external_buffer_groups();
  const auto* subgraphs = model.subgraphs();
  if (!buffers || !subgraphs) return;

  using BufferPtr = decltype(buffers->Get(0));
  using GroupPtr = decltype(groups ? groups->Get(0) : nullptr);
  absl::flat_hash_map<uint32_t, BufferPtr> buffer_lookup;
  for (int i = 0; i < buffers->size(); ++i) {
    BufferPtr buffer = buffers->Get(i);
    if (buffer) buffer_lookup.emplace(buffer->id(), buffer);
  }

  absl::flat_hash_map<uint32_t, GroupPtr> group_lookup;
  if (groups) {
    for (int i = 0; i < groups->size(); ++i) {
      GroupPtr group = groups->Get(i);
      if (group) {
        group_lookup.emplace(i, group);
      }
    }
  }

  for (int sg = 0; sg < subgraphs->size(); ++sg) {
    const auto* subgraph = subgraphs->Get(sg);
    if (!subgraph || !subgraph->tensors()) continue;

    for (int t = 0; t < subgraph->tensors()->size(); ++t) {
      const auto* tensor_fb = subgraph->tensors()->Get(t);
      if (!tensor_fb || tensor_fb->external_buffer() == 0) continue;

      auto buffer_it = buffer_lookup.find(tensor_fb->external_buffer());
      if (buffer_it == buffer_lookup.end()) continue;
      const BufferPtr buffer = buffer_it->second;

      LiteRtWeightInfo info;
      info.subgraph_index = static_cast<uint16_t>(sg);
      info.tensor_index = static_cast<uint16_t>(t);
      info.external_buffer_id = buffer->id();
      info.group_id = buffer->group();
      info.offset = buffer->offset();
      info.length = buffer->length();
      info.packing = buffer->packing()
                         ? absl::string_view(buffer->packing()->string_view())
                         : absl::string_view();

      absl::StatusOr<LiteRtRankedTensorType> ranked_type =
          BuildRankedTensorType(*tensor_fb);
      if (!ranked_type.ok()) {
        continue;
      }
      info.tensor_type = *ranked_type;
      info.gpu_buffer_type = SelectGpuBufferType();

      auto group_it = group_lookup.find(info.group_id);
      if (group_it != group_lookup.end() && group_it->second &&
          group_it->second->name()) {
        group_paths.try_emplace(info.group_id, group_it->second->name()->str());
      }

      entries.emplace(info.external_buffer_id, Entry{infos.size()});
      infos.push_back(info);
    }
  }
}

class LiteRtWeightLoader : public WeightLoader {
 public:
  LiteRtWeightLoader(
      const tflite::Model* model, std::optional<std::string> model_directory,
      std::unique_ptr<litert::ScopedWeightSource> scoped_weight_source)
      : model_directory_(std::move(model_directory)),
        scoped_weight_source_(std::move(scoped_weight_source)) {
    ParseFlatBuffer(*model, infos_, entries_, group_paths_);
    if (scoped_weight_source_) {
      for (const auto& [group_id, group_path] : group_paths_) {
        auto section_it = scoped_weight_source_->sections.find(group_path);
        if (section_it != scoped_weight_source_->sections.end()) {
          group_sections_.emplace(group_id, section_it->second);
        }
      }
    }
  }

  ~LiteRtWeightLoader() override {
    for (auto& [_, entry] : entries_) {
      ReleaseEntry(entry);
    }
  }

  absl::Span<const WeightInfo> GetWeightInfo() const override {
    if (weight_info_cache_.empty() && !infos_.empty()) {
      weight_info_cache_.reserve(infos_.size());
      for (const auto& info : infos_) {
        // Copy only the base WeightInfo fields.
        weight_info_cache_.push_back(static_cast<const WeightInfo&>(info));
      }
    }
    return absl::MakeConstSpan(weight_info_cache_);
  }

  const WeightInfo* FindWeightInfoByBuffer(
      uint32_t external_buffer_id) const override {
    auto it = entries_.find(external_buffer_id);
    if (it == entries_.end()) {
      return nullptr;
    }
    return &infos_[it->second.info_index];
  }

  absl::Status PrepareAccess(const WeightAccessRequest& request,
                             LiteRtEnvironmentT* env) override {
    for (auto& [tensor_id, entry] : entries_) {
      const LiteRtWeightInfo& info = infos_[entry.info_index];
      WeightSource source;
      auto section_it = scoped_weight_source_
                            ? group_sections_.find(info.group_id)
                            : group_sections_.end();
      if (section_it != group_sections_.end()) {
        source.kind = WeightSource::Kind::kScopedFile;
        source.section = &section_it->second;
        source.scoped_source = scoped_weight_source_.get();
      } else {
        absl::StatusOr<std::string> path =
            ResolveGroupPath(info.group_id, group_paths_, model_directory_);
        if (!path.ok()) {
          return path.status();
        }
        source.kind = WeightSource::Kind::kFilePath;
        source.path = std::move(*path);
      }

      if (request.cpu) {
        absl::Status status = EnsureCpuTensorBuffer(entry, info, source);
        if (!status.ok()) {
          return status;
        }
      }
      if (request.opencl) {
#if LITERT_HAS_OPENCL_SUPPORT
        absl::Status status =
            EnsureOpenClTensorBuffer(entry, info, source, env);
        if (!status.ok()) {
          return status;
        }
#else
        return absl::UnimplementedError(
            "OpenCL support not enabled in this build");
#endif
      }
    }
    return absl::OkStatus();
  }

  absl::Status SetExternalWeightByBuffer(uint32_t external_buffer_id,
                                         WeightAccess access) override {
    auto it = entries_.find(external_buffer_id);
    if (it == entries_.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unknown external buffer id %u", external_buffer_id));
    }
    ReleaseEntry(it->second);
    it->second.access = std::move(access);
    return absl::OkStatus();
  }

  const WeightAccess* GetExternalWeightByBuffer(
      uint32_t external_buffer_id) const override {
    auto it = entries_.find(external_buffer_id);
    if (it == entries_.end() || !it->second.access) {
      return nullptr;
    }
    return &(*it->second.access);
  }

 private:
  std::optional<std::string> model_directory_;
  std::vector<LiteRtWeightInfo> infos_;
  absl::flat_hash_map<uint32_t, Entry> entries_;
  absl::flat_hash_map<uint32_t, std::string> group_paths_;
  std::unique_ptr<litert::ScopedWeightSource> scoped_weight_source_;
  absl::flat_hash_map<uint32_t, litert::ScopedWeightSection> group_sections_;
  mutable std::vector<WeightInfo> weight_info_cache_;
};

}  // namespace

std::unique_ptr<WeightLoader> CreateLiteRtWeightLoader(
    const tflite::Model* flatbuffer, std::optional<std::string> model_directory,
    std::unique_ptr<litert::ScopedWeightSource> scoped_weight_source) {
  return std::make_unique<LiteRtWeightLoader>(
      flatbuffer, std::move(model_directory), std::move(scoped_weight_source));
}

}  // namespace weight_loader
