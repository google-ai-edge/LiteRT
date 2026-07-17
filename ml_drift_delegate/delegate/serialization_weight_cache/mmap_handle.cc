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

#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196
#include <sys/stat.h>

#include <functional>
#include <memory>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define F_OK 0
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/windows_util.h"

namespace ml_drift {

void swap(MMapHandle& a, MMapHandle& b) {
  using std::swap;
  swap(a.size_, b.size_);
  swap(a.offset_, b.offset_);
  swap(a.offset_page_adjustment_, b.offset_page_adjustment_);
  swap(a.data_, b.data_);
}

MMapHandle::~MMapHandle() { UnMap(); }

MMapHandle::MMapHandle(MMapHandle&& other) { swap(*this, other); }

MMapHandle& MMapHandle::operator=(MMapHandle&& other) {
  swap(*this, other);
  return *this;
}

absl::Status MMapHandle::Map(const char* path, const size_t offset,
                             size_t size) {
  return this->Map(FileDescriptor::Open(path, O_RDONLY),  // NOLINT: b/332641196
                   offset, size, path);
}

absl::Status MMapHandle::Map(const FileDescriptor& fd, const size_t offset,
                             size_t size, const char* const path) {
  this->UnMap();

  if (!fd.IsValid()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot mmap invalid file descriptor ", fd.Value(), " ('",
                     path, "')."));
  }

#if defined(_WIN64)
  struct _stat64 file_stats;
  if (_fstat64(fd.Value(), &file_stats) != 0) {
    return absl::InternalError(
        absl::StrCat("Could not access file stats to get size ('", path,
                     "'): ", strerror(errno)));
  }
#else   // defined(_WIN64)
  struct stat file_stats;
  if (fstat(fd.Value(), &file_stats) != 0) {
    return absl::InternalError(
        absl::StrCat("Could not access file stats to get size ('", path,
                     "'): ", strerror(errno)));
  }
#endif  // defined(_WIN64)

  // This will reset data_ and size_ on return until is is deactivated.
  ScopeGuard unmap_on_error([this] { UnMap(); });
  if (offset > file_stats.st_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot mmap beyond EOF: offset=", offset,
                     ", file_size=", file_stats.st_size, " ('", path, "')."));
  }
  // If size is 0, we will map the entire rest of the file.
  if (size == 0) {
    size = file_stats.st_size - offset;
  }
  if (size == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot mmap 0 bytes (file is empty or offset is at EOF): offset=",
        offset, ", file_size=", file_stats.st_size, " ('", path, "')."));
  }
  if (size > file_stats.st_size || offset > file_stats.st_size - size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot mmap beyond EOF: offset=", offset, ", size=", size,
                     ", file_size=", file_stats.st_size, " ('", path, "')."));
  }

  size_ = size;
  offset_ = offset;
#if defined(_WIN32)
  HANDLE osf_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd.Value()));
  if (osf_handle == INVALID_HANDLE_VALUE) {
    return absl::InternalError(
        _strerror("Could not convert file descriptor to file handle"));
  }

  std::string name = path;
  if (name == kUnspecifiedPath) {
    name.clear();
  } else {
    for (int i = 0; i < name.size(); ++i) {
      if (name[i] == '\\') {
        name[i] = '_';
      }
    }
  }
  file_mapping_ =
      CreateFileMappingA(osf_handle, /*lpFileMappingAttributes=*/nullptr,
                         /*flProtect=*/PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/name.c_str());
  if (file_mapping_ == NULL) {
    return absl::InternalError(absl::StrCat("Could not create a file mapping: ",
                                            GetLastErrorString()));
  }

  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);

  offset_page_adjustment_ = offset_ % sys_info.dwAllocationGranularity;

  const size_t adjusted_offset = offset - offset_page_adjustment_;
  const DWORD file_offset_high =
      sizeof(DWORD) < sizeof(adjusted_offset)
          ? (adjusted_offset >> CHAR_BIT * sizeof(DWORD))
          : 0;
  const DWORD file_offset_low = static_cast<DWORD>(adjusted_offset);

  data_ = static_cast<uint8_t*>(MapViewOfFile(file_mapping_, FILE_MAP_READ,
                                              file_offset_high, file_offset_low,
                                              /*dwNumberOfBytesToMap=*/0));

  if (data_ == nullptr) {
    return absl::InternalError(absl::StrCat("Could not map file ('", path,
                                            "'): ", GetLastErrorString()));
  }
#else   // defined(_WIN32)
  offset_page_adjustment_ = offset_ % getpagesize();
  data_ = static_cast<uint8_t*>(
      mmap(/*addr=*/nullptr, size_ + offset_page_adjustment_,
           PROT_READ,   // NOLINT: b/332641196
           MAP_SHARED,  // NOLINT: b/332641196
           fd.Value(), offset_ - offset_page_adjustment_));
  if (data_ == MAP_FAILED) {
    return absl::InternalError(
        absl::StrCat("Could not mmap file ('", path, "'): ", strerror(errno)));
  }
#endif  // defined(_WIN32)
  unmap_on_error.Deactivate();
  return absl::OkStatus();
}

#if defined(_WIN32)
void MMapHandle::UnMap(uint8_t* data, size_t size,
                       size_t offset_page_adjustment, HANDLE file_mapping) {
  if (!data) return;
  UnmapViewOfFile(data);
  CloseHandle(file_mapping);
}
#else   // defined(_WIN32)
void MMapHandle::UnMap(uint8_t* data, size_t size,
                       size_t offset_page_adjustment) {
  if (!data) return;
  munmap(data, size + offset_page_adjustment);
}
#endif  // defined(_WIN32)

void MMapHandle::UnMap() {
#if defined(_WIN32)
  UnMap(data_, size_, offset_page_adjustment_, file_mapping_);
#else   // defined(_WIN32)
  UnMap(data_, size_, offset_page_adjustment_);
#endif  // defined(_WIN32)
  ResetWithoutUnmapping();
}

void MMapHandle::ResetWithoutUnmapping() {
  data_ = nullptr;
  offset_ = 0;
  offset_page_adjustment_ = 0;
  size_ = 0;
#if defined(_WIN32)
  file_mapping_ = 0;
#endif  // defined(_WIN32)
}

#if defined(_WIN32)
ReleaseDataCallback MMapHandle::TakeOwnership() {
  uint8_t* data = data_;
  size_t size = size_;
  size_t offset_page_adjustment = offset_page_adjustment_;
  HANDLE file_mapping = file_mapping_;
  auto release_data_callback = std::make_unique<std::function<void()>>(
      [data, size, offset_page_adjustment, file_mapping]() {
        UnMap(data, size, offset_page_adjustment, file_mapping);
      });

  ResetWithoutUnmapping();
  return release_data_callback;
}
#else   // defined(_WIN32)
ReleaseDataCallback MMapHandle::TakeOwnership() {
  uint8_t* data = data_;
  size_t size = size_;
  size_t offset_page_adjustment = offset_page_adjustment_;
  auto release_data_callback = std::make_unique<std::function<void()>>(
      [data, size, offset_page_adjustment]() {
        UnMap(data, size, offset_page_adjustment);
      });

  ResetWithoutUnmapping();
  return release_data_callback;
}
#endif  // defined(_WIN32)

}  // namespace ml_drift
