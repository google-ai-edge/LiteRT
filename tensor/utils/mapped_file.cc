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

#include "tensor/utils/mapped_file.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::tensor {
namespace {

#if defined(_WIN32)
size_t PageSize() {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return static_cast<size_t>(info.dwPageSize);
}
#else
size_t PageSize() {
  const int64_t page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 0;
}
#endif

}  // namespace

absl::StatusOr<MappedFile> MappedFile::OpenReadOnly(absl::string_view path) {
  const std::string path_str(path);
#if defined(_WIN32)
  const DWORD attributes = GetFileAttributesA(path_str.c_str());
  if (attributes != INVALID_FILE_ATTRIBUTES &&
      (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Not a regular file: ", path));
  }
  HANDLE file = CreateFileA(path_str.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            /*lpSecurityAttributes=*/nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, /*hTemplateFile=*/nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return absl::NotFoundError(absl::StrCat("Could not open file: ", path,
                                            " (error ", GetLastError(), ")"));
  }
  if (GetFileType(file) != FILE_TYPE_DISK) {
    CloseHandle(file);
    return absl::InvalidArgumentError(
        absl::StrCat("Not a regular file: ", path));
  }
  LARGE_INTEGER file_size = {};
  if (!GetFileSizeEx(file, &file_size)) {
    CloseHandle(file);
    return absl::InternalError(absl::StrCat("Could not stat file: ", path));
  }
  if (file_size.QuadPart <= 0) {
    CloseHandle(file);
    return absl::InvalidArgumentError(absl::StrCat("Empty file: ", path));
  }
  HANDLE mapping =
      CreateFileMappingA(file, /*lpFileMappingAttributes=*/nullptr,
                         PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/nullptr);
  // The view keeps the file and the mapping object alive once mapped.
  CloseHandle(file);
  if (mapping == nullptr) {
    return absl::InternalError(absl::StrCat("Could not map file: ", path,
                                            " (error ", GetLastError(), ")"));
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, /*dwFileOffsetHigh=*/0,
                             /*dwFileOffsetLow=*/0,
                             /*dwNumberOfBytesToMap=*/0);
  CloseHandle(mapping);
  if (view == nullptr) {
    return absl::InternalError(absl::StrCat("Could not map file: ", path,
                                            " (error ", GetLastError(), ")"));
  }
  return MappedFile(static_cast<const uint8_t*>(view),
                    static_cast<size_t>(file_size.QuadPart));
#else
  const int fd = open(path_str.c_str(), O_RDONLY);
  if (fd == -1) {
    return absl::NotFoundError(absl::StrCat("Could not open file: ", path));
  }
  struct stat sb = {};
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return absl::InternalError(absl::StrCat("Could not stat file: ", path));
  }
  if (!S_ISREG(sb.st_mode)) {
    close(fd);
    return absl::InvalidArgumentError(
        absl::StrCat("Not a regular file: ", path));
  }
  const size_t file_size = static_cast<size_t>(sb.st_size);
  if (file_size == 0) {
    close(fd);
    return absl::InvalidArgumentError(absl::StrCat("Empty file: ", path));
  }
  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) {
    return absl::InternalError(absl::StrCat("Could not mmap file: ", path));
  }
  return MappedFile(static_cast<const uint8_t*>(mapped), file_size);
#endif
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    Reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

MappedFile::~MappedFile() { Reset(); }

void MappedFile::Reset() {
  if (data_ == nullptr) return;
#if defined(_WIN32)
  UnmapViewOfFile(data_);
#else
  munmap(const_cast<uint8_t*>(data_), size_);
#endif
  data_ = nullptr;
  size_ = 0;
}

void MappedFile::AdviseDontNeed(size_t offset, size_t length) const {
  if (data_ == nullptr || length == 0 || offset >= size_) return;
  const size_t page_size = PageSize();
  if (page_size == 0) return;
  const size_t len = length < size_ - offset ? length : size_ - offset;
  // Only whole pages inside the range are released, so that the pages it
  // shares with its neighbors stay resident.
  const size_t aligned_start = (offset + page_size - 1) & ~(page_size - 1);
  const size_t aligned_end = (offset + len) & ~(page_size - 1);
  if (aligned_end <= aligned_start) return;
  void* start = const_cast<uint8_t*>(data_) + aligned_start;
  const size_t aligned_len = aligned_end - aligned_start;
#if defined(_WIN32)
  // Unlocking pages that are not locked removes them from the working set.
  (void)VirtualUnlock(start, aligned_len);
#else
  (void)madvise(start, aligned_len, MADV_DONTNEED);
#endif
}

}  // namespace litert::tensor
