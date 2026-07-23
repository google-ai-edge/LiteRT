/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stddef.h>

#include <cerrno>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "tflite/converter/allocation.h"
#include "tflite/converter/core/api/error_reporter.h"
#include "tflite/util.h"

namespace tflite {
namespace {

#if defined(_WIN32)

// On Windows a mapped view of a file is represented by the address returned
// from MapViewOfFile(). A failed mapping is reported as nullptr (there is no
// MAP_FAILED sentinel), and the open file descriptor uses the MSVC CRT.
constexpr void* kFailedMmap = nullptr;

// Returns the platform allocation granularity. On Windows the file offset
// passed to MapViewOfFile() must be a multiple of dwAllocationGranularity
// (typically 64KB), which is coarser than the page size used on POSIX.
size_t GetMapOffsetAlignment() {
  SYSTEM_INFO sys_info;
  ::GetSystemInfo(&sys_info);
  return sys_info.dwAllocationGranularity;
}

// Splits a 64-bit value into the high/low 32-bit parts expected by the
// Win32 file-mapping APIs.
struct HighLow {
  DWORD high;
  DWORD low;

  static HighLow From(uint64_t value) {
    return HighLow{static_cast<DWORD>(value >> 32),
                   static_cast<DWORD>(value & 0xFFFFFFFFULL)};
  }
};

// _dup(-1) (and other negative fds) trips the MSVC CRT invalid-parameter
// handler, which aborts the process by default. Guard so a bad fd is reported
// as an error the same way it is on POSIX, where dup(-1) simply returns -1.
int DupFd(int fd) { return fd < 0 ? -1 : _dup(fd); }
int CloseFd(int fd) { return _close(fd); }
int OpenReadOnly(const char* filename) {
  return _open(filename, _O_RDONLY | _O_BINARY);
}

size_t GetFdSizeBytes(int fd) {
  if (fd < 0) {
    return 0;
  }
  int64_t size = _filelengthi64(fd);
  if (size < 0) {
    return 0;
  }
  return static_cast<size_t>(size);
}

#else  // !_WIN32

void* const kFailedMmap = MAP_FAILED;

size_t GetMapOffsetAlignment() { return sysconf(_SC_PAGE_SIZE); }

int DupFd(int fd) { return dup(fd); }
int CloseFd(int fd) { return close(fd); }
int OpenReadOnly(const char* filename) { return open(filename, O_RDONLY); }

size_t GetFdSizeBytes(int fd) {
  if (fd < 0) {
    return 0;
  }

  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }

  return fd_stat.st_size;
}

#endif  // _WIN32

}  // namespace

MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, OpenReadOnly(filename), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, DupFd(fd), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(const char* filename, size_t offset,
                               size_t length, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, OpenReadOnly(filename), offset, length,
                     map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, size_t offset, size_t length,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, DupFd(fd), offset, length, map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               bool map_private)
    : MMAPAllocation(error_reporter, owned_fd, /*offset=*/0,
                     /*length=*/GetFdSizeBytes(owned_fd), map_private) {}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               size_t offset, size_t length, bool map_private)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(kFailedMmap),
      buffer_size_bytes_(length) {
  if (owned_fd < 0) {
    return;
  }

  const size_t alignment = GetMapOffsetAlignment();
  offset_in_buffer_ = offset % alignment;
  offset_of_buffer_in_file_ = offset - offset_in_buffer_;

  size_t file_size = GetFdSizeBytes(mmap_fd_);
  CheckedInt<size_t> checked_length_offset =
      CheckedInt<size_t>(length) + offset;
  // A zero-length mapping is rejected on both platforms: POSIX mmap(len=0)
  // fails with EINVAL, and on Windows MapViewOfFile(size=0) would instead map
  // to the end of the file -- an inconsistency, so reject it explicitly.
  if (length == 0 || checked_length_offset.Overflow() ||
      checked_length_offset.Value() > file_size) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Asked to mmap '%d' bytes from fd '%d' at offset "
                         "'%d'. This is over the length of file '%d'.",
                         length, mmap_fd_, offset, file_size);
    return;
  }

#if defined(_WIN32)
  HANDLE osf_handle = reinterpret_cast<HANDLE>(_get_osfhandle(mmap_fd_));
  if (osf_handle == INVALID_HANDLE_VALUE) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to obtain file handle for fd '%d'.", mmap_fd_);
    return;
  }

  // map_private => copy-on-write writable mapping; otherwise read-only shared.
  const DWORD protect = map_private ? PAGE_WRITECOPY : PAGE_READONLY;
  const DWORD access = map_private ? FILE_MAP_COPY : FILE_MAP_READ;

  // A maximum size of 0 maps the whole file; the view below restricts the
  // window that is actually mapped into the address space.
  HANDLE file_mapping = ::CreateFileMappingA(osf_handle, /*attributes=*/nullptr,
                                             protect, /*sizeHigh=*/0,
                                             /*sizeLow=*/0, /*name=*/nullptr);
  if (file_mapping == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "CreateFileMapping of fd '%d' failed with error '%lu'.",
                         mmap_fd_, ::GetLastError());
    return;
  }

  HighLow file_offset = HighLow::From(offset_of_buffer_in_file_);
  mmapped_buffer_ = ::MapViewOfFile(file_mapping, access, file_offset.high,
                                    file_offset.low,
                                    /*dwNumberOfBytesToMap=*/length +
                                        offset_in_buffer_);
  // The view holds its own reference to the section object, so the mapping
  // handle can be released immediately; the view stays valid until unmapped.
  ::CloseHandle(file_mapping);

  if (mmapped_buffer_ == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "MapViewOfFile of fd '%d' at offset '%d' failed with "
                         "error '%lu'.",
                         mmap_fd_, offset, ::GetLastError());
    return;
  }
#else   // !_WIN32
  mmapped_buffer_ = mmap(nullptr, /*__len=*/length + offset_in_buffer_,
                         map_private ? (PROT_READ | PROT_WRITE) : PROT_READ,
                         map_private ? MAP_PRIVATE : MAP_SHARED, mmap_fd_,
                         /*__offset=*/offset_of_buffer_in_file_);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Mmap of '%d' at offset '%d' failed with error '%d'.",
                         mmap_fd_, offset, errno);
    return;
  }
#endif  // _WIN32
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
#if defined(_WIN32)
    ::UnmapViewOfFile(const_cast<void*>(mmapped_buffer_));
#else
    munmap(const_cast<void*>(mmapped_buffer_),
           buffer_size_bytes_ + offset_in_buffer_);
#endif
  }
  if (mmap_fd_ >= 0) {
    CloseFd(mmap_fd_);
  }
}

const void* MMAPAllocation::base() const {
  return reinterpret_cast<const void*>(
      reinterpret_cast<const char*>(mmapped_buffer_) + offset_in_buffer_);
}

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != kFailedMmap; }

bool MMAPAllocation::IsSupported() { return true; }

}  // namespace tflite
