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

#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"

#include <fcntl.h>
#include <sys/types.h>

#if defined(_WIN32)
#include <io.h>
#define F_OK 0
#else  // defined(_WIN32)
#include <unistd.h>
#endif  // defined(_WIN32)

// We currently use the memfd_create system call to create in-memory files which
// is only supported on Linux and Android.
#if defined(__linux__) || defined(__ANDROID__)
#ifndef TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED
// Some systems have syscall.h but don't define the SYS_memfd_create macro. We
// detect those by actually doing the include and checking for its definition.
#ifdef SYS_memfd_create
#define TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED 1
#endif  // SYS_memfd_create
#endif  // TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED
#endif  // defined(__linux__) || defined(__ANDROID__)

#include <cstdio>

#if !TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED
#include "tflite/logger.h"
#include "tflite/minimal_logging.h"
#endif

namespace ml_drift {

FileDescriptor FileDescriptor::Duplicate() const {
  if (!IsValid()) {
    return FileDescriptor(-1);
  }
  return FileDescriptor(dup(fd_));
}

void FileDescriptor::Reset(int new_fd) {
  if (fd_ == new_fd) {
    return;
  }
  if (IsValid()) {
    close(fd_);
  }
  fd_ = new_fd;
}

FileDescriptor::Offset FileDescriptor::GetPos() const {
#if defined(_WIN32)
  return _lseeki64(fd_, 0, SEEK_CUR);
#else   // defined(_WIN32)
  return lseek(fd_, 0, SEEK_CUR);
#endif  // defined(_WIN32)
}

FileDescriptor::Offset FileDescriptor::SetPos(size_t position) const {
#if defined(_WIN32)
  return _lseeki64(fd_, position, SEEK_SET);
#else   // defined(_WIN32)
  return lseek(fd_, position, SEEK_SET);
#endif  // defined(_WIN32)
}

FileDescriptor::Offset FileDescriptor::SetPosFromEnd(size_t offset) const {
#if defined(_WIN32)
  return _lseeki64(fd_, offset, SEEK_END);
#else   // defined(_WIN32)
  return lseek(fd_, offset, SEEK_END);
#endif  // defined(_WIN32)
}

FileDescriptor::Offset FileDescriptor::MovePos(size_t offset) const {
#if defined(_WIN32)
  return _lseeki64(fd_, offset, SEEK_CUR);
#else   // defined(_WIN32)
  return lseek(fd_, offset, SEEK_CUR);
#endif  // defined(_WIN32)
}

FileDescriptor FileDescriptor::Open(const char* path, int flags, mode_t mode) {
#if defined(_WIN32)
  if (!(flags & O_TEXT)) {
    flags |= O_BINARY;
  }
#endif  // defined(_WIN32)
  return FileDescriptor(open(path, flags, mode));
}

void FileDescriptor::Close() { Reset(-1); }

bool FileDescriptor::Read(void* dst, size_t count) const {
  char* dst_it = reinterpret_cast<char*>(dst);
  while (count > 0) {
    const auto bytes = read(fd_, dst_it, count);
    if (bytes == -1 /* error */ || bytes == 0 /* EOF */) {
      return false;
    }
    count -= bytes;
    dst_it += bytes;
  }
  return true;
}

bool FileDescriptor::Write(const void* src, size_t count) const {
  const char* src_it = reinterpret_cast<const char*>(src);
  while (count > 0) {
    const auto bytes = write(fd_, src_it, count);
    if (bytes == -1) {
      return false;
    }
    count -= bytes;
    src_it += bytes;
  }
  return true;
}

bool FileDescriptor::Truncate(size_t size) const {
#if defined(_WIN32)
  return _chsize_s(fd_, size) == 0;
#else   // defined(_WIN32)
  return ftruncate(fd_, size) == 0;
#endif  // defined(_WIN32)
}

bool InMemoryFileDescriptorAvailable() {
#if TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED
  // Test if the syscall memfd_create is available.
  const int test_fd = syscall(SYS_memfd_create, "test fd", 0);
  if (test_fd != -1) {
    close(test_fd);
    return true;
  }
#endif
  return false;
}

FileDescriptor CreateInMemoryFileDescriptor(const char* path) {
#ifdef TFLITE_MLDRIFT_IN_MEMORY_FILE_ENABLED
  return FileDescriptor(
      syscall(SYS_memfd_create, "MLDrift in-memory weight cache", 0));
#else
  // TODO is it okay to use this?
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                  "MLDrift weight cache: in-memory cache is not enabled for "
                  "this build.");
  return FileDescriptor(-1);
#endif
}

}  // namespace ml_drift
