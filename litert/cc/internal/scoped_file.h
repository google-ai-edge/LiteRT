// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_FILE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_FILE_H_

#if defined(_WIN32)
#include <Windows.h>
#endif

#include <cstddef>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {

// A file wrapper that will automatically close on deletion.
class ScopedFile {
 public:
#if defined(_WIN32)
  using PlatformFile = HANDLE;
  static const PlatformFile kInvalidPlatformFile;
#else
  using PlatformFile = int;
  static constexpr PlatformFile kInvalidPlatformFile = -1;
#endif

  static absl::StatusOr<ScopedFile> Open(absl::string_view path);
  static absl::StatusOr<ScopedFile> OpenWritable(absl::string_view path);

  ScopedFile() : file_(kInvalidPlatformFile) {}
  explicit ScopedFile(PlatformFile file) : file_(file) {}
  ~ScopedFile() {
    if (IsValid()) {
      CloseFile(file_);
    }
  }

  ScopedFile(ScopedFile&& other) : file_(other.ReleasePlatformFile()) {}
  ScopedFile& operator=(ScopedFile&& other) {
    if (IsValid()) {
      CloseFile(file_);
    }
    file_ = other.ReleasePlatformFile();
    return *this;
  }

  ScopedFile(const ScopedFile&) = delete;
  ScopedFile& operator=(const ScopedFile&) = delete;

  PlatformFile file() const { return file_; }
  bool IsValid() const { return file_ != kInvalidPlatformFile; }

  // Returns the number of bytes of the file.
  static absl::StatusOr<size_t> GetSize(PlatformFile file);
  absl::StatusOr<size_t> GetSize() const { return GetSize(file_); }

  // Returns a ScopedFile pointing to the same underlying file as `this`.
  absl::StatusOr<ScopedFile> Duplicate();

  // Releases ownership of the current file as a C file descriptor.
  //
  // Windows notes:
  // Releases ownership of the operating system file HANDLE and returns the
  // corresponding C file descriptor.
  //
  // Note: Currently, this function only works if the file can be re-opened in
  // read/write mode.
  //
  // Warning: Files opened in asynchronous mode (`FILE_FLAG_OVERLAPPED`) are not
  // supported. Windows' POSIX C implementation does not support such I/O
  // operations. This function tries to detect such invalid use case and return
  // an error but doesn't guarantee it.
  //
  // Warning: If successful, the returned file descriptor owns the file which
  // means it will need to be closed using `_close`.
  //
  // Warning: While it is possible to get a HANDLE back from the file
  // descriptor, **ownership will stay with the file descriptor**.
  absl::StatusOr<int> Release();

 private:
  PlatformFile ReleasePlatformFile() {
    PlatformFile temp = file_;
    file_ = kInvalidPlatformFile;
    return temp;
  }

  // Platform-specific file operations requiring platform-specific
  // implementations. It may be assumed by the implementation that the passed
  // `PlatformFile` is valid. This must be ensured by the caller.
  static void CloseFile(PlatformFile file);
  static absl::StatusOr<size_t> GetSizeImpl(PlatformFile file);

  PlatformFile file_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_FILE_H_
