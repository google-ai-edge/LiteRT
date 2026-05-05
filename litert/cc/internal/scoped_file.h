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

#include <cstddef>
#include <string>
#include <string_view>

#ifndef LITERT_NO_ABSL
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#endif  // LITERT_NO_ABSL
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_expected.h"

/// @file
/// @brief Defines a file wrapper that automatically closes on destruction.

namespace litert {

#ifndef LITERT_NO_ABSL
template <typename T>
using ScopedFileStatusOr = absl::StatusOr<T>;
#else
template <typename T>
using ScopedFileStatusOr = Expected<T>;
#endif  // LITERT_NO_ABSL

/// @brief A file wrapper that ensures the underlying file handle is
/// automatically closed when the object goes out of scope.
class ScopedFile {
 public:
#if defined(_WIN32)
  using PlatformFile = void*;
  static const PlatformFile kInvalidPlatformFile;
#else
  using PlatformFile = int;
  static constexpr PlatformFile kInvalidPlatformFile = -1;
#endif

  static ScopedFileStatusOr<ScopedFile> Open(StringView path);
  static ScopedFileStatusOr<ScopedFile> OpenWritable(StringView path);

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

  /// @brief Returns the size of the file in bytes.
  static ScopedFileStatusOr<size_t> GetSize(PlatformFile file);
  ScopedFileStatusOr<size_t> GetSize() const { return GetSize(file_); }

  /// @brief Returns a `ScopedFile` pointing to the same underlying file.
  ScopedFileStatusOr<ScopedFile> Duplicate();

  /// @brief Releases ownership of the current file as a C file descriptor.
  ///
  /// @note Windows-specific behavior:
  /// Releases ownership of the operating system file `HANDLE` and returns the
  /// corresponding C file descriptor.
  ///
  /// This function currently only works if the file can be re-opened in
  /// read/write mode.
  ///
  /// @warning Files opened in asynchronous mode (`FILE_FLAG_OVERLAPPED`) are
  /// not supported. Windows' POSIX C implementation does not support such I/O
  /// operations. This function attempts to detect and return an error for this
  /// invalid use case, but it is not guaranteed.
  ///
  /// @warning If successful, the returned file descriptor owns the file and
  /// must be closed using `_close`.
  ///
  /// @warning While it is possible to get a `HANDLE` back from the file
  /// descriptor, **ownership will remain with the file descriptor**.
  ScopedFileStatusOr<int> Release();

 private:
  PlatformFile ReleasePlatformFile() {
    PlatformFile temp = file_;
    file_ = kInvalidPlatformFile;
    return temp;
  }

  /// @brief Platform-specific file operations.
  ///
  /// The implementation can assume that the passed `PlatformFile` is valid.
  /// This must be ensured by the caller.
  static void CloseFile(PlatformFile file);
  static ScopedFileStatusOr<size_t> GetSizeImpl(PlatformFile file);

  PlatformFile file_;
};

namespace internal::scoped_file_detail {

inline bool IsFileValid(ScopedFile::PlatformFile file) {
  return file != ScopedFile::kInvalidPlatformFile;
}

}  // namespace internal::scoped_file_detail

inline ScopedFileStatusOr<size_t> ScopedFile::GetSize(PlatformFile file) {
  if (!internal::scoped_file_detail::IsFileValid(file)) {
#ifndef LITERT_NO_ABSL
    return absl::FailedPreconditionError("Scoped file is not valid");
#else
    return Unexpected(Status::kErrorInvalidArgument,
                      "Scoped file is not valid");
#endif  // LITERT_NO_ABSL
  }
  return GetSizeImpl(file);
}

}  // namespace litert

#if defined(_WIN32)
#include "litert/cc/internal/scoped_file_win.h"
#else
#include "litert/cc/internal/scoped_file_posix.h"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_FILE_H_
