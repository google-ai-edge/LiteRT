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
#include <string_view>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl

/// @file
/// @brief Defines a file wrapper that automatically closes on destruction.

namespace litert {

/// @brief A file wrapper that ensures the underlying file handle is
/// automatically closed when the object goes out of scope.
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

#ifdef LITERT_NO_ABSL
  static absl::StatusOr<ScopedFile> Open(std::string_view path) {
    return Open(absl::string_view(path.data(), path.size()));
  }

  static absl::StatusOr<ScopedFile> OpenWritable(std::string_view path) {
    return OpenWritable(absl::string_view(path.data(), path.size()));
  }
#endif // LITERT_NO_ABSL

  ScopedFile() : file_(kInvalidPlatformFile) {}
  explicit ScopedFile(PlatformFile file) : file_(file) {}
  ~ScopedFile() {
    if (IsValid()) {
      CloseFile(file_);
    }
  }

  ScopedFile(ScopedFile &&other) : file_(other.ReleasePlatformFile()) {}
  ScopedFile &operator=(ScopedFile &&other) {
    if (IsValid()) {
      CloseFile(file_);
    }
    file_ = other.ReleasePlatformFile();
    return *this;
  }

  ScopedFile(const ScopedFile &) = delete;
  ScopedFile &operator=(const ScopedFile &) = delete;

  PlatformFile file() const { return file_; }
  bool IsValid() const { return file_ != kInvalidPlatformFile; }

  /// @brief Returns the size of the file in bytes.
  static absl::StatusOr<size_t> GetSize(PlatformFile file);
  absl::StatusOr<size_t> GetSize() const { return GetSize(file_); }

  /// @brief Returns a `ScopedFile` pointing to the same underlying file.
  absl::StatusOr<ScopedFile> Duplicate();

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
  absl::StatusOr<int> Release();

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
  static absl::StatusOr<size_t> GetSizeImpl(PlatformFile file);

  PlatformFile file_;
};

namespace {

inline bool IsFileValid(ScopedFile::PlatformFile file) {
  return file != ScopedFile::kInvalidPlatformFile;
}

} // namespace

inline absl::StatusOr<size_t> ScopedFile::GetSize(PlatformFile file) {
  if (!IsFileValid(file)) {
    return absl::FailedPreconditionError("Scoped file is not valid");
  }
  return GetSizeImpl(file);
}

} // namespace litert

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {
namespace {

inline std::wstring Utf8ToWideChar(absl::string_view utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.data(),
                                          (int)utf8str.size(), nullptr, 0);
  std::wstring ws_translated_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.data(), (int)utf8str.size(),
                      &ws_translated_str[0], size_required);
  return ws_translated_str;
}

inline absl::StatusOr<ScopedFile> OpenImpl(absl::string_view path,
                                           DWORD file_attribute_flag) {
  std::wstring ws_path = Utf8ToWideChar(path);

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  DWORD access = GENERIC_READ;
  if (file_attribute_flag == FILE_ATTRIBUTE_NORMAL) {
    access |= GENERIC_WRITE;
  }
  HANDLE hfile = ::CreateFileW(ws_path.c_str(), access, share_mode, nullptr,
                               OPEN_EXISTING, file_attribute_flag, nullptr);
  if (hfile == INVALID_HANDLE_VALUE) {
    return absl::UnknownError(absl::StrCat("Failed to open: ", path));
  }
  return ScopedFile(hfile);
}

inline std::string GetLastErrorString() {
  const DWORD error = GetLastError();
  LPSTR message_buffer = nullptr;
  const DWORD chars_written = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      reinterpret_cast<LPSTR>(&message_buffer), 0, NULL);
  if (chars_written > 0 && message_buffer != nullptr) {
    std::string error_message = message_buffer;
    LocalFree(message_buffer);
    while (!error_message.empty() && std::isspace(error_message.back())) {
      error_message.pop_back();
    }
    return error_message;
  }
  return std::to_string(error);
}

} // namespace

inline const HANDLE ScopedFile::kInvalidPlatformFile = INVALID_HANDLE_VALUE;

inline absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_READONLY);
}

inline absl::StatusOr<ScopedFile>
ScopedFile::OpenWritable(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_NORMAL);
}

inline void ScopedFile::CloseFile(HANDLE file) { ::CloseHandle(file); }

inline absl::StatusOr<size_t> ScopedFile::GetSizeImpl(HANDLE file) {
  LARGE_INTEGER size;
  if (!::GetFileSizeEx(file, &size)) {
    return absl::UnknownError("Failed to get file size");
  }
  return static_cast<size_t>(size.QuadPart);
}

inline absl::StatusOr<ScopedFile> ScopedFile::Duplicate() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }
  HANDLE duplicated;
  if (!DuplicateHandle(GetCurrentProcess(), file_, GetCurrentProcess(),
                       &duplicated, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    return absl::FailedPreconditionError("Could not duplicate handle: " +
                                         GetLastErrorString());
  }
  return ScopedFile(duplicated);
}

inline absl::StatusOr<int> ScopedFile::Release() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }

  char buffer[1];
  if (!ReadFile(file_, &buffer, /*nNumberOfBytesToRead=*/0,
                /*lpNumberOfBytesRead=*/nullptr, /*lpOverlapped=*/nullptr)) {
    return absl::FailedPreconditionError(
        "Could not convert asynchronous handle to C file descriptor: " +
        GetLastErrorString());
  }

  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(file_),
                                 /*flags=*/_O_RDWR | _O_BINARY);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, "Could not convert HANDLE to a C file descriptor");
  }
  ReleasePlatformFile();
  return fd;
}

} // namespace litert
#else
#include <fcntl.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {

inline absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open() failed: ", path));
  }
  return ScopedFile(fd);
}

inline absl::StatusOr<ScopedFile>
ScopedFile::OpenWritable(absl::string_view path) {
  int fd = open(path.data(), O_RDWR);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open() failed: ", path));
  }
  return ScopedFile(fd);
}

inline void ScopedFile::CloseFile(int file) { close(file); }

inline absl::StatusOr<size_t> ScopedFile::GetSizeImpl(int file) {
  struct stat info;
  int result = fstat(file, &info);
  if (result < 0) {
    return absl::ErrnoToStatus(errno, "Failed to get file size");
  }
  return info.st_size;
}

inline absl::StatusOr<ScopedFile> ScopedFile::Duplicate() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }
  return ScopedFile(dup(file_));
}

inline absl::StatusOr<int> ScopedFile::Release() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }
  return ReleasePlatformFile();
}

} // namespace litert
#endif

#endif // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_FILE_H_
