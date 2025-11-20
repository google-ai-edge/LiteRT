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

#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include <cerrno>
#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/scoped_file.h"

namespace litert {
namespace {
std::wstring Utf8ToWideChar(absl::string_view utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.data(),
                                          (int)utf8str.size(), nullptr, 0);
  std::wstring ws_translated_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.data(), (int)utf8str.size(),
                      &ws_translated_str[0], size_required);
  return ws_translated_str;
}

absl::StatusOr<ScopedFile> OpenImpl(absl::string_view path,
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
}  // namespace

const HANDLE ScopedFile::kInvalidPlatformFile = INVALID_HANDLE_VALUE;

// static
absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_READONLY);
}

// static
absl::StatusOr<ScopedFile> ScopedFile::OpenWritable(absl::string_view path) {
  return OpenImpl(path, FILE_ATTRIBUTE_NORMAL);
}

// static
void ScopedFile::CloseFile(HANDLE file) { ::CloseHandle(file); }

// static
absl::StatusOr<size_t> ScopedFile::GetSizeImpl(HANDLE file) {
  LARGE_INTEGER size;
  if (!::GetFileSizeEx(file, &size)) {
    return absl::UnknownError("Failed to get file size");
  }
  return static_cast<size_t>(size.QuadPart);
}

namespace {

// Returns a string holding the error message corresponding to the code returned
// by `GetLastError()`.
//
// TODO: Extract to a separate helper file for Windows utilities.
std::string GetLastErrorString() {
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
    // Remove trailing whitespace
    while (!error_message.empty() && std::isspace(error_message.back())) {
      error_message.pop_back();
    }
    return error_message;
  }
  // https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes#system-error-codes
  return std::to_string(error);
}

}  // namespace

absl::StatusOr<ScopedFile> ScopedFile::Duplicate() {
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

absl::StatusOr<int> ScopedFile::Release() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }

  // We test whether we can read the file in a synchronous manner by attempting
  // a 0 byte read without specifying the "overlapped" parameter.
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

}  // namespace litert
