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
#include <sys/stat.h>

#include <cerrno>
#include <cstddef>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/scoped_file.h"

namespace litert {

// static
absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open() failed: ", path));
  }
  return ScopedFile(fd);
}

// static
absl::StatusOr<ScopedFile> ScopedFile::OpenWritable(absl::string_view path) {
  int fd = open(path.data(), O_RDWR);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open() failed: ", path));
  }
  return ScopedFile(fd);
}

// static
void ScopedFile::CloseFile(int file) { close(file); }

// static
absl::StatusOr<size_t> ScopedFile::GetSizeImpl(int file) {
  struct stat info;
  int result = fstat(file, &info);
  if (result < 0) {
    return absl::ErrnoToStatus(errno, "Failed to get file size");
  }
  return info.st_size;
}

absl::StatusOr<ScopedFile> ScopedFile::Duplicate() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }
  return ScopedFile(dup(file_));
}

absl::StatusOr<int> ScopedFile::Release() {
  if (!IsValid()) {
    return absl::InvalidArgumentError("File is not opened.");
  }
  return ReleasePlatformFile();
}

}  // namespace litert
