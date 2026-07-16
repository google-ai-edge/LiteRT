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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_TESTING_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_TESTING_UTIL_H_

#include <fcntl.h>  // IWYU pragma: keep b/332641196
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>  // IWYU pragma: keep b/332641196
#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace mldrift {

namespace testing_util {

// Wraps a call to `mkstemp` to create temporary files.
class TempFileDesc {
 public:
  static constexpr struct AutoClose {
  } kAutoClose{};

#if defined(_WIN32)
  explicit TempFileDesc(abslstd::string_view path) : path_(path), fd_() {
    char filename[L_tmpnam_s];
    errno_t err = tmpnam_s(filename, L_tmpnam_s);
    if (err) {
      fprintf(stderr, "Could not create temporary filename.\n");
      std::abort();
    }
    path_ = filename;
    fd_ = open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
    if (fd_ < 0) {
      fprintf(stderr, "Could not create temporary filename.\n");
      std::abort();
    }
  }
#else   // defined(_WIN32)
  explicit TempFileDesc(absl::string_view path)
      : path_(path), fd_(mkstemp(path_.data())) {  // NOLINT: b/332641196
    if (GetFd() < 0) {
      perror("Could not create temporary file");
    }
  }
#endif  // defined(_WIN32)

  TempFileDesc(absl::string_view path, AutoClose) : TempFileDesc(path) {
    Close();
  }

  TempFileDesc(const TempFileDesc&) = delete;
  TempFileDesc& operator=(const TempFileDesc&) = delete;

  friend void swap(TempFileDesc& a, TempFileDesc& b) {
    std::swap(a.path_, b.path_);
    std::swap(a.fd_, b.fd_);
  }

  TempFileDesc(TempFileDesc&& other) { swap(*this, other); }
  TempFileDesc& operator=(TempFileDesc&& other) {
    swap(*this, other);
    return *this;
  }

  ~TempFileDesc() { Close(); }

  void Close() {
    if (GetFd() >= 0) {
      close(fd_);  // NOLINT: b/332641196
      fd_ = -1;
    }
  }

  const std::string& GetPath() const { return path_; }

  const char* GetCPath() const { return path_.c_str(); }

  int GetFd() const { return fd_; }

  bool IsOpen() const { return fd_ >= 0; }

 private:
  std::string path_;
  int fd_ = -1;
};

}  // namespace testing_util
}  // namespace mldrift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_SERIALIZATION_WEIGHT_CACHE_TESTING_UTIL_H_
