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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MAPPED_FILE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MAPPED_FILE_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::tensor {

// A read-only memory mapping of a whole regular file, portable across POSIX
// (`mmap`) and Windows (`CreateFileMapping` / `MapViewOfFile`). The mapping is
// released on destruction; the file handle is closed once the file is mapped.
class MappedFile {
 public:
  // Maps the whole file at `path`. Fails if the file does not exist, is not a
  // regular file, is empty, or cannot be mapped.
  static absl::StatusOr<MappedFile> OpenReadOnly(absl::string_view path);

  MappedFile() = default;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  absl::Span<const uint8_t> span() const { return {data_, size_}; }
  bool is_mapped() const { return data_ != nullptr; }

  // Hints the OS that the whole pages inside `[offset, offset + length)` are no
  // longer needed in resident memory (`MADV_DONTNEED` on POSIX, working set
  // trimming on Windows). The mapping stays valid: the pages are read back
  // from the file if accessed again. Out-of-range bytes are ignored.
  void AdviseDontNeed(size_t offset, size_t length) const;

  // Unmaps the file, if mapped.
  void Reset();

 private:
  MappedFile(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_UTILS_MAPPED_FILE_H_
