// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_FILESYSTEM_H_
#define ODML_LITERT_LITERT_CORE_FILESYSTEM_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

// Generic file operations. Try to encapsulate the std filesystem header as much
// as possible because its technically unapproved.

namespace litert::internal {

// Append all given subpaths together (e.g. os.path.join).
std::string Join(const std::vector<absl::string_view>& paths);

// Make a new empty file at the given path.
void Touch(absl::string_view path);

// Does this file exist.
bool Exists(absl::string_view path);

// Is this a directory.
bool IsDir(absl::string_view path);

// Get size of file.
Expected<size_t> Size(absl::string_view path);

// Load the bytes of the file at given path.
Expected<OwningBufferRef<uint8_t>> LoadBinaryFile(absl::string_view path);

// List all files in the directory at the given path.
Expected<std::vector<std::string>> ListDir(absl::string_view path);

// Get filename from path string.
Expected<std::string> Filename(absl::string_view path);

// Make a new directory at the given path.
Expected<void> MkDir(absl::string_view path);

// Get the parent directory of the given path.
Expected<std::string> Parent(absl::string_view path);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_FILESYSTEM_H_
