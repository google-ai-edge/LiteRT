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

#include "tensor/utils/file_utils.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <system_error>  // NOLINT

#include "absl/random/random.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::tensor {

absl::StatusOr<std::string> CreateTempFile(absl::string_view prefix) {
  std::error_code ec;
  auto temp_dir = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Failed to get temp directory: %s", ec.message()));
  }

  absl::InsecureBitGen gen;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto temp_path =
        temp_dir /
        absl::StrCat(prefix, "_", absl::Hex(absl::Uniform<uint64_t>(gen)));
    if (!std::filesystem::exists(temp_path)) {
      std::ofstream(temp_path).close();
      return temp_path.generic_string();
    }
  }
  return absl::InternalError("Failed to create unique temp file");
}

absl::Status RemoveFile(absl::string_view path) {
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(std::string(path)), ec);
  if (ec) {
    return absl::InternalError(absl::StrFormat(
        "Failed to remove file: %s, error: %s", path, ec.message()));
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor
