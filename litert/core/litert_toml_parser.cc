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

#include "litert/core/litert_toml_parser.h"

#include <cstddef>
#include <cstdint>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace internal {

namespace {

// Helper to trim whitespace from the beginning and end of a string view.
absl::string_view TrimWhitespace(absl::string_view str) {
  const char* whitespace = " \t\n\r";
  size_t start = str.find_first_not_of(whitespace);
  if (start == absl::string_view::npos) {
    return "";
  }
  size_t end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}

bool IsCommentOrEmpty(absl::string_view line) {
  absl::string_view trimmed = TrimWhitespace(line);
  return trimmed.empty() || trimmed[0] == '#';
}

}  // namespace

Expected<bool> ParseTomlBool(absl::string_view value) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  return Unexpected(kLiteRtStatusErrorInvalidArgument, "Invalid boolean value");
}

Expected<int64_t> ParseTomlInt(absl::string_view value) {
  int64_t result;
  if (absl::SimpleAtoi(value, &result)) {
    return result;
  }
  return Unexpected(kLiteRtStatusErrorInvalidArgument, "Invalid integer value");
}

LiteRtStatus ParseToml(absl::string_view data, TomlCallback callback) {
  size_t pos = 0;
  while (pos < data.size()) {
    size_t line_end = data.find('\n', pos);
    absl::string_view line = (line_end == absl::string_view::npos)
                                 ? data.substr(pos)
                                 : data.substr(pos, line_end - pos);
    pos = (line_end == absl::string_view::npos) ? data.size() : line_end + 1;

    absl::string_view trimmed_line = TrimWhitespace(line);
    if (IsCommentOrEmpty(trimmed_line)) {
      continue;
    }

    size_t eq_pos = trimmed_line.find('=');
    if (eq_pos == absl::string_view::npos) {
      continue;  // Ignore lines without '='
    }

    absl::string_view key = TrimWhitespace(trimmed_line.substr(0, eq_pos));
    absl::string_view value = TrimWhitespace(trimmed_line.substr(eq_pos + 1));

    LITERT_RETURN_IF_ERROR(callback(key, value));
  }
  return kLiteRtStatusOk;
}

}  // namespace internal
}  // namespace litert
