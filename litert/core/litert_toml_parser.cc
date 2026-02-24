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
#include <string>
#include <vector>

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

Expected<std::string> ParseTomlString(absl::string_view value) {
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    // Basic unquoting. Does not handle complex escape sequences yet.
    return std::string(value.substr(1, value.size() - 2));
  }
  if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
    return std::string(value.substr(1, value.size() - 2));
  }
  return Unexpected(kLiteRtStatusErrorInvalidArgument, "Invalid string value");
}

std::vector<absl::string_view> SplitTomlArrayElements(absl::string_view inner) {
  std::vector<absl::string_view> elements;
  size_t start = 0;
  bool in_quotes = false;
  char quote_char = '\0';

  for (size_t i = 0; i <= inner.size(); ++i) {
    bool is_end = (i == inner.size());
    if (!is_end) {
      char c = inner[i];
      if (c == '"' || c == '\'') {
        if (!in_quotes) {
          in_quotes = true;
          quote_char = c;
        } else if (c == quote_char) {
          in_quotes = false;
        }
      }
    }

    if (is_end || (inner[i] == ',' && !in_quotes)) {
      absl::string_view elem = inner.substr(start, i - start);
      elem = TrimWhitespace(elem);
      if (!elem.empty()) {
        elements.push_back(elem);
      }
      start = i + 1;
    }
  }
  return elements;
}

Expected<std::vector<std::string>> ParseTomlStringArray(
    absl::string_view value) {
  if (value.empty() || value.front() != '[' || value.back() != ']') {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid array format");
  }

  absl::string_view inner = value.substr(1, value.size() - 2);
  std::vector<std::string> result;
  // Empty array
  if (TrimWhitespace(inner).empty()) {
    return result;
  }

  for (absl::string_view elem : SplitTomlArrayElements(inner)) {
    auto parsed_str = ParseTomlString(elem);
    if (!parsed_str.HasValue()) {
      return parsed_str.Error();
    }
    result.push_back(*parsed_str);
  }

  return result;
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

    // Remove quotes from string values.
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
      value = value.substr(1, value.size() - 2);
    }

    LITERT_RETURN_IF_ERROR(callback(key, value));
  }
  return kLiteRtStatusOk;
}

}  // namespace internal
}  // namespace litert
