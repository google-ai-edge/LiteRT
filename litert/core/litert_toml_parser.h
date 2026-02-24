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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_TOML_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_TOML_PARSER_H_

#include <cstdint>
#include <functional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace internal {

// Helper to parse a boolean value.
Expected<bool> ParseTomlBool(absl::string_view value);

// Helper to parse an integer value.
Expected<int64_t> ParseTomlInt(absl::string_view value);

// Helper to parse a string value.
// Note: Basic unquoting is supported for either "" or '' quotes.
// Escape sequences are currently not supported.
Expected<std::string> ParseTomlString(absl::string_view value);

// Helper to parse an array of string values.
// Supports comma-separated strings within brackets (e.g., `["a", "b"]`).
// Commas inside quoted strings (e.g., `["a,b"]`) are correctly preserved,
// matching strictly on either `"` or `'` quotation characters.
Expected<std::vector<std::string>> ParseTomlStringArray(
    absl::string_view value);

// Callback function type for handling key-value pairs.
using TomlCallback =
    std::function<LiteRtStatus(absl::string_view key, absl::string_view value)>;

// Parses a simple key-value TOML string.
//
// The callback is invoked for each valid key-value pair found.
// The syntax supported is very basic:
// - Lines starting with '#' are comments.
// - Empty lines are ignored.
// - Keys and values are separated by '='.
// - Whitespace around keys and values is trimmed.
//
// Note: This is not full featured TOML parser, and only supports a subset of
// TOML syntax for LiteRT Options.
// TODO: b/484095144 - Finalize the name before the next release.
LiteRtStatus ParseToml(absl::string_view data, TomlCallback callback);

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_TOML_PARSER_H_
