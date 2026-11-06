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

#include "litert/core/litert_options_encoder.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace internal {
namespace {

Expected<void> ValidateKey(absl::string_view key) {
  if (key.empty()) {
    return Unexpected(Status::kErrorInvalidArgument,
                      "Option key must not be empty.");
  }
  for (char c : key) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t' ||
        c == '#') {
      return Unexpected(
          Status::kErrorInvalidArgument,
          absl::StrCat("Option key contains an unsupported character: ", key));
    }
  }
  return {};
}

Expected<void> ValidateStringValue(absl::string_view key,
                                   absl::string_view value) {
  for (char c : value) {
    if (c == '"' || c == '\n' || c == '\r') {
      return Unexpected(
          Status::kErrorInvalidArgument,
          absl::StrCat("Value of option '", key,
                       "' contains an unsupported character (\", newline or "
                       "carriage return)."));
    }
  }
  return {};
}

template <typename SpanType>
Expected<void> EncodeStringArrayImpl(std::string& buffer, absl::string_view key,
                                     SpanType values) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  for (absl::string_view value : values) {
    LITERT_RETURN_IF_ERROR(ValidateStringValue(key, value));
  }
  absl::StrAppend(&buffer, key, " = [");
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      absl::StrAppend(&buffer, ", ");
    }
    absl::StrAppend(&buffer, "\"", values[i], "\"");
  }
  absl::StrAppend(&buffer, "]\n");
  return {};
}

}  // namespace

Expected<void> OptionsEncoder::EncodeBool(absl::string_view key, bool value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  absl::StrAppend(&buffer_, key, " = ", value ? "true" : "false", "\n");
  return {};
}

Expected<void> OptionsEncoder::EncodeInt(absl::string_view key, int64_t value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  absl::StrAppend(&buffer_, key, " = ", value, "\n");
  return {};
}

Expected<void> OptionsEncoder::EncodeString(absl::string_view key,
                                            absl::string_view value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  LITERT_RETURN_IF_ERROR(ValidateStringValue(key, value));
  absl::StrAppend(&buffer_, key, " = \"", value, "\"\n");
  return {};
}

Expected<void> OptionsEncoder::EncodeStringArray(
    absl::string_view key, absl::Span<absl::string_view> values) {
  return EncodeStringArrayImpl(buffer_, key, values);
}

Expected<void> OptionsEncoder::EncodeStringArray(
    absl::string_view key, absl::Span<const std::string> values) {
  return EncodeStringArrayImpl(buffer_, key, values);
}

std::string OptionsEncoder::Release() {
  std::string result = std::move(buffer_);
  buffer_.clear();
  return result;
}

}  // namespace internal
}  // namespace litert
