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
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"

namespace litert {
namespace internal {
namespace {

LiteRtStatus ValidateKey(absl::string_view key) {
  if (key.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (char c : key) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t' ||
        c == '#') {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ValidateStringValue(absl::string_view value) {
  for (char c : value) {
    if (c == '"' || c == '\n' || c == '\r') {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

template <typename SpanType>
LiteRtStatus EncodeStringArrayImpl(std::string& buffer, absl::string_view key,
                                   SpanType values) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  for (const auto& value : values) {
    LITERT_RETURN_IF_ERROR(ValidateStringValue(value));
  }
  absl::StrAppend(&buffer, key, " = [");
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      absl::StrAppend(&buffer, ", ");
    }
    absl::StrAppend(&buffer, "\"", values[i], "\"");
  }
  absl::StrAppend(&buffer, "]\n");
  return kLiteRtStatusOk;
}

}  // namespace

LiteRtStatus OptionsEncoder::EncodeBool(absl::string_view key, bool value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  absl::StrAppend(&buffer_, key, " = ", value ? "true" : "false", "\n");
  return kLiteRtStatusOk;
}

LiteRtStatus OptionsEncoder::EncodeInt(absl::string_view key, int64_t value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  absl::StrAppend(&buffer_, key, " = ", value, "\n");
  return kLiteRtStatusOk;
}

LiteRtStatus OptionsEncoder::EncodeString(absl::string_view key,
                                          absl::string_view value) {
  LITERT_RETURN_IF_ERROR(ValidateKey(key));
  LITERT_RETURN_IF_ERROR(ValidateStringValue(value));
  absl::StrAppend(&buffer_, key, " = \"", value, "\"\n");
  return kLiteRtStatusOk;
}

LiteRtStatus OptionsEncoder::EncodeStringArray(
    absl::string_view key, absl::Span<const absl::string_view> values) {
  return EncodeStringArrayImpl(buffer_, key, values);
}

LiteRtStatus OptionsEncoder::EncodeStringArray(
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
