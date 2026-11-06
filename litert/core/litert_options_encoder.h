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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_OPTIONS_ENCODER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_OPTIONS_ENCODER_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

namespace litert {
namespace internal {

// Encodes flat key-value LiteRT option payloads (`key = value\n`).
//
// Note: Escape sequences are not supported by LiteRT's option decoder, so keys
// containing whitespace/`=`/`#`/newlines and string values containing `"` or
// newlines are rejected with `Status::kErrorInvalidArgument`.
//
// Each `Encode*` method validates its input before writing, so on failure the
// buffer is left unchanged.
class OptionsEncoder {
 public:
  // Appends `key = true` or `key = false`.
  Expected<void> EncodeBool(absl::string_view key, bool value);

  // Appends `key = <value>` with `value` written in decimal.
  Expected<void> EncodeInt(absl::string_view key, int64_t value);

  // Appends `key = "<value>"`. Backslashes are written as-is (e.g. Windows
  // paths).
  Expected<void> EncodeString(absl::string_view key, absl::string_view value);

  // Appends `key = ["<v0>", "<v1>", ...]`. An empty `values` writes `key = []`.
  // Every element is validated like an `EncodeString` value.
  Expected<void> EncodeStringArray(absl::string_view key,
                                   absl::Span<absl::string_view> values);
  Expected<void> EncodeStringArray(absl::string_view key,
                                   absl::Span<const std::string> values);

  // Returns true if nothing has been encoded yet.
  bool Empty() const { return buffer_.empty(); }

  // Returns the encoded payload. The view is invalidated by the next
  // `Encode*`, `Release` or `Clear` call.
  absl::string_view Str() const { return buffer_; }

  // Returns the encoded payload and leaves the encoder empty.
  std::string Release();

  // Discards everything encoded so far.
  void Clear() { buffer_.clear(); }

 private:
  std::string buffer_;
};

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_OPTIONS_ENCODER_H_
