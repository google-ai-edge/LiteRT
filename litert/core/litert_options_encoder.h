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
#include "litert/c/litert_common.h"

namespace litert {
namespace internal {

// Encodes flat key-value LiteRT option payloads (`key = value\n`).
//
// Note: Escape sequences are not supported by LiteRT's option decoder, so keys
// containing whitespace/`=`/`#`/newlines and string values containing `"` or
// newlines are rejected with `kLiteRtStatusErrorInvalidArgument`.
class OptionsEncoder {
 public:
  LiteRtStatus EncodeBool(absl::string_view key, bool value);
  LiteRtStatus EncodeInt(absl::string_view key, int64_t value);
  LiteRtStatus EncodeString(absl::string_view key, absl::string_view value);
  LiteRtStatus EncodeStringArray(absl::string_view key,
                                 absl::Span<const absl::string_view> values);
  LiteRtStatus EncodeStringArray(absl::string_view key,
                                 absl::Span<const std::string> values);

  bool Empty() const { return buffer_.empty(); }
  absl::string_view Str() const { return buffer_; }
  std::string Release();
  void Clear() { buffer_.clear(); }

 private:
  std::string buffer_;
};

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_LITERT_OPTIONS_ENCODER_H_
