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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_OPTIONS_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_OPTIONS_HELPER_H_

#include <string.h>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {
namespace internal {

inline void MakeCStringPayload(absl::string_view toml_str, void** payload,
                               void (**payload_deleter)(void*)) {
  *payload = new char[toml_str.size() + 1];
  memcpy(*payload, toml_str.data(), toml_str.size());
  static_cast<char*>(*payload)[toml_str.size()] = '\0';
  *payload_deleter = [](void* p) { delete[] static_cast<char*>(p); };
}

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_OPTIONS_HELPER_H_
