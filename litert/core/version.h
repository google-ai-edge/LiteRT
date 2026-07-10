// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_VERSION_H_
#define ODML_LITERT_LITERT_CORE_VERSION_H_

#include <cstddef>

#include "litert/c/litert_common.h"

#define LITERT_IS_MEMBER_SUPPORTED(struct_ptr, member)                    \
  ((struct_ptr) != nullptr &&                                             \
   (struct_ptr)->size >                                                   \
       offsetof(typename std::remove_pointer<decltype(struct_ptr)>::type, \
                member))

namespace litert::internal {

// Return true if two API versions are the same.
inline bool IsSameVersion(const LiteRtApiVersion& v1,
                          const LiteRtApiVersion& v2) {
  return (v1.major == v2.major) && (v1.minor == v2.minor) &&
         (v1.patch == v2.patch);
}

// Return true if a given API version is the same as the current runtime.
inline bool IsSameVersionAsRuntime(const LiteRtApiVersion& v) {
  return IsSameVersion(v, {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                           LITERT_API_VERSION_PATCH});
}

// Return true if the vendor version is compatible with the runtime version.
// Compatibility rules:
// 1. Major version must match exactly.
// 2. If major is 0, minor version must match exactly (pre-1.0 breaking
// changes).
// 3. If major >= 1, runtime minor must be greater than or equal to vendor minor
//    (backward compatibility).
// 4. Patch versions are ignored for compatibility.
inline bool IsCompatibleVersion(const LiteRtApiVersion& vendor_version,
                                const LiteRtApiVersion& runtime_version) {
  if (vendor_version.major != runtime_version.major) {
    return false;
  }
  if (runtime_version.major == 0) {
    return vendor_version.minor == runtime_version.minor;
  }
  return runtime_version.minor >= vendor_version.minor;
}

// Return true if the given vendor API version is compatible with the current
// runtime.
inline bool IsCompatibleVersionAsRuntime(const LiteRtApiVersion& v) {
  return IsCompatibleVersion(
      v, {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
          LITERT_API_VERSION_PATCH});
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_VERSION_H_
