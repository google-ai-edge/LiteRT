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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_WINDOWS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_WINDOWS_UTIL_H_

/* We always keep this header as Linux linter don't see anything being used and
 * want to remove it...*/
// IWYU pragma: always_keep

#if defined(_WIN32)

#include <string>

namespace ml_drift {

// Returns a string holding the error message corresponding to the code returned
// by `GetLastError()`.
std::string GetLastErrorString();

};  // namespace ml_drift

#endif  // defined(_WIN32)

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_SERIALIZATION_WEIGHT_CACHE_WINDOWS_UTIL_H_
