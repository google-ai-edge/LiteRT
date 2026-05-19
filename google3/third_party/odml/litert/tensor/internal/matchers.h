/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MATCHERS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MATCHERS_H_

#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"

namespace testing::litert {

[[maybe_unused]] inline uintptr_t GetPtrInt(const void* d) {
  return reinterpret_cast<uintptr_t>(d);
}

[[maybe_unused]] inline uint64_t GetPtrInt(uint64_t d) { return d; }

[[maybe_unused]] inline int64_t GetPtrInt(int64_t d) { return d; }

template <class T, class F>
uintptr_t GetPtrInt(const std::unique_ptr<T, F>& d) {
  return GetPtrInt(d.get());
}

// Matches a pointer against the requested alignment.
MATCHER_P(AlignmentIs, alignment,
          "is aligned to " + std::to_string(alignment) + " bytes.") {
  const auto misalignment = GetPtrInt(arg) % alignment;
  *result_listener << "which is misaligned by " << misalignment << " bytes.";
  return !misalignment;
}

}  // namespace testing::litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_MATCHERS_H_
