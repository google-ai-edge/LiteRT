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


#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_

#include "litert/c/litert_common.h"

namespace litert {

enum class HwAccelerators : int {
  kNone = kLiteRtHwAcceleratorNone,
  kCpu = kLiteRtHwAcceleratorCpu,
  kGpu = kLiteRtHwAcceleratorGpu,
  kNpu = kLiteRtHwAcceleratorNpu,
#if defined(__EMSCRIPTEN__)
  kWebNn = kLiteRtHwAcceleratorWebNn,
#endif  // __EMSCRIPTEN__
};

// Type-safe bit field for HwAccelerators.
struct HwAcceleratorSet {
  int value;

  explicit HwAcceleratorSet(int val) : value(val) {}
};

inline HwAcceleratorSet operator|(HwAccelerators lhs, HwAccelerators rhs) {
  return HwAcceleratorSet(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline HwAcceleratorSet operator|(HwAcceleratorSet lhs, HwAccelerators rhs) {
  return HwAcceleratorSet(lhs.value | static_cast<int>(rhs));
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_COMMON_H_
