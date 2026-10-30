// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_I2_U2_CONVERT_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_I2_U2_CONVERT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl

namespace litert::openvino {

// Signed i2 weights are rewritten to unsigned u2 before reaching OpenVINO's
// NPU/CPU paths (OpenVINO never sees i2). This is the single definition of that
// rewrite, shared by the frontend graph rewrite (GraphIteratorDelegate) and the
// NPU weight-sharing pool (WeightBank). Both MUST produce identical bytes: the
// shared pool is what NPUW's weightless loader memcpy's into the u2 Constant at
// runtime, so a pool holding anything other than these bytes corrupts weights.
//
// XOR flips each 2-bit element's MSB, which is adding 2^(bits-1) mod 2^bits
// ([-2..1] -> [0..3]). The byte layout (4 elements per byte) is unchanged; the
// zero points must be shifted by kI2ToU2ZeroPointOffset to keep dequantized
// values the same (that shift lives in the quantization info, not the bytes).
inline constexpr uint8_t kI2ToU2XorMask = 0xAA;
inline constexpr int64_t kI2ToU2ZeroPointOffset = 2;

inline std::vector<uint8_t> ConvertI2BytesToU2(
    absl::Span<const uint8_t> i2_bytes) {
  std::vector<uint8_t> u2_bytes(i2_bytes.begin(), i2_bytes.end());
  for (auto& byte : u2_bytes) {
    byte ^= kI2ToU2XorMask;
  }
  return u2_bytes;
}

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_I2_U2_CONVERT_H_
