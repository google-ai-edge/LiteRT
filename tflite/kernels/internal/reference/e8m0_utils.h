/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_E8M0_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_E8M0_UTILS_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "tflite/kernels/internal/compatibility.h"

namespace tflite {
namespace reference_ops {

// E8M0 is an 8-bit unsigned floating point format with 8 exponent bits and
// 0 mantissa bits. In accordance with OCP Microscaling Formats (MX) and
// the Gemma5 compression quant_schema definition:
// - Bias = 127
// - Raw value 0xFF (255) represents NaN.
// - Raw values 0..254 represent exact powers of two: 2^(raw_value - 127).
// - There are no subnormals, sign bit, or zero representation.
//
// In TFLite models, scales are passed as standard FLOAT32 tensors and packed
// or unpacked as E8M0FNU. In IEEE-754 single-precision float (FLOAT32), bits
// 30..23 store the 8-bit exponent with bias 127. When decoding a packed FLOAT32
// scale as E8M0, the 23 mantissa bits are ignored.

// Unpacks a FLOAT32 value into a raw 8-bit E8M0 byte by extracting the 8
// exponent bits (bits 30..23) and ignoring mantissa bits (bits 22..0).
inline uint8_t UnpackFloat32ToE8M0(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint8_t>((bits >> 23) & 0xFF);
}

// Packs a raw 8-bit E8M0 byte into a FLOAT32 value with 0 mantissa bits.
// For raw values 1..254, this is mathematically identical to
// 2^(raw_e8m0 - 127).
inline float PackE8M0ToFloat32(uint8_t raw_e8m0) {
  uint32_t bits = static_cast<uint32_t>(raw_e8m0) << 23;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Decodes the raw 8-bit E8M0 value into its unbiased integer exponent
// (E - 127).
inline int DecodeE8M0Exponent(uint8_t raw_e8m0) {
  return static_cast<int>(raw_e8m0) - 127;
}

// Decodes the exponent directly from a packed FLOAT32 scale, ignoring mantissa.
inline int DecodePackedFloat32Exponent(float value) {
  return DecodeE8M0Exponent(UnpackFloat32ToE8M0(value));
}

// Converts a raw E8M0 byte into its exact float representation.
inline float DecodeE8M0Scale(uint8_t raw_e8m0) {
  if (raw_e8m0 == 0xFF) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f, DecodeE8M0Exponent(raw_e8m0));
}

// Decodes a packed FLOAT32 scale into its exact power-of-two float, ignoring
// mantissa.
inline float DecodePackedFloat32Scale(float value) {
  return DecodeE8M0Scale(UnpackFloat32ToE8M0(value));
}

// Scales a float value by an E8M0 power-of-two scale factor via ldexp.
inline float ScaleByE8M0(float value, uint8_t raw_e8m0) {
  if (raw_e8m0 == 0xFF) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(value, DecodeE8M0Exponent(raw_e8m0));
}

// Scales a float value by a packed FLOAT32 E8M0 scale factor, ignoring
// mantissa bits.
inline float ScaleByPackedFloat32(float value, float packed_scale) {
  return ScaleByE8M0(value, UnpackFloat32ToE8M0(packed_scale));
}

// Scales an integer value by an E8M0 scale factor using exact bit shifts.
// Positive shifts are shifted left; negative shifts are shifted right with
// round-half-up (nearest) rounding.
template <typename IntType>
inline IntType ShiftByE8M0(IntType value, uint8_t raw_e8m0) {
  if (raw_e8m0 == 0xFF) {
    return 0;
  }
  int shift = DecodeE8M0Exponent(raw_e8m0);
  if (shift == 0) {
    return value;
  }
  constexpr int kBitWidth = static_cast<int>(sizeof(IntType) * 8);
  using UnsignedType = std::make_unsigned_t<IntType>;
  if (shift > 0) {
    if (shift >= kBitWidth) {
      return 0;
    }
    return static_cast<IntType>(static_cast<UnsignedType>(value) << shift);
  }
  int right_shift = -shift;
  if (right_shift >= kBitWidth) {
    return 0;
  }
  UnsignedType round_offset = UnsignedType(1) << (right_shift - 1);
  return static_cast<IntType>(
      (static_cast<UnsignedType>(value) + round_offset) >> right_shift);
}

// Encodes a positive float value to the nearest E8M0 raw byte.
inline uint8_t FloatToE8M0(float value) {
  if (std::isnan(value)) return 0xFF;
  if (value <= 0.0f) return 0;
  if (std::isinf(value)) return 254;
  float log2_val = std::log2(value);
  int rounded_exp = static_cast<int>(std::round(log2_val));
  int biased_exp = rounded_exp + 127;
  if (biased_exp < 0) return 0;
  if (biased_exp > 254) return 254;
  return static_cast<uint8_t>(biased_exp);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_E8M0_UTILS_H_
