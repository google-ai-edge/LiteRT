// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
#ifdef __ANDROID__
#include <arm_neon.h>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
namespace qnn {
constexpr uint32_t kUint16ZeroPoint = -std::numeric_limits<std::int16_t>::min();
constexpr uint32_t kQuantBitWidth4 = 4;

template <typename...>
inline constexpr bool always_false = false;
template <typename T>
T Quantize(const float val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Quantize function.");
  return std::round(val / scale) + zero_point;
}

template <typename T>
float Dequantize(const T val, const float scale, const int32_t zero_point) {
  static_assert(std::is_integral<T>::value,
                "Integral required in Dequantize function.");
  return scale * (val - zero_point);
}

template <typename T>
using EnableIfInt16OrUint16Ptr = std::enable_if_t<
    std::is_same_v<T, int16_t*> || std::is_same_v<T, uint16_t*>, bool>;
// Converts data between quantized UINT16 and INT16 formats.
template <typename T, EnableIfInt16OrUint16Ptr<T> = true>
void ToggleMostSignificantBit(T src, size_t size) {
  std::uint16_t* data = reinterpret_cast<std::uint16_t*>(src);
#ifdef __ANDROID__
  const uint16x8_t mask = vdupq_n_u16(0x8000);
  size_t i = 0;
  for (; i + 8 < size; i += 8) {
    uint16x8_t uin = vld1q_u16(data + i);
    uint16x8_t out = veorq_u16(uin, mask);
    vst1q_u16(data + i, out);
  }
  for (; i < size; ++i) {
    data[i] ^= 0x8000;
  }
#else
  for (size_t i = 0; i < size; ++i) {
    data[i] ^= 0x8000;
  }
#endif
}

void ConvertDataFromInt4ToInt8(const void* src, std::vector<std::int8_t>& dst,
                               size_t num_bytes);
}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_MISCS_H_
