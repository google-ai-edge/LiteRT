// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/qualcomm/core/utils/miscs.h"

#ifdef __ANDROID__
#include <arm_neon.h>
#endif
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
namespace qnn {
#ifdef __ANDROID__
void neon_xor(std::uint16_t* input, size_t tensor_size) {
  const uint16x8_t mask = vdupq_n_u16(0x8000);
  size_t i = 0;
  for (; i + 8 < tensor_size; i += 8) {
    uint16x8_t uin = vld1q_u16(input + i);
    uint16x8_t out = veorq_u16(uin, mask);
    vst1q_u16(input + i, out);
  }
  for (; i < tensor_size; ++i) {
    input[i] = input[i] ^ 0x8000;
  }
}
#endif

void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data + kUint16ZeroPoint);
  }
}

void ConvertDataFromInt16toUInt16(absl::Span<std::int16_t> src) {
#ifdef __ANDROID__
  neon_xor(reinterpret_cast<std::uint16_t*>(src.data()), src.size());
#else
  for (auto& data : src) {
    data += kUint16ZeroPoint;
  }
#endif
}

void ConvertDataFromUInt16toInt16(absl::Span<std::uint16_t> src) {
#ifdef __ANDROID__
  neon_xor(reinterpret_cast<std::uint16_t*>(src.data()), src.size());
#else
  for (auto& data : src) {
    data -= kUint16ZeroPoint;
  }
#endif
}

void ConvertDataFromInt4ToInt8(const void* src, std::vector<std::int8_t>& dst,
                               size_t num_bytes) {
  dst.clear();
  dst.reserve(num_bytes * 2);
  const std::uint8_t* byte_data = static_cast<const std::uint8_t*>(src);
  for (size_t i = 0; i < num_bytes; i++) {
    std::uint8_t byte = byte_data[i];
    std::int8_t lower = byte & 0x0F;
    std::int8_t upper = (byte >> 4) & 0x0F;
    if (lower & 0x08) lower |= 0xF0;
    if (upper & 0x08) upper |= 0xF0;
    dst.emplace_back(lower);
    dst.emplace_back(upper);
  }
}
}  // namespace qnn
