// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/qualcomm/core/utils/miscs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
namespace qnn {

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
