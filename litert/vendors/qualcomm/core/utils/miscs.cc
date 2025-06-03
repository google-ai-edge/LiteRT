// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/qualcomm/core/utils/miscs.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <system_error>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/log.h"
namespace qnn {
void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data + kUint16ZeroPoint);
  }
}
void ConvertDataFromUInt16toInt16(absl::Span<const std::uint16_t> src,
                                  std::vector<std::int16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data - kUint16ZeroPoint);
  }
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

bool CreateDirectoryRecursive(const std::filesystem::path& dir_name) {
  std::error_code err;
  err.clear();
  if (!std::filesystem::create_directories(dir_name, err)) {
    if (std::filesystem::exists(dir_name)) {
      err.clear();
      return true;
    }
    if (err) {
      QNN_LOG_ERROR("%s", err.message().c_str());
    }
    return false;
  }
  return true;
}

bool IsStrEndsWith(std::string_view str, std::string_view suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace qnn
