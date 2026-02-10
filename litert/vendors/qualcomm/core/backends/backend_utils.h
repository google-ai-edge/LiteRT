// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {

struct PowerConfig {
  static constexpr uint32_t kSleepMinLatency = 40;
  static constexpr uint32_t kSleepLowLatency = 100;
  static constexpr uint32_t kSleepMediumLatency = 1000;
  static constexpr uint32_t kSleepHighLatency = 2000;
  static constexpr uint32_t kSleepMaxLatency = 65535;
  static constexpr uint32_t kDcvsDisable = 0;
  static constexpr uint32_t kDcvsEnable = 1;
  // default rpc control latency - 0 us
  static constexpr uint32_t kRpcControlLatency = 0;
  // default rpc polling time for high power modes - 9999 us
  static constexpr uint32_t kRpcPollingTimeHighPower = 9999;
};

template <typename T, std::size_t N>
void SetNullTermPtrArray(absl::Span<const T> src,
                         std::array<const T*, N>& dst) {
  size_t min_size = std::min(src.size(), dst.size() - 1);
  for (std::size_t i = 0; i < min_size; ++i) {
    dst[i] = &src[i];
  }
  dst[min_size] = nullptr;
}

inline std::optional<SocInfo> FindSocInfo(const SnapdragonModel& soc_model) {
  for (auto i = 0; i < kNumSocInfos; ++i) {
    if (soc_model == kSocInfos[i].soc_model) {
      return kSocInfos[i];
    }
  }
  return std::nullopt;
}

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_BACKEND_UTILS_H_
