// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_BACKEND_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_BACKEND_UTILS_H_

#include <array>
#include <cstdint>
#include <type_traits>

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
void SetNullTermPtrArray(
    const std::array<T, N>& src,
    std::array<std::add_pointer_t<std::add_const_t<T>>, N + 1>& dst) {
  for (std::size_t i = 0; i < N; ++i) {
    dst[i] = &src[i];
  }
  dst[N] = nullptr;
}

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_BACKEND_UTILS_H_
