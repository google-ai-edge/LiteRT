// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SDK_VERSION_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SDK_VERSION_H_

#include <tuple>

#include "litert/cc/litert_expected.h"

namespace litert::qnn {

struct SdkVersion {
  int major, minor, patch;

  friend constexpr bool operator==(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return std::tie(lhs.major, lhs.minor, lhs.patch) ==
           std::tie(rhs.major, rhs.minor, rhs.patch);
  }
  friend constexpr bool operator!=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(const SdkVersion& lhs,
                                  const SdkVersion& rhs) noexcept {
    return std::tie(lhs.major, lhs.minor, lhs.patch) <
           std::tie(rhs.major, rhs.minor, rhs.patch);
  }
  friend constexpr bool operator>(const SdkVersion& lhs,
                                  const SdkVersion& rhs) noexcept {
    return rhs < lhs;
  }
  friend constexpr bool operator<=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(rhs < lhs);
  }
  friend constexpr bool operator>=(const SdkVersion& lhs,
                                   const SdkVersion& rhs) noexcept {
    return !(lhs < rhs);
  }
};

// Parses a QNN SDK build ID string (e.g. "v2.37.0") into an SdkVersion.
Expected<SdkVersion> ParseSdkVersion(const char* build_id);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SDK_VERSION_H_
