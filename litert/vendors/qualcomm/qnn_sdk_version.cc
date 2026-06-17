// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_sdk_version.h"

#include <charconv>
#include <string_view>
#include <system_error>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::qnn {

Expected<SdkVersion> ParseSdkVersion(const char* build_id) {
  // Generic parse-failure result.
  const auto parsing_error =
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to parse build ID");

  if (!build_id) return parsing_error;

  std::string_view version_str = build_id;

  // Require the 'v' prefix, then strip it.
  if (version_str.empty() || version_str.front() != 'v') {
    return parsing_error;
  }
  version_str.remove_prefix(1);

  SdkVersion version{};
  const char* current = version_str.data();
  const char* const end = version_str.data() + version_str.size();

  auto parse_component = [&current, &end](int& component) {
    auto [ptr, ec] = std::from_chars(current, end, component);
    if (ec != std::errc()) {
      return false;
    }
    current = ptr;
    return true;
  };

  // Expect "major.minor.patch".
  if (!parse_component(version.major)) return parsing_error;

  if (current == end || *current++ != '.') return parsing_error;
  if (!parse_component(version.minor)) return parsing_error;

  if (current == end || *current++ != '.') return parsing_error;
  if (!parse_component(version.patch)) return parsing_error;

  return version;
}

}  // namespace litert::qnn
