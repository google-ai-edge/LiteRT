// Copyright 2025 Google LLC.
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

#include "litert/tools/flags/vendors/qualcomm_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_qualcomm_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsLogLevel* options, std::string* error) {
  if (text == "off") {
    *options = kLiteRtQualcommLogOff;
    return true;
  }
  if (text == "error") {
    *options = kLiteRtQualcommLogLevelError;
    return true;
  }
  if (text == "warn") {
    *options = kLiteRtQualcommLogLevelWarn;
    return true;
  }
  if (text == "info") {
    *options = kLiteRtQualcommLogLevelInfo;
    return true;
  }
  if (text == "verbose") {
    *options = kLiteRtQualcommLogLevelVerbose;
    return true;
  }
  if (text == "debug") {
    *options = kLiteRtQualcommLogLevelDebug;
    return true;
  }
  *error = "Unknown log level";
  return false;
}

std::string AbslUnparseFlag(LiteRtQualcommOptionsLogLevel options) {
  switch (options) {
    case kLiteRtQualcommLogOff:
      return "off";
    case kLiteRtQualcommLogLevelError:
      return "error";
    case kLiteRtQualcommLogLevelWarn:
      return "warn";
    case kLiteRtQualcommLogLevelInfo:
      return "info";
    case kLiteRtQualcommLogLevelVerbose:
      return "verbose";
    case kLiteRtQualcommLogLevelDebug:
      return "debug";
  }
}

bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsPowerMode* options,
                   std::string* error) {
  if (text == "unknown") {
    *options = kLiteRtQualcommPowerModeUnknown;
    return true;
  }
  if (text == "performance") {
    *options = kLiteRtQualcommPowerModePerformance;
    return true;
  }
  if (text == "power_saver") {
    *options = kLiteRtQualcommPowerModePowerSaver;
    return true;
  }
  *error = "Unknown power mode";
  return false;
}

std::string AbslUnparseFlag(LiteRtQualcommOptionsPowerMode options) {
  switch (options) {
    case kLiteRtQualcommPowerModeUnknown:
      return "unknown";
    case kLiteRtQualcommPowerModePerformance:
      return "performance";
    case kLiteRtQualcommPowerModePowerSaver:
      return "power_saver";
  }
}

ABSL_FLAG(bool, enable_weight_sharing, true,
          "Whether to enable weight sharing, this is unsupported on mobile "
          "platforms.");

ABSL_FLAG(LiteRtQualcommOptionsLogLevel, qualcomm_log_level,
          kLiteRtQualcommLogLevelInfo, "Log level for Qualcomm.");

ABSL_FLAG(LiteRtQualcommOptionsPowerMode, qualcomm_power_mode,
          kLiteRtQualcommPowerModeUnknown, "Power preference for HTP device.");

// NOLINTEND(*alien-types*)
