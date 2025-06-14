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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_qualcomm_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(LiteRtQualcommOptionsLogLevel, qualcomm_log_level,
          kLiteRtQualcommLogLevelInfo, "Log level for Qualcomm.");

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

ABSL_FLAG(bool, qualcomm_enable_weight_sharing, false,
          "Whether to enable weight sharing, this is unsupported on mobile "
          "platforms.");

ABSL_FLAG(bool, qualcomm_use_htp_preference, false,
          "Whether to transform a litert op into the HTP prefered pattern.");

ABSL_FLAG(bool, qualcomm_use_qint16_as_quint16, false,
          "Whether to automatically convert a quantized int16 model into a "
          "quantized uin16 model.");

ABSL_FLAG(LiteRtQualcommOptionsHtpPerformanceMode,
          qualcomm_htp_performance_mode,
          kLiteRtQualcommHtpPerformanceModeBurst, "HTP performance mode.");

ABSL_FLAG(std::vector<std::string>, qualcomm_dump_tensor_ids, {},
          "Debug Feature. Ids to dump as outputs. Comma-separated list of "
          "string. Use -1 to dump all op outputs.");

bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsHtpPerformanceMode* options,
                   std::string* error) {
  if (text == "default") {
    *options = kLiteRtQualcommHtpPerformanceModeDefault;
    return true;
  }
  if (text == "sustained_high_performance") {
    *options = kLiteRtQualcommHtpPerformanceModeSustainedHighPerformance;
    return true;
  }
  if (text == "burst") {
    *options = kLiteRtQualcommHtpPerformanceModeBurst;
    return true;
  }
  if (text == "high_performance") {
    *options = kLiteRtQualcommHtpPerformanceModeHighPerformance;
    return true;
  }
  if (text == "power_saver") {
    *options = kLiteRtQualcommHtpPerformanceModePowerSaver;
    return true;
  }
  if (text == "low_power_saver") {
    *options = kLiteRtQualcommHtpPerformanceModeLowPowerSaver;
    return true;
  }
  if (text == "high_power_saver") {
    *options = kLiteRtQualcommHtpPerformanceModeHighPowerSaver;
    return true;
  }
  if (text == "low_balanced") {
    *options = kLiteRtQualcommHtpPerformanceModeLowBalanced;
    return true;
  }
  if (text == "balanced") {
    *options = kLiteRtQualcommHtpPerformanceModeBalanced;
    return true;
  }
  if (text == "extreme_power_saver") {
    *options = kLiteRtQualcommHtpPerformanceModeExtremePowerSaver;
    return true;
  }
  *error = "Unknown htp performance mode";
  return false;
}

std::string AbslUnparseFlag(LiteRtQualcommOptionsHtpPerformanceMode options) {
  switch (options) {
    case kLiteRtQualcommHtpPerformanceModeDefault:
      return "default";
    case kLiteRtQualcommHtpPerformanceModeSustainedHighPerformance:
      return "sustained_high_performance";
    case kLiteRtQualcommHtpPerformanceModeBurst:
      return "burst";
    case kLiteRtQualcommHtpPerformanceModeHighPerformance:
      return "high_performance";
    case kLiteRtQualcommHtpPerformanceModePowerSaver:
      return "power_saver";
    case kLiteRtQualcommHtpPerformanceModeLowPowerSaver:
      return "low_power_saver";
    case kLiteRtQualcommHtpPerformanceModeHighPowerSaver:
      return "high_power_saver";
    case kLiteRtQualcommHtpPerformanceModeLowBalanced:
      return "low_balanced";
    case kLiteRtQualcommHtpPerformanceModeBalanced:
      return "balanced";
    case kLiteRtQualcommHtpPerformanceModeExtremePowerSaver:
      return "extreme_power_saver";
  }
}

ABSL_FLAG(LiteRtQualcommOptionsProfiling, qualcomm_profiling,
          kLiteRtQualcommProfilingOff, "QNN profiling mode");

bool AbslParseFlag(absl::string_view text,
                   LiteRtQualcommOptionsProfiling* options,
                   std::string* error) {
  if (text == "off") {
    *options = kLiteRtQualcommProfilingOff;
    return true;
  }
  if (text == "basic") {
    *options = kLiteRtQualcommProfilingBasic;
    return true;
  }
  if (text == "detailed") {
    *options = kLiteRtQualcommProfilingDetailed;
    return true;
  }
  if (text == "linting") {
    *options = kLiteRtQualcommProfilingLinting;
    return true;
  }
  *error = "Unknown htp performance mode";
  return false;
}

std::string AbslUnparseFlag(LiteRtQualcommOptionsProfiling options) {
  switch (options) {
    case kLiteRtQualcommProfilingOff:
      return "off";
    case kLiteRtQualcommProfilingBasic:
      return "basic";
    case kLiteRtQualcommProfilingDetailed:
      return "detailed";
    case kLiteRtQualcommProfilingLinting:
      return "linting";
  }
}

ABSL_FLAG(std::string, qualcomm_qnn_json_path, "",
          "Qnn JSON path. If provided, you can obtain Qnn IR in Qnn JSON "
          "format.");

// NOLINTEND(*alien-types*)

namespace litert::qualcomm {

Expected<QualcommOptions> QualcommOptionsFromFlags() {
  LITERT_ASSIGN_OR_RETURN(auto opts, QualcommOptions::Create());

  const auto weight_share = absl::GetFlag(FLAGS_qualcomm_enable_weight_sharing);
  opts.SetEnableWeightSharing(weight_share);

  const auto log_level = absl::GetFlag(FLAGS_qualcomm_log_level);
  opts.SetLogLevel(log_level);

  const auto use_htp_preference =
      absl::GetFlag(FLAGS_qualcomm_use_htp_preference);
  opts.SetUseHtpPreference(use_htp_preference);

  const auto use_qint16_as_quint16 =
      absl::GetFlag(FLAGS_qualcomm_use_qint16_as_quint16);
  opts.SetUseQint16AsQuint16(use_qint16_as_quint16);

  const auto htp_performance_mode =
      absl::GetFlag(FLAGS_qualcomm_htp_performance_mode);
  opts.SetHtpPerformanceMode(htp_performance_mode);

  const auto profiling = absl::GetFlag(FLAGS_qualcomm_profiling);
  opts.SetProfiling(profiling);

  const auto dump_tensor_ids = absl::GetFlag(FLAGS_qualcomm_dump_tensor_ids);
  std::vector<std::int32_t> int32_ids;
  std::for_each(dump_tensor_ids.begin(), dump_tensor_ids.end(),
                [&int32_ids](const std::string& id) {
                  int32_ids.push_back(std::stoi(id));
                });
  opts.SetDumpTensorIds(int32_ids);

  const auto qnn_json_path = absl::GetFlag(FLAGS_qualcomm_qnn_json_path);
  opts.SetQnnJsonPath(qnn_json_path.c_str());

  return opts;
}

}  // namespace litert::qualcomm
