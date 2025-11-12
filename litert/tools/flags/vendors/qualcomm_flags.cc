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
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_qualcomm_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(litert::qualcomm::QualcommOptions::LogLevel, qualcomm_log_level,
          litert::qualcomm::QualcommOptions::LogLevel::kInfo,
          "Log level for Qualcomm.");

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text, QualcommOptions::LogLevel* options,
                   std::string* error) {
  if (text == "off") {
    *options = QualcommOptions::LogLevel::kOff;
    return true;
  }
  if (text == "error") {
    *options = QualcommOptions::LogLevel::kError;
    return true;
  }
  if (text == "warn") {
    *options = QualcommOptions::LogLevel::kWarn;
    return true;
  }
  if (text == "info") {
    *options = QualcommOptions::LogLevel::kInfo;
    return true;
  }
  if (text == "verbose") {
    *options = QualcommOptions::LogLevel::kVerbose;
    return true;
  }
  if (text == "debug") {
    *options = QualcommOptions::LogLevel::kDebug;
    return true;
  }
  *error = "Unknown log level";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::LogLevel options) {
  switch (options) {
    case QualcommOptions::LogLevel::kOff:
      return "off";
    case QualcommOptions::LogLevel::kError:
      return "error";
    case QualcommOptions::LogLevel::kWarn:
      return "warn";
    case QualcommOptions::LogLevel::kInfo:
      return "info";
    case QualcommOptions::LogLevel::kVerbose:
      return "verbose";
    case QualcommOptions::LogLevel::kDebug:
      return "debug";
  }
}
}  // namespace litert::qualcomm

ABSL_FLAG(bool, qualcomm_enable_weight_sharing, false,
          "Whether to enable weight sharing, this is unsupported on mobile "
          "platforms.");

ABSL_FLAG(bool, qualcomm_use_htp_preference, false,
          "Whether to transform a litert op into the HTP prefered pattern.");

ABSL_FLAG(bool, qualcomm_use_qint16_as_quint16, false,
          "Whether to automatically convert a quantized int16 model into a "
          "quantized uin16 model.");

// Default should be
// litert::qualcomm::QualcommOptions::HtpPerformanceMode::kDefault since we need
// default performance model during compilation.
ABSL_FLAG(litert::qualcomm::QualcommOptions::HtpPerformanceMode,
          qualcomm_htp_performance_mode,
          litert::qualcomm::QualcommOptions::HtpPerformanceMode::kDefault,
          "HTP performance mode.");

ABSL_FLAG(std::vector<std::string>, qualcomm_dump_tensor_ids, {},
          "Debug Feature. Ids to dump as outputs. Comma-separated list of "
          "string. Use -1 to dump all op outputs.");

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::HtpPerformanceMode* options,
                   std::string* error) {
  if (text == "default") {
    *options = QualcommOptions::HtpPerformanceMode::kDefault;
    return true;
  }
  if (text == "sustained_high_performance") {
    *options = QualcommOptions::HtpPerformanceMode::kSustainedHighPerformance;
    return true;
  }
  if (text == "burst") {
    *options = QualcommOptions::HtpPerformanceMode::kBurst;
    return true;
  }
  if (text == "high_performance") {
    *options = QualcommOptions::HtpPerformanceMode::kHighPerformance;
    return true;
  }
  if (text == "power_saver") {
    *options = QualcommOptions::HtpPerformanceMode::kPowerSaver;
    return true;
  }
  if (text == "low_power_saver") {
    *options = QualcommOptions::HtpPerformanceMode::kLowPowerSaver;
    return true;
  }
  if (text == "high_power_saver") {
    *options = QualcommOptions::HtpPerformanceMode::kHighPowerSaver;
    return true;
  }
  if (text == "low_balanced") {
    *options = QualcommOptions::HtpPerformanceMode::kLowBalanced;
    return true;
  }
  if (text == "balanced") {
    *options = QualcommOptions::HtpPerformanceMode::kBalanced;
    return true;
  }
  if (text == "extreme_power_saver") {
    *options = QualcommOptions::HtpPerformanceMode::kExtremePowerSaver;
    return true;
  }
  *error = "Unknown htp performance mode";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::HtpPerformanceMode options) {
  switch (options) {
    case QualcommOptions::HtpPerformanceMode::kDefault:
      return "default";
    case QualcommOptions::HtpPerformanceMode::kSustainedHighPerformance:
      return "sustained_high_performance";
    case QualcommOptions::HtpPerformanceMode::kBurst:
      return "burst";
    case QualcommOptions::HtpPerformanceMode::kHighPerformance:
      return "high_performance";
    case QualcommOptions::HtpPerformanceMode::kPowerSaver:
      return "power_saver";
    case QualcommOptions::HtpPerformanceMode::kLowPowerSaver:
      return "low_power_saver";
    case QualcommOptions::HtpPerformanceMode::kHighPowerSaver:
      return "high_power_saver";
    case QualcommOptions::HtpPerformanceMode::kLowBalanced:
      return "low_balanced";
    case QualcommOptions::HtpPerformanceMode::kBalanced:
      return "balanced";
    case QualcommOptions::HtpPerformanceMode::kExtremePowerSaver:
      return "extreme_power_saver";
  }
}

}  // namespace litert::qualcomm

ABSL_FLAG(litert::qualcomm::QualcommOptions::Profiling, qualcomm_profiling,
          litert::qualcomm::QualcommOptions::Profiling::kOff,
          "QNN profiling mode");

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text, QualcommOptions::Profiling* options,
                   std::string* error) {
  if (text == "off") {
    *options = QualcommOptions::Profiling::kOff;
    return true;
  }
  if (text == "basic") {
    *options = QualcommOptions::Profiling::kBasic;
    return true;
  }
  if (text == "detailed") {
    *options = QualcommOptions::Profiling::kDetailed;
    return true;
  }
  if (text == "linting") {
    *options = QualcommOptions::Profiling::kLinting;
    return true;
  }
  if (text == "optrace") {
    *options = QualcommOptions::Profiling::kOptrace;
    return true;
  }
  *error = "Unknown htp performance mode";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::Profiling options) {
  switch (options) {
    case QualcommOptions::Profiling::kOff:
      return "off";
    case QualcommOptions::Profiling::kBasic:
      return "basic";
    case QualcommOptions::Profiling::kDetailed:
      return "detailed";
    case QualcommOptions::Profiling::kLinting:
      return "linting";
    case QualcommOptions::Profiling::kOptrace:
      return "optrace";
  }
}

}  // namespace litert::qualcomm

ABSL_FLAG(
    std::string, qualcomm_ir_json_dir, "",
    "Qnn IR JSON directory. If provided, you can obtain Qnn IR in Qnn JSON "
    "format.");

ABSL_FLAG(
    std::string, qualcomm_dlc_dir, "",
    "DLC directory. If provided, you can obtain Qnn graphs in DLC format.");

ABSL_FLAG(uint32_t, qualcomm_vtcm_size, 0,
          "The vtcm size of the target device. If this option is set to 0, the "
          "max size of vtcm size will be used.");

ABSL_FLAG(uint32_t, qualcomm_num_hvx_thread, 0,
          "The number of hvx threads for the target device. If this option is "
          "set to 0, the max number of hvx threads will be used.");

ABSL_FLAG(litert::qualcomm::QualcommOptions::OptimizationLevel,
          qualcomm_optimization_level,
          litert::qualcomm::QualcommOptions::OptimizationLevel::
              kOptimizeForInferenceO3,
          "QNN optimization level");

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::OptimizationLevel* optimization_level,
                   std::string* error) {
  if (text == "O1") {
    *optimization_level =
        QualcommOptions::OptimizationLevel::kOptimizeForInference;
    return true;
  }
  if (text == "O2") {
    *optimization_level =
        QualcommOptions::OptimizationLevel::kOptimizeForPrepare;
    return true;
  }
  if (text == "O3") {
    *optimization_level =
        QualcommOptions::OptimizationLevel::kOptimizeForInferenceO3;
    return true;
  }
  *error = "Unknown optimization level";
  return false;
}

std::string AbslUnparseFlag(
    QualcommOptions::OptimizationLevel optimization_level) {
  switch (optimization_level) {
    case QualcommOptions::OptimizationLevel::kOptimizeForInference:
      return "O1";
    case QualcommOptions::OptimizationLevel::kOptimizeForPrepare:
      return "O2";
    case QualcommOptions::OptimizationLevel::kOptimizeForInferenceO3:
      return "O3";
  }
}

}  // namespace litert::qualcomm

ABSL_FLAG(bool, qualcomm_use_conv_hmx, true,
          "When using short conv hmx, one might have better performance, but "
          "convolution that have short depth and/or weights that are not "
          "symmetric could exhibit inaccurate results.");

ABSL_FLAG(bool, qualcomm_use_fold_relu, true,
          "When using fold relu, one might have better performance. This "
          "optimization is correct when quantization ranges for convolution "
          "are equal to or are subset of the Relu operation.");

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

  const std::string ir_json_dir = absl::GetFlag(FLAGS_qualcomm_ir_json_dir);
  opts.SetIrJsonDir(ir_json_dir);

  const std::string dlc_dir = absl::GetFlag(FLAGS_qualcomm_dlc_dir);
  opts.SetDlcDir(dlc_dir);

  const auto vtcm_size = absl::GetFlag(FLAGS_qualcomm_vtcm_size);
  opts.SetVtcmSize(vtcm_size);

  const auto num_hvx_threads = absl::GetFlag(FLAGS_qualcomm_num_hvx_thread);
  opts.SetNumHvxThreads(num_hvx_threads);

  const auto optimization_level =
      absl::GetFlag(FLAGS_qualcomm_optimization_level);
  opts.SetOptimizationLevel(optimization_level);

  const auto use_conv_hmx = absl::GetFlag(FLAGS_qualcomm_use_conv_hmx);
  opts.SetUseConvHMX(use_conv_hmx);

  const auto use_fold_relu = absl::GetFlag(FLAGS_qualcomm_use_fold_relu);
  opts.SetUseFoldReLU(use_fold_relu);

  return opts;
}

}  // namespace litert::qualcomm
