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
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/tools/flags/flag_types.h"

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

ABSL_FLAG(litert::qualcomm::QualcommOptions::DspPerformanceMode,
          qualcomm_dsp_performance_mode,
          litert::qualcomm::QualcommOptions::DspPerformanceMode::kDefault,
          "DSP performance mode.");

ABSL_FLAG(::litert::tools::IntList, qualcomm_dump_tensor_ids, {},
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

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::DspPerformanceMode* options,
                   std::string* error) {
  if (text == "default") {
    *options = QualcommOptions::DspPerformanceMode::kDefault;
    return true;
  }
  if (text == "sustained_high_performance") {
    *options = QualcommOptions::DspPerformanceMode::kSustainedHighPerformance;
    return true;
  }
  if (text == "burst") {
    *options = QualcommOptions::DspPerformanceMode::kBurst;
    return true;
  }
  if (text == "high_performance") {
    *options = QualcommOptions::DspPerformanceMode::kHighPerformance;
    return true;
  }
  if (text == "power_saver") {
    *options = QualcommOptions::DspPerformanceMode::kPowerSaver;
    return true;
  }
  if (text == "low_power_saver") {
    *options = QualcommOptions::DspPerformanceMode::kLowPowerSaver;
    return true;
  }
  if (text == "high_power_saver") {
    *options = QualcommOptions::DspPerformanceMode::kHighPowerSaver;
    return true;
  }
  if (text == "low_balanced") {
    *options = QualcommOptions::DspPerformanceMode::kLowBalanced;
    return true;
  }
  if (text == "balanced") {
    *options = QualcommOptions::DspPerformanceMode::kBalanced;
    return true;
  }
  *error = "Unknown dsp performance mode";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::DspPerformanceMode options) {
  switch (options) {
    case QualcommOptions::DspPerformanceMode::kDefault:
      return "default";
    case QualcommOptions::DspPerformanceMode::kSustainedHighPerformance:
      return "sustained_high_performance";
    case QualcommOptions::DspPerformanceMode::kBurst:
      return "burst";
    case QualcommOptions::DspPerformanceMode::kHighPerformance:
      return "high_performance";
    case QualcommOptions::DspPerformanceMode::kPowerSaver:
      return "power_saver";
    case QualcommOptions::DspPerformanceMode::kLowPowerSaver:
      return "low_power_saver";
    case QualcommOptions::DspPerformanceMode::kHighPowerSaver:
      return "high_power_saver";
    case QualcommOptions::DspPerformanceMode::kLowBalanced:
      return "low_balanced";
    case QualcommOptions::DspPerformanceMode::kBalanced:
      return "balanced";
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

ABSL_FLAG(litert::qualcomm::QualcommOptions::GraphPriority,
          qualcomm_graph_priority,
          litert::qualcomm::QualcommOptions::GraphPriority::kDefault,
          "QNN graph priority, If the option is set to 'default', the "
          "QNN_PRIORITY_DEFAULT (Equal to QNN_PRIORITY_NORMAL) will be used.");

ABSL_FLAG(std::string, qualcomm_saver_output_dir, "",
          "Saver output directory. If provided, you can obtain saver_output.c "
          "and params.bin. Saver records all QNN API calls into files which "
          "can be replayed on any QNN backend. See Qualcomm "
          "AI Runtime (QAIRT) SDK document for more details.");

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

bool AbslParseFlag(absl::string_view text,
                   QualcommOptions::GraphPriority* graph_priority,
                   std::string* error) {
  if (text == "default") {
    *graph_priority = QualcommOptions::GraphPriority::kDefault;
    return true;
  }
  if (text == "low") {
    *graph_priority = QualcommOptions::GraphPriority::kLow;
    return true;
  }
  if (text == "normal") {
    *graph_priority = QualcommOptions::GraphPriority::kNormal;
    return true;
  }
  if (text == "normal_high") {
    *graph_priority = QualcommOptions::GraphPriority::kNormalHigh;
    return true;
  }
  if (text == "high") {
    *graph_priority = QualcommOptions::GraphPriority::kHigh;
    return true;
  }
  *error = "Unknown graph priority";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::GraphPriority graph_priority) {
  switch (graph_priority) {
    case QualcommOptions::GraphPriority::kDefault:
      return "default";
    case QualcommOptions::GraphPriority::kLow:
      return "low";
    case QualcommOptions::GraphPriority::kNormal:
      return "normal";
    case QualcommOptions::GraphPriority::kNormalHigh:
      return "normal_high";
    case QualcommOptions::GraphPriority::kHigh:
      return "high";
  }
  return "default";
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
  return "O1";
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

ABSL_FLAG(litert::qualcomm::QualcommOptions::Backend, qualcomm_backend,
          litert::qualcomm::QualcommOptions::Backend::kHtp,
          "QNN backend to use.");

namespace litert::qualcomm {

bool AbslParseFlag(absl::string_view text, QualcommOptions::Backend* options,
                   std::string* error) {
  if (text == "gpu") {
    *options = QualcommOptions::Backend::kGpu;
    return true;
  }
  if (text == "htp") {
    *options = QualcommOptions::Backend::kHtp;
    return true;
  }
  if (text == "dsp") {
    *options = QualcommOptions::Backend::kDsp;
    return true;
  }
  if (text == "ir") {
    *options = QualcommOptions::Backend::kIr;
    return true;
  }
  *error = "Unknown QNN backend";
  return false;
}

std::string AbslUnparseFlag(QualcommOptions::Backend options) {
  switch (options) {
    case QualcommOptions::Backend::kUndefined:
      return "undefined";
    case QualcommOptions::Backend::kGpu:
      return "gpu";
    case QualcommOptions::Backend::kHtp:
      return "htp";
    case QualcommOptions::Backend::kDsp:
      return "dsp";
    case QualcommOptions::Backend::kIr:
      return "ir";
  }
}

}  // namespace litert::qualcomm
// NOLINTEND(*alien-types*)

namespace litert::qualcomm {

Expected<void> UpdateQualcommOptionsFromFlags(QualcommOptions& opts) {
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

  const auto dsp_performance_mode =
      absl::GetFlag(FLAGS_qualcomm_dsp_performance_mode);
  opts.SetDspPerformanceMode(dsp_performance_mode);

  const auto profiling = absl::GetFlag(FLAGS_qualcomm_profiling);
  opts.SetProfiling(profiling);

  const auto dump_tensor_ids =
      absl::GetFlag(FLAGS_qualcomm_dump_tensor_ids).elements;
  opts.SetDumpTensorIds(dump_tensor_ids);

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

  const auto graph_priority = absl::GetFlag(FLAGS_qualcomm_graph_priority);
  opts.SetGraphPriority(graph_priority);

  const auto use_conv_hmx = absl::GetFlag(FLAGS_qualcomm_use_conv_hmx);
  opts.SetUseConvHMX(use_conv_hmx);

  const auto use_fold_relu = absl::GetFlag(FLAGS_qualcomm_use_fold_relu);
  opts.SetUseFoldReLU(use_fold_relu);

  const auto qnn_backend = absl::GetFlag(FLAGS_qualcomm_backend);
  opts.SetBackend(qnn_backend);

  const std::string saver_output_dir =
      absl::GetFlag(FLAGS_qualcomm_saver_output_dir);
  opts.SetSaverOutputDir(saver_output_dir);

  return {};
}

}  // namespace litert::qualcomm
