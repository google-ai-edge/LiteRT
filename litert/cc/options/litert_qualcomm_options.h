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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::qualcomm {

// Wraps a LiteRtQualcommOptions object for convenience.
class QualcommOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() {
    return LiteRtQualcommOptionsGetIdentifier();
  }

  static Expected<QualcommOptions> Create(OpaqueOptions& options);

  static Expected<QualcommOptions> Create();

  // This determines the logging level of all underlying qualcomm sdk libraries.
  // Does not effect litert logging. Defaults to INFO.
  enum class LogLevel : int {
    kOff = kLiteRtQualcommLogOff,
    kError = kLiteRtQualcommLogLevelError,
    kWarn = kLiteRtQualcommLogLevelWarn,
    kInfo = kLiteRtQualcommLogLevelInfo,
    kVerbose = kLiteRtQualcommLogLevelVerbose,
    kDebug = kLiteRtQualcommLogLevelDebug,
  };

  void SetLogLevel(LogLevel log_level);
  LogLevel GetLogLevel();

  // This option controls whether to convert a LiteRt operation to QNN
  // operations which are preferred by the HTP backend. Defaults to false.
  enum class HtpPerformanceMode : int {
    kDefault = kLiteRtQualcommHtpPerformanceModeDefault,
    kSustainedHighPerformance =
        kLiteRtQualcommHtpPerformanceModeSustainedHighPerformance,
    kBurst = kLiteRtQualcommHtpPerformanceModeBurst,
    kHighPerformance = kLiteRtQualcommHtpPerformanceModeHighPerformance,
    kPowerSaver = kLiteRtQualcommHtpPerformanceModePowerSaver,
    kLowPowerSaver = kLiteRtQualcommHtpPerformanceModeLowPowerSaver,
    kHighPowerSaver = kLiteRtQualcommHtpPerformanceModeHighPowerSaver,
    kLowBalanced = kLiteRtQualcommHtpPerformanceModeLowBalanced,
    kBalanced = kLiteRtQualcommHtpPerformanceModeBalanced,
    kExtremePowerSaver = kLiteRtQualcommHtpPerformanceModeExtremePowerSaver,
  };

  void SetHtpPerformanceMode(HtpPerformanceMode htp_performance_mode);
  HtpPerformanceMode GetHtpPerformanceMode();

  void SetUseHtpPreference(bool use_htp_preference);
  bool GetUseHtpPreference();

  // This option controls whether to convert a quantized int16 model to a
  // quantized uint16 model. Defaults to false.
  void SetUseQint16AsQuint16(bool use_qin16_as_quint16);
  bool GetUseQint16AsQuint16();

  // Weight sharing indicates whether different subgraphs may share weight
  // tensors. This is only supported on x86 AOT. Defaults to false.
  void SetEnableWeightSharing(bool weight_sharing_enabled);
  bool GetEnableWeightSharing();

  // When using short conv hmx, one might have better performance, but
  // convolution that have short depth and/or weights that are not symmetric
  // could exhibit inaccurate results.
  void SetUseConvHMX(bool use_conv_hmx);
  bool GetUseConvHMX();

  // When using fold relu, one might have better performance. This optimization
  // is correct when quantization ranges for convolution are equal to or are
  // subset of the Relu operation.
  void SetUseFoldReLU(bool use_fold_relu);
  bool GetUseFoldReLU();

  // This option controls the profiling level. A higher level results in a more
  // detailed report after execution. Defaults to off.
  enum class Profiling : int {
    kOff = kLiteRtQualcommProfilingOff,
    kBasic = kLiteRtQualcommProfilingBasic,
    kDetailed = kLiteRtQualcommProfilingDetailed,
    kLinting = kLiteRtQualcommProfilingLinting,
    kOptrace = kLiteRtQualcommProfilingOptrace,
  };

  void SetProfiling(Profiling profiling);
  Profiling GetProfiling();

  void SetDumpTensorIds(const std::vector<std::int32_t>& ids);
  std::vector<std::int32_t> GetDumpTensorIds();

  void SetIrJsonDir(const std::string& ir_json_dir);
  absl::string_view GetIrJsonDir();

  void SetDlcDir(const std::string& dlc_dir);
  absl::string_view GetDlcDir();

  void SetVtcmSize(std::uint32_t vtcm_size);
  std::uint32_t GetVtcmSize();

  void SetNumHvxThreads(std::uint32_t num_hvx_threads);
  std::uint32_t GetNumHvxThreads();

  enum class OptimizationLevel : int {
    kOptimizeForInference = kHtpOptimizeForInference,
    kOptimizeForPrepare = kHtpOptimizeForPrepare,
    kOptimizeForInferenceO3 = kHtpOptimizeForInferenceO3,
  };

  void SetOptimizationLevel(OptimizationLevel optimization_level);
  OptimizationLevel GetOptimizationLevel();

  enum class GraphPriority : int {
    kDefault = kLiteRTQualcommGraphPriorityDefault,
    kLow = kLiteRTQualcommGraphPriorityLow,
    kNormal = kLiteRTQualcommGraphPriorityNormal,
    kNormalHigh = kLiteRTQualcommGraphPriorityNormalHigh,
    kHigh = kLiteRTQualcommGraphPriorityHigh,
  };

  void SetGraphPriority(GraphPriority graph_priority);
  GraphPriority GetGraphPriority();

 private:
  LiteRtQualcommOptions Data() const;
};

}  // namespace litert::qualcomm

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
