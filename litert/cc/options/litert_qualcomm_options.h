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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::qualcomm {

/// @brief Defines the C++ wrapper for Qualcomm-specific LiteRT options.
class QualcommOptions {
 public:
  QualcommOptions() : options_(nullptr) {}
  explicit QualcommOptions(LrtQualcommOptions options) : options_(options) {}
  ~QualcommOptions() {
    if (options_) {
      LrtDestroyQualcommOptions(options_);
    }
  }

  QualcommOptions(QualcommOptions&& other) noexcept : options_(other.options_) {
    other.options_ = nullptr;
  }
  QualcommOptions& operator=(QualcommOptions&& other) noexcept {
    if (this != &other) {
      if (options_) LrtDestroyQualcommOptions(options_);
      options_ = other.options_;
      other.options_ = nullptr;
    }
    return *this;
  }

  // Delete copy constructor and assignment
  QualcommOptions(const QualcommOptions&) = delete;
  QualcommOptions& operator=(const QualcommOptions&) = delete;

  LrtQualcommOptions Get() const { return options_; }
  LrtQualcommOptions Release() {
    auto* res = options_;
    options_ = nullptr;
    return res;
  }

  static const char* Discriminator() {
    return LrtQualcommOptionsGetIdentifier();
  }

  static Expected<QualcommOptions> Create() {
    LrtQualcommOptions c_options;
    LITERT_RETURN_IF_ERROR(LrtCreateQualcommOptions(&c_options));
    return QualcommOptions(c_options);
  }

  /// @brief Determines the logging level of all underlying Qualcomm SDK
  /// libraries.
  ///
  /// This does not affect LiteRT logging. Defaults to `kInfo`.
  enum class LogLevel : int {
    kOff = kLiteRtQualcommLogOff,
    kError = kLiteRtQualcommLogLevelError,
    kWarn = kLiteRtQualcommLogLevelWarn,
    kInfo = kLiteRtQualcommLogLevelInfo,
    kVerbose = kLiteRtQualcommLogLevelVerbose,
    kDebug = kLiteRtQualcommLogLevelDebug,
  };

  void SetLogLevel(LogLevel log_level) {
    LrtQualcommOptionsSetLogLevel(
        options_, static_cast<LrtQualcommOptionsLogLevel>(log_level));
  }
  LogLevel GetLogLevel() {
    LrtQualcommOptionsLogLevel val;
    auto status = LrtQualcommOptionsGetLogLevel(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return LogLevel::kInfo;
    }
    return static_cast<LogLevel>(val);
  }

  /// @brief This option controls whether to convert a LiteRT operation to QNN
  /// operations that are preferred by the HTP backend. Defaults to `false`.
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

  void SetHtpPerformanceMode(HtpPerformanceMode htp_performance_mode) {
    LrtQualcommOptionsSetHtpPerformanceMode(
        options_, static_cast<LrtQualcommOptionsHtpPerformanceMode>(
                      htp_performance_mode));
  }
  HtpPerformanceMode GetHtpPerformanceMode() {
    LrtQualcommOptionsHtpPerformanceMode val;
    auto status = LrtQualcommOptionsGetHtpPerformanceMode(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return HtpPerformanceMode::kDefault;
    }
    return static_cast<HtpPerformanceMode>(val);
  }

  enum class DspPerformanceMode : int {
    kDefault = kLiteRtQualcommDspPerformanceModeDefault,
    kSustainedHighPerformance =
        kLiteRtQualcommDspPerformanceModeSustainedHighPerformance,
    kBurst = kLiteRtQualcommDspPerformanceModeBurst,
    kHighPerformance = kLiteRtQualcommDspPerformanceModeHighPerformance,
    kPowerSaver = kLiteRtQualcommDspPerformanceModePowerSaver,
    kLowPowerSaver = kLiteRtQualcommDspPerformanceModeLowPowerSaver,
    kHighPowerSaver = kLiteRtQualcommDspPerformanceModeHighPowerSaver,
    kLowBalanced = kLiteRtQualcommDspPerformanceModeLowBalanced,
    kBalanced = kLiteRtQualcommDspPerformanceModeBalanced,
  };

  void SetDspPerformanceMode(DspPerformanceMode dsp_performance_mode) {
    LrtQualcommOptionsSetDspPerformanceMode(
        options_, static_cast<LrtQualcommOptionsDspPerformanceMode>(
                      dsp_performance_mode));
  }
  DspPerformanceMode GetDspPerformanceMode() {
    LrtQualcommOptionsDspPerformanceMode val;
    auto status = LrtQualcommOptionsGetDspPerformanceMode(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return DspPerformanceMode::kDefault;
    }
    return static_cast<DspPerformanceMode>(val);
  }

  [[deprecated("This option is deprecated and will be no-op.")]]
  void SetUseHtpPreference(bool use_htp_preference) {
    LrtQualcommOptionsSetUseHtpPreference(options_, use_htp_preference);
  }
  [[deprecated("This option is deprecated and will be no-op.")]]
  bool GetUseHtpPreference() {
    bool val;
    auto status = LrtQualcommOptionsGetUseHtpPreference(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return false;
    }
    return val;
  }

  /// @brief This option controls whether to convert a quantized int16 model to
  /// a quantized uint16 model. Defaults to `false`.
  [[deprecated("This option is deprecated and will be no-op.")]]
  void SetUseQint16AsQuint16(bool use_qin16_as_quint16) {
    LrtQualcommOptionsSetUseQint16AsQuint16(options_, use_qin16_as_quint16);
  }
  [[deprecated("This option is deprecated and will be no-op.")]]
  bool GetUseQint16AsQuint16() {
    bool val;
    auto status = LrtQualcommOptionsGetUseQint16AsQuint16(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return false;
    }
    return val;
  }

  /// @brief This option controls whether to convert bias tensors of
  /// FullyConnected and Conv2D Ops from int64 to int32. Defaults to `true`.
  void SetUseInt64BiasAsInt32(bool use_int64_bias_as_int32) {
    LrtQualcommOptionsSetUseInt64BiasAsInt32(options_, use_int64_bias_as_int32);
  }
  bool GetUseInt64BiasAsInt32() {
    bool val;
    auto status = LrtQualcommOptionsGetUseInt64BiasAsInt32(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return true;
    }
    return val;
  }

  /// @brief Indicates whether different subgraphs may share weight tensors.
  ///
  /// This is only supported on x86 AOT. Defaults to `false`.
  void SetEnableWeightSharing(bool weight_sharing_enabled) {
    LrtQualcommOptionsSetEnableWeightSharing(options_, weight_sharing_enabled);
  }
  bool GetEnableWeightSharing() {
    bool val;
    auto status = LrtQualcommOptionsGetEnableWeightSharing(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return false;
    }
    return val;
  }

  /// @brief Just-In-Time allows passing the QNN Context directly from the
  /// compiler plugin to the dispatcher in-memory, bypassing graph finalization.
  /// Defaults to `false`.
  void SetEnableJustInTime(bool enable_just_in_time) {
    LrtQualcommOptionsSetEnableJustInTime(options_, enable_just_in_time);
  }
  bool GetEnableJustInTime() {
    bool val;
    auto status = LrtQualcommOptionsGetEnableJustInTime(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return false;
    }
    return val;
  }

  /// @brief When using short conv hmx, one might have better performance, but
  /// convolutions with short depth and/or non-symmetric weights could exhibit
  /// inaccurate results.
  void SetUseConvHMX(bool use_conv_hmx) {
    LrtQualcommOptionsSetUseConvHMX(options_, use_conv_hmx);
  }
  bool GetUseConvHMX() {
    bool val;
    auto status = LrtQualcommOptionsGetUseConvHMX(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return true;
    }
    return val;
  }

  /// @brief When using fold relu, one might have better performance.
  ///
  /// This optimization is correct when quantization ranges for convolution are
  /// equal to or are a subset of the Relu operation.
  void SetUseFoldReLU(bool use_fold_relu) {
    LrtQualcommOptionsSetUseFoldReLU(options_, use_fold_relu);
  }
  bool GetUseFoldReLU() {
    bool val;
    auto status = LrtQualcommOptionsGetUseFoldReLU(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return true;
    }
    return val;
  }

  /// @brief This option controls the profiling level.
  ///
  /// A higher level results in a more detailed report after execution.
  /// Defaults to `kOff`.
  enum class Profiling : int {
    kOff = kLiteRtQualcommProfilingOff,
    kBasic = kLiteRtQualcommProfilingBasic,
    kDetailed = kLiteRtQualcommProfilingDetailed,
    kLinting = kLiteRtQualcommProfilingLinting,
    kOptrace = kLiteRtQualcommProfilingOptrace,
  };

  void SetProfiling(Profiling profiling) {
    LrtQualcommOptionsSetProfiling(
        options_, static_cast<LrtQualcommOptionsProfiling>(profiling));
  }
  Profiling GetProfiling() {
    LrtQualcommOptionsProfiling val;
    auto status = LrtQualcommOptionsGetProfiling(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return Profiling::kOff;
    }
    return static_cast<Profiling>(val);
  }

  void SetDumpTensorIds(const std::vector<std::int32_t>& ids) {
    LrtQualcommOptionsSetDumpTensorIds(options_, ids.data(), ids.size());
  }
  std::vector<std::int32_t> GetDumpTensorIds() {
    const std::int32_t* ids;
    size_t number_of_ids;
    auto status =
        LrtQualcommOptionsGetDumpTensorIds(options_, &ids, &number_of_ids);
    if (status == kLiteRtStatusErrorNotFound) {
      return {};
    }
    return std::vector<std::int32_t>(ids, ids + number_of_ids);
  }

  void SetIrJsonDir(const std::string& ir_json_dir) {
    LrtQualcommOptionsSetIrJsonDir(options_, ir_json_dir.c_str());
  }
  absl::string_view GetIrJsonDir() {
    const char* val;
    auto status = LrtQualcommOptionsGetIrJsonDir(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return "";
    }
    return val;
  }

  void SetDlcDir(const std::string& dlc_dir) {
    LrtQualcommOptionsSetDlcDir(options_, dlc_dir.c_str());
  }
  absl::string_view GetDlcDir() {
    const char* val;
    auto status = LrtQualcommOptionsGetDlcDir(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return "";
    }
    return val;
  }

  void SetVtcmSize(std::uint32_t vtcm_size) {
    LrtQualcommOptionsSetVtcmSize(options_, vtcm_size);
  }
  std::uint32_t GetVtcmSize() {
    std::uint32_t val;
    auto status = LrtQualcommOptionsGetVtcmSize(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return 0;
    }
    return val;
  }

  void SetNumHvxThreads(std::uint32_t num_hvx_threads) {
    LrtQualcommOptionsSetNumHvxThreads(options_, num_hvx_threads);
  }
  std::uint32_t GetNumHvxThreads() {
    std::uint32_t val;
    auto status = LrtQualcommOptionsGetNumHvxThreads(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return 0;
    }
    return val;
  }

  enum class OptimizationLevel : int {
    kOptimizeForInference = kHtpOptimizeForInference,
    kOptimizeForPrepare = kHtpOptimizeForPrepare,
    kOptimizeForInferenceO3 = kHtpOptimizeForInferenceO3,
  };

  void SetOptimizationLevel(OptimizationLevel optimization_level) {
    LrtQualcommOptionsSetOptimizationLevel(
        options_,
        static_cast<LrtQualcommOptionsOptimizationLevel>(optimization_level));
  }
  OptimizationLevel GetOptimizationLevel() {
    LrtQualcommOptionsOptimizationLevel val;
    auto status = LrtQualcommOptionsGetOptimizationLevel(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return OptimizationLevel::kOptimizeForInferenceO3;
    }
    return static_cast<OptimizationLevel>(val);
  }

  enum class GraphPriority : int {
    kDefault = kLiteRTQualcommGraphPriorityDefault,
    kLow = kLiteRTQualcommGraphPriorityLow,
    kNormal = kLiteRTQualcommGraphPriorityNormal,
    kNormalHigh = kLiteRTQualcommGraphPriorityNormalHigh,
    kHigh = kLiteRTQualcommGraphPriorityHigh,
  };

  void SetGraphPriority(GraphPriority graph_priority) {
    LrtQualcommOptionsSetGraphPriority(
        options_, static_cast<LrtQualcommOptionsGraphPriority>(graph_priority));
  }
  GraphPriority GetGraphPriority() {
    LrtQualcommOptionsGraphPriority val;
    auto status = LrtQualcommOptionsGetGraphPriority(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return GraphPriority::kDefault;
    }
    return static_cast<GraphPriority>(val);
  }

  enum class Backend : int {
    kUndefined = kLiteRtQualcommBackendUndefined,
    kGpu = kLiteRtQualcommBackendGpu,
    kHtp = kLiteRtQualcommBackendHtp,
    kDsp = kLiteRtQualcommBackendDsp,
    kIr = kLiteRtQualcommBackendIr,
  };

  void SetBackend(Backend backend) {
    LrtQualcommOptionsSetBackend(
        options_, static_cast<LrtQualcommOptionsBackend>(backend));
  }
  Backend GetBackend() {
    LrtQualcommOptionsBackend val;
    auto status = LrtQualcommOptionsGetBackend(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return Backend::kHtp;
    }
    return static_cast<Backend>(val);
  }

  void SetSaverOutputDir(const std::string& saver_output_dir) {
    LrtQualcommOptionsSetSaverOutputDir(options_, saver_output_dir.c_str());
  }
  absl::string_view GetSaverOutputDir() {
    const char* val;
    auto status = LrtQualcommOptionsGetSaverOutputDir(options_, &val);
    if (status == kLiteRtStatusErrorNotFound) {
      return "";
    }
    return val;
  }

  enum class GraphIOTensorMemType : int {
    kRaw = kLiteRtQualcommGraphIOTensorMemTypeRaw,
    kMemHandle = kLiteRtQualcommGraphIOTensorMemTypeMemHandle,
  };

  void SetGraphIOTensorMemType(GraphIOTensorMemType mem_type) {
    LrtQualcommOptionsSetGraphIOTensorMemType(
        options_,
        static_cast<LrtQualcommOptionsGraphIOTensorMemType>(mem_type));
  }

  GraphIOTensorMemType GetGraphIOTensorMemType() const {
    LrtQualcommOptionsGraphIOTensorMemType val;
    auto status = LrtQualcommOptionsGetGraphIOTensorMemType(options_, &val);
    if (status != kLiteRtStatusOk) {
      return GraphIOTensorMemType::kMemHandle;
    }
    return static_cast<GraphIOTensorMemType>(val);
  }

 private:
  LrtQualcommOptions options_;
};

}  // namespace litert::qualcomm

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
