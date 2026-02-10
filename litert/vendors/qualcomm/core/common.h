// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "QnnLog.h"  // from @qairt

// c++ enum and wrapper without dependency.
namespace qnn {

enum class LogLevel {
  kOff = 0,
  kError = 1,
  kWarn = 2,
  kInfo = 3,
  kVerbose = 4,
  kDebug = 5,
};

enum class Profiling {
  kOff = 0,
  kBasic = 1,
  kDetailed = 2,
  kLinting = 3,
  kOptrace = 4
};

enum class BackendType {
  kUndefinedBackend = 0,
  kGpuBackend,
  kHtpBackend,
  kDspBackend,
  kIrBackend,
};

enum class HtpPerformanceMode {
  kDefault = 0,
  kSustainedHighPerformance = 1,
  kBurst = 2,
  kHighPerformance = 3,
  kPowerSaver = 4,
  kLowPowerSaver = 5,
  kHighPowerSaver = 6,
  kLowBalanced = 7,
  kBalanced = 8,
  kExtremePowerSaver = 9,
};

enum class DspPerformanceMode {
  kDefault = 0,
  kSustainedHighPerformance = 1,
  kBurst = 2,
  kHighPerformance = 3,
  kPowerSaver = 4,
  kLowPowerSaver = 5,
  kHighPowerSaver = 6,
  kLowBalanced = 7,
  kBalanced = 8,
};

enum class OptimizationLevel {
  kHtpOptimizeForInference = 0,
  kHtpOptimizeForPrepare = 1,
  kHtpOptimizeForInferenceO3 = 2,
};

enum class GraphPriority {
  kDefault = 0,
  kLow = 1,
  kNormal = 2,
  kNormalHigh = 3,
  kHigh = 4,
};

class Options {
 public:
  Options() = default;

  void SetLogLevel(LogLevel log_level);
  LogLevel GetLogLevel() const;

  void SetBackendType(BackendType backend_type);
  BackendType GetBackendType() const;

  void SetProfiling(Profiling profiling);
  Profiling GetProfiling() const;

  void SetUseHtpPreference(bool use_htp_preference);
  bool GetUseHtpPreference() const;

  void SetUseQint16AsQuint16(bool use_qint16_as_quint16);
  bool GetUseQint16AsQuint16() const;

  void SetEnableWeightSharing(bool enable_weight_sharing);
  bool GetEnableWeightSharing() const;

  void SetUseConvHMX(bool use_conv_hmx);
  bool GetUseConvHMX() const;

  void SetUseFoldReLU(bool use_fold_relu);
  bool GetUseFoldReLU() const;

  void SetHtpPerformanceMode(HtpPerformanceMode htp_performance_mode);
  HtpPerformanceMode GetHtpPerformanceMode() const;

  void SetDspPerformanceMode(DspPerformanceMode dsp_performance_mode);
  DspPerformanceMode GetDspPerformanceMode() const;

  // for per-layer dump
  void SetDumpTensorIds(const std::vector<std::int32_t>& ids);
  std::vector<std::int32_t> GetDumpTensorIds() const;

  absl::string_view GetIrJsonDir() const;
  void SetIrJsonDir(absl::string_view ir_json_dir);

  absl::string_view GetDlcDir() const;
  void SetDlcDir(absl::string_view dlc_dir);

  std::uint32_t GetVtcmSize() const;
  void SetVtcmSize(std::uint32_t vtcm_size);

  std::uint32_t GetNumHvxThreads() const;
  void SetNumHvxThreads(std::uint32_t num_hvx_threads);

  void SetOptimizationLevel(OptimizationLevel optimization_level);
  OptimizationLevel GetOptimizationLevel() const;

  void SetGraphPriority(GraphPriority graph_priority);
  GraphPriority GetGraphPriority() const;

  std::string Dump() const;

  absl::string_view GetSaverOutputDir() const;
  void SetSaverOutputDir(absl::string_view saver_output_dir);

 private:
  LogLevel log_level_ = LogLevel::kInfo;
  BackendType backend_type_ = BackendType::kHtpBackend;
  Profiling profiling_ = Profiling::kOff;
  bool use_htp_preference_ = false;
  bool use_qint16_as_quint16_ = false;
  bool enable_weight_sharing_ = false;
  bool use_conv_hmx_ = true;
  bool use_fold_relu_ = true;
  HtpPerformanceMode htp_performance_mode_ = HtpPerformanceMode::kDefault;
  DspPerformanceMode dsp_performance_mode_ = DspPerformanceMode::kDefault;
  std::vector<std::int32_t> dump_tensor_ids_;
  std::string ir_json_dir_;
  std::string dlc_dir_;
  std::uint32_t vtcm_size_ = 0;
  std::uint32_t num_hvx_threads_ = 0;
  OptimizationLevel optimization_level_ =
      OptimizationLevel::kHtpOptimizeForInferenceO3;
  GraphPriority graph_priority_ = GraphPriority::kDefault;
  std::string saver_output_dir_;
};

// Gets a default logger implementation to stdout.
// This is used when initializing qnn logging.
QnnLog_Callback_t GetDefaultStdOutLogger();

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
