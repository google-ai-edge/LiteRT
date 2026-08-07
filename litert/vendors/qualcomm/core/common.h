// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "QnnLog.h"  // from @qairt
#include "absl/strings/string_view.h"  // from @com_google_absl

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
  kLpaiBackend,
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

enum class GraphIOTensorMemType {
  kRaw = 0,
  kMemHandle = 1,
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

enum class HtpPerfCtrlMode {
  kManual = 0,  // default: upvote at init, downvote at destroy
  kAuto = 1,    // per-inference upvote + 300ms debounced downvote
};

enum class DspPerfCtrlMode {
  kManual = 0,
  kAuto = 1,
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

struct CustomOpPackage {
  std::string name;
  std::string interface_provider;
  std::string compile_package_path;
  std::string dispatch_package_path;
  // QNN backend target at dispatch time (e.g., "HTP", "GPU").
  std::string target;
};

enum class GpuPrecision {
  kUserProvided = 0,
  kFp32 = 1,
  kFp16 = 2,
  kHybrid = 3,
};

enum class GpuPerformanceMode {
  kDefault = 0,
  kHigh = 1,
  kNormal = 2,
  kLow = 3,
};

enum class LpaiClientPerfType {
  kDefault = 0,
  kRealTime = 1,
  kNonRealTime = 2,
};

enum class LpaiCoreAffinityType {
  kDefault = 0,
  kSoft = 1,
  kHard = 2,
};

enum class LpaiTarget {
  kX86 = 0,
  kArm = 1,
  kAdsp = 2,
  kTensilica = 3,
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

  void SetUseInt64BiasAsInt32(bool use_int64_bias_as_int32);
  bool GetUseInt64BiasAsInt32() const;

  void SetEnableWeightSharing(bool enable_weight_sharing);
  bool GetEnableWeightSharing() const;

  void SetEnableJustInTime(bool enable_just_in_time);
  bool GetEnableJustInTime() const;

  void SetUseConvHMX(bool use_conv_hmx);
  bool GetUseConvHMX() const;

  void SetUseFoldReLU(bool use_fold_relu);
  bool GetUseFoldReLU() const;

  void SetHtpPPoint(std::int32_t htp_p_point);
  std::int32_t GetHtpPPoint() const;

  void SetHtpDlbc(bool htp_dlbc);
  bool GetHtpDlbc() const;

  void SetHtpDlbcWeights(bool htp_dlbc_weights);
  bool GetHtpDlbcWeights() const;

  void SetHtpPerformanceMode(HtpPerformanceMode htp_performance_mode);
  HtpPerformanceMode GetHtpPerformanceMode() const;

  void SetDspPerformanceMode(DspPerformanceMode dsp_performance_mode);
  DspPerformanceMode GetDspPerformanceMode() const;

  void SetHtpPerfCtrlMode(HtpPerfCtrlMode htp_perf_ctrl_mode);
  HtpPerfCtrlMode GetHtpPerfCtrlMode() const;

  void SetDspPerfCtrlMode(DspPerfCtrlMode dsp_perf_ctrl_mode);
  DspPerfCtrlMode GetDspPerfCtrlMode() const;

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

  void SetGpuPrecision(GpuPrecision gpu_precision);
  GpuPrecision GetGpuPrecision() const;

  void SetGpuPerformanceMode(GpuPerformanceMode gpu_performance_mode);
  GpuPerformanceMode GetGpuPerformanceMode() const;

  std::string Dump() const;

  absl::string_view GetSaverOutputDir() const;
  void SetSaverOutputDir(absl::string_view saver_output_dir);

  void SetGraphIOTensorMemType(GraphIOTensorMemType mem_type);
  GraphIOTensorMemType GetGraphIOTensorMemType() const;

  absl::string_view GetSchematicDir() const;
  void SetSchematicDir(absl::string_view schematic_dir);

  void SetCustomOpPackage(absl::string_view name,
                          absl::string_view interface_provider,
                          absl::string_view compile_package_path,
                          absl::string_view dispatch_package_path,
                          absl::string_view target);
  const CustomOpPackage& GetCustomOpPackage() const;

  // LPAI options.
  void SetLpaiTarget(LpaiTarget lpai_target);
  LpaiTarget GetLpaiTarget() const;

  void SetLpaiFps(std::uint32_t lpai_fps);
  std::uint32_t GetLpaiFps() const;

  void SetLpaiFtrtRatio(std::uint32_t lpai_ftrt_ratio);
  std::uint32_t GetLpaiFtrtRatio() const;

  void SetLpaiClientPerfType(LpaiClientPerfType lpai_client_perf_type);
  LpaiClientPerfType GetLpaiClientPerfType() const;

  void SetLpaiCoreAffinityType(LpaiCoreAffinityType lpai_core_affinity_type);
  LpaiCoreAffinityType GetLpaiCoreAffinityType() const;

  void SetLpaiCoreSelection(std::uint32_t lpai_core_selection);
  std::uint32_t GetLpaiCoreSelection() const;

 private:
  LogLevel log_level_ = LogLevel::kInfo;
  BackendType backend_type_ = BackendType::kHtpBackend;
  Profiling profiling_ = Profiling::kOff;
  bool use_int64_bias_as_int32_ = true;
  bool enable_weight_sharing_ = false;
  bool enable_just_in_time_ = false;
  bool use_conv_hmx_ = true;
  bool use_fold_relu_ = true;
  std::int32_t htp_p_point_ = 0;
  bool htp_dlbc_ = false;
  bool htp_dlbc_weights_ = false;
  HtpPerformanceMode htp_performance_mode_ = HtpPerformanceMode::kDefault;
  DspPerformanceMode dsp_performance_mode_ = DspPerformanceMode::kDefault;
  HtpPerfCtrlMode htp_perf_ctrl_mode_ = HtpPerfCtrlMode::kManual;
  DspPerfCtrlMode dsp_perf_ctrl_mode_ = DspPerfCtrlMode::kManual;
  std::vector<std::int32_t> dump_tensor_ids_;
  std::string ir_json_dir_;
  std::string dlc_dir_;
  std::uint32_t vtcm_size_ = 0;
  std::uint32_t num_hvx_threads_ = 0;
  OptimizationLevel optimization_level_ =
      OptimizationLevel::kHtpOptimizeForInferenceO3;
  GraphPriority graph_priority_ = GraphPriority::kDefault;
  GpuPrecision gpu_precision_ = GpuPrecision::kFp16;
  GpuPerformanceMode gpu_performance_mode_ = GpuPerformanceMode::kHigh;
  std::string saver_output_dir_;
  GraphIOTensorMemType graph_io_tensor_mem_type_ =
      GraphIOTensorMemType::kMemHandle;
  std::string schematic_dir_;
  // Currently we only support one custom op package.
  CustomOpPackage custom_op_package_;
  LpaiTarget lpai_target_ = LpaiTarget::kAdsp;
  std::uint32_t lpai_fps_ = 1;  // QNN_LPAI_GRAPH_DEFAULT_FPS
  std::uint32_t lpai_ftrt_ratio_ = 10;  // QNN_LPAI_GRAPH_DEFAULT_FTRT_RATIO
  LpaiClientPerfType lpai_client_perf_type_ = LpaiClientPerfType::kDefault;
  LpaiCoreAffinityType lpai_core_affinity_type_ = LpaiCoreAffinityType::kDefault;
  std::uint32_t lpai_core_selection_ = 0;
};

// Gets a default logger implementation to stdout.
// This is used when initializing qnn logging.
QnnLog_Callback_t GetDefaultStdOutLogger();

struct SdkVersion {
  int major = 0;
  int minor = 0;
  int patch = 0;

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

// Parses a QNN build ID string of the form "vMAJOR.MINOR.PATCH" into an
// SdkVersion. Returns std::nullopt on parsing failure.
std::optional<SdkVersion> ParseSdkVersion(const char* build_id);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
