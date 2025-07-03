// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

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

enum class Profiling { kOff = 0, kBasic = 1, kDetailed = 2 };

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

class Options {
 public:
  Options() = default;

  void SetLogLevel(const LogLevel log_level);
  LogLevel GetLogLevel() const;

  void SetProfiling(const Profiling profiling);
  Profiling GetProfiling() const;

  void SetUseHtpPreference(const bool use_htp_preference);
  bool GetUseHtpPreference() const;

  void SetUseQint16AsQuint16(const bool use_qint16_as_quint16);
  bool GetUseQint16AsQuint16() const;

  void SetEnableWeightSharing(const bool enable_weight_sharing);
  bool GetEnableWeightSharing() const;

  void SetHtpPerformanceMode(const HtpPerformanceMode htp_performance_mode);
  HtpPerformanceMode GetHtpPerformanceMode() const;

  // for per-layer dump
  void SetDumpTensorIds(const std::vector<std::int32_t>& ids);
  std::vector<std::int32_t> GetDumpTensorIds() const;

  const absl::string_view GetQnnJsonPath() const;
  void SetQnnJsonPath(const char* qnn_json_path);

  std::string Dump() const;

 private:
  LogLevel log_level_ = LogLevel::kInfo;
  Profiling profiling_ = Profiling::kOff;
  bool use_htp_preference_ = false;
  bool use_qint16_as_quint16_ = false;
  bool enable_weight_sharing_ = false;
  HtpPerformanceMode htp_performance_mode_ = HtpPerformanceMode::kDefault;
  std::vector<std::int32_t> dump_tensor_ids_;
  std::string qnn_json_path_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
