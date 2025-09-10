// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace qnn {

void Options::SetLogLevel(const LogLevel log_level) { log_level_ = log_level; }

LogLevel Options::GetLogLevel() const { return log_level_; }

void Options::SetProfiling(const Profiling profiling) {
  profiling_ = profiling;
}

Profiling Options::GetProfiling() const { return profiling_; }

void Options::SetUseHtpPreference(const bool use_htp_preference) {
  use_htp_preference_ = use_htp_preference;
}

bool Options::GetUseHtpPreference() const { return use_htp_preference_; }

void Options::SetUseQint16AsQuint16(const bool use_qint16_as_quint16) {
  use_qint16_as_quint16_ = use_qint16_as_quint16;
}

bool Options::GetUseQint16AsQuint16() const { return use_qint16_as_quint16_; }

void Options::SetEnableWeightSharing(const bool enable_weight_sharing) {
  enable_weight_sharing_ = enable_weight_sharing;
}

bool Options::GetEnableWeightSharing() const { return enable_weight_sharing_; }

void Options::SetHtpPerformanceMode(
    const HtpPerformanceMode htp_performance_mode) {
  htp_performance_mode_ = htp_performance_mode;
}

HtpPerformanceMode Options::GetHtpPerformanceMode() const {
  return htp_performance_mode_;
}

void Options::SetDumpTensorIds(const std::vector<std::int32_t>& ids) {
  dump_tensor_ids_ = ids;
}

std::vector<std::int32_t> Options::GetDumpTensorIds() const {
  return dump_tensor_ids_;
}

absl::string_view Options::GetIrJsonDir() const { return ir_json_dir_; }

void Options::SetIrJsonDir(absl::string_view ir_json_dir) {
  ir_json_dir_ = ir_json_dir;
}

std::uint32_t Options::GetVtcmSize() const { return vtcm_size_; }

void Options::SetVtcmSize(std::uint32_t vtcm_size) { vtcm_size_ = vtcm_size; }

std::uint32_t Options::GetNumHvxThreads() const { return num_hvx_threads_; }

void Options::SetNumHvxThreads(std::uint32_t num_hvx_threads) {
  num_hvx_threads_ = num_hvx_threads;
}

OptimizationLevel Options::GetOptimizationLevel() const {
  return optimization_level_;
}

void Options::SetOptimizationLevel(OptimizationLevel optimization_level) {
  optimization_level_ = optimization_level;
}

std::string Options::Dump() const {
  static constexpr absl::string_view kQnnOptionsDumpFormat =
      "\
::qnn::Options:\n\
LogLevel: %d\n\
Profiling: %d\n\
UseHtpPreference: %v\n\
UseQint16AsQuint16: %v\n\
EnableWeightSharing: %v\n\
HtpPerformanceMode: %d\n\
DumpTensorIds: %s\n\
IrJsonDir: %s\n\
VtcmSize: %d\n\
HvxThread: %d\n\
OptimizationLevel: %d\n";  // NOLINT

  std::string dump_tensor_ids = absl::StrJoin(dump_tensor_ids_, ",");

  return absl::StrFormat(kQnnOptionsDumpFormat, log_level_, profiling_,
                         use_htp_preference_, use_qint16_as_quint16_,
                         enable_weight_sharing_, htp_performance_mode_,
                         dump_tensor_ids, ir_json_dir_, vtcm_size_,
                         num_hvx_threads_, optimization_level_);
}

}  // namespace qnn
