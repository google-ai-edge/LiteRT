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

const absl::string_view Options::GetQnnJsonPath() const {
  return qnn_json_path_;
}

void Options::SetQnnJsonPath(const char* qnn_json_path) {
  qnn_json_path_ = qnn_json_path;
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
DumpTensorIds: %s\n";  // NOLINT

  std::string dump_tensor_ids = absl::StrJoin(dump_tensor_ids_, ",");

  return absl::StrFormat(kQnnOptionsDumpFormat, log_level_, profiling_,
                         use_htp_preference_, use_qint16_as_quint16_,
                         enable_weight_sharing_, htp_performance_mode_,
                         dump_tensor_ids);
}

}  // namespace qnn
