// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/options/litert_qualcomm_options.h"

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

std::string Options::Dump() const {
  static constexpr absl::string_view kQnnOptionsDumpFormat =
      "\
::qnn::Options:\n\
LogLevel: %d\n\
Profiling: %d\n\
UseHtpPreference: %v\n\
UseQint16AsQuint16: %v\n\
EnableWeightSharing: %v\n\
HtpPerformanceMode: %d\n";

  return absl::StrFormat(kQnnOptionsDumpFormat, log_level_, profiling_,
                         use_htp_preference_, use_qint16_as_quint16_,
                         enable_weight_sharing_, htp_performance_mode_);
}

LiteRtStatus InitQnnOptions(
    ::qnn::Options& qnn_options,
    litert::qualcomm::QualcommOptions& qualcomm_options) {
  qnn_options.SetLogLevel(
      static_cast<::qnn::LogLevel>(qualcomm_options.GetLogLevel()));
  qnn_options.SetProfiling(
      static_cast<::qnn::Profiling>(qualcomm_options.GetProfiling()));
  qnn_options.SetUseHtpPreference(qualcomm_options.GetUseHtpPreference());
  qnn_options.SetUseQint16AsQuint16(qualcomm_options.GetUseQint16AsQuint16());
  qnn_options.SetEnableWeightSharing(qualcomm_options.GetEnableWeightSharing());
  qnn_options.SetHtpPerformanceMode(static_cast<::qnn::HtpPerformanceMode>(
      qualcomm_options.GetHtpPerformanceMode()));
  LITERT_LOG(LITERT_INFO, "\n%s", qnn_options.Dump().data());
  return kLiteRtStatusOk;
}

}  // namespace qnn
