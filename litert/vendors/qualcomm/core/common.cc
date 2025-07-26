// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <cstdint>
#include <string>
#include <vector>

#include "QnnLog.h"                    // from @qairt
#include "absl/strings/str_format.h"   // from @com_google_absl
#include "absl/strings/str_join.h"     // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

#include <cstdarg>
#include <cstdio>
#include <iostream>

namespace qnn {
namespace {

void DefaultStdOutLogger(const char* fmt, QnnLog_Level_t level,
                         uint64_t timestamp, va_list argp) {
  const char* levelStr = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      levelStr = " ERROR ";
      break;
    case QNN_LOG_LEVEL_WARN:
      levelStr = "WARNING";
      break;
    case QNN_LOG_LEVEL_INFO:
      levelStr = "  INFO ";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      levelStr = " DEBUG ";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      levelStr = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      levelStr = "UNKNOWN";
      break;
  }
  char buffer1[256];
  char buffer2[256];
  double ms = timestamp;
  snprintf(buffer1, sizeof(buffer1), "%8.1fms [%-7s] ", ms, levelStr);
  buffer1[sizeof(buffer1) - 1] = 0;
  vsnprintf(buffer2, sizeof(buffer2), fmt, argp);
  buffer2[sizeof(buffer1) - 2] = 0;
  printf("%s %s", buffer1, buffer2);
}

}  // namespace

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

QnnLog_Callback_t GetDefaultStdOutLogger() { return DefaultStdOutLogger; }

}  // namespace qnn
