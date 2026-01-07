// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

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

void Options::SetLogLevel(LogLevel log_level) { log_level_ = log_level; }

LogLevel Options::GetLogLevel() const { return log_level_; }

void Options::SetBackendType(BackendType backend_type) {
  backend_type_ = backend_type;
}

BackendType Options::GetBackendType() const { return backend_type_; }

void Options::SetProfiling(Profiling profiling) { profiling_ = profiling; }

Profiling Options::GetProfiling() const { return profiling_; }

void Options::SetUseHtpPreference(bool use_htp_preference) {
  use_htp_preference_ = use_htp_preference;
}

bool Options::GetUseHtpPreference() const { return use_htp_preference_; }

void Options::SetUseQint16AsQuint16(bool use_qint16_as_quint16) {
  use_qint16_as_quint16_ = use_qint16_as_quint16;
}

bool Options::GetUseQint16AsQuint16() const { return use_qint16_as_quint16_; }

void Options::SetEnableWeightSharing(bool enable_weight_sharing) {
  enable_weight_sharing_ = enable_weight_sharing;
}

bool Options::GetEnableWeightSharing() const { return enable_weight_sharing_; }

void Options::SetUseConvHMX(bool use_conv_hmx) { use_conv_hmx_ = use_conv_hmx; }

bool Options::GetUseConvHMX() const { return use_conv_hmx_; }

void Options::SetUseFoldReLU(bool use_fold_relu) {
  use_fold_relu_ = use_fold_relu;
}

bool Options::GetUseFoldReLU() const { return use_fold_relu_; }

void Options::SetHtpPerformanceMode(HtpPerformanceMode htp_performance_mode) {
  htp_performance_mode_ = htp_performance_mode;
}

HtpPerformanceMode Options::GetHtpPerformanceMode() const {
  return htp_performance_mode_;
}

void Options::SetDspPerformanceMode(DspPerformanceMode dsp_performance_mode) {
  dsp_performance_mode_ = dsp_performance_mode;
}

DspPerformanceMode Options::GetDspPerformanceMode() const {
  return dsp_performance_mode_;
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

absl::string_view Options::GetDlcDir() const { return dlc_dir_; }

void Options::SetDlcDir(absl::string_view dlc_dir) { dlc_dir_ = dlc_dir; }

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

GraphPriority Options::GetGraphPriority() const { return graph_priority_; }

void Options::SetGraphPriority(GraphPriority graph_priority) {
  graph_priority_ = graph_priority;
}

absl::string_view Options::GetSaverOutputDir() const {
  return saver_output_dir_;
}

void Options::SetSaverOutputDir(absl::string_view saver_output_dir) {
  saver_output_dir_ = saver_output_dir;
}

std::string Options::Dump() const {
  static constexpr absl::string_view kQnnOptionsDumpFormat =
      "\
::qnn::Options:\n\
LogLevel: %d\n\
BackendType: %d\n\
Profiling: %d\n\
UseHtpPreference: %v\n\
UseQint16AsQuint16: %v\n\
EnableWeightSharing: %v\n\
UseConvHMX: %v\n\
UseFoldReLU: %v\n\
HtpPerformanceMode: %d\n\
DspPerformanceMode: %d\n\
DumpTensorIds: %s\n\
IrJsonDir: %s\n\
DlcDir: %s\n\
VtcmSize: %d\n\
HvxThread: %d\n\
OptimizationLevel: %d\n\
GraphPriority: %d\n\
SaverOutputDir: %s\n";  // NOLINT

  std::string dump_tensor_ids = absl::StrJoin(dump_tensor_ids_, ",");

  return absl::StrFormat(
      kQnnOptionsDumpFormat, log_level_, backend_type_, profiling_,
      use_htp_preference_, use_qint16_as_quint16_, enable_weight_sharing_,
      use_conv_hmx_, use_fold_relu_, htp_performance_mode_,
      dsp_performance_mode_, dump_tensor_ids, ir_json_dir_, dlc_dir_,
      vtcm_size_, num_hvx_threads_, optimization_level_, graph_priority_,
      saver_output_dir_);
}

QnnLog_Callback_t GetDefaultStdOutLogger() { return DefaultStdOutLogger; }

}  // namespace qnn
