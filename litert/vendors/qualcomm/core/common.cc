// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <charconv>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#endif  // defined(__ANDROID__)

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace qnn {
namespace {

#if defined(__ANDROID__)
int GetAndroidLogPriority(QnnLog_Level_t level) {
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      return ANDROID_LOG_ERROR;
    case QNN_LOG_LEVEL_WARN:
      return ANDROID_LOG_WARN;
    case QNN_LOG_LEVEL_INFO:
      return ANDROID_LOG_INFO;
    case QNN_LOG_LEVEL_VERBOSE:
      return ANDROID_LOG_VERBOSE;
    case QNN_LOG_LEVEL_DEBUG:
      return ANDROID_LOG_DEBUG;
    case QNN_LOG_LEVEL_MAX:
      return ANDROID_LOG_UNKNOWN;
  }
}
#endif  // defined(__ANDROID__)

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

#if defined(__ANDROID__)
  // Log to Android logcat.
  va_list argp_copy;
  va_copy(argp_copy, argp);
  __android_log_vprint(GetAndroidLogPriority(level), "qnn", fmt, argp_copy);
  va_end(argp_copy);
#endif  // defined(__ANDROID__)

  // Also print to stdout for console output.
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

// AbslStringify overloads so %v renders these enums as "Name(N)" in Dump().
// In namespace qnn, not the anonymous namespace above -- ADL skips anonymous
// namespaces, so %v would silently print the integer instead. Kept out of
// common.h so the rendering stays local to this TU. No default case: a new
// enumerator then trips -Werror=switch.
template <typename Sink>
void AbslStringify(Sink& sink, LogLevel v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case LogLevel::kOff:
      name = "Off";
      break;
    case LogLevel::kError:
      name = "Error";
      break;
    case LogLevel::kWarn:
      name = "Warn";
      break;
    case LogLevel::kInfo:
      name = "Info";
      break;
    case LogLevel::kVerbose:
      name = "Verbose";
      break;
    case LogLevel::kDebug:
      name = "Debug";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, BackendType v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case BackendType::kUndefinedBackend:
      name = "Undefined";
      break;
    case BackendType::kGpuBackend:
      name = "Gpu";
      break;
    case BackendType::kHtpBackend:
      name = "Htp";
      break;
    case BackendType::kDspBackend:
      name = "Dsp";
      break;
    case BackendType::kIrBackend:
      name = "Ir";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, Profiling v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case Profiling::kOff:
      name = "Off";
      break;
    case Profiling::kBasic:
      name = "Basic";
      break;
    case Profiling::kDetailed:
      name = "Detailed";
      break;
    case Profiling::kLinting:
      name = "Linting";
      break;
    case Profiling::kOptrace:
      name = "Optrace";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, HtpPerformanceMode v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case HtpPerformanceMode::kDefault:
      name = "Default";
      break;
    case HtpPerformanceMode::kSustainedHighPerformance:
      name = "SustainedHighPerformance";
      break;
    case HtpPerformanceMode::kBurst:
      name = "Burst";
      break;
    case HtpPerformanceMode::kHighPerformance:
      name = "HighPerformance";
      break;
    case HtpPerformanceMode::kPowerSaver:
      name = "PowerSaver";
      break;
    case HtpPerformanceMode::kLowPowerSaver:
      name = "LowPowerSaver";
      break;
    case HtpPerformanceMode::kHighPowerSaver:
      name = "HighPowerSaver";
      break;
    case HtpPerformanceMode::kLowBalanced:
      name = "LowBalanced";
      break;
    case HtpPerformanceMode::kBalanced:
      name = "Balanced";
      break;
    case HtpPerformanceMode::kExtremePowerSaver:
      name = "ExtremePowerSaver";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, DspPerformanceMode v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case DspPerformanceMode::kDefault:
      name = "Default";
      break;
    case DspPerformanceMode::kSustainedHighPerformance:
      name = "SustainedHighPerformance";
      break;
    case DspPerformanceMode::kBurst:
      name = "Burst";
      break;
    case DspPerformanceMode::kHighPerformance:
      name = "HighPerformance";
      break;
    case DspPerformanceMode::kPowerSaver:
      name = "PowerSaver";
      break;
    case DspPerformanceMode::kLowPowerSaver:
      name = "LowPowerSaver";
      break;
    case DspPerformanceMode::kHighPowerSaver:
      name = "HighPowerSaver";
      break;
    case DspPerformanceMode::kLowBalanced:
      name = "LowBalanced";
      break;
    case DspPerformanceMode::kBalanced:
      name = "Balanced";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, HtpPerfCtrlMode v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case HtpPerfCtrlMode::kManual:
      name = "Manual";
      break;
    case HtpPerfCtrlMode::kAuto:
      name = "Auto";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, DspPerfCtrlMode v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case DspPerfCtrlMode::kManual:
      name = "Manual";
      break;
    case DspPerfCtrlMode::kAuto:
      name = "Auto";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, OptimizationLevel v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case OptimizationLevel::kHtpOptimizeForInference:
      name = "HtpOptimizeForInference";
      break;
    case OptimizationLevel::kHtpOptimizeForPrepare:
      name = "HtpOptimizeForPrepare";
      break;
    case OptimizationLevel::kHtpOptimizeForInferenceO3:
      name = "HtpOptimizeForInferenceO3";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, GraphPriority v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case GraphPriority::kDefault:
      name = "Default";
      break;
    case GraphPriority::kLow:
      name = "Low";
      break;
    case GraphPriority::kNormal:
      name = "Normal";
      break;
    case GraphPriority::kNormalHigh:
      name = "NormalHigh";
      break;
    case GraphPriority::kHigh:
      name = "High";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, GpuPrecision v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case GpuPrecision::kUserProvided:
      name = "UserProvided";
      break;
    case GpuPrecision::kFp32:
      name = "Fp32";
      break;
    case GpuPrecision::kFp16:
      name = "Fp16";
      break;
    case GpuPrecision::kHybrid:
      name = "Hybrid";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, GpuPerformanceMode v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case GpuPerformanceMode::kDefault:
      name = "Default";
      break;
    case GpuPerformanceMode::kHigh:
      name = "High";
      break;
    case GpuPerformanceMode::kNormal:
      name = "Normal";
      break;
    case GpuPerformanceMode::kLow:
      name = "Low";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

template <typename Sink>
void AbslStringify(Sink& sink, GraphIOTensorMemType v) {
  absl::string_view name = "Unknown";
  switch (v) {
    case GraphIOTensorMemType::kRaw:
      name = "Raw";
      break;
    case GraphIOTensorMemType::kMemHandle:
      name = "MemHandle";
      break;
  }
  absl::Format(&sink, "%s(%d)", name, static_cast<int>(v));
}

void Options::SetLogLevel(LogLevel log_level) { log_level_ = log_level; }

LogLevel Options::GetLogLevel() const { return log_level_; }

void Options::SetBackendType(BackendType backend_type) {
  backend_type_ = backend_type;
}

BackendType Options::GetBackendType() const { return backend_type_; }

void Options::SetProfiling(Profiling profiling) { profiling_ = profiling; }

Profiling Options::GetProfiling() const { return profiling_; }

void Options::SetUseInt64BiasAsInt32(bool use_int64_bias_as_int32) {
  use_int64_bias_as_int32_ = use_int64_bias_as_int32;
}

bool Options::GetUseInt64BiasAsInt32() const {
  return use_int64_bias_as_int32_;
}

void Options::SetEnableWeightSharing(bool enable_weight_sharing) {
  enable_weight_sharing_ = enable_weight_sharing;
  // Mutually exclusive with DLBC weights (QAIRT 2.36+); weight sharing wins.
  // Enforced here too so the outcome is order-independent.
  if (enable_weight_sharing_) {
    htp_dlbc_weights_ = false;
  }
}

bool Options::GetEnableWeightSharing() const { return enable_weight_sharing_; }

void Options::SetEnableJustInTime(bool enable_just_in_time) {
  enable_just_in_time_ = enable_just_in_time;
}

bool Options::GetEnableJustInTime() const { return enable_just_in_time_; }

void Options::SetUseConvHMX(bool use_conv_hmx) { use_conv_hmx_ = use_conv_hmx; }

bool Options::GetUseConvHMX() const { return use_conv_hmx_; }

void Options::SetUseFoldReLU(bool use_fold_relu) {
  use_fold_relu_ = use_fold_relu;
}

bool Options::GetUseFoldReLU() const { return use_fold_relu_; }

void Options::SetHtpPPoint(std::int32_t htp_p_point) {
  htp_p_point_ = htp_p_point;
}

std::int32_t Options::GetHtpPPoint() const { return htp_p_point_; }

void Options::SetHtpDlbc(bool htp_dlbc) { htp_dlbc_ = htp_dlbc; }

bool Options::GetHtpDlbc() const { return htp_dlbc_; }

void Options::SetHtpDlbcWeights(bool htp_dlbc_weights) {
  // DLBC weights is mutually exclusive with weight sharing (QAIRT 2.36+).
  htp_dlbc_weights_ = htp_dlbc_weights && !enable_weight_sharing_;
}

bool Options::GetHtpDlbcWeights() const { return htp_dlbc_weights_; }

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

void Options::SetHtpPerfCtrlMode(HtpPerfCtrlMode htp_perf_ctrl_mode) {
  htp_perf_ctrl_mode_ = htp_perf_ctrl_mode;
}

HtpPerfCtrlMode Options::GetHtpPerfCtrlMode() const {
  return htp_perf_ctrl_mode_;
}

void Options::SetDspPerfCtrlMode(DspPerfCtrlMode dsp_perf_ctrl_mode) {
  dsp_perf_ctrl_mode_ = dsp_perf_ctrl_mode;
}

DspPerfCtrlMode Options::GetDspPerfCtrlMode() const {
  return dsp_perf_ctrl_mode_;
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

void Options::SetGpuPrecision(GpuPrecision gpu_precision) {
  gpu_precision_ = gpu_precision;
}

GpuPrecision Options::GetGpuPrecision() const { return gpu_precision_; }

void Options::SetGpuPerformanceMode(GpuPerformanceMode gpu_performance_mode) {
  gpu_performance_mode_ = gpu_performance_mode;
}

GpuPerformanceMode Options::GetGpuPerformanceMode() const {
  return gpu_performance_mode_;
}

absl::string_view Options::GetSaverOutputDir() const {
  return saver_output_dir_;
}

void Options::SetSaverOutputDir(absl::string_view saver_output_dir) {
  saver_output_dir_ = saver_output_dir;
}

void Options::SetGraphIOTensorMemType(GraphIOTensorMemType mem_type) {
  graph_io_tensor_mem_type_ = mem_type;
}

GraphIOTensorMemType Options::GetGraphIOTensorMemType() const {
  return graph_io_tensor_mem_type_;
}

absl::string_view Options::GetSchematicDir() const { return schematic_dir_; }

void Options::SetSchematicDir(absl::string_view schematic_dir) {
  schematic_dir_ = schematic_dir;
}

void Options::SetCustomOpPackage(absl::string_view name,
                                 absl::string_view interface_provider,
                                 absl::string_view compile_package_path,
                                 absl::string_view dispatch_package_path,
                                 absl::string_view target) {
  custom_op_package_.name = name;
  custom_op_package_.interface_provider = interface_provider;
  custom_op_package_.compile_package_path = compile_package_path;
  custom_op_package_.dispatch_package_path = dispatch_package_path;
  custom_op_package_.target = target;
}

const CustomOpPackage& Options::GetCustomOpPackage() const {
  return custom_op_package_;
}

std::string Options::Dump() const {
  // Grouped by category; append a field() line to the right section to add one.
  std::string out =
      "\n"
      "+------------------------------------------+\n"
      "|              ::qnn::Options              |\n"
      "+------------------------------------------+\n";

  // Renders "<indent><name> : <value>". indent is 2 for top-level options, 4
  // for nested struct members; the name width (26 - indent) keeps the " : "
  // aligned across both. %v renders enums as "Name(N)" via the overloads above.
  // One option per line keeps edits conflict-free.
  const auto field = [&out](int indent, absl::string_view name,
                            const auto& value) {
    absl::StrAppendFormat(&out, "%*s%-*s : %v\n", indent, "", 26 - indent, name,
                          value);
  };

  // --- GENERAL ---
  absl::StrAppend(&out, "[GENERAL]\n");
  field(2, "LogLevel", log_level_);
  field(2, "BackendType", backend_type_);
  field(2, "Profiling", profiling_);
  field(2, "UseInt64BiasAsInt32", use_int64_bias_as_int32_);
  field(2, "EnableWeightSharing", enable_weight_sharing_);
  field(2, "EnableJustInTime", enable_just_in_time_);
  field(2, "GraphPriority", graph_priority_);
  field(2, "GraphIOTensorMemType", graph_io_tensor_mem_type_);
  field(2, "CustomOpPackage", absl::string_view());
  field(4, "name", custom_op_package_.name);
  field(4, "interface_provider", custom_op_package_.interface_provider);
  field(4, "compile_package_path", custom_op_package_.compile_package_path);
  field(4, "dispatch_package_path", custom_op_package_.dispatch_package_path);
  field(4, "target", custom_op_package_.target);

  // --- HTP ---
  absl::StrAppend(&out, "[HTP]\n");
  field(2, "UseConvHMX", use_conv_hmx_);
  field(2, "UseFoldReLU", use_fold_relu_);
  field(2, "HtpDlbc", htp_dlbc_);
  field(2, "HtpDlbcWeights", htp_dlbc_weights_);
  field(2, "HtpPPoint", htp_p_point_);
  field(2, "HtpPerformanceMode", htp_performance_mode_);
  field(2, "HtpPerfCtrlMode", htp_perf_ctrl_mode_);
  field(2, "VtcmSize", vtcm_size_);
  field(2, "NumHvxThreads", num_hvx_threads_);
  field(2, "OptimizationLevel", optimization_level_);

  // --- IR ---
  absl::StrAppend(&out, "[IR]\n");
  field(2, "IrJsonDir", ir_json_dir_);
  field(2, "DlcDir", dlc_dir_);

  // --- SAVER ---
  absl::StrAppend(&out, "[SAVER]\n");
  field(2, "SaverOutputDir", saver_output_dir_);

  // --- DSP ---
  absl::StrAppend(&out, "[DSP]\n");
  field(2, "DspPerformanceMode", dsp_performance_mode_);
  field(2, "DspPerfCtrlMode", dsp_perf_ctrl_mode_);

  // --- GPU ---
  absl::StrAppend(&out, "[GPU]\n");
  field(2, "GpuPerformanceMode", gpu_performance_mode_);
  field(2, "GpuPrecision", gpu_precision_);

  // --- DEBUG ---
  absl::StrAppend(&out, "[DEBUG]\n");
  field(2, "DumpTensorIds", absl::StrJoin(dump_tensor_ids_, ","));
  field(2, "SchematicDir", schematic_dir_);

  // Strip the trailing space empty values leave after " : ".
  return absl::StrReplaceAll(out, {{" \n", "\n"}});
}

QnnLog_Callback_t GetDefaultStdOutLogger() { return DefaultStdOutLogger; }

std::optional<SdkVersion> ParseSdkVersion(const char* build_id) {
  if (!build_id) return std::nullopt;

  std::string_view version_str = build_id;

  // Check for and remove the 'v' prefix.
  if (version_str.empty() || version_str.front() != 'v') {
    return std::nullopt;
  }
  version_str.remove_prefix(1);

  SdkVersion version{};
  const char* current = version_str.data();
  const char* const end = version_str.data() + version_str.size();

  auto parse_component = [&current, &end](int& component) {
    auto [ptr, ec] = std::from_chars(current, end, component);
    if (ec != std::errc()) {
      return false;
    }
    current = ptr;
    return true;
  };

  // Parse major, minor, and patch versions, checking for dots in between.
  if (!parse_component(version.major)) return std::nullopt;

  if (current == end || *current++ != '.') return std::nullopt;
  if (!parse_component(version.minor)) return std::nullopt;

  if (current == end || *current++ != '.') return std::nullopt;
  if (!parse_component(version.patch)) return std::nullopt;

  return version;
}

}  // namespace qnn
