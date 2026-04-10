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

#include "litert/cc/options/litert_qualcomm_options.h"

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

// Create a new QualcommOptions instance.
Expected<QualcommOptions> QualcommOptions::Create() {
  LrtQualcommOptions c_options;
  LITERT_RETURN_IF_ERROR(LrtCreateQualcommOptions(&c_options));
  return QualcommOptions(c_options);
}

void QualcommOptions::SetLogLevel(LogLevel log_level) {
  LrtQualcommOptionsSetLogLevel(
      options_, static_cast<LrtQualcommOptionsLogLevel>(log_level));
}

QualcommOptions::LogLevel QualcommOptions::GetLogLevel() {
  LrtQualcommOptionsLogLevel val;
  auto status = LrtQualcommOptionsGetLogLevel(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return LogLevel::kInfo;
  }
  return static_cast<LogLevel>(val);
}

void QualcommOptions::SetHtpPerformanceMode(
    HtpPerformanceMode htp_performance_mode) {
  LrtQualcommOptionsSetHtpPerformanceMode(
      options_,
      static_cast<LrtQualcommOptionsHtpPerformanceMode>(htp_performance_mode));
}

QualcommOptions::HtpPerformanceMode QualcommOptions::GetHtpPerformanceMode() {
  LrtQualcommOptionsHtpPerformanceMode val;
  auto status = LrtQualcommOptionsGetHtpPerformanceMode(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return HtpPerformanceMode::kDefault;
  }
  return static_cast<HtpPerformanceMode>(val);
}

void QualcommOptions::SetDspPerformanceMode(
    DspPerformanceMode dsp_performance_mode) {
  LrtQualcommOptionsSetDspPerformanceMode(
      options_,
      static_cast<LrtQualcommOptionsDspPerformanceMode>(dsp_performance_mode));
}

QualcommOptions::DspPerformanceMode QualcommOptions::GetDspPerformanceMode() {
  LrtQualcommOptionsDspPerformanceMode val;
  auto status = LrtQualcommOptionsGetDspPerformanceMode(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return DspPerformanceMode::kDefault;
  }
  return static_cast<DspPerformanceMode>(val);
}

void QualcommOptions::SetUseHtpPreference(bool use_htp_preference) {
  LrtQualcommOptionsSetUseHtpPreference(options_, use_htp_preference);
}

bool QualcommOptions::GetUseHtpPreference() {
  bool val;
  auto status = LrtQualcommOptionsGetUseHtpPreference(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return false;
  }
  return val;
}

void QualcommOptions::SetUseQint16AsQuint16(bool use_qin16_as_quint16) {
  LrtQualcommOptionsSetUseQint16AsQuint16(options_, use_qin16_as_quint16);
}

bool QualcommOptions::GetUseQint16AsQuint16() {
  bool val;
  auto status = LrtQualcommOptionsGetUseQint16AsQuint16(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return false;
  }
  return val;
}

void QualcommOptions::SetUseInt64BiasAsInt32(bool use_int64_bias_as_int32) {
  LrtQualcommOptionsSetUseInt64BiasAsInt32(options_, use_int64_bias_as_int32);
}

bool QualcommOptions::GetUseInt64BiasAsInt32() {
  bool val;
  auto status = LrtQualcommOptionsGetUseInt64BiasAsInt32(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return true;
  }
  return val;
}

void QualcommOptions::SetEnableWeightSharing(bool weight_sharing_enabled) {
  LrtQualcommOptionsSetEnableWeightSharing(options_, weight_sharing_enabled);
}

bool QualcommOptions::GetEnableWeightSharing() {
  bool val;
  auto status = LrtQualcommOptionsGetEnableWeightSharing(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return false;
  }
  return val;
}

void QualcommOptions::SetUseConvHMX(bool use_conv_hmx) {
  LrtQualcommOptionsSetUseConvHMX(options_, use_conv_hmx);
}

bool QualcommOptions::GetUseConvHMX() {
  bool val;
  auto status = LrtQualcommOptionsGetUseConvHMX(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return true;
  }
  return val;
}

void QualcommOptions::SetUseFoldReLU(bool use_fold_relu) {
  LrtQualcommOptionsSetUseFoldReLU(options_, use_fold_relu);
}

bool QualcommOptions::GetUseFoldReLU() {
  bool val;
  auto status = LrtQualcommOptionsGetUseFoldReLU(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return true;
  }
  return val;
}

void QualcommOptions::SetProfiling(Profiling profiling) {
  LrtQualcommOptionsSetProfiling(
      options_, static_cast<LrtQualcommOptionsProfiling>(profiling));
}

QualcommOptions::Profiling QualcommOptions::GetProfiling() {
  LrtQualcommOptionsProfiling val;
  auto status = LrtQualcommOptionsGetProfiling(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return Profiling::kOff;
  }
  return static_cast<Profiling>(val);
}

void QualcommOptions::SetDumpTensorIds(const std::vector<std::int32_t>& ids) {
  LrtQualcommOptionsSetDumpTensorIds(options_, ids.data(), ids.size());
}

std::vector<std::int32_t> QualcommOptions::GetDumpTensorIds() {
  const std::int32_t* ids;
  size_t number_of_ids;
  auto status =
      LrtQualcommOptionsGetDumpTensorIds(options_, &ids, &number_of_ids);
  if (status == kLiteRtStatusErrorNotFound) {
    return {};
  }
  return std::vector<std::int32_t>(ids, ids + number_of_ids);
}

void QualcommOptions::SetIrJsonDir(const std::string& ir_json_dir) {
  LrtQualcommOptionsSetIrJsonDir(options_, ir_json_dir.c_str());
}

absl::string_view QualcommOptions::GetIrJsonDir() {
  const char* val;
  auto status = LrtQualcommOptionsGetIrJsonDir(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetDlcDir(const std::string& dlc_dir) {
  LrtQualcommOptionsSetDlcDir(options_, dlc_dir.c_str());
}

absl::string_view QualcommOptions::GetDlcDir() {
  const char* val;
  auto status = LrtQualcommOptionsGetDlcDir(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetVtcmSize(std::uint32_t vtcm_size) {
  LrtQualcommOptionsSetVtcmSize(options_, vtcm_size);
}

std::uint32_t QualcommOptions::GetVtcmSize() {
  std::uint32_t val;
  auto status = LrtQualcommOptionsGetVtcmSize(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return 0;
  }
  return val;
}

void QualcommOptions::SetNumHvxThreads(std::uint32_t num_hvx_threads) {
  LrtQualcommOptionsSetNumHvxThreads(options_, num_hvx_threads);
}

std::uint32_t QualcommOptions::GetNumHvxThreads() {
  std::uint32_t val;
  auto status = LrtQualcommOptionsGetNumHvxThreads(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return 0;
  }
  return val;
}

void QualcommOptions::SetOptimizationLevel(
    OptimizationLevel optimization_level) {
  LrtQualcommOptionsSetOptimizationLevel(
      options_,
      static_cast<LrtQualcommOptionsOptimizationLevel>(optimization_level));
}

QualcommOptions::OptimizationLevel QualcommOptions::GetOptimizationLevel() {
  LrtQualcommOptionsOptimizationLevel val;
  auto status = LrtQualcommOptionsGetOptimizationLevel(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return OptimizationLevel::kOptimizeForInferenceO3;
  }
  return static_cast<OptimizationLevel>(val);
}

void QualcommOptions::SetGraphPriority(GraphPriority graph_priority) {
  LrtQualcommOptionsSetGraphPriority(
      options_, static_cast<LrtQualcommOptionsGraphPriority>(graph_priority));
}

QualcommOptions::GraphPriority QualcommOptions::GetGraphPriority() {
  LrtQualcommOptionsGraphPriority val;
  auto status = LrtQualcommOptionsGetGraphPriority(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return GraphPriority::kDefault;
  }
  return static_cast<GraphPriority>(val);
}

void QualcommOptions::SetBackend(Backend backend) {
  LrtQualcommOptionsSetBackend(options_,
                               static_cast<LrtQualcommOptionsBackend>(backend));
}

QualcommOptions::Backend QualcommOptions::GetBackend() {
  LrtQualcommOptionsBackend val;
  auto status = LrtQualcommOptionsGetBackend(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return Backend::kHtp;
  }
  return static_cast<Backend>(val);
}

void QualcommOptions::SetSaverOutputDir(const std::string& saver_output_dir) {
  LrtQualcommOptionsSetSaverOutputDir(options_, saver_output_dir.c_str());
}

absl::string_view QualcommOptions::GetSaverOutputDir() {
  const char* val;
  auto status = LrtQualcommOptionsGetSaverOutputDir(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetCustomOpPackageName(const std::string& name) {
  LrtQualcommOptionsSetCustomOpPackageName(options_, name.c_str());
}

std::string_view QualcommOptions::GetCustomOpPackageName() {
  const char* val;
  auto status = LrtQualcommOptionsGetCustomOpPackageName(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetCustomOpPackagePath(const std::string& path) {
  LrtQualcommOptionsSetCustomOpPackagePath(options_, path.c_str());
}

std::string_view QualcommOptions::GetCustomOpPackagePath() {
  const char* val;
  auto status = LrtQualcommOptionsGetCustomOpPackagePath(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetCustomOpPackageTarget(const std::string& target) {
  LrtQualcommOptionsSetCustomOpPackageTarget(options_, target.c_str());
}

std::string_view QualcommOptions::GetCustomOpPackageTarget() {
  const char* val;
  auto status = LrtQualcommOptionsGetCustomOpPackageTarget(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

void QualcommOptions::SetCustomOpPackageInterfaceProvider(
    const std::string& interface_provider) {
  LrtQualcommOptionsSetCustomOpPackageInterfaceProvider(
      options_, interface_provider.c_str());
}

std::string_view QualcommOptions::GetCustomOpPackageInterfaceProvider() {
  const char* val;
  auto status =
      LrtQualcommOptionsGetCustomOpPackageInterfaceProvider(options_, &val);
  if (status == kLiteRtStatusErrorNotFound) {
    return "";
  }
  return val;
}

}  // namespace litert::qualcomm
