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
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert::qualcomm {

LiteRtQualcommOptions QualcommOptions::Data() const {
  LiteRtQualcommOptions options;
  internal::AssertOk(LiteRtQualcommOptionsGet, Get(), &options);
  return options;
}

Expected<QualcommOptions> QualcommOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtQualcommOptionsCreate(&options));
  return QualcommOptions(options, litert::OwnHandle::kYes);
}

void QualcommOptions::SetLogLevel(QualcommOptions::LogLevel log_level) {
  internal::AssertOk(LiteRtQualcommOptionsSetLogLevel, Data(),
                     static_cast<LiteRtQualcommOptionsLogLevel>(log_level));
}

QualcommOptions::LogLevel QualcommOptions::GetLogLevel() {
  LiteRtQualcommOptionsLogLevel log_level;
  internal::AssertOk(LiteRtQualcommOptionsGetLogLevel, Data(), &log_level);
  return static_cast<QualcommOptions::LogLevel>(log_level);
}

void QualcommOptions::SetHtpPerformanceMode(
    QualcommOptions::HtpPerformanceMode htp_performance_mode) {
  internal::AssertOk(LiteRtQualcommOptionsSetHtpPerformanceMode, Data(),
                     static_cast<LiteRtQualcommOptionsHtpPerformanceMode>(
                         htp_performance_mode));
}

QualcommOptions::HtpPerformanceMode QualcommOptions::GetHtpPerformanceMode() {
  LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode;
  internal::AssertOk(LiteRtQualcommOptionsGetHtpPerformanceMode, Data(),
                     &htp_performance_mode);
  return static_cast<QualcommOptions::HtpPerformanceMode>(htp_performance_mode);
}

void QualcommOptions::SetEnableWeightSharing(bool weight_sharing_enabled) {
  internal::AssertOk(LiteRtQualcommOptionsSetEnableWeightSharing, Data(),
                     weight_sharing_enabled);
}

bool QualcommOptions::GetEnableWeightSharing() {
  bool enable_weight_sharing;
  internal::AssertOk(LiteRtQualcommOptionsGetEnableWeightSharing, Data(),
                     &enable_weight_sharing);
  return enable_weight_sharing;
}

void QualcommOptions::SetUseHtpPreference(bool use_htp_preference) {
  internal::AssertOk(LiteRtQualcommOptionsSetUseHtpPreference, Data(),
                     use_htp_preference);
}

bool QualcommOptions::GetUseHtpPreference() {
  bool use_htp_preference;
  internal::AssertOk(LiteRtQualcommOptionsGetUseHtpPreference, Data(),
                     &use_htp_preference);
  return use_htp_preference;
}

void QualcommOptions::SetUseQint16AsQuint16(bool use_qin16_as_quint16) {
  internal::AssertOk(LiteRtQualcommOptionsSetUseQint16AsQuint16, Data(),
                     use_qin16_as_quint16);
}

bool QualcommOptions::GetUseQint16AsQuint16() {
  bool use_qin16_as_quint16;
  internal::AssertOk(LiteRtQualcommOptionsGetUseQint16AsQuint16, Data(),
                     &use_qin16_as_quint16);
  return use_qin16_as_quint16;
}

void QualcommOptions::SetProfiling(QualcommOptions::Profiling profiling) {
  internal::AssertOk(LiteRtQualcommOptionsSetProfiling, Data(),
                     static_cast<LiteRtQualcommOptionsProfiling>(profiling));
}

QualcommOptions::Profiling QualcommOptions::GetProfiling() {
  LiteRtQualcommOptionsProfiling profiling;
  internal::AssertOk(LiteRtQualcommOptionsGetProfiling, Data(), &profiling);
  return static_cast<QualcommOptions::Profiling>(profiling);
}

void QualcommOptions::SetDumpTensorIds(const std::vector<std::int32_t>& ids) {
  internal::AssertOk(LiteRtQualcommOptionsSetDumpTensorIds, Data(), ids.data(),
                     ids.size());
}

std::vector<std::int32_t> QualcommOptions::GetDumpTensorIds() {
  std::vector<std::int32_t> dump_ids;
  std::int32_t* ids = nullptr;
  std::uint32_t number_of_ids = 0;
  internal::AssertOk(LiteRtQualcommOptionsGetDumpTensorIds, Data(), &ids,
                     &number_of_ids);
  if (ids == nullptr) {
    return dump_ids;
  }
  dump_ids.reserve(number_of_ids);
  for (size_t i = 0; i < number_of_ids; i++) {
    dump_ids.emplace_back(ids[i]);
  }
  return dump_ids;
}

void QualcommOptions::SetIrJsonDir(const std::string& ir_json_dir) {
  internal::AssertOk(LiteRtQualcommOptionsSetIrJsonDir, Data(),
                     ir_json_dir.c_str());
}

absl::string_view QualcommOptions::GetIrJsonDir() {
  const char* ir_json_dir;
  internal::AssertOk(LiteRtQualcommOptionsGetIrJsonDir, Data(), &ir_json_dir);
  return absl::string_view(ir_json_dir);
}

void QualcommOptions::SetDlcDir(const std::string& dlc_dir) {
  internal::AssertOk(LiteRtQualcommOptionsSetDlcDir, Data(), dlc_dir.c_str());
}

absl::string_view QualcommOptions::GetDlcDir() {
  const char* dlc_dir;
  internal::AssertOk(LiteRtQualcommOptionsGetDlcDir, Data(), &dlc_dir);
  return absl::string_view(dlc_dir);
}

void QualcommOptions::SetVtcmSize(std::uint32_t vtcm_size) {
  internal::AssertOk(LiteRtQualcommOptionsSetVtcmSize, Data(), vtcm_size);
}

std::uint32_t QualcommOptions::GetVtcmSize() {
  std::uint32_t vtcm_size;
  internal::AssertOk(LiteRtQualcommOptionsGetVtcmSize, Data(), &vtcm_size);
  return vtcm_size;
}

void QualcommOptions::SetNumHvxThreads(std::uint32_t num_hvx_threads) {
  internal::AssertOk(LiteRtQualcommOptionsSetNumHvxThreads, Data(),
                     num_hvx_threads);
}

std::uint32_t QualcommOptions::GetNumHvxThreads() {
  std::uint32_t num_hvx_threads;
  internal::AssertOk(LiteRtQualcommOptionsGetNumHvxThreads, Data(),
                     &num_hvx_threads);
  return num_hvx_threads;
}

void QualcommOptions::SetOptimizationLevel(
    QualcommOptions::OptimizationLevel optimization_level) {
  internal::AssertOk(
      LiteRtQualcommOptionsSetOptimizationLevel, Data(),
      static_cast<LiteRtQualcommOptionsOptimizationLevel>(optimization_level));
}

QualcommOptions::OptimizationLevel QualcommOptions::GetOptimizationLevel() {
  LiteRtQualcommOptionsOptimizationLevel optimization_level;
  internal::AssertOk(LiteRtQualcommOptionsGetOptimizationLevel, Data(),
                     &optimization_level);
  return static_cast<QualcommOptions::OptimizationLevel>(optimization_level);
}

void QualcommOptions::SetGraphPriority(
    QualcommOptions::GraphPriority graph_priority) {
  internal::AssertOk(
      LiteRtQualcommOptionsSetGraphPriority, Data(),
      static_cast<LiteRtQualcommOptionsGraphPriority>(graph_priority));
}

QualcommOptions::GraphPriority QualcommOptions::GetGraphPriority() {
  LiteRtQualcommOptionsGraphPriority graph_priority;
  internal::AssertOk(LiteRtQualcommOptionsGetGraphPriority, Data(),
                     &graph_priority);
  return static_cast<QualcommOptions::GraphPriority>(graph_priority);
}

void QualcommOptions::SetUseConvHMX(bool use_conv_hmx) {
  internal::AssertOk(LiteRtQualcommOptionsSetUseConvHMX, Data(), use_conv_hmx);
}

bool QualcommOptions::GetUseConvHMX() {
  bool use_conv_hmx;
  internal::AssertOk(LiteRtQualcommOptionsGetUseConvHMX, Data(), &use_conv_hmx);
  return use_conv_hmx;
}

void QualcommOptions::SetUseFoldReLU(bool use_fold_relu) {
  internal::AssertOk(LiteRtQualcommOptionsSetUseFoldReLU, Data(),
                     use_fold_relu);
}

bool QualcommOptions::GetUseFoldReLU() {
  bool use_fold_relu;
  internal::AssertOk(LiteRtQualcommOptionsGetUseFoldReLU, Data(),
                     &use_fold_relu);
  return use_fold_relu;
}

Expected<QualcommOptions> QualcommOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return QualcommOptions(options.Get(), OwnHandle::kNo);
}

}  // namespace litert::qualcomm
