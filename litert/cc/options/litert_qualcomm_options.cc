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
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
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

void QualcommOptions::SetLogLevel(LiteRtQualcommOptionsLogLevel log_level) {
  internal::AssertOk(LiteRtQualcommOptionsSetLogLevel, Data(), log_level);
}

LiteRtQualcommOptionsLogLevel QualcommOptions::GetLogLevel() {
  LiteRtQualcommOptionsLogLevel log_level;
  internal::AssertOk(LiteRtQualcommOptionsGetLogLevel, Data(), &log_level);
  return log_level;
}

void QualcommOptions::SetHtpPerformanceMode(
    LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode) {
  internal::AssertOk(LiteRtQualcommOptionsSetHtpPerformanceMode, Data(),
                     htp_performance_mode);
}

LiteRtQualcommOptionsHtpPerformanceMode
QualcommOptions::GetHtpPerformanceMode() {
  LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode;
  internal::AssertOk(LiteRtQualcommOptionsGetHtpPerformanceMode, Data(),
                     &htp_performance_mode);
  return htp_performance_mode;
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

void QualcommOptions::SetProfiling(LiteRtQualcommOptionsProfiling profiling) {
  internal::AssertOk(LiteRtQualcommOptionsSetProfiling, Data(), profiling);
}

LiteRtQualcommOptionsProfiling QualcommOptions::GetProfiling() {
  LiteRtQualcommOptionsProfiling profiling;
  internal::AssertOk(LiteRtQualcommOptionsGetProfiling, Data(), &profiling);
  return profiling;
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

void QualcommOptions::SetQnnJsonPath(const char* qnn_json_path) {
  internal::AssertOk(LiteRtQualcommOptionsSetQnnJsonPath, Data(),
                     qnn_json_path);
}

const char* QualcommOptions::GetQnnJsonPath() {
  const char* qnn_json_path;
  internal::AssertOk(LiteRtQualcommOptionsGetQnnJsonPath, Data(),
                     &qnn_json_path);
  return qnn_json_path;
}

Expected<QualcommOptions> QualcommOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return QualcommOptions(options.Get(), OwnHandle::kNo);
}

}  // namespace litert::qualcomm
