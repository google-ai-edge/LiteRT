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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_

#include <cstdint>
#include <vector>

#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::qualcomm {

// Wraps a LiteRtQualcommOptions object for convenience.
class QualcommOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() {
    return LiteRtQualcommOptionsGetIdentifier();
  }

  static Expected<QualcommOptions> Create(OpaqueOptions& options);

  static Expected<QualcommOptions> Create();

  void SetLogLevel(LiteRtQualcommOptionsLogLevel log_level);
  LiteRtQualcommOptionsLogLevel GetLogLevel();

  void SetHtpPerformanceMode(
      LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode);
  LiteRtQualcommOptionsHtpPerformanceMode GetHtpPerformanceMode();

  void SetUseHtpPreference(bool use_htp_preference);
  bool GetUseHtpPreference();

  void SetUseQint16AsQuint16(bool use_qin16_as_quint16);
  bool GetUseQint16AsQuint16();

  void SetEnableWeightSharing(bool weight_sharing_enabled);
  bool GetEnableWeightSharing();

  void SetProfiling(LiteRtQualcommOptionsProfiling profiling);
  LiteRtQualcommOptionsProfiling GetProfiling();

  void SetDumpTensorIds(const std::vector<std::int32_t>& ids);
  std::vector<std::int32_t> GetDumpTensorIds();

  void SetQnnJsonPath(const char* profiling);
  const char* GetQnnJsonPath();

 private:
  LiteRtQualcommOptions Data() const;
};

}  // namespace litert::qualcomm

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
