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

#include "litert/cc/options/litert_qualcomm_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
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

void QualcommOptions::SetLogLevel(QualcommOptions::LogLevel log_level) {
  internal::AssertOk(LiteRtQualcommOptionsSetLogLevel, Data(), log_level);
}

QualcommOptions::LogLevel QualcommOptions::GetLogLevel() {
  QualcommOptions::LogLevel log_level;
  internal::AssertOk(LiteRtQualcommOptionsGetLogLevel, Data(), &log_level);
  return log_level;
}

void QualcommOptions::SetPowerMode(QualcommOptions::PowerMode power_mode) {
  internal::AssertOk(LiteRtQualcommOptionsSetPowerMode, Data(), power_mode);
}

QualcommOptions::PowerMode QualcommOptions::GetPowerMode() {
  QualcommOptions::PowerMode power_mode;
  internal::AssertOk(LiteRtQualcommOptionsGetPowerMode, Data(), &power_mode);
  return power_mode;
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

Expected<QualcommOptions> QualcommOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return QualcommOptions(options.Get(), OwnHandle::kNo);
}

namespace {}  // namespace

}  // namespace litert::qualcomm
