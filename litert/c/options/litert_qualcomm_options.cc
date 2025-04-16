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

#include "litert/c/options/litert_qualcomm_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

struct LiteRtQualcommOptionsT {
  LiteRtQualcommOptionsLogLevel log_level = kLiteRtQualcommLogLevelInfo;
  bool enable_weight_sharing = true;
  LiteRtQualcommOptionsPowerMode power_mode =
      kLiteRtQualcommPowerModePerformance;
};

LiteRtStatus LiteRtQualcommOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtQualcommOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtQualcommOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtQualcommOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtQualcommOptionsGetIdentifier() { return "qualcomm"; }

LiteRtStatus LiteRtQualcommOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtQualcommOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtQualcommOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtQualcommOptions>(payload);

  return kLiteRtStatusOk;
}

// GLOBAL OPTIONS //////////////////////////////////////////////////////////////

// log_level -------------------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->log_level = log_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel* log_level) {
  if (log_level == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *log_level = options->log_level;

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// enable_weight_sharing -------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetEnableWeightSharing(
    LiteRtQualcommOptions options, bool enable_weight_sharing) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->enable_weight_sharing = enable_weight_sharing;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetEnableWeightSharing(
    LiteRtQualcommOptions options, bool* enable_weight_sharing) {
  if (enable_weight_sharing == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *enable_weight_sharing = options->enable_weight_sharing;

  return kLiteRtStatusOk;
}

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

// power_mode ------------------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode power_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->power_mode = power_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode* power_mode) {
  if (power_mode == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *power_mode = options->power_mode;

  return kLiteRtStatusOk;
}

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
