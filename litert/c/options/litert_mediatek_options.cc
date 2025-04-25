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
#include "litert/c/options/litert_mediatek_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
struct LiteRtMediatekOptionsT {
  LiteRtMediatekOptionsNeronSDKVersionType neron_sdk_version =
      kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8;
};
LiteRtStatus LiteRtMediatekOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtMediatekOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtMediatekOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtMediatekOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}
const char* LiteRtMediatekOptionsGetIdentifier() { return "mediatek"; }

LiteRtStatus LiteRtMediatekOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtMediatekOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtMediatekOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtMediatekOptionsT*>(payload);
  return kLiteRtStatusOk;
}
// COMPILATION OPTIONS /////////////////////////////////////////////////////////
// float_truncation_type -------------------------------------------------------
LiteRtStatus LiteRtMediatekOptionsSetNeronSDKVersionType(
    LiteRtMediatekOptionsT* options,
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->neron_sdk_version = sdk_version_type;
  return kLiteRtStatusOk;
}
LiteRtStatus LiteRtMediatekOptionsGetNeronSDKVersionType(
    LiteRtMediatekOptionsT* options,
    LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type) {
  if (options == nullptr || sdk_version_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version_type = options->neron_sdk_version;
  return kLiteRtStatusOk;
}

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
namespace litert::mediatek {
const char* MediatekOptions::Discriminator() {
  return LiteRtMediatekOptionsGetIdentifier();
}
Expected<MediatekOptions> MediatekOptions::Create(OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return MediatekOptions(options.Get(), OwnHandle::kNo);
}
Expected<MediatekOptions> MediatekOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtMediatekOptionsCreate(&options));
  return MediatekOptions(options, OwnHandle::kYes);
}

void MediatekOptions::SetNeronSDKVersionType(
    LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type) {
  internal::AssertOk(LiteRtMediatekOptionsSetNeronSDKVersionType, Data(),
                     sdk_version_type);
}

LiteRtMediatekOptionsNeronSDKVersionType
MediatekOptions::GetNeronSDKVersionType() {
  LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type;
  LiteRtMediatekOptions options_data = Data();
  internal::AssertOk(LiteRtMediatekOptionsGetNeronSDKVersionType, options_data,
                     &sdk_version_type);
  return sdk_version_type;
}

LiteRtMediatekOptions MediatekOptions::Data() const {
  LiteRtMediatekOptions options_data;
  internal::AssertOk(LiteRtMediatekOptionsGet, Get(), &options_data);
  return options_data;
}
}  // namespace litert::mediatek
