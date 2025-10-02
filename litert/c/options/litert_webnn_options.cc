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

#include "litert/c/options/litert_webnn_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

using ::litert::ErrorStatusBuilder;

struct LiteRtWebNnOptionsPayloadT {
  static constexpr const absl::string_view kIdentifier = "webnn_payload";

  LiteRtWebNnDeviceType device_type =
      LiteRtWebNnDeviceType::kLiteRtWebNnDeviceTypeCpu;
  LiteRtWebNnPowerPreference power_preference =
      LiteRtWebNnPowerPreference::kLiteRtWebNnPowerPreferenceDefault;
  LiteRtWebNnPrecision precision =
      LiteRtWebNnPrecision::kLiteRtWebNnPrecisionFp32;
};

namespace litert {
namespace {

litert::Expected<LiteRtWebNnOptionsPayloadT*> GetPayload(
    LiteRtOpaqueOptions options) {
  const char* identifier = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  LITERT_RETURN_IF_ERROR(identifier == LiteRtWebNnOptionsPayloadT::kIdentifier,
                         ErrorStatusBuilder::InvalidArgument())
      << "Payload stored in accelerator options is incompatible. Got "
      << identifier << ", expected " << LiteRtWebNnOptionsPayloadT::kIdentifier
      << ".";

  LiteRtWebNnOptionsPayloadT* payload;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsData(options, reinterpret_cast<void**>(&payload)));
  return payload;
}

}  // namespace
}  // namespace litert

LiteRtStatus LiteRtCreateWebNnOptions(LiteRtOpaqueOptions* options) {
  auto payload = std::make_unique<LiteRtWebNnOptionsPayloadT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtWebNnOptionsPayloadT::kIdentifier.data(), payload.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtWebNnOptionsPayloadT*>(payload);
      },
      options));
  payload.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetWebNnOptionsDevicePreference(
    LiteRtOpaqueOptions webnn_options, LiteRtWebNnDeviceType device_type) {
  LITERT_ASSIGN_OR_RETURN(LiteRtWebNnOptionsPayloadT * payload,
                          litert::GetPayload(webnn_options));
  payload->device_type = device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetWebNnOptionsPowerPreference(
    LiteRtOpaqueOptions webnn_options,
    LiteRtWebNnPowerPreference power_preference) {
  LITERT_ASSIGN_OR_RETURN(LiteRtWebNnOptionsPayloadT * payload,
                          litert::GetPayload(webnn_options));
  payload->power_preference = power_preference;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetWebNnOptionsPrecision(
    LiteRtOpaqueOptions webnn_options, LiteRtWebNnPrecision precision) {
  LITERT_ASSIGN_OR_RETURN(LiteRtWebNnOptionsPayloadT * payload,
                          litert::GetPayload(webnn_options));
  payload->precision = precision;
  return kLiteRtStatusOk;
}

const char* LiteRtGetWebNnOptionsPayloadIdentifier() {
  return LiteRtWebNnOptionsPayloadT::kIdentifier.data();
}

LiteRtStatus LiteRtGetWebNnOptionsDevicePreference(
    LiteRtWebNnDeviceType* device_type, LiteRtWebNnOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(device_type, ErrorStatusBuilder::InvalidArgument())
      << "`device_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *device_type = payload->device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetWebNnOptionsPowerPreference(
    LiteRtWebNnPowerPreference* power_preference,
    LiteRtWebNnOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(power_preference,
                         ErrorStatusBuilder::InvalidArgument())
      << "`power_preference` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *power_preference = payload->power_preference;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetWebNnOptionsPrecision(
    LiteRtWebNnPrecision* precision, LiteRtWebNnOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *precision = payload->precision;
  return kLiteRtStatusOk;
}
