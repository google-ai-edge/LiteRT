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

#include "litert/cc/options/litert_webnn_options.h"

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_webnn_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

const char* WebNnOptions::GetPayloadIdentifier() {
  return LiteRtGetWebNnOptionsPayloadIdentifier();
}

Expected<WebNnOptions> WebNnOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateWebNnOptions(&options));
  return WebNnOptions(options, OwnHandle::kYes);
}

LiteRtStatus WebNnOptions::SetDevicePreference(
    LiteRtWebNnDeviceType device_type) {
  return LiteRtSetWebNnOptionsDevicePreference(Get(), device_type);
}

LiteRtStatus WebNnOptions::SetPowerPreference(
    LiteRtWebNnPowerPreference power_preference) {
  return LiteRtSetWebNnOptionsPowerPreference(Get(), power_preference);
}

LiteRtStatus WebNnOptions::SetPrecision(LiteRtWebNnPrecision precision) {
  return LiteRtSetWebNnOptionsPrecision(Get(), precision);
}

}  // namespace litert
