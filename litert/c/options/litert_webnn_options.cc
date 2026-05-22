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

#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdlib>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

using ::litert::ErrorStatusBuilder;
using ::litert::internal::ParseToml;
using ::litert::internal::ParseTomlInt;
struct LrtWebNnOptions {
  LiteRtWebNnDeviceType device_type = kLiteRtWebNnDeviceTypeCpu;
  LiteRtWebNnPowerPreference power_preference =
      kLiteRtWebNnPowerPreferenceDefault;
  LiteRtWebNnPrecision precision = kLiteRtWebNnPrecisionFp32;
};

LiteRtStatus LrtCreateWebNnOptions(LrtWebNnOptions** options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtWebNnOptions();
  if (!*options) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  return kLiteRtStatusOk;
}

void LrtDestroyWebNnOptions(LrtWebNnOptions* options) {
  if (options) {
    delete options;
  }
}

LiteRtStatus LrtGetOpaqueWebNnOptionsData(const LrtWebNnOptions* options,
                                          const char** identifier,
                                          void** payload,
                                          void (**payload_deleter)(void*)) {
  if (!options || !identifier || !payload || !payload_deleter) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::stringstream ss;
  ss << "device_type = " << static_cast<int>(options->device_type) << "\n";
  ss << "power_preference = " << static_cast<int>(options->power_preference)
     << "\n";
  ss << "precision = " << static_cast<int>(options->precision) << "\n";

  *identifier = "webnn_options_string";
  std::string toml_str = ss.str();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateWebNnOptionsFromToml(const char* toml_string,
                                           LrtWebNnOptions** options) {
  if (!toml_string || !options) return kLiteRtStatusErrorInvalidArgument;
  *options = new LrtWebNnOptions();
  if (!*options) return kLiteRtStatusErrorMemoryAllocationFailure;
  absl::string_view toml_view(toml_string);
  if (toml_view.empty()) return kLiteRtStatusOk;

  auto status = ParseToml(
      toml_view,
      [&](absl::string_view key, absl::string_view value) -> LiteRtStatus {
        if (key == "device_type") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->device_type = static_cast<LiteRtWebNnDeviceType>(*res);
        } else if (key == "power_preference") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->power_preference =
              static_cast<LiteRtWebNnPowerPreference>(*res);
        } else if (key == "precision") {
          auto res = ParseTomlInt(value);
          if (!res) return kLiteRtStatusErrorInvalidArgument;
          (*options)->precision = static_cast<LiteRtWebNnPrecision>(*res);
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    delete *options;
    *options = nullptr;
    return status;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetWebNnOptionsDevicePreference(
    LrtWebNnOptions* options, LiteRtWebNnDeviceType device_type) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->device_type = device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetWebNnOptionsDevicePreference(
    const LrtWebNnOptions* options, LiteRtWebNnDeviceType* device_type) {
  if (!options || !device_type) return kLiteRtStatusErrorInvalidArgument;
  *device_type = options->device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetWebNnOptionsPowerPreference(
    LrtWebNnOptions* options, LiteRtWebNnPowerPreference power_preference) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->power_preference = power_preference;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetWebNnOptionsPowerPreference(
    const LrtWebNnOptions* options,
    LiteRtWebNnPowerPreference* power_preference) {
  if (!options || !power_preference) return kLiteRtStatusErrorInvalidArgument;
  *power_preference = options->power_preference;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtSetWebNnOptionsPrecision(LrtWebNnOptions* options,
                                         LiteRtWebNnPrecision precision) {
  if (!options) return kLiteRtStatusErrorInvalidArgument;
  options->precision = precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetWebNnOptionsPrecision(const LrtWebNnOptions* options,
                                         LiteRtWebNnPrecision* precision) {
  if (!options || !precision) return kLiteRtStatusErrorInvalidArgument;
  *precision = options->precision;
  return kLiteRtStatusOk;
}
