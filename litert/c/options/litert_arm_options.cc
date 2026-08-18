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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

#include "litert/c/options/litert_arm_options.h"

#include <sstream>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtArmOptionsT {
  bool enable_just_in_time = false;
};

const char* LrtArmOptionsGetIdentifier() { return "Arm"; }

LiteRtStatus LrtCreateArmOptions(LrtArmOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *options = new LrtArmOptionsT;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateArmOptionsFromToml(const char* toml_payload,
                                         LrtArmOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(LrtCreateArmOptions(options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  LrtArmOptionsT& options_ref = **options;
  auto status = litert::internal::ParseToml(
      toml_payload,
      [&options_ref](absl::string_view key,
                     absl::string_view value) -> LiteRtStatus {
        if (key == "enable_just_in_time") {
          LITERT_ASSIGN_OR_RETURN(options_ref.enable_just_in_time,
                                  litert::internal::ParseTomlBool(value));
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyArmOptions(*options);
    *options = nullptr;
  }

  return status;
}

void LrtDestroyArmOptions(LrtArmOptions options) { delete options; }

LiteRtStatus LrtGetOpaqueArmOptionsData(LrtArmOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *identifier = LrtArmOptionsGetIdentifier();

  std::ostringstream toml;
  toml << "enable_just_in_time = "
       << (options->enable_just_in_time ? "true" : "false") << "\n";

  litert::internal::MakeCStringPayload(toml.str(), payload, payload_deleter);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmOptionsSetEnableJustInTime(LrtArmOptions options,
                                              bool enable_just_in_time) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_just_in_time = enable_just_in_time;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtArmOptionsGetEnableJustInTime(LrtArmOptions options,
                                              bool* enable_just_in_time) {
  if (options == nullptr || enable_just_in_time == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_just_in_time = options->enable_just_in_time;
  return kLiteRtStatusOk;
}
