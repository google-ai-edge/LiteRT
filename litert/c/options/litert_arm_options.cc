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

struct LiteRtArmOptionsT {
  bool enable_just_in_time = false;
};

const char* LiteRtArmOptionsGetIdentifier() { return "Arm"; }

LiteRtStatus LiteRtArmOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* arm_options = new LiteRtArmOptionsT;
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtArmOptionsGetIdentifier(), arm_options,
      [](void* payload) { delete reinterpret_cast<LiteRtArmOptions>(payload); },
      options));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtArmOptionsGet(LiteRtOpaqueOptions options,
                                 LiteRtArmOptions* arm_options) {
  if (options == nullptr || arm_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      options, LiteRtArmOptionsGetIdentifier(), &payload));
  *arm_options = reinterpret_cast<LiteRtArmOptions>(payload);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtGetOpaqueArmOptionsData(LiteRtOpaqueOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtArmOptions arm_options = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtArmOptionsGet(options, &arm_options));

  std::ostringstream toml;
  toml << "enable_just_in_time = "
       << (arm_options->enable_just_in_time ? "true" : "false") << "\n";

  const std::string toml_payload = toml.str();
  *identifier = LiteRtArmOptionsGetIdentifier();
  litert::internal::MakeCStringPayload(absl::string_view(toml_payload), payload,
                                       payload_deleter);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtArmOptionsSetEnableJustInTime(LiteRtArmOptions options,
                                                 bool enable_just_in_time) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_just_in_time = enable_just_in_time;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtArmOptionsGetEnableJustInTime(LiteRtArmOptions options,
                                                 bool* enable_just_in_time) {
  if (options == nullptr || enable_just_in_time == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_just_in_time = options->enable_just_in_time;
  return kLiteRtStatusOk;
}
