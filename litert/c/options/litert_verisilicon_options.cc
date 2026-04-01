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
#include "litert/c/options/litert_verisilicon_options.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

using litert::internal::ParseToml;
struct LrtVerisiliconOptionsT {
  std::optional<uint32_t> device_index = 0;
  std::optional<uint32_t> core_index = 0;
  std::optional<uint32_t> time_out = 0;
  std::optional<uint32_t> profile_level = 0;
  std::optional<bool> dump_nbg = false;
};

LiteRtStatus LrtCreateVerisiliconOptions(LrtVerisiliconOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtVerisiliconOptionsT;
  return kLiteRtStatusOk;

}

void LrtDestroyVerisiliconOptions(LrtVerisiliconOptions options) { delete options; }

LiteRtStatus LrtCreateVerisiliconOptionsFromToml(const char* toml_payload,
                                              LrtVerisiliconOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LrtVerisiliconOptions parsed_options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateVerisiliconOptions(&parsed_options));

  if (toml_payload == nullptr || toml_payload[0] == '\0') {
    return kLiteRtStatusOk;
  }

  auto status = ::litert::internal::ParseToml(
      toml_payload,
      [&parsed_options](absl::string_view key, absl::string_view value) -> LiteRtStatus {
        if (key == "device_index") {
          LITERT_ASSIGN_OR_RETURN(auto device_index,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtVerisiliconOptionsSetDeviceIndex(parsed_options, static_cast<uint32_t>(device_index));
        }
        if (key == "core_index") {
          LITERT_ASSIGN_OR_RETURN(auto core_index,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtVerisiliconOptionsSetCoreIndex(parsed_options, static_cast<uint32_t>(core_index));
        }
        if (key == "time_out") {
          LITERT_ASSIGN_OR_RETURN(auto time,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtVerisiliconOptionsSetTimeOut(parsed_options, static_cast<uint32_t>(time));
        }
        if (key == "profile_level") {
          LITERT_ASSIGN_OR_RETURN(auto profile_level,
                                  ::litert::internal::ParseTomlInt(value));
          return LrtVerisiliconOptionsSetProfileLevel(parsed_options, static_cast<uint32_t>(profile_level));
        }
        if (key == "dump_nbg") {
          LITERT_ASSIGN_OR_RETURN(auto dump_nbg,
                                  ::litert::internal::ParseTomlBool(value));
          return LrtVerisiliconOptionsSetDumpNBG(parsed_options, dump_nbg);
        }


        // Ignore unknown keys to allow for forward compatibility.
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyVerisiliconOptions(parsed_options);
    parsed_options = nullptr;
  }
  *options = parsed_options;
  return status;
}

LiteRtStatus LrtGetOpaqueVerisiliconOptionsData(const LrtVerisiliconOptions options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Construct a TOML string from the options.
  std::string toml_str;
  if (options->device_index.has_value()) {
    absl::StrAppendFormat(&toml_str, "device_index = %d\n",
                          static_cast<unsigned int>(*options->device_index));
  }
  if (options->core_index.has_value()) {
    absl::StrAppendFormat(
        &toml_str, "core_index = %d\n",
        static_cast<unsigned int>(*options->core_index));
  }
  if (options->time_out.has_value()) {
    absl::StrAppendFormat(
        &toml_str, "time_out = %d\n",
        static_cast<unsigned int>(*options->time_out));
  }
  if (options->profile_level.has_value()) {
    absl::StrAppendFormat(
        &toml_str, "profile_level = %d\n",
        static_cast<unsigned int>(*options->profile_level));
  }
  if (options->dump_nbg.has_value()) {
    absl::StrAppendFormat(
        &toml_str, "dump_nbg = %s\n",
        (*options->dump_nbg ? "true" : "false"));
  }

  *identifier = LrtVerisiliconOptionsGetIdentifier();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);
  return kLiteRtStatusOk;
}

const char* LrtVerisiliconOptionsGetIdentifier() { return "verisilicon"; }

// device index ----------------------------------------------------------
LiteRtStatus LrtVerisiliconOptionsSetDeviceIndex(
    LrtVerisiliconOptions options,
    unsigned int device_index) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->device_index = device_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtVerisiliconOptionsGetDeviceIndex(
    const LrtVerisiliconOptions options,
    unsigned int* device_index) {
  if (options == nullptr || device_index == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_index = options->device_index.value_or(0);
  return kLiteRtStatusOk;
}

// core index ----------------------------------------------------------
LiteRtStatus LrtVerisiliconOptionsSetCoreIndex(
    LrtVerisiliconOptions options,
    unsigned int core_index) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->core_index = core_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtVerisiliconOptionsGetCoreIndex(
    const LrtVerisiliconOptions options,
    unsigned int* core_index) {
  if (options == nullptr || core_index == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *core_index = options->core_index.value_or(0);
  return kLiteRtStatusOk;
}

// Time Out ----------------------------------------------------------
LiteRtStatus LrtVerisiliconOptionsSetTimeOut(
    LrtVerisiliconOptions options,
    unsigned int time) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->time_out = time;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtVerisiliconOptionsGetTimeOut(
    const LrtVerisiliconOptions options,
    unsigned int* time) {
  if (options == nullptr || time == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *time = options->time_out.value_or(0);
  return kLiteRtStatusOk;
}

// profile level ----------------------------------------------------------
LiteRtStatus LrtVerisiliconOptionsSetProfileLevel(
    LrtVerisiliconOptions options,
    unsigned int level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->profile_level = level;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtVerisiliconOptionsGetProfileLevel(
    const LrtVerisiliconOptions options,
    unsigned int* level) {
  if (options == nullptr || level == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *level = options->profile_level.value_or(0);
  return kLiteRtStatusOk;
}

// Dump NBG ----------------------------------------------------------
LiteRtStatus LrtVerisiliconOptionsSetDumpNBG(
    LrtVerisiliconOptions options,
    bool enable) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dump_nbg = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtVerisiliconOptionsGetDumpNBG(
    const LrtVerisiliconOptions options,
    bool* enable) {
  if (options == nullptr || enable == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable = options->dump_nbg.value_or(false);
  return kLiteRtStatusOk;
}