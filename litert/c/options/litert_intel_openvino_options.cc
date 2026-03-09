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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/c/options/litert_intel_openvino_options.h"

#include <stdlib.h>
#include <string.h>  // NOLINT: To use strdup in some environments.

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtIntelOpenVinoOptionsT {
  std::optional<LiteRtIntelOpenVinoDeviceType> device_type;
  std::optional<LiteRtIntelOpenVinoPerformanceMode> performance_mode;
  // Store custom configuration options as key-value pairs
  std::vector<std::pair<std::string, std::string>> configs_map_options;
};

LiteRtStatus LrtIntelOpenVinoOptionsCreate(LrtIntelOpenVinoOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtIntelOpenVinoOptionsT{};
  return kLiteRtStatusOk;
}

void LrtDestroyIntelOpenVinoOptions(LrtIntelOpenVinoOptions options) {
  delete options;
}

static const char kIdentifier[] = "intel_openvino";

LiteRtStatus LrtGetOpaqueIntelOpenVinoOptionsData(
    LrtIntelOpenVinoOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*)) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::stringstream ss;
  if (options->device_type.has_value()) {
    ss << "device_type = " << options->device_type.value() << "\n";
  }
  if (options->performance_mode.has_value()) {
    ss << "performance_mode = " << options->performance_mode.value() << "\n";
  }
  for (const auto& pair : options->configs_map_options) {
    ss << "configs_map." << pair.first << " = \"" << pair.second << "\"\n";
  }

  *identifier = LrtGetIntelOpenVinoOptionsIdentifier();
  *payload = strdup(ss.str().c_str());
  *payload_deleter = [](void* p) { free(p); };

  return kLiteRtStatusOk;
}

const char* LrtGetIntelOpenVinoOptionsIdentifier() { return kIdentifier; }

LiteRtStatus LrtCreateIntelOpenVinoOptionsFromToml(
    const char* payload, LrtIntelOpenVinoOptions* options) {
  if (options == nullptr || payload == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LrtIntelOpenVinoOptions local_options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtIntelOpenVinoOptionsCreate(&local_options));

  absl::string_view payload_sv(payload);

  auto status = litert::internal::ParseToml(
      payload_sv,
      [local_options](absl::string_view key,
                      absl::string_view value) -> LiteRtStatus {
        if (key == "device_type") {
          auto parsed_int = litert::internal::ParseTomlInt(value);
          if (!parsed_int.HasValue()) return parsed_int.Error().Status();
          return LrtIntelOpenVinoOptionsSetDeviceType(
              local_options,
              static_cast<LiteRtIntelOpenVinoDeviceType>(*parsed_int));
        } else if (key == "performance_mode") {
          auto parsed_int = litert::internal::ParseTomlInt(value);
          if (!parsed_int.HasValue()) return parsed_int.Error().Status();
          return LrtIntelOpenVinoOptionsSetPerformanceMode(
              local_options,
              static_cast<LiteRtIntelOpenVinoPerformanceMode>(*parsed_int));
        } else if (absl::StartsWith(key, "configs_map.")) {
          std::string config_key(key.substr(12));
          std::string config_value(value);
          return LrtIntelOpenVinoOptionsSetConfigsMapOption(
              local_options, config_key.c_str(), config_value.c_str());
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyIntelOpenVinoOptions(local_options);
    return status;
  }

  *options = local_options;
  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// device_type ----------------------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetDeviceType(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoDeviceType device_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->device_type = device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetDeviceType(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoDeviceType* device_type) {
  if (options == nullptr || device_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->device_type.has_value()) {
    *device_type = kLiteRtIntelOpenVinoDeviceTypeNPU;
  } else {
    *device_type = options->device_type.value();
  }
  return kLiteRtStatusOk;
}

// performance_mode -----------------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->performance_mode = performance_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode* performance_mode) {
  if (options == nullptr || performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->performance_mode.has_value()) {
    *performance_mode = kLiteRtIntelOpenVinoPerformanceModeLatency;
  } else {
    *performance_mode = options->performance_mode.value();
  }
  return kLiteRtStatusOk;
}

// configs_map_options --------------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetConfigsMapOption(
    LrtIntelOpenVinoOptions options, const char* key, const char* value) {
  if (options == nullptr || key == nullptr || value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Check if the key already exists and update it, otherwise add new
  for (auto& pair : options->configs_map_options) {
    if (pair.first == key) {
      pair.second = value;
      return kLiteRtStatusOk;
    }
  }
  options->configs_map_options.emplace_back(key, value);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int* num_options) {
  if (options == nullptr || num_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_options = static_cast<int>(options->configs_map_options.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetConfigsMapOption(
    LrtIntelOpenVinoOptions options, int index, const char** key,
    const char** value) {
  if (options == nullptr || key == nullptr || value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (index < 0 ||
      index >= static_cast<int>(options->configs_map_options.size())) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& pair = options->configs_map_options[index];
  *key = pair.first.c_str();
  *value = pair.second.c_str();
  return kLiteRtStatusOk;
}
