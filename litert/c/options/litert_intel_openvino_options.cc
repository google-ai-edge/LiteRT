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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/cache/hash_util.h"

struct LiteRtIntelOpenVinoOptionsT {
  LiteRtIntelOpenVinoDeviceType device_type = kLiteRtIntelOpenVinoDeviceTypeNPU;
  LiteRtIntelOpenVinoPerformanceMode performance_mode =
      kLiteRtIntelOpenVinoPerformanceModeLatency;
  // Store custom configuration options as key-value pairs
  std::vector<std::pair<std::string, std::string>> configs_map_options;
};

LiteRtStatus LiteRtIntelOpenVinoOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtIntelOpenVinoOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtIntelOpenVinoOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtIntelOpenVinoOptions>(payload);
      },
      options));

  auto intel_openvino_hash = [](const void* payload) -> uint64_t {
    const LiteRtIntelOpenVinoOptionsT* opts =
        reinterpret_cast<const LiteRtIntelOpenVinoOptionsT*>(payload);
    uint64_t ans = 0;
    litert::HashCombine(ans, opts->device_type, opts->performance_mode);
    // Hash the configs_map_options
    for (const auto& pair : opts->configs_map_options) {
      litert::HashCombine(ans, pair.first, pair.second);
    }
    return ans;
  };
  LITERT_RETURN_IF_ERROR(
      LiteRtSetOpaqueOptionsHash(*options, intel_openvino_hash));

  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtIntelOpenVinoOptionsGetIdentifier() {
  return "intel_openvino";
}

LiteRtStatus LiteRtIntelOpenVinoOptionsGet(
    LiteRtOpaqueOptions options, LiteRtIntelOpenVinoOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtIntelOpenVinoOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtIntelOpenVinoOptionsT*>(payload);
  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// device_type ----------------------------------------------------------------
LiteRtStatus LiteRtIntelOpenVinoOptionsSetDeviceType(
    LiteRtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoDeviceType device_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->device_type = device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtIntelOpenVinoOptionsGetDeviceType(
    LiteRtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoDeviceType* device_type) {
  if (options == nullptr || device_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_type = options->device_type;
  return kLiteRtStatusOk;
}

// performance_mode -----------------------------------------------------------
LiteRtStatus LiteRtIntelOpenVinoOptionsSetPerformanceMode(
    LiteRtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->performance_mode = performance_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtIntelOpenVinoOptionsGetPerformanceMode(
    LiteRtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode* performance_mode) {
  if (options == nullptr || performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *performance_mode = options->performance_mode;
  return kLiteRtStatusOk;
}

// configs_map_options --------------------------------------------------------
LiteRtStatus LiteRtIntelOpenVinoOptionsSetConfigsMapOption(
    LiteRtIntelOpenVinoOptions options, const char* key, const char* value) {
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

LiteRtStatus LiteRtIntelOpenVinoOptionsGetNumConfigsMapOptions(
    LiteRtIntelOpenVinoOptions options, int* num_options) {
  if (options == nullptr || num_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_options = static_cast<int>(options->configs_map_options.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtIntelOpenVinoOptionsGetConfigsMapOption(
    LiteRtIntelOpenVinoOptions options, int index, const char** key,
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
