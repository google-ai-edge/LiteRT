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

#include "litert/tools/flags/vendors/intel_openvino_flags.h"

#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(LiteRtIntelOpenVinoDeviceType, intel_openvino_device_type,
          kLiteRtIntelOpenVinoDeviceTypeNPU,
          "Device type for Intel OpenVINO inference (cpu, gpu, npu, auto).");

ABSL_FLAG(LiteRtIntelOpenVinoPerformanceMode, intel_openvino_performance_mode,
          kLiteRtIntelOpenVinoPerformanceModeLatency,
          "Performance mode for Intel OpenVINO inference (latency, throughput, "
          "cumulative_throughput).");

ABSL_FLAG(std::string, intel_openvino_configs_map, "",
          "Configuration options for Intel OpenVINO as comma-separated "
          "key=value pairs "
          "(e.g., 'INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/cache').");

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoDeviceType* options, std::string* error) {
  if (text == "cpu") {
    *options = kLiteRtIntelOpenVinoDeviceTypeCPU;
    return true;
  }

  if (text == "gpu") {
    *options = kLiteRtIntelOpenVinoDeviceTypeGPU;
    return true;
  }

  if (text == "npu") {
    *options = kLiteRtIntelOpenVinoDeviceTypeNPU;
    return true;
  }

  if (text == "auto") {
    *options = kLiteRtIntelOpenVinoDeviceTypeAUTO;
    return true;
  }

  *error = "Unknown Intel OpenVINO device type";
  return false;
}

std::string AbslUnparseFlag(LiteRtIntelOpenVinoDeviceType options) {
  switch (options) {
    case kLiteRtIntelOpenVinoDeviceTypeCPU:
      return "cpu";
    case kLiteRtIntelOpenVinoDeviceTypeGPU:
      return "gpu";
    case kLiteRtIntelOpenVinoDeviceTypeNPU:
      return "npu";
    case kLiteRtIntelOpenVinoDeviceTypeAUTO:
      return "auto";
  }
}

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoPerformanceMode* options,
                   std::string* error) {
  if (text == "latency") {
    *options = kLiteRtIntelOpenVinoPerformanceModeLatency;
    return true;
  }
  if (text == "throughput") {
    *options = kLiteRtIntelOpenVinoPerformanceModeThroughput;
    return true;
  }
  if (text == "cumulative_throughput") {
    *options = kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput;
    return true;
  }

  *error = "Unknown Intel OpenVINO performance mode";
  return false;
}

std::string AbslUnparseFlag(LiteRtIntelOpenVinoPerformanceMode options) {
  switch (options) {
    case kLiteRtIntelOpenVinoPerformanceModeLatency:
      return "latency";
    case kLiteRtIntelOpenVinoPerformanceModeThroughput:
      return "throughput";
    case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
      return "cumulative_throughput";
  }
}

// NOLINTEND(*alien-types*)

namespace litert::intel_openvino {

Expected<IntelOpenVinoOptions> IntelOpenVinoOptionsFromFlags() {
  LITERT_ASSIGN_OR_RETURN(auto options, IntelOpenVinoOptions::Create());

  options.SetDeviceType(absl::GetFlag(FLAGS_intel_openvino_device_type));
  options.SetPerformanceMode(
      absl::GetFlag(FLAGS_intel_openvino_performance_mode));

  // Parse configs map options from the flag (comma-separated key=value pairs)
  const std::string configs_map_str =
      absl::GetFlag(FLAGS_intel_openvino_configs_map);
  if (!configs_map_str.empty()) {
    std::vector<std::string> config_pairs =
        absl::StrSplit(configs_map_str, ',');
    for (const auto& pair : config_pairs) {
      std::vector<std::string> key_value = absl::StrSplit(pair, '=');
      if (key_value.size() == 2) {
        options.SetConfigsMapOption(key_value[0].c_str(), key_value[1].c_str());
      } else {
        LITERT_LOG(
            LITERT_WARNING,
            "Ignoring malformed config pair: '%s'. Expected format: key=value",
            pair.c_str());
      }
    }
  }

  return options;
}

}  // namespace litert::intel_openvino
