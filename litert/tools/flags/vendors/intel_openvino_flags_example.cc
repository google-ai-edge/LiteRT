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

// Example usage of Intel OpenVINO flags in LiteRT

#include <iostream>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/tools/flags/vendors/intel_openvino_flags.h"

using litert::intel_openvino::IntelOpenVinoOptions;
using litert::intel_openvino::IntelOpenVinoOptionsFromFlags;

// Example command line usage:
// ./your_binary \
//   --intel_openvino_device_type=npu \
//   --intel_openvino_performance_mode=latency \
//   --intel_openvino_configs_map="INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/cache"

void ExampleIntelOpenVinoFlagsUsage(int argc, char** argv) {
  // Parse command line flags
  absl::ParseCommandLine(argc, argv);

  // Create Intel OpenVINO options from parsed flags
  auto options_result = IntelOpenVinoOptionsFromFlags();
  if (!options_result) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to create Intel OpenVINO options from flags");
    return;
  }

  auto options = std::move(options_result.Value());

  // Display the configured options
  std::cout << "Intel OpenVINO Configuration:\n";
  std::cout << "  Device Type: ";
  switch (options.GetDeviceType()) {
    case kLiteRtIntelOpenVinoDeviceTypeCPU:
      std::cout << "CPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeGPU:
      std::cout << "GPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeNPU:
      std::cout << "NPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeAUTO:
      std::cout << "AUTO";
      break;
  }
  std::cout << "\n";

  std::cout << "  Performance Mode: ";
  switch (options.GetPerformanceMode()) {
    case kLiteRtIntelOpenVinoPerformanceModeLatency:
      std::cout << "Latency";
      break;
    case kLiteRtIntelOpenVinoPerformanceModeThroughput:
      std::cout << "Throughput";
      break;
    case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
      std::cout << "Cumulative Throughput";
      break;
  }
  std::cout << "\n";

  // Now you can use these options to configure your Intel OpenVINO compiler
  // plugin
  LITERT_LOG(LITERT_INFO, "Intel OpenVINO options configured successfully");
}

int main(int argc, char** argv) {
  ExampleIntelOpenVinoFlagsUsage(argc, argv);
  return 0;
}
