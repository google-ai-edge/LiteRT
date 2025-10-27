// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

// Example usage of Intel OpenVINO options in LiteRT

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/options/litert_intel_openvino_options.h"

using litert::intel_openvino::IntelOpenVinoOptions;

void example_intel_openvino_options_usage() {
  // Create Intel OpenVINO options
  auto options_result = IntelOpenVinoOptions::Create();
  if (!options_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Intel OpenVINO options");
    return;
  }
  auto options = std::move(options_result.Value());

  // Configure device type (NPU, CPU, GPU, AUTO)
  options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeNPU);

  // Configure performance mode
  options.SetPerformanceMode(kLiteRtIntelOpenVinoPerformanceModeLatency);

  // Set custom OpenVINO configuration properties
  // Configure inference precision
  options.SetConfigsMapOption("INFERENCE_PRECISION_HINT", "f16");

  // Set NPU compilation parameters
  options.SetConfigsMapOption("NPU_COMPILATION_MODE_PARAMS",
                              "compute-layers-with-higher-precision=Sigmoid");

  // Enable model caching for faster subsequent loads
  options.SetConfigsMapOption("CACHE_DIR", "/tmp/ov_cache");

  // Read back configured values
  auto device_type = options.GetDeviceType();
  auto performance_mode = options.GetPerformanceMode();

  LITERT_LOG(LITERT_INFO,
             "Intel OpenVINO Options - Device: %d, Performance: %d",
             device_type, performance_mode);
  LITERT_LOG(LITERT_INFO, "Custom configurations applied via configs_map");
}
