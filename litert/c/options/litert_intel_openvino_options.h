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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtIntelOpenVinoOptions);

// Create an Intel OpenVINO options object.
LiteRtStatus LrtIntelOpenVinoOptionsCreate(LrtIntelOpenVinoOptions* options);

// Destroy the options object.
void LrtDestroyIntelOpenVinoOptions(LrtIntelOpenVinoOptions options);

// Serializes intel openvino options and returns the components needed to create
// opaque options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions`.
LiteRtStatus LrtGetOpaqueIntelOpenVinoOptionsData(
    LrtIntelOpenVinoOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*));

// Gets the identifier for Intel OpenVINO options stored in opaque options.
const char* LrtGetIntelOpenVinoOptionsIdentifier();

// Parses a TOML string into the C API representation.
LiteRtStatus LrtCreateIntelOpenVinoOptionsFromToml(
    const char* payload, LrtIntelOpenVinoOptions* options);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// device_type ----------------------------------------------------------------
typedef enum LiteRtIntelOpenVinoDeviceType {
  kLiteRtIntelOpenVinoDeviceTypeCPU = 0,
  kLiteRtIntelOpenVinoDeviceTypeGPU = 1,
  kLiteRtIntelOpenVinoDeviceTypeNPU = 2,
  kLiteRtIntelOpenVinoDeviceTypeAUTO = 3,
} LiteRtIntelOpenVinoDeviceType;

LiteRtStatus LrtIntelOpenVinoOptionsSetDeviceType(
    LrtIntelOpenVinoOptions options,
    enum LiteRtIntelOpenVinoDeviceType device_type);

LiteRtStatus LrtIntelOpenVinoOptionsGetDeviceType(
    LrtIntelOpenVinoOptions options,
    enum LiteRtIntelOpenVinoDeviceType* device_type);

// performance_mode -----------------------------------------------------------

// Configures OpenVINO devices to optimize for performance or efficiency.
// See ov::hint::PerformanceMode in OpenVINO. By default, it
// will use LATENCY mode.

typedef enum LiteRtIntelOpenVinoPerformanceMode {
  /* Optimize for low latency */
  kLiteRtIntelOpenVinoPerformanceModeLatency = 0,
  /* Optimize for high throughput */
  kLiteRtIntelOpenVinoPerformanceModeThroughput = 1,
  /* Optimize for cumulative throughput */
  kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput = 2,
} LiteRtIntelOpenVinoPerformanceMode;

LiteRtStatus LrtIntelOpenVinoOptionsSetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode performance_mode);

LiteRtStatus LrtIntelOpenVinoOptionsGetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode* performance_mode);

// configs_map ----------------------------------------------------------------

// Set a custom configuration option with a string key-value pair.
// The key and value strings are copied internally, so their lifetime does not
// need to extend beyond this function call.
LiteRtStatus LrtIntelOpenVinoOptionsSetConfigsMapOption(
    LrtIntelOpenVinoOptions options, const char* key, const char* value);

// Get the number of custom configuration options
LiteRtStatus LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int* num_options);

// Get a custom configuration option by index.
// The returned key and value pointers point to internal string data
// and are valid for the lifetime of the options object.
// The caller should not free these pointers.
LiteRtStatus LrtIntelOpenVinoOptionsGetConfigsMapOption(
    LrtIntelOpenVinoOptions options, int index, const char** key,
    const char** value);

#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
