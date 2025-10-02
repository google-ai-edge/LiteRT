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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_WEBNN_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_WEBNN_OPTIONS_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maps to third_party/tflite_webnn_delegate/src/delegate_options.h.
typedef enum {
  // The CPU device type.
  kLiteRtWebNnDeviceTypeCpu = 0,
  // The GPU device type.
  kLiteRtWebNnDeviceTypeGpu = 1,
  // The NPU device type.
  kLiteRtWebNnDeviceTypeNpu = 2,
} LiteRtWebNnDeviceType;

typedef enum {
  // The default power preference.
  kLiteRtWebNnPowerPreferenceDefault = 0,
  // The high performance power preference.
  kLiteRtWebNnPowerPreferenceHighPerformance = 1,
  // The low power power preference.
  kLiteRtWebNnPowerPreferenceLowPower = 2,
} LiteRtWebNnPowerPreference;

typedef enum {
  // The FP32 precision.
  kLiteRtWebNnPrecisionFp32 = 0,
  // The FP16 precision.
  kLiteRtWebNnPrecisionFp16 = 1,
} LiteRtWebNnPrecision;

// Create a LiteRtOpaqueOptions object holding WebNN accelerator
// options.
LiteRtStatus LiteRtCreateWebNnOptions(LiteRtOpaqueOptions* options);

// Sets the device type for WebNN.
LiteRtStatus LiteRtSetWebNnOptionsDevicePreference(
    LiteRtOpaqueOptions webnn_options, LiteRtWebNnDeviceType device_type);

// Sets the power preference for WebNN.
LiteRtStatus LiteRtSetWebNnOptionsPowerPreference(
    LiteRtOpaqueOptions webnn_options,
    LiteRtWebNnPowerPreference power_preference);

// Sets the precision for WebNN.
LiteRtStatus LiteRtSetWebNnOptionsPrecision(LiteRtOpaqueOptions webnn_options,
                                            LiteRtWebNnPrecision precision);


// Declarations below this point are meant to be used by accelerator code.
LITERT_DEFINE_HANDLE(LiteRtWebNnOptionsPayload);

const char* LiteRtGetWebNnOptionsPayloadIdentifier();

// Gets the device type for WebNN.
LiteRtStatus LiteRtGetWebNnOptionsDevicePreference(
    LiteRtWebNnDeviceType* device_type, LiteRtWebNnOptionsPayload payload);

// Gets the power preference for WebNN.
LiteRtStatus LiteRtGetWebNnOptionsPowerPreference(
    LiteRtWebNnPowerPreference* power_preference,
    LiteRtWebNnOptionsPayload payload);

// Gets the precision for WebNN.
LiteRtStatus LiteRtGetWebNnOptionsPrecision(LiteRtWebNnPrecision* precision,
                                            LiteRtWebNnOptionsPayload payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_WEBNN_OPTIONS_H_
