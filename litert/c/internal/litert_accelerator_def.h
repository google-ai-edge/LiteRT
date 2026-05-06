// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_DEF_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_DEF_H_

#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// The version of the LiteRT accelerator definition.
// This version is used to ensure the ABI compatibility of the accelerator
// definition. Any changes to the LiteRtAcceleratorDef struct and HwMemoryInfo
// struct should be accompanied by an update to this version.
#define LITERT_ACCELERATOR_DEF_CURRENT_VERSION 1

// A struct that contains the data and functions that are used to define an
// accelerator. Refer litert_accelerator_registration.h for more details.
typedef struct {
  int version;  // Version of the accelerator definition
                // Current runtime only supports version 1.

  LiteRtStatus (*get_name)(LiteRtAccelerator accelerator, const char** name);
  LiteRtStatus (*get_version)(LiteRtAccelerator accelerator,
                              LiteRtApiVersion* version);
  LiteRtStatus (*get_hardware_support)(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware);
  LiteRtStatus (*is_tflite_delegate_responsible_for_jit_compilation)(
      LiteRtAccelerator accelerator, bool* does_jit_compilation);
  LiteRtStatus (*create_delegate)(LiteRtRuntimeContext* runtime_context,
                                  LiteRtEnvironment env,
                                  LiteRtAccelerator accelerator,
                                  LiteRtOptions options,
                                  LiteRtDelegateWrapper* delegate_wrapper);
  void (*destroy_delegate)(LiteRtRuntimeContext* runtime_context,
                           LiteRtDelegateWrapper delegate_wrapper);
  LiteRtStatus (*start_metrics_collection)(
      LiteRtRuntimeContext* runtime_context, LiteRtDelegateWrapper delegate,
      int detail_level);
  LiteRtStatus (*stop_metrics_collection)(LiteRtRuntimeContext* runtime_context,
                                          LiteRtDelegateWrapper delegate,
                                          LiteRtMetrics metrics);

  LiteRtCustomTensorBufferHandlersDef buffer_handlers;
} LiteRtAcceleratorDefV1;

typedef LiteRtAcceleratorDefV1 LiteRtAcceleratorDef;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_DEF_H_
