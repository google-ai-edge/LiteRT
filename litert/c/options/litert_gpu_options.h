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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_PAYLOAD_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_PAYLOAD_H_

#include <stdbool.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#ifdef __cplusplus
extern "C" {
#endif

// Create a LiteRtOpaqueOptions object holding GPU accelerator
// options.
LiteRtStatus LiteRtCreateGpuOptions(LiteRtOpaqueOptions* options);

// Enables the GPU accelerator constant tensor sharing.
LiteRtStatus LiteRtSetGpuOptionsConstantTensorSharing(
    LiteRtOpaqueOptions gpu_options, bool enable);

// Enables the GPU accelerator infinite float capping.
LiteRtStatus LiteRtSetGpuOptionsInfiniteFloatCapping(
    LiteRtOpaqueOptions gpu_options, bool enable);

// Enables the GPU accelerator benchmark mode.
LiteRtStatus LiteRtSetGpuOptionsBenchmarkMode(LiteRtOpaqueOptions gpu_options,
                                              bool enable);

// Enables the GPU accelerator allow src quantized FC conv ops.
LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LiteRtOpaqueOptions gpu_accelerator_options, bool enable);

// Sets the GPU accelerator precision. e.g. FP16, FP32, etc.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegatePrecision precision);

// Sets the GPU buffer storage type.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegateBufferStorageType buffer_storage_type);

// Declarations below this point are meant to be used by accelerator code.

LITERT_DEFINE_HANDLE(LiteRtGpuOptionsPayload);

const char* LiteRtGetGpuOptionsPayloadIdentifier();

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload);

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* type, LiteRtGpuOptionsPayload payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_PAYLOAD_H_
