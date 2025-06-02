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

#ifdef __cplusplus
extern "C" {
#endif

// Create a LiteRtOpaqueOptions object holding GPU accelerator
// options.
LiteRtStatus LiteRtCreateGpuOptions(LiteRtOpaqueOptions* options);

// Enables the GPU accelerator constant tensor sharing to enable sharing of
// constant tensors between different subgraphs.
LiteRtStatus LiteRtSetGpuOptionsConstantTensorSharing(
    LiteRtOpaqueOptions gpu_options, bool enable);

// Enables the GPU accelerator infinite float capping to enforce capping
// inf/-inf to max float values for the softmax input and padding
LiteRtStatus LiteRtSetGpuOptionsInfiniteFloatCapping(
    LiteRtOpaqueOptions gpu_options, bool enable);

// Enables the GPU accelerator benchmark mode. This will disable some
// optimizations that are not needed for benchmarking.
LiteRtStatus LiteRtSetGpuOptionsBenchmarkMode(LiteRtOpaqueOptions gpu_options,
                                              bool enable);

// Set to true to run in no immutable external tensors mode. This prevents GPU
// Accelerator from using immutable external tensors.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LiteRtSetGpuOptionsNoImmutableExternalTensorsMode(
    LiteRtOpaqueOptions gpu_options, bool enable);

// This enables dynamic range quantization of the input tensor for large sized
// fully connected and convolution operations, if the device supports it. This
// will result in accuracy loss, since the input tensor will be quantized to
// 8-bit. Turning this on will also increase the initialization time to
// calculate some extra constant tensor. `enable_constant_tensors_sharing` must
// be true to use this.
LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LiteRtOpaqueOptions gpu_accelerator_options, bool enable);

// Sets the GPU accelerator precision. e.g. FP16, FP32, etc.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegatePrecision precision);

// Sets the GPU buffer storage type. If true, the delegate will use buffer
// storage type.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegateBufferStorageType buffer_storage_type);

// If true, the delegate will prefer to use textures rather than buffers for
// weights. Use option when weights in texture has better performance.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    LiteRtOpaqueOptions gpu_accelerator_options, bool prefer_texture_weights);

// The nul-terminated directory to use for serialization.
// Whether serialization actually happens or not is dependent on backend used
// and validity of this directory.
// Set to nullptr implies the delegate will not try serialization.
//
// NOTE: Users should ensure that this directory is private to the app to
// avoid data access issues.
// Delegate stores the pointer to the string and doesn't take ownership of the
// memory. The string memory must outlive the `gpu_accelerator_options` object.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir(
    LiteRtOpaqueOptions gpu_accelerator_options, const char* serialization_dir);

// The unique nul-terminated token string that acts as a 'namespace' for
// all serialization entries.
// Should be unique to a particular model (graph & constants).
// For an example of how to generate this from a TFLite model, see
// StrFingerprint() in lite/delegates/serialization.h.
//
// Set to nullptr implies the delegate will not try serialization.
// Delegate stores the pointer to the string and doesn't take ownership of the
// memory. The string memory must outlive the `gpu_accelerator_options` object.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsModelCacheKey(
    LiteRtOpaqueOptions gpu_accelerator_options, const char* model_cache_key);

// When set to true AND the serialization_dir and model_cache_key are also set,
// the delegate will serialize the program cache.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    LiteRtOpaqueOptions gpu_accelerator_options, bool serialize_program_cache);

// Set to true to serialize immutable external tensors. By default only the
// non-external tensors are serialized.
LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    LiteRtOpaqueOptions gpu_accelerator_options,
    bool serialize_external_tensors);

// Declarations below this point are meant to be used by accelerator code.

LITERT_DEFINE_HANDLE(LiteRtGpuOptionsPayload);

const char* LiteRtGetGpuOptionsPayloadIdentifier();

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsNoImmutableExternalTensorsMode(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* type, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    bool* prefer_texture_weights, LiteRtGpuOptionsPayload payload);

// Returns serialization directory.
// The returned string pointer is owned by the user of
// LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir() API.
LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializationDir(
    const char** serialization_dir, LiteRtGpuOptionsPayload payload);

// Returns model cache key.
// The returned string pointer is owned by the user of
// LiteRtSetGpuAcceleratorCompilationOptionsModelCacheKey() API.
LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsModelCacheKey(
    const char** model_cache_key, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    bool* serialize_program_cache, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    bool* serialize_external_tensors, LiteRtGpuOptionsPayload payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_PAYLOAD_H_
