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

// Sets the GPU backend.
LiteRtStatus LiteRtSetGpuOptionsGpuBackend(LiteRtOpaqueOptions gpu_options,
                                           LiteRtGpuBackend backend);

// Set to true to run in external tensors mode. This allows GPU
// Accelerator to always use external tensors (PHWC4 format) as inputs and
// outputs. This mode mostly gives a slightly lower performance but it reduces
// additional GPU-GPU copies for input and output tensors.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LiteRtSetGpuOptionsExternalTensorsMode(
    LiteRtOpaqueOptions gpu_options, bool enable);

// Add a prefix pattern to match external tensors. When ExternalTensorsMode is
// not used (default behavior), all input and output tensors requires PHWC4
// layout conversion. This pattern is useful for state tensors to reduce the
// layout conversion. For example, if the prefix pattern is "kv_cache_", then
// all tensors whose names begin with "kv_cache_" will be exempted.
LiteRtStatus LiteRtAddGpuOptionsExternalTensorPattern(
    LiteRtOpaqueOptions gpu_options, const char* pattern);

// Sets the GPU priority. Low priority helps to unblock UI workloads.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LiteRtSetGpuOptionsGpuPriority(LiteRtOpaqueOptions gpu_options,
                                            LiteRtGpuPriority priority);

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

// Sets whether to madvise the original shared tensors after use. Note that
// this boolean flag is to disable madvise which is enabled by default.
LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    LiteRtOpaqueOptions gpu_accelerator_options,
    bool madvise_original_shared_tensors);

// Sets the number of steps of command buffer preparations.
LiteRtStatus
LiteRtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    LiteRtOpaqueOptions gpu_accelerator_options,
    int num_steps_of_command_buffer_preparations);

// Sets whether to use Metal argument buffers.
LiteRtStatus LiteRtSetGpuOptionsUseMetalArgumentBuffers(
    LiteRtOpaqueOptions gpu_options, bool use_metal_argument_buffers);

// Sets the wait type.
LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtOpaqueOptions gpu_accelerator_options, LiteRtGpuWaitType wait_type);

// Sets the preferred device substring.
LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    LiteRtOpaqueOptions gpu_accelerator_options,
    const char* preferred_device_substr);

// Sets the number of threads for webgpu upload.
LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    LiteRtOpaqueOptions gpu_accelerator_options, int num_threads_to_upload);

// Sets the number of threads for webgpu kernel compilation.
LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    LiteRtOpaqueOptions gpu_accelerator_options, int num_threads_to_compile);

// Sets whether to convert weights on GPU. It's an experimental feature.
LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    LiteRtOpaqueOptions gpu_accelerator_options, bool convert_weights_on_gpu);

// Sets the hint to fully delegate to single delegate.
// This is an ADVANCED option and should only be set if every subgraph is
// known to be fully delegated to a single delegate. This flag can be used to
// skip unnecessary memory allocations.
LiteRtStatus LiteRtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
    LiteRtOpaqueOptions gpu_options,
    bool hint_fully_delegated_to_single_delegate);

// Declarations below this point are meant to be used by accelerator code.

LITERT_DEFINE_HANDLE(LiteRtGpuOptionsPayload);

const char* LiteRtGetGpuOptionsPayloadIdentifier();

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsGpuBackend(LiteRtGpuBackend* backend,
                                           LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsExternalTensorsMode(
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

LiteRtStatus LiteRtGetNumGpuAcceleratorCompilationOptionsExternalTensorPatterns(
    int* num_patterns, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsExternalTensorPattern(
    const char** external_tensor_pattern, int pattern_index,
    LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsGpuPriority(LiteRtGpuPriority* priority,
                                            LiteRtGpuOptionsPayload payload);

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    bool* madvise_original_shared_tensors, LiteRtGpuOptionsPayload payload);

LiteRtStatus
LiteRtGetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    int* num_steps_of_command_buffer_preparations,
    LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsUseMetalArgumentBuffers(
    LiteRtGpuOptionsPayload payload, bool* use_metal_argument_buffers);

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtGpuWaitType* wait_type, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    const char** preferred_device_substr, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    int* num_threads_to_upload, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    int* num_threads_to_compile, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    bool* convert_weights_on_gpu, LiteRtGpuOptionsPayload payload);

LiteRtStatus LiteRtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
    bool* hint_fully_delegated_to_single_delegate,
    LiteRtGpuOptionsPayload payload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_GPU_ACCELERATOR_OPTIONS_PAYLOAD_H_
