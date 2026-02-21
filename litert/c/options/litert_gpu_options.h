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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GPU_OPTIONS_H_

#include <stdbool.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LrtGpuOptions LrtGpuOptions;

// Creates a GPU options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyGpuOptions`.
LiteRtStatus LrtCreateGpuOptions(LrtGpuOptions** options);

// Creates a GPU options object from a TOML string.
// The caller is responsible for freeing the returned options using
// `LrtDestroyGpuOptions`.
LiteRtStatus LrtCreateGpuOptionsFromToml(const char* toml_string,
                                         LrtGpuOptions** options);

// Destroys a GPU options object.
void LrtDestroyGpuOptions(LrtGpuOptions* options);

// Serializes GPU options and returns the components needed to create opaque
// options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions`.
LiteRtStatus LrtGetOpaqueGpuOptionsData(const LrtGpuOptions* options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*));

// Enables the GPU accelerator constant tensor sharing to enable sharing of
// constant tensors between different subgraphs.
LiteRtStatus LrtSetGpuOptionsConstantTensorsSharing(LrtGpuOptions* gpu_options,
                                                    bool enable);

// Enables the GPU accelerator infinite float capping to enforce capping
// inf/-inf to max float values for the softmax input and padding
LiteRtStatus LrtSetGpuOptionsInfiniteFloatCapping(LrtGpuOptions* gpu_options,
                                                  bool enable);

// Enables the GPU accelerator benchmark mode. This will disable some
// optimizations that are not needed for benchmarking.
LiteRtStatus LrtSetGpuOptionsBenchmarkMode(LrtGpuOptions* gpu_options,
                                           bool enable);

// Sets the GPU backend.
LiteRtStatus LrtSetGpuOptionsGpuBackend(LrtGpuOptions* gpu_options,
                                        LiteRtGpuBackend backend);

// Set to true to run in external tensors mode. This allows GPU
// Accelerator to always use external tensors (PHWC4 format) as inputs and
// outputs. This mode mostly gives a slightly lower performance but it reduces
// additional GPU-GPU copies for input and output tensors.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LrtSetGpuOptionsExternalTensorsMode(LrtGpuOptions* gpu_options,
                                                 bool enable);

// Add a prefix pattern to match external tensors. When ExternalTensorsMode is
// not used (default behavior), all input and output tensors requires PHWC4
// layout conversion. This pattern is useful for state tensors to reduce the
// layout conversion. For example, if the prefix pattern is "kv_cache_", then
// all tensors whose names begin with "kv_cache_" will be exempted.
LiteRtStatus LrtAddGpuOptionsExternalTensorPattern(LrtGpuOptions* gpu_options,
                                                   const char* pattern);

// Add a prefix pattern to match buffer storage tensors. When this pattern is
// matched, the tensor will use buffer storage type.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LrtAddGpuOptionsBufferStorageTensorPattern(
    LrtGpuOptions* gpu_options, const char* pattern);

// Sets the GPU priority. Low priority helps to unblock UI workloads.
//
// WARNING: This is an experimental feature and subject to change.
LiteRtStatus LrtSetGpuOptionsGpuPriority(LrtGpuOptions* gpu_options,
                                         LiteRtGpuPriority priority);

// This enables dynamic range quantization of the input tensor for large sized
// fully connected and convolution operations, if the device supports it. This
// will result in accuracy loss, since the input tensor will be quantized to
// 8-bit. Turning this on will also increase the initialization time to
// calculate some extra constant tensor. `enable_constant_tensors_sharing` must
// be true to use this.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LrtGpuOptions* gpu_options, bool enable);

// Sets the GPU accelerator precision. e.g. FP16, FP32, etc.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsPrecision(
    LrtGpuOptions* gpu_options, LiteRtDelegatePrecision precision);

// Sets the GPU buffer storage type. If true, the delegate will use buffer
// storage type.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LrtGpuOptions* gpu_options,
    LiteRtDelegateBufferStorageType buffer_storage_type);

// If true, the delegate will prefer to use textures rather than buffers for
// weights. Use option when weights in texture has better performance.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    LrtGpuOptions* gpu_options, bool prefer_texture_weights);

// The nul-terminated directory to use for serialization.
// Whether serialization actually happens or not is dependent on backend used
// and validity of this directory.
// Set to nullptr implies the delegate will not try serialization.
//
// NOTE: Users should ensure that this directory is private to the app to
// avoid data access issues.
// Delegate stores the pointer to the string and doesn't take ownership of the
// memory. The string memory must outlive the `gpu_options` object.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializationDir(
    LrtGpuOptions* gpu_options, const char* serialization_dir);

// The unique nul-terminated token string that acts as a 'namespace' for
// all serialization entries.
// Should be unique to a particular model (graph & constants).
// For an example of how to generate this from a TFLite model, see
// StrFingerprint() in lite/delegates/serialization.h.
//
// Set to nullptr implies the delegate will not try serialization.
// Delegate stores the pointer to the string and doesn't take ownership of the
// memory. The string memory must outlive the `gpu_options` object.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsModelCacheKey(
    LrtGpuOptions* gpu_options, const char* model_cache_key);

// The file descriptor to use for program caching.
// If set, the delegate will use this file descriptor to read and write the
// program cache.
// If it is not set, the delegate will use the serialization_dir + model_token
// to determine where to read and write the program cache from.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsProgramCacheFd(
    LrtGpuOptions* gpu_options, int program_cache_fd);

// When set to true AND the serialization_dir and model_cache_key are also set,
// the delegate will serialize the program cache.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    LrtGpuOptions* gpu_options, bool serialize_program_cache);

// If true, only the compiled programs will be cached.
// If false, gpu graph info including work group sizes (and all compiled
// programs depending on backend) will be cached.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
    LrtGpuOptions* gpu_options, bool cache_only_compiled_programs);

// Set to true to serialize immutable external tensors. By default only the
// non-external tensors are serialized.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    LrtGpuOptions* gpu_options, bool serialize_external_tensors);

// Sets whether to madvise the original shared tensors after use. Note that
// this boolean flag is to disable madvise which is enabled by default.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    LrtGpuOptions* gpu_options, bool madvise_original_shared_tensors);

// Sets whether to disable Vulkan kernel shader optimization.
LiteRtStatus LrtSetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    LrtGpuOptions* gpu_options, bool disable_shader_optimization);

// Sets the number of steps of command buffer preparations.
LiteRtStatus
LrtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    LrtGpuOptions* gpu_options, int num_steps_of_command_buffer_preparations);

// Sets whether to use Metal argument buffers.
LiteRtStatus LrtSetGpuOptionsUseMetalArgumentBuffers(
    LrtGpuOptions* gpu_options, bool use_metal_argument_buffers);

// Sets the wait type.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsWaitType(
    LrtGpuOptions* gpu_options, LiteRtGpuWaitType wait_type);

// Sets the preferred device substring.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    LrtGpuOptions* gpu_options, const char* preferred_device_substr);

// Sets the number of threads for webgpu upload.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    LrtGpuOptions* gpu_options, int num_threads_to_upload);

// Sets the number of threads for webgpu kernel shader compilation.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    LrtGpuOptions* gpu_options, int num_threads_to_compile);

// Sets whether to convert weights on GPU. It's an experimental feature.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    LrtGpuOptions* gpu_options, bool convert_weights_on_gpu);

// Sets whether to wait for weights conversion on GPU complete.
// It's an experimental feature and should only be used when converting
// weights on GPU.
LiteRtStatus LrtSetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
    LrtGpuOptions* gpu_options, bool wait);

// Sets the hint to fully delegate to single delegate.
// This is an ADVANCED option and should only be set if every subgraph is
// known to be fully delegated to a single delegate. This flag can be used to
// skip unnecessary memory allocations.
LiteRtStatus LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
    LrtGpuOptions* gpu_options, bool hint_fully_delegated_to_single_delegate);

// Declarations below this point are meant to be used by accelerator code.

const char* LrtGetGpuOptionsIdentifier();

LiteRtStatus LrtGetGpuOptionsConstantTensorsSharing(
    bool* enabled, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsInfiniteFloatCapping(bool* enabled,
                                                  const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsBenchmarkMode(bool* enabled,
                                           const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsGpuBackend(LiteRtGpuBackend* backend,
                                        const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsExternalTensorsMode(bool* enabled,
                                                 const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* type, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    bool* prefer_texture_weights, const LrtGpuOptions* options);

// Returns serialization directory.
// The returned string pointer is owned by the user of
// LrtSetGpuAcceleratorCompilationOptionsSerializationDir() API.
LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
    const char** serialization_dir, const LrtGpuOptions* options);

// Returns model cache key.
// The returned string pointer is owned by the user of
// LrtSetGpuAcceleratorCompilationOptionsModelCacheKey() API.
LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
    const char** model_cache_key, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
    int* program_cache_fd, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    bool* serialize_program_cache, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
    bool* cache_only_compiled_programs, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    bool* serialize_external_tensors, const LrtGpuOptions* options);

LiteRtStatus LrtGetNumGpuAcceleratorCompilationOptionsExternalTensorPatterns(
    int* num_patterns, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsExternalTensorPattern(
    const char** external_tensor_pattern, int pattern_index,
    const LrtGpuOptions* options);

LiteRtStatus
LrtGetNumGpuAcceleratorCompilationOptionsBufferStorageTensorPatterns(
    int* num_patterns, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsBufferStorageTensorPattern(
    const char** buffer_storage_tensor_pattern, int pattern_index,
    const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsGpuPriority(LiteRtGpuPriority* priority,
                                         const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    bool* madvise_original_shared_tensors, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    bool* disable_shader_optimization, const LrtGpuOptions* options);

LiteRtStatus
LrtGetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    int* num_steps_of_command_buffer_preparations,
    const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsUseMetalArgumentBuffers(
    const LrtGpuOptions* options, bool* use_metal_argument_buffers);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtGpuWaitType* wait_type, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    const char** preferred_device_substr, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    int* num_threads_to_upload, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    int* num_threads_to_compile, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    bool* convert_weights_on_gpu, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
    bool* wait, const LrtGpuOptions* options);

LiteRtStatus LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
    bool* hint_fully_delegated_to_single_delegate,
    const LrtGpuOptions* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_GPU_OPTIONS_H_
