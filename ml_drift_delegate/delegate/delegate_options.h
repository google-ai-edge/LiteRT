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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_OPTIONS_H_

#include "ml_drift_delegate/delegate/precision.h"

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu_cpp.h>
#endif  // __EMSCRIPTEN__

#ifdef __cplusplus
#include <memory>
#include <set>
#include <string>

namespace weight_loader {
class WeightLoader;
}  // namespace weight_loader

extern "C" {
#endif  // __cplusplus

// Forward declaration of MLDriftDelegateOptions for C APIs.
struct MlDriftDelegateOptions;
typedef struct MlDriftDelegateOptions MlDriftDelegateOptions;

// Forward declaration of LiteRtRuntimeContext.
struct LiteRtRuntimeContext;

#ifdef __cplusplus
}  // extern "C"

typedef enum {
  // waitUntilCompleted
  kGpuDelegateWaitTypePassive,
  // Minimize latency. It uses active spinning instead of mutex and consumes
  // additional CPU resources.
  kGpuDelegateWaitTypeActive,
  // Useful when the output is used with GPU pipeline then or if external
  // command encoder is set.
  kGpuDelegateWaitTypeDoNotWait,
  // Tries to avoid GPU sleep mode.
} GpuDelegateWaitType;

typedef enum {
  kGpuNormalPriority = 0,  // Default priority.
  kGpuLowPriority = 1,     // Low priority help to unblock UI workloads.
} GpuPriority;

struct MlDriftDelegateOptions {
  MlDriftDelegatePrecision precision;

  // If true, only delegates the node range of `debug_first_delegate_node_index`
  // and `debug_last_delegate_node_index`.
  // Note: This is for debugging purpose.
  bool debug_delegate_partition;
  // This sets the index of the first node that could be delegated.
  int debug_first_delegate_node_index;
  // This sets the index of the last node that could be delegated.
  int debug_last_delegate_node_index;
  // Allows sharing of constant tensors between different subgraphs.
  bool enable_constant_tensors_sharing;
  // If true, the delegate will improve tuning time, but inference can be
  // slower.
  bool enable_fast_tuning;
  // If true, the delegate will enable op profiling.
  bool enable_op_profiling;
  // If true, the delegate will enable gpu op profiling detailed report.
  bool enable_op_profiling_detailed_report;
  // Set to enforce capping inf/-inf to max float values for the softmax input
  // and padding.
  bool enable_infinite_float_capping;

  // The nul-terminated directory to use for serialization.
  // Whether serialization actually happens or not is dependent on backend used
  // and validity of this directory.
  // Set to nullptr implies the delegate will not try serialization.
  //
  // NOTE: Users should ensure that this directory is private to the app to
  // avoid data access issues.
  // Delegate copies the string and doesn't take ownership of the memory.
  const char* serialization_dir;

  // The unique nul-terminated token string that acts as a 'namespace' for
  // all serialization entries.
  // Should be unique to a particular model (graph & constants).
  // For an example of how to generate this from a TFLite model, see
  // StrFingerprint() in lite/delegates/serialization.h.
  //
  // Set to nullptr implies the delegate will not try serialization.
  // Delegate copies the string and doesn't take ownership of the memory.
  const char* model_token;

  // The file descriptor to use for program caching.
  // If set, the delegate will use this file descriptor to read and write the
  // program cache.
  // If it is not set, the delegate will use the serialization_dir + model_token
  // to determine where to read and write the program cache from.
  int program_cache_fd;
  // The file descriptor to use for weight caching.
  // If set, the delegate will use this file descriptor to read and write the
  // weight cache.
  // If it is not set, the delegate will use the serialization_dir + model_token
  // to determine where to read and write the weight cache from.
  int weight_cache_fd;

  // When set to true AND the serialization_dir and model_token are also
  // set, the delegate will serialize the program cache.
  bool serialize_program_cache;

  // Set to true to serialize immutable external tensors. By default only the
  // non-external tensors are serialized.
  bool serialize_external_tensors;

  // If true, the delegate will prefer to use textures rather than buffers for
  // weights. Use option when weights in texture has better performance.
  bool prefer_texture_weights;

  // Set to true to enable uploading tensor weights directly without processing.
  // This requires the model file to have pre-processed weights.
  //
  // WARNING: This differs from the typical serialization path because the
  // pre-processed immutable external tensors are stored in the model file
  // itself and not in a separate serialization directory. This option reduces
  // the disk space required and the memory usage on the first run. However, if
  // the prepacked weights become incompatible with the current ML Drift
  // kernels, there is no fallback path. Due to this risk, this option is
  // intended for ADVANCED USERS only.
  bool has_prepacked_external_tflite_tensors;

  // This enables dynamic range quantization of the input tensor for large
  // sized fully connected and convolution operations, if the device supports
  // it.
  // This will result in accuracy loss, since the input tensor will be
  // quantized to 8-bit.
  // Turning this on will also increase the initialization time to calculate
  // some extra constant tensor.
  // `enable_constant_tensors_sharing` must be true to use this.
  bool allow_src_quantized_fc_conv_ops;

  // If true, the delegate hints waiting for completion. This is for some
  // backends , e.g. OpenCL on AMD and Mali GPUs, to wait for all the enqueued
  // commands to be completed after each invoke. This feature is only applied to
  // the OpenCL backend and the goal is to fix a known quality issue on AMD and
  // Mali GPUs. By default, it is false. Set this to true can help to fix the
  // quality issue, it will reduce the performance around 10% for prefill and
  // decode.
  bool hint_waiting_for_completion;

  // If true, the delegate will run in benchmark mode.
  // This will disable some optimizations that are not needed for benchmarking.
  bool litert_benchmark_mode;

  // If true, it uses input and output tensors directly as external tensors.
  // External tensors are PHWC4 format, so no additional conversion is needed to
  // use them.
  // If false, it converts user provided GPU input and outputs to PHWC4 format.
  // This mode is default behavior since it provides slightly better
  // performance and easier to use.
  bool litert_external_tensors_mode;

  // Prefix pattern of the tensor name that is used for external tensors. When
  // it matches, those tensors won't use litert_no_external_tensors_mode.
  // Which means these input and output tensors are binded directly to the
  // InferenceContext.
  // Even when external tensors mode is not enabled, this list of patterns will
  // still allow to use external tensors.
  std::set<std::string> litert_external_tensor_patterns;

  // If true, the delegate will use buffer storage type.
  // By default, the delegate will try to use the fastest storage type for the
  // device. In the case of TEXTURE_2D type (non-Apple devices), this can cause
  // increased memory consumption. Turn this flag on to force BUFFER storage
  // type, if memory is a higher concern than latency.
  bool use_buffer_storage_type;

  // Prefix pattern of the tensor name that is used for buffer storage type.
  // When it matches, those tensors will use buffer storage type.
  //
  // WARNING: This option is experimental and subject to change.
  std::set<std::string> litert_buffer_storage_tensor_patterns;

  // If true, the delegate will madvise the original tensor memory after use.
  bool madvise_original_shared_tensors;

  // The priority of the GPU task.
  GpuPriority gpu_priority;

  // Non-owning pointer set by LiteRT to expose the shared WeightLoader.
  weight_loader::WeightLoader* weight_loader;

  // Wait type options on synchronous execution mode, i.e when
  // IsAsyncExecutionMode() returns false. It's meaningful only when delegate
  // calls DelegateKernelLitert::HandleOutputs(), e.g. OpenCL and WebGPU.
  GpuDelegateWaitType wait_type;

  // OpenCL and WebGPU only.
  //
  // ML Drift's optimal performance requires the weights to be arranged in
  // layouts, which is typically different from TFL flatbuffer's weight layouts.
  // If true, the weights rearrangement, if supported, will utilize GPU.
  // Otherwise, it will be always done with CPU.
  bool convert_weights_on_gpu;

  // If true, the delegate will wait for weights conversion on GPU complete
  // during initialization. It's meaningful only when
  // convert_weights_on_gpu is true.
  bool wait_for_weights_conversion_complete;

  // When program_cache is enabled (i.e. either program_cache_fd > 0 or
  // serialize_program_cache is true), this flag determines whether the program
  // cache has only the compiled shader programs or not.
  bool cache_compiled_programs_only;

  // WebGPU and Vulkan only.
  //
  // Number of steps to prepare command buffers in advance.
  // 0 (default value) = No command buffer preparation in advance. It must be 0
  //     when any GPU resource bindings are changed during inference.
  // 1 = Prepare one step ahead assuming that all the gpu resource bindings are
  //     the same as the previous step.
  // 2 = Prepare two steps ahead. It can be used when gpu resource bindings are
  //     the same as the previous previous step, e.g. LLM inferences which swaps
  //     input and output KV caches.
  int num_steps_of_command_buffer_preparations;

  // WebGPU only.
  //
#ifdef __EMSCRIPTEN__
  wgpu::Device webGpuDevice;
  wgpu::AdapterInfo webGpuAdapterInfo;
#endif  // __EMSCRIPTEN__

  // If true, for each subgraph delegated, the delegate will allocate a WebGpu
  // tensor as the GPU memory for each input tensor and output tensor and the
  // WebGpu tensor will be associated with the TFLite tensor's BufferHandle.
  // Note: if the model is not fully delegated to ML Drift Delegate, the input
  // and output tensors to be allocated with GPU memory are each partition's
  // input and output tensors, rather than the model's input and output tensors.
  //
  // The WebGpu tensor can be accessed by calling
  // `tflite::ml_drift::webgpu::GetSpatialTensor` with the TFLite tensor's
  // BufferHandle.
  //
  // Please see `DelegateWithAllocatingIoOnGpu` test to check how it's used.
  bool allocate_gpu_memory_for_io_tensors;

  // Preferred WebGPU device name substring, case-insensitive.
  // If not empty, the adapter which the device name contains the substring will
  // be chosen.
  // If empty, the device will be determined by other factors.
  std::string preferred_device_substr;

  // Set to true to hint that the delegate is fully delegated to a single
  // delegate.
  bool hint_fully_delegated_to_single_delegate;

  // Number of threads for webgpu upload.
  int num_threads_to_upload;
  // Number of threads for webgpu kernel shader compilation.
  int num_threads_to_compile;

  // Vulkan only.
  //
  // If true, the delegate will disable kernel shader optimization.
  bool disable_shader_optimization;

#ifdef __APPLE__
  // Metal only.
  //
  // If true, the delegate will use Metal argument buffers.
  // More details:
  // https://developer.apple.com/documentation/Metal/managing-groups-of-resources-with-argument-buffers
  // Metal argument buffer is a specific type of buffer that contains pointers
  // to other resources (texture, buffers, etc). For some LLMs, we
  // encountered a buffer-out-of-bounds issue where we exhausted the available
  // buffer binding slots (indices) in the Metal Shading Language (MSL)
  // argument table for a single compute function. In this case, we can use
  // Metal argument buffers to reduce the number of buffers bound to the
  // argument table.
  // The capability of Metal argument buffers depends on the GPU Tier (Tier 1
  // vs. Tier 2).
  // Tier 1 (Older iOS / Some Intel Macs): Cannot access arrays of
  // textures/samplers by index. This means ML Drift will not work correctly
  // with argument buffers on devices that only support Tier 1 and use
  // textures. Textures are not often used in ML Drift with Metal. Therefore, we
  // will disable the use_metal_argument_buffers option on these Tier 1 devices.
  // Tier 2 (Apple Silicon M-series, Recent A-series, Discrete AMD): For Tier 2,
  // argument buffers can be mutable so that the GPU and CPU can both modify
  // their contents at any time.
  bool use_metal_argument_buffers;
  // If true, the delegate will use MTLResidencySet to prevent memory swapping.
  bool enable_metal_residency_set;
#endif  // __APPLE__

  // LiteRT Runtime Context.
  struct LiteRtRuntimeContext* runtime_context;

  // If > 0, specifies the kernel (op) batch size, for a flush.
  int kernel_batch_size = 0;

  // Pointer to SharedTensorMaps provided by the client to sharing GPU tensors,
  // weights across delegates.
  void* shared_tensor_maps_from_client = nullptr;
};

namespace litert::ml_drift {
using MlDriftDelegateOptionsPtr = std::unique_ptr<MlDriftDelegateOptions>;
}  // namespace litert::ml_drift

#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_OPTIONS_H_
