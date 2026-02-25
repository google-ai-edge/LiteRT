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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GPU_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT GPU options.
class GpuOptions : public litert::OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static Expected<GpuOptions> Create();
  static const char* GetPayloadIdentifier();

  LiteRtStatus EnableConstantTensorSharing(bool enabled);
  LiteRtStatus EnableInfiniteFloatCapping(bool enabled);
  LiteRtStatus EnableBenchmarkMode(bool enabled);
  LiteRtStatus EnableAllowSrcQuantizedFcConvOps(bool enabled);

  enum class Precision : int {
    kDefault = kLiteRtDelegatePrecisionDefault,
    kFp16 = kLiteRtDelegatePrecisionFp16,
    kFp32 = kLiteRtDelegatePrecisionFp32,
  };
  Expected<void> SetPrecision(Precision precision);

  enum class BufferStorageType : int {
    kDefault = kLiteRtDelegateBufferStorageTypeDefault,
    kBuffer = kLiteRtDelegateBufferStorageTypeBuffer,
    kTexture2D = kLiteRtDelegateBufferStorageTypeTexture2D,
  };
  Expected<void> SetBufferStorageType(BufferStorageType type);

  LiteRtStatus SetPreferTextureWeights(bool prefer_texture_weights);
  LiteRtStatus SetSerializationDir(const char* serialization_dir);
  LiteRtStatus SetModelCacheKey(const char* model_cache_key);
  LiteRtStatus SetProgramCacheFd(int program_cache_fd);
  LiteRtStatus SetSerializeProgramCache(bool serialize_program_cache);
  LiteRtStatus SetSerializeExternalTensors(bool serialize_external_tensors);
  LiteRtStatus EnableExternalTensorsMode(bool enabled);
  LiteRtStatus AddExternalTensorPattern(const char* pattern);
  LiteRtStatus AddBufferStorageTensorPattern(const char* pattern);

  enum class Backend : int {
    kAutomatic = kLiteRtGpuBackendAutomatic,
    kOpenCl = kLiteRtGpuBackendOpenCl,
    kWebGpu = kLiteRtGpuBackendWebGpu,
    kOpenGl = kLiteRtGpuBackendOpenGl,
  };
  Expected<void> SetBackend(Backend backend);

  enum class Priority : int {
    kDefault = kLiteRtGpuPriorityDefault,
    kLow = kLiteRtGpuPriorityLow,
    kNormal = kLiteRtGpuPriorityNormal,
    kHigh = kLiteRtGpuPriorityHigh,
  };
  Expected<void> SetPriority(Priority priority);

  LiteRtStatus SetMadviseOriginalSharedTensors(
      bool madvise_original_shared_tensors);

  /// @brief Sets the number of steps to prepare WebGPU or Vulkan command
  /// buffers in advance.
  ///
  /// - `0` (default): No command buffer preparation in advance. This must be
  ///   used when any GPU resource bindings are changed during inference.
  /// - `1`: Prepare one step ahead, assuming all GPU resource bindings are the
  ///   same as the previous step.
  /// - `2`: Prepare two steps ahead. This can be used when GPU resource
  ///   bindings are the same as two steps prior (e.g., LLM inferences that
  ///   swap input and output KV caches).
  LiteRtStatus SetNumStepsOfCommandBufferPreparations(
      int num_steps_of_command_buffer_preparations);

#ifdef __APPLE__
  /// @brief Sets whether to use Metal argument buffers.
  /// @warning This is only applicable to the Metal backend.
  LiteRtStatus SetUseMetalArgumentBuffers(bool use_metal_argument_buffers);
#endif  // __APPLE__

  /// @brief Sets the wait type for synchronous execution mode.
  ///
  /// This is ignored for asynchronous execution mode.
  enum class SyncExecutionModeWaitType : int {
    /// Wait type will be automatically determined by the delegate.
    kDefault = kLiteRtGpuWaitTypeDefault,
    /// Blocked waiting for the GPU to finish.
    kPassive = kLiteRtGpuWaitTypePassive,
    /// Active busy-waiting for the GPU to finish.
    kActive = kLiteRtGpuWaitTypeActive,
    /// Do not wait for the GPU to finish. Relies on other synchronization
    /// methods like barriers or in-order queues. Not recommended for new use
    /// cases; use asynchronous execution mode instead.
    kDoNotWait = kLiteRtGpuWaitTypeDoNotWait,
  };
  LiteRtStatus SetSyncExecutionModeWaitType(
      SyncExecutionModeWaitType wait_type);

  /// @brief Sets the preferred WebGPU device substring.
  LiteRtStatus SetPreferredDeviceSubstr(const char* preferred_device_substr);

  /// @brief Sets the number of threads for WebGPU upload.
  LiteRtStatus SetNumThreadsToUpload(int num_threads_to_upload);

  /// @brief Sets the number of threads for WebGPU kernel shader compilation.
  LiteRtStatus SetNumThreadsToCompile(int num_threads_to_compile);

  /// @brief Sets whether to convert weights on the GPU.
  /// @note This is an experimental feature.
  LiteRtStatus SetConvertWeightsOnGpu(bool convert_weights_on_gpu);

  /// @brief Sets a hint that the graph will be fully delegated to a single
  /// delegate.
  LiteRtStatus SetHintFullyDelegatedToSingleDelegate(
      bool hint_fully_delegated_to_single_delegate);

  /// @brief Sets whether to disable Vulkan kernel shader optimization.
  LiteRtStatus DisableShaderOptimization(bool disable);

  /// @brief Wait for weights conversion on GPU complete.
  /// @note This is an experimental feature and should only be used when
  /// converting weights on GPU.
  LiteRtStatus WaitForWeightsConversionComplete(bool wait);

  /// @brief Cache only the compiled programs. If true, only the compiled
  /// programs will be cached. If false, gpu graph info including work group
  /// sizes (and all compiled programs depending on backend) will be cached.
  LiteRtStatus CacheCompiledProgramsOnly(bool only);
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GPU_OPTIONS_H_
