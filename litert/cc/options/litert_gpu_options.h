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
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT GPU options.
class GpuOptions {
 public:
  GpuOptions() : options_(nullptr) {}
  explicit GpuOptions(LrtGpuOptions* options) : options_(options) {}
  ~GpuOptions() {
    if (options_) {
      LrtDestroyGpuOptions(options_);
    }
  }

  GpuOptions(GpuOptions&& other) noexcept : options_(other.options_) {
    other.options_ = nullptr;
  }
  GpuOptions& operator=(GpuOptions&& other) noexcept {
    if (this != &other) {
      if (options_) LrtDestroyGpuOptions(options_);
      options_ = other.options_;
      other.options_ = nullptr;
    }
    return *this;
  }

  // Delete copy constructor and assignment
  GpuOptions(const GpuOptions&) = delete;
  GpuOptions& operator=(const GpuOptions&) = delete;

  LrtGpuOptions* Get() const { return options_; }
  LrtGpuOptions* Release() {
    auto* res = options_;
    options_ = nullptr;
    return res;
  }

  static Expected<GpuOptions> Create() {
    LrtGpuOptions* options;
    LITERT_RETURN_IF_ERROR(LrtCreateGpuOptions(&options));
    return GpuOptions(options);
  }
  static const char* GetPayloadIdentifier() {
    return LrtGetGpuOptionsIdentifier();
  }

  LiteRtStatus EnableConstantTensorSharing(bool enabled) {
    return LrtSetGpuOptionsConstantTensorsSharing(options_, enabled);
  }
  LiteRtStatus EnableInfiniteFloatCapping(bool enabled) {
    return LrtSetGpuOptionsInfiniteFloatCapping(options_, enabled);
  }
  LiteRtStatus EnableBenchmarkMode(bool enabled) {
    return LrtSetGpuOptionsBenchmarkMode(options_, enabled);
  }
  LiteRtStatus EnableAllowSrcQuantizedFcConvOps(bool enabled) {
    return LrtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
        options_, enabled);
  }
  LiteRtStatus HintWaitingForCompletion(bool wait) {
    return LrtSetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion(options_,
                                                                      wait);
  }

  enum class Precision : int {
    kDefault = kLiteRtDelegatePrecisionDefault,
    kFp16 = kLiteRtDelegatePrecisionFp16,
    kFp32 = kLiteRtDelegatePrecisionFp32,
  };
  Expected<void> SetPrecision(Precision precision) {
    LITERT_RETURN_IF_ERROR(LrtSetGpuAcceleratorCompilationOptionsPrecision(
        options_, static_cast<LiteRtDelegatePrecision>(precision)));
    return {};
  }

  enum class BufferStorageType : int {
    kDefault = kLiteRtDelegateBufferStorageTypeDefault,
    kBuffer = kLiteRtDelegateBufferStorageTypeBuffer,
    kTexture2D = kLiteRtDelegateBufferStorageTypeTexture2D,
  };
  Expected<void> SetBufferStorageType(BufferStorageType type) {
    LITERT_RETURN_IF_ERROR(
        LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
            options_, static_cast<LiteRtDelegateBufferStorageType>(type)));
    return {};
  }

  LiteRtStatus SetPreferTextureWeights(bool prefer_texture_weights) {
    return LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
        options_, prefer_texture_weights);
  }
  LiteRtStatus SetSerializationDir(const char* serialization_dir) {
    return LrtSetGpuAcceleratorCompilationOptionsSerializationDir(
        options_, serialization_dir);
  }
  LiteRtStatus SetModelCacheKey(const char* model_cache_key) {
    return LrtSetGpuAcceleratorCompilationOptionsModelCacheKey(options_,
                                                               model_cache_key);
  }
  LiteRtStatus SetProgramCacheFd(int program_cache_fd) {
    return LrtSetGpuAcceleratorCompilationOptionsProgramCacheFd(
        options_, program_cache_fd);
  }
  LiteRtStatus SetSerializeProgramCache(bool serialize_program_cache) {
    return LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
        options_, serialize_program_cache);
  }
  LiteRtStatus SetSerializeExternalTensors(bool serialize_external_tensors) {
    return LrtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
        options_, serialize_external_tensors);
  }
  LiteRtStatus EnableExternalTensorsMode(bool enabled) {
    return LrtSetGpuOptionsExternalTensorsMode(options_, enabled);
  }
  LiteRtStatus AddExternalTensorPattern(const char* pattern) {
    return LrtAddGpuOptionsExternalTensorPattern(options_, pattern);
  }
  LiteRtStatus AddBufferStorageTensorPattern(const char* pattern) {
    return LrtAddGpuOptionsBufferStorageTensorPattern(options_, pattern);
  }

  enum class Backend : int {
    kAutomatic = kLiteRtGpuBackendAutomatic,
    kOpenCl = kLiteRtGpuBackendOpenCl,
    kWebGpu = kLiteRtGpuBackendWebGpu,
    kOpenGl = kLiteRtGpuBackendOpenGl,
  };
  Expected<void> SetBackend(Backend backend) {
    LITERT_RETURN_IF_ERROR(LrtSetGpuOptionsGpuBackend(
        options_, static_cast<LiteRtGpuBackend>(backend)));
    return {};
  }

  enum class Priority : int {
    kDefault = kLiteRtGpuPriorityDefault,
    kLow = kLiteRtGpuPriorityLow,
    kNormal = kLiteRtGpuPriorityNormal,
    kHigh = kLiteRtGpuPriorityHigh,
  };
  Expected<void> SetPriority(Priority priority) {
    LITERT_RETURN_IF_ERROR(LrtSetGpuOptionsGpuPriority(
        options_, static_cast<LiteRtGpuPriority>(priority)));
    return {};
  }

  LiteRtStatus SetMadviseOriginalSharedTensors(
      bool madvise_original_shared_tensors) {
    return LrtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
        options_, madvise_original_shared_tensors);
  }

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
      int num_steps_of_command_buffer_preparations) {
    return LrtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
        options_, num_steps_of_command_buffer_preparations);
  }

#ifdef __APPLE__
  /// @brief Sets whether to use Metal argument buffers.
  /// @warning This is only applicable to the Metal backend.
  LiteRtStatus SetUseMetalArgumentBuffers(bool use_metal_argument_buffers) {
    return LrtSetGpuOptionsUseMetalArgumentBuffers(options_,
                                                   use_metal_argument_buffers);
  }
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
      SyncExecutionModeWaitType wait_type) {
    return LrtSetGpuAcceleratorRuntimeOptionsWaitType(
        options_, static_cast<LiteRtGpuWaitType>(wait_type));
  }

  /// @brief Sets the preferred WebGPU device substring.
  LiteRtStatus SetPreferredDeviceSubstr(const char* preferred_device_substr) {
    return LrtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
        options_, preferred_device_substr);
  }

  /// @brief Sets the number of threads for WebGPU upload.
  LiteRtStatus SetNumThreadsToUpload(int num_threads_to_upload) {
    return LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
        options_, num_threads_to_upload);
  }

  /// @brief Sets the number of threads for WebGPU kernel shader compilation.
  LiteRtStatus SetNumThreadsToCompile(int num_threads_to_compile) {
    return LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
        options_, num_threads_to_compile);
  }

  /// @brief Sets whether to convert weights on the GPU.
  /// @note This is an experimental feature.
  LiteRtStatus SetConvertWeightsOnGpu(bool convert_weights_on_gpu) {
    return LrtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
        options_, convert_weights_on_gpu);
  }

  /// @brief Sets a hint that the graph will be fully delegated to a single
  /// delegate.
  LiteRtStatus SetHintFullyDelegatedToSingleDelegate(
      bool hint_fully_delegated_to_single_delegate) {
    return LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
        options_, hint_fully_delegated_to_single_delegate);
  }

  /// @brief Sets whether to disable Vulkan kernel shader optimization.
  LiteRtStatus DisableShaderOptimization(bool disable) {
    return LrtSetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
        options_, disable);
  }

  /// @brief Wait for weights conversion on GPU complete.
  /// @note This is an experimental feature and should only be used when
  /// converting weights on GPU.
  LiteRtStatus WaitForWeightsConversionComplete(bool wait) {
    return LrtSetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
        options_, wait);
  }

  /// @brief Cache only the compiled programs. If true, only the compiled
  /// programs will be cached. If false, gpu graph info including work group
  /// sizes (and all compiled programs depending on backend) will be cached.
  LiteRtStatus CacheCompiledProgramsOnly(bool only) {
    return LrtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
        options_, only);
  }

  /// @brief Sets the kernel (op) batch size, for a flush.
  /// @note Only for OpenCL backend.
  LiteRtStatus SetKernelBatchSize(int kernel_batch_size) {
    return LrtSetGpuAcceleratorRuntimeOptionsKernelBatchSize(options_,
                                                             kernel_batch_size);
  }

 private:
  LrtGpuOptions* options_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GPU_OPTIONS_H_
