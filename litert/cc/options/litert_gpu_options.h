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

// Builds a GPU option object that can be passed to LiteRT CompiledModel
// creation.
//
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
  [[deprecated("Use SetPrecision above instead.")]]
  LiteRtStatus SetDelegatePrecision(LiteRtDelegatePrecision precision);

  enum class BufferStorageType : int {
    kDefault = kLiteRtDelegateBufferStorageTypeDefault,
    kBuffer = kLiteRtDelegateBufferStorageTypeBuffer,
    kTexture2D = kLiteRtDelegateBufferStorageTypeTexture2D,
  };
  Expected<void> SetBufferStorageType(BufferStorageType type);
  [[deprecated("Use SetBufferStorageType above instead.")]]
  LiteRtStatus SetBufferStorageType(LiteRtDelegateBufferStorageType type);

  LiteRtStatus SetPreferTextureWeights(bool prefer_texture_weights);
  LiteRtStatus SetSerializationDir(const char* serialization_dir);
  LiteRtStatus SetModelCacheKey(const char* model_cache_key);
  LiteRtStatus SetSerializeProgramCache(bool serialize_program_cache);
  LiteRtStatus SetSerializeExternalTensors(bool serialize_external_tensors);
  LiteRtStatus EnableExternalTensorsMode(bool enabled);
  LiteRtStatus AddExternalTensorPattern(const char* pattern);

  enum class Backend : int {
    kAutomatic = kLiteRtGpuBackendAutomatic,
    kOpenCl = kLiteRtGpuBackendOpenCl,
    kWebGpu = kLiteRtGpuBackendWebGpu,
    kOpenGl = kLiteRtGpuBackendOpenGl,
  };
  Expected<void> SetBackend(Backend backend);
  [[deprecated("Use SetBackend above instead.")]]
  LiteRtStatus SetGpuBackend(LiteRtGpuBackend backend);

  enum class Priority : int {
    kDefault = kLiteRtGpuPriorityDefault,
    kLow = kLiteRtGpuPriorityLow,
    kNormal = kLiteRtGpuPriorityNormal,
    kHigh = kLiteRtGpuPriorityHigh,
  };
  Expected<void> SetPriority(Priority priority);
  [[deprecated("Use SetPriority above instead.")]]
  LiteRtStatus SetGpuPriority(LiteRtGpuPriority priority);

  LiteRtStatus SetMadviseOriginalSharedTensors(
      bool madvise_original_shared_tensors);

  // Sets the number of steps to prepare WebGPU command buffers in advance.
  // 0 (default value) = No command buffer preparation in advance. It must be 0
  //     when any GPU resource bindings are changed during inference.
  // 1 = Prepare one step ahead assuming that all the gpu resource bindings are
  //     the same as the previous step.
  // 2 = Prepare two steps ahead. It can be used when gpu resource bindings are
  //     the same as the previous previous step, e.g. LLM inferences which swaps
  //     input and output KV caches.
  LiteRtStatus SetNumStepsOfCommandBufferPreparations(
      int num_steps_of_command_buffer_preparations);
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_GPU_OPTIONS_H_
