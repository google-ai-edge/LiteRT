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

#include "litert/cc/options/litert_gpu_options.h"

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

const char* GpuOptions::GetPayloadIdentifier() {
  return LrtGetGpuOptionsIdentifier();
}

Expected<GpuOptions> GpuOptions::Create() {
  LrtGpuOptions* options;
  LITERT_RETURN_IF_ERROR(LrtCreateGpuOptions(&options));
  return GpuOptions(options);
}

LiteRtStatus GpuOptions::EnableConstantTensorSharing(bool enabled) {
  return LrtSetGpuOptionsConstantTensorsSharing(options_, enabled);
}

LiteRtStatus GpuOptions::EnableInfiniteFloatCapping(bool enabled) {
  return LrtSetGpuOptionsInfiniteFloatCapping(options_, enabled);
}

LiteRtStatus GpuOptions::EnableBenchmarkMode(bool enabled) {
  return LrtSetGpuOptionsBenchmarkMode(options_, enabled);
}

Expected<void> GpuOptions::SetBackend(Backend backend) {
  LITERT_RETURN_IF_ERROR(LrtSetGpuOptionsGpuBackend(
      options_, static_cast<LiteRtGpuBackend>(backend)));
  return {};
}

LiteRtStatus GpuOptions::EnableAllowSrcQuantizedFcConvOps(bool enabled) {
  return LrtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
      options_, enabled);
}

Expected<void> GpuOptions::SetPrecision(Precision precision) {
  LITERT_RETURN_IF_ERROR(LrtSetGpuAcceleratorCompilationOptionsPrecision(
      options_, static_cast<LiteRtDelegatePrecision>(precision)));
  return {};
}

Expected<void> GpuOptions::SetBufferStorageType(BufferStorageType type) {
  LITERT_RETURN_IF_ERROR(
      LrtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
          options_, static_cast<LiteRtDelegateBufferStorageType>(type)));
  return {};
}

LiteRtStatus GpuOptions::SetPreferTextureWeights(bool prefer_texture_weights) {
  return LrtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      options_, prefer_texture_weights);
}

LiteRtStatus GpuOptions::SetSerializationDir(const char* serialization_dir) {
  return LrtSetGpuAcceleratorCompilationOptionsSerializationDir(
      options_, serialization_dir);
}

LiteRtStatus GpuOptions::SetModelCacheKey(const char* model_cache_key) {
  return LrtSetGpuAcceleratorCompilationOptionsModelCacheKey(options_,
                                                             model_cache_key);
}

LiteRtStatus GpuOptions::SetProgramCacheFd(int program_cache_fd) {
  return LrtSetGpuAcceleratorCompilationOptionsProgramCacheFd(options_,
                                                              program_cache_fd);
}

LiteRtStatus GpuOptions::SetSerializeProgramCache(
    bool serialize_program_cache) {
  return LrtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      options_, serialize_program_cache);
}

LiteRtStatus GpuOptions::SetSerializeExternalTensors(
    bool serialize_external_tensors) {
  return LrtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
      options_, serialize_external_tensors);
}

LiteRtStatus GpuOptions::EnableExternalTensorsMode(bool enabled) {
  return LrtSetGpuOptionsExternalTensorsMode(options_, enabled);
}

LiteRtStatus GpuOptions::AddExternalTensorPattern(const char* pattern) {
  return LrtAddGpuOptionsExternalTensorPattern(options_, pattern);
}

LiteRtStatus GpuOptions::AddBufferStorageTensorPattern(const char* pattern) {
  return LrtAddGpuOptionsBufferStorageTensorPattern(options_, pattern);
}

Expected<void> GpuOptions::SetPriority(Priority priority) {
  LITERT_RETURN_IF_ERROR(LrtSetGpuOptionsGpuPriority(
      options_, static_cast<LiteRtGpuPriority>(priority)));
  return {};
}

LiteRtStatus GpuOptions::SetMadviseOriginalSharedTensors(
    bool madvise_original_shared_tensors) {
  return LrtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
      options_, madvise_original_shared_tensors);
}

LiteRtStatus GpuOptions::SetNumStepsOfCommandBufferPreparations(
    int num_steps_of_command_buffer_preparations) {
  return LrtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
      options_, num_steps_of_command_buffer_preparations);
}

#ifdef __APPLE__
LiteRtStatus GpuOptions::SetUseMetalArgumentBuffers(
    bool use_metal_argument_buffers) {
  return LrtSetGpuOptionsUseMetalArgumentBuffers(options_,
                                                 use_metal_argument_buffers);
}
#endif  // __APPLE__

LiteRtStatus GpuOptions::SetSyncExecutionModeWaitType(
    SyncExecutionModeWaitType wait_type) {
  return LrtSetGpuAcceleratorRuntimeOptionsWaitType(
      options_, static_cast<LiteRtGpuWaitType>(wait_type));
}

LiteRtStatus GpuOptions::SetPreferredDeviceSubstr(
    const char* preferred_device_substr) {
  return LrtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
      options_, preferred_device_substr);
}

LiteRtStatus GpuOptions::SetNumThreadsToUpload(int num_threads_to_upload) {
  return LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
      options_, num_threads_to_upload);
}

LiteRtStatus GpuOptions::SetNumThreadsToCompile(int num_threads_to_compile) {
  return LrtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
      options_, num_threads_to_compile);
}

LiteRtStatus GpuOptions::SetConvertWeightsOnGpu(bool convert_weights_on_gpu) {
  return LrtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
      options_, convert_weights_on_gpu);
}

LiteRtStatus GpuOptions::SetHintFullyDelegatedToSingleDelegate(
    bool hint_fully_delegated_to_single_delegate) {
  return LrtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
      options_, hint_fully_delegated_to_single_delegate);
}

LiteRtStatus GpuOptions::DisableShaderOptimization(bool disable) {
  return LrtSetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
      options_, disable);
}

LiteRtStatus GpuOptions::WaitForWeightsConversionComplete(bool wait) {
  return LrtSetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
      options_, wait);
}

LiteRtStatus GpuOptions::CacheCompiledProgramsOnly(bool only) {
  return LrtSetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
      options_, only);
}

}  // namespace litert
