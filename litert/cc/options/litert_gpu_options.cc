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
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

const char* GpuOptions::GetPayloadIdentifier() {
  return LiteRtGetGpuOptionsPayloadIdentifier();
}

Expected<GpuOptions> GpuOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateGpuOptions(&options));
  return GpuOptions(options, OwnHandle::kYes);
}

LiteRtStatus GpuOptions::EnableConstantTensorSharing(bool enabled) {
  return LiteRtSetGpuOptionsConstantTensorSharing(Get(), enabled);
}

LiteRtStatus GpuOptions::EnableInfiniteFloatCapping(bool enabled) {
  return LiteRtSetGpuOptionsInfiniteFloatCapping(Get(), enabled);
}

LiteRtStatus GpuOptions::EnableBenchmarkMode(bool enabled) {
  return LiteRtSetGpuOptionsBenchmarkMode(Get(), enabled);
}

Expected<void> GpuOptions::SetBackend(Backend backend) {
  LITERT_RETURN_IF_ERROR(LiteRtSetGpuOptionsGpuBackend(
      Get(), static_cast<LiteRtGpuBackend>(backend)));
  return {};
}

LiteRtStatus GpuOptions::SetGpuBackend(LiteRtGpuBackend backend) {
  return LiteRtSetGpuOptionsGpuBackend(Get(), backend);
}

LiteRtStatus GpuOptions::EnableAllowSrcQuantizedFcConvOps(bool enabled) {
  return LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
      Get(), enabled);
}

Expected<void> GpuOptions::SetPrecision(Precision precision) {
  LITERT_RETURN_IF_ERROR(LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
      Get(), static_cast<LiteRtDelegatePrecision>(precision)));
  return {};
}

LiteRtStatus GpuOptions::SetDelegatePrecision(
    LiteRtDelegatePrecision precision) {
  return LiteRtSetGpuAcceleratorCompilationOptionsPrecision(Get(), precision);
}

Expected<void> GpuOptions::SetBufferStorageType(BufferStorageType type) {
  LITERT_RETURN_IF_ERROR(
      LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
          Get(), static_cast<LiteRtDelegateBufferStorageType>(type)));
  return {};
}

LiteRtStatus GpuOptions::SetBufferStorageType(
    LiteRtDelegateBufferStorageType type) {
  return LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(Get(),
                                                                       type);
}

LiteRtStatus GpuOptions::SetPreferTextureWeights(bool prefer_texture_weights) {
  return LiteRtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
      Get(), prefer_texture_weights);
}

LiteRtStatus GpuOptions::SetSerializationDir(const char* serialization_dir) {
  return LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir(
      Get(), serialization_dir);
}

LiteRtStatus GpuOptions::SetModelCacheKey(const char* model_cache_key) {
  return LiteRtSetGpuAcceleratorCompilationOptionsModelCacheKey(
      Get(), model_cache_key);
}

LiteRtStatus GpuOptions::SetSerializeProgramCache(
    bool serialize_program_cache) {
  return LiteRtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      Get(), serialize_program_cache);
}

LiteRtStatus GpuOptions::SetSerializeExternalTensors(
    bool serialize_external_tensors) {
  return LiteRtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
      Get(), serialize_external_tensors);
}

LiteRtStatus GpuOptions::EnableExternalTensorsMode(bool enabled) {
  return LiteRtSetGpuOptionsExternalTensorsMode(Get(), enabled);
}

LiteRtStatus GpuOptions::AddExternalTensorPattern(const char* pattern) {
  return LiteRtAddGpuOptionsExternalTensorPattern(Get(), pattern);
}

Expected<void> GpuOptions::SetPriority(Priority priority) {
  LITERT_RETURN_IF_ERROR(LiteRtSetGpuOptionsGpuPriority(
      Get(), static_cast<LiteRtGpuPriority>(priority)));
  return {};
}

LiteRtStatus GpuOptions::SetGpuPriority(LiteRtGpuPriority priority) {
  return LiteRtSetGpuOptionsGpuPriority(Get(), priority);
}

LiteRtStatus GpuOptions::SetMadviseOriginalSharedTensors(
    bool madvise_original_shared_tensors) {
  return LiteRtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
      Get(), madvise_original_shared_tensors);
}

LiteRtStatus GpuOptions::SetNumStepsOfCommandBufferPreparations(
    int num_steps_of_command_buffer_preparations) {
  return
      LiteRtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
          Get(), num_steps_of_command_buffer_preparations);
}

}  // namespace litert
