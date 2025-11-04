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

#include "litert/c/options/litert_gpu_options.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

using ::litert::ErrorStatusBuilder;

struct LiteRtGpuOptionsPayloadT {
  // Increment the minor version every time a field is added.
  static constexpr const absl::string_view kIdentifier = "gpu_payload";

  bool enable_constant_tensor_sharing = false;
  bool enable_infinite_float_capping = false;
  bool benchmark_mode = false;
  // Added in version 1.2.0.
  bool allow_src_quantized_fc_conv_ops = false;
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  LiteRtDelegateBufferStorageType buffer_storage_type =
      kLiteRtDelegateBufferStorageTypeDefault;
  // If true, the delegate will prefer to use textures rather than buffers for
  // weights. Use option when weights in texture has better performance.
  bool prefer_texture_weights = false;
  // The nul-terminated directory to use for serialization.
  const char* serialization_dir = nullptr;
  // The unique nul-terminated token string that acts as a 'namespace' for
  // all serialization entries.
  const char* model_cache_key = nullptr;
  // When set to true AND the serialization_dir and model_cache_key are also
  // set, the delegate will serialize the program cache.
  bool serialize_program_cache = true;
  // Set to true to serialize immutable external tensors. By default only the
  // non-external tensors are serialized.
  bool serialize_external_tensors = false;
  // Set to true to run in no external tensors mode. This enables GPU
  // Accelerator using external tensors (PHWC4 format) directly as inputs and
  // outputs.
  bool experimental_external_tensors_mode = false;
  // List of external tensor patterns which are not affected by the no immutable
  // external tensors mode.
  std::vector<std::string> external_tensor_patterns;
  // Added in version 1.4.0.
  // GPU backend to use.
  LiteRtGpuBackend gpu_backend = kLiteRtGpuBackendAutomatic;
  // Added in version 2.0.2a1.
  // GPU priority to use.
  LiteRtGpuPriority gpu_priority = kLiteRtGpuPriorityNormal;
  // Added in version 2.0.2a1.
  // Set to true to madvise the original shared tensors after use.
  bool madvise_original_shared_tensors = false;
  // Added in version 2.0.2a1.
  // Number of steps to prepare WebGPU command buffers in advance.
  int num_steps_of_command_buffer_preparations = 0;
};

namespace litert {
namespace {

litert::Expected<LiteRtGpuOptionsPayloadT*> GetPayload(
    LiteRtOpaqueOptions options) {
  const char* identifier = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  LITERT_RETURN_IF_ERROR(identifier == LiteRtGpuOptionsPayloadT::kIdentifier,
                         ErrorStatusBuilder::InvalidArgument())
      << "Payload stored in accelerator options is incompatible. Got "
      << identifier << ", expected " << LiteRtGpuOptionsPayloadT::kIdentifier
      << ".";

  LiteRtGpuOptionsPayloadT* payload;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsData(options, reinterpret_cast<void**>(&payload)));
  return payload;
}

}  // namespace
}  // namespace litert

LiteRtStatus LiteRtCreateGpuOptions(LiteRtOpaqueOptions* options) {
  auto payload = std::make_unique<LiteRtGpuOptionsPayloadT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGpuOptionsPayloadT::kIdentifier.data(), payload.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtGpuOptionsPayloadT*>(payload);
      },
      options));
  payload.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsConstantTensorSharing(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->enable_constant_tensor_sharing = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsInfiniteFloatCapping(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->enable_infinite_float_capping = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsBenchmarkMode(LiteRtOpaqueOptions gpu_options,
                                              bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->benchmark_mode = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsGpuBackend(LiteRtOpaqueOptions gpu_options,
                                           LiteRtGpuBackend backend) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->gpu_backend = backend;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsExternalTensorsMode(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->experimental_external_tensors_mode = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddGpuOptionsExternalTensorPattern(
    LiteRtOpaqueOptions gpu_options, const char* pattern) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->external_tensor_patterns.push_back(std::string(pattern));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsGpuPriority(LiteRtOpaqueOptions gpu_options,
                                            LiteRtGpuPriority priority) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->gpu_priority = priority;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LiteRtOpaqueOptions gpu_accelerator_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->allow_src_quantized_fc_conv_ops = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegatePrecision precision) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->precision = precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegateBufferStorageType buffer_storage_type) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->buffer_storage_type = buffer_storage_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    LiteRtOpaqueOptions gpu_accelerator_options, bool prefer_texture_weights) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->prefer_texture_weights = prefer_texture_weights;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir(
    LiteRtOpaqueOptions gpu_accelerator_options,
    const char* serialization_dir) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->serialization_dir = serialization_dir;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsModelCacheKey(
    LiteRtOpaqueOptions gpu_accelerator_options, const char* model_cache_key) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->model_cache_key = model_cache_key;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    LiteRtOpaqueOptions gpu_accelerator_options, bool serialize_program_cache) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->serialize_program_cache = serialize_program_cache;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    LiteRtOpaqueOptions gpu_accelerator_options,
    bool serialize_external_tensors) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->serialize_external_tensors = serialize_external_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    LiteRtOpaqueOptions gpu_accelerator_options,
    bool madvise_original_shared_tensors) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->madvise_original_shared_tensors = madvise_original_shared_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    LiteRtOpaqueOptions gpu_accelerator_options,
    int num_steps_of_command_buffer_preparations) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->num_steps_of_command_buffer_preparations =
      num_steps_of_command_buffer_preparations;
  return kLiteRtStatusOk;
}

const char* LiteRtGetGpuOptionsPayloadIdentifier() {
  return LiteRtGpuOptionsPayloadT::kIdentifier.data();
}

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_constant_tensor_sharing;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_infinite_float_capping;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsGpuBackend(LiteRtGpuBackend* backend,
                                           LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(backend, ErrorStatusBuilder::InvalidArgument())
      << "`backend` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *backend = payload->gpu_backend;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->benchmark_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsExternalTensorsMode(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->experimental_external_tensors_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsGpuPriority(LiteRtGpuPriority* priority,
                                            LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(priority, ErrorStatusBuilder::InvalidArgument())
      << "`priority` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *priority = payload->gpu_priority;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->allow_src_quantized_fc_conv_ops;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *precision = payload->precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* buffer_storage_type,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(buffer_storage_type,
                         ErrorStatusBuilder::InvalidArgument())
      << "`use_buffer_storage_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *buffer_storage_type = payload->buffer_storage_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
    bool* prefer_texture_weights, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(prefer_texture_weights,
                         ErrorStatusBuilder::InvalidArgument())
      << "`prefer_texture_weights` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *prefer_texture_weights = payload->prefer_texture_weights;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializationDir(
    const char** serialization_dir, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(serialization_dir,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialization_dir` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *serialization_dir = payload->serialization_dir;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsModelCacheKey(
    const char** model_cache_key, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(model_cache_key, ErrorStatusBuilder::InvalidArgument())
      << "`model_cache_key` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *model_cache_key = payload->model_cache_key;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
    bool* serialize_program_cache, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(serialize_program_cache,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialize_program_cache` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *serialize_program_cache = payload->serialize_program_cache;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
    bool* serialize_external_tensors, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(serialize_external_tensors,
                         ErrorStatusBuilder::InvalidArgument())
      << "`serialize_external_tensors` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *serialize_external_tensors = payload->serialize_external_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumGpuAcceleratorCompilationOptionsExternalTensorPatterns(
    int* num_patterns, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(num_patterns, ErrorStatusBuilder::InvalidArgument())
      << "`num_patterns` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *num_patterns = payload->external_tensor_patterns.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsExternalTensorPattern(
    const char** external_tensor_pattern, int pattern_index,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(external_tensor_pattern,
                         ErrorStatusBuilder::InvalidArgument())
      << "`external_tensor_pattern` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *external_tensor_pattern =
      payload->external_tensor_patterns[pattern_index].c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
    bool* madvise_original_shared_tensors,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(madvise_original_shared_tensors,
                         ErrorStatusBuilder::InvalidArgument())
      << "`madvise_original_shared_tensors` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *madvise_original_shared_tensors = payload->madvise_original_shared_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtGetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
    int* num_steps_of_command_buffer_preparations,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(num_steps_of_command_buffer_preparations,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_steps_of_command_buffer_preparations` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *num_steps_of_command_buffer_preparations =
      payload->num_steps_of_command_buffer_preparations;
  return kLiteRtStatusOk;
}
