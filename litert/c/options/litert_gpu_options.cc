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
  // The null-terminated directory to use for serialization.
  // If program_cache_fd is set, this field is ignored for the program cache.
  const char* serialization_dir = nullptr;
  // The unique null-terminated token string that acts as a 'namespace' for
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
  // Number of steps to prepare WebGPU or Vulkan command buffers in advance.
  int num_steps_of_command_buffer_preparations = 0;
  // Set to true to use Metal argument buffers.
  bool use_metal_argument_buffers = false;
  // Added in version 2.0.2a1.
  LiteRtGpuWaitType wait_type = kLiteRtGpuWaitTypeDefault;
  // Added in version 2.0.2a1.
  // Preferred WebGPU device name substring, case-insensitive.
  // If not empty, the adapter which the device name contains the substring will
  // be chosen.
  // If empty, the device will be determined by other factors.
  std::string preferred_device_substr;
  // Added in version 2.0.2a1.
  // Set to true to hint that the delegate is fully delegated to a single
  // delegate.
  bool hint_fully_delegated_to_single_delegate = false;
  // Added in version 2.0.2a1.
  // Number of threads for WebGPU upload.
  int num_threads_to_upload = 0;
  // Added in version 2.0.2a1.
  // Number of threads for WebGPU kernel shader compilation.
  int num_threads_to_compile = 0;
  // Added in version 2.0.2a1.
  // Whether to convert weights on GPU.
  // It is not supported by the all backends so this flag is ignored when using
  // non-OpenCL and non-WebGPU backends.
  bool convert_weights_on_gpu = false;
  // Added in version 2.1.0.
  // Whether to disable Vulkan kernel shader optimization to reduce init time.
  bool disable_shader_optimization = false;
  // The file descriptor to use for program caching. If set, it overrides the
  // serialization_dir.
  int program_cache_fd = -1;
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

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsProgramCacheFd(
    LiteRtOpaqueOptions gpu_accelerator_options, int program_cache_fd) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->program_cache_fd = program_cache_fd;
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

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    LiteRtOpaqueOptions gpu_accelerator_options,
    bool disable_shader_optimization) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->disable_shader_optimization = disable_shader_optimization;
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

LiteRtStatus LiteRtSetGpuOptionsUseMetalArgumentBuffers(
    LiteRtOpaqueOptions gpu_options, bool use_metal_argument_buffers) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->use_metal_argument_buffers = use_metal_argument_buffers;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtOpaqueOptions gpu_accelerator_options, LiteRtGpuWaitType wait_type) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->wait_type = wait_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    LiteRtOpaqueOptions gpu_accelerator_options,
    const char* preferred_device_substr) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->preferred_device_substr = preferred_device_substr;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    LiteRtOpaqueOptions gpu_accelerator_options, int num_threads_to_upload) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->num_threads_to_upload = num_threads_to_upload;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    LiteRtOpaqueOptions gpu_accelerator_options, int num_threads_to_compile) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->num_threads_to_compile = num_threads_to_compile;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    LiteRtOpaqueOptions gpu_accelerator_options, bool convert_weights_on_gpu) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_accelerator_options));
  payload->convert_weights_on_gpu = convert_weights_on_gpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsHintFullyDelegatedToSingleDelegate(
    LiteRtOpaqueOptions gpu_options,
    bool hint_fully_delegated_to_single_delegate) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->hint_fully_delegated_to_single_delegate =
      hint_fully_delegated_to_single_delegate;
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

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
    int* program_cache_fd, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(program_cache_fd,
                         ErrorStatusBuilder::InvalidArgument())
      << "`program_cache_fd` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *program_cache_fd = payload->program_cache_fd;
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
    bool* madvise_original_shared_tensors, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(madvise_original_shared_tensors,
                         ErrorStatusBuilder::InvalidArgument())
      << "`madvise_original_shared_tensors` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *madvise_original_shared_tensors = payload->madvise_original_shared_tensors;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
    bool* disable_shader_optimization, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(disable_shader_optimization,
                         ErrorStatusBuilder::InvalidArgument())
      << "`disable_shader_compilation_optimization` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *disable_shader_optimization = payload->disable_shader_optimization;
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

LiteRtStatus LiteRtGetGpuOptionsUseMetalArgumentBuffers(
    LiteRtGpuOptionsPayload payload, bool* use_metal_argument_buffers) {
  LITERT_RETURN_IF_ERROR(use_metal_argument_buffers,
                         ErrorStatusBuilder::InvalidArgument())
      << "`use_metal_argument_buffers` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *use_metal_argument_buffers = payload->use_metal_argument_buffers;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsWaitType(
    LiteRtGpuWaitType* wait_type, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(wait_type, ErrorStatusBuilder::InvalidArgument())
      << "`wait_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *wait_type = payload->wait_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
    const char** preferred_device_substr, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(preferred_device_substr,
                         ErrorStatusBuilder::InvalidArgument())
      << "`preferred_device_substr` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *preferred_device_substr = payload->preferred_device_substr.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
    int* num_threads_to_upload, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(num_threads_to_upload,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_upload` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *num_threads_to_upload = payload->num_threads_to_upload;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
    int* num_threads_to_compile, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(num_threads_to_compile,
                         ErrorStatusBuilder::InvalidArgument())
      << "`num_threads_to_compile` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *num_threads_to_compile = payload->num_threads_to_compile;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
    bool* convert_weights_on_gpu, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(convert_weights_on_gpu,
                         ErrorStatusBuilder::InvalidArgument())
      << "`convert_weights_on_gpu` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *convert_weights_on_gpu = payload->convert_weights_on_gpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
    bool* hint_fully_delegated_to_single_delegate,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(hint_fully_delegated_to_single_delegate,
                         ErrorStatusBuilder::InvalidArgument())
      << "`hint_fully_delegated_to_single_delegate` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *hint_fully_delegated_to_single_delegate =
      payload->hint_fully_delegated_to_single_delegate;
  return kLiteRtStatusOk;
}
