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

#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"

#include <memory>
#include <utility>

#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_runtime_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"

namespace litert::ml_drift {
namespace {

MlDriftDelegatePrecision GetMlDriftPrecision(
    LiteRtDelegatePrecision precision) {
  switch (precision) {
    case kLiteRtDelegatePrecisionDefault:
      return kDefault;
    case kLiteRtDelegatePrecisionFp16:
      return kFp16;
    case kLiteRtDelegatePrecisionFp32:
      return kFp32;
  }
}

}  // namespace

LrtGpuOptions* GetGpuOptionsPayload(LiteRtRuntimeContext* runtime_context,
                                    LiteRtOptions options) {
  LiteRtOpaqueOptions opaque_options;
  auto status = runtime_context->get_opaque_options(options, &opaque_options);
  if (status != kLiteRtStatusOk) return nullptr;

  void* options_data = nullptr;
  auto found_status = runtime_context->find_opaque_options_data(
      opaque_options, LrtGetGpuOptionsIdentifier(), &options_data);
  if (found_status != kLiteRtStatusOk) {
    return nullptr;
  }

  LrtGpuOptions* gpu_options_payload = nullptr;
  if (LrtCreateGpuOptionsFromToml(reinterpret_cast<const char*>(options_data),
                                  &gpu_options_payload) != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse GPU options TOML payload");
    return nullptr;
  }
  return gpu_options_payload;
}

LiteRtStatus CreateDelegate(
    LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env,
    LiteRtAccelerator accelerator, LrtGpuOptions* gpu_options_payload,
    std::unique_ptr<MlDriftDelegateOptions> gpu_delegate_options,
    DelegateCreator delegate_creator, TfLiteDelegatePtr& delegate) {
  if (delegate_creator == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Propagate the min logger severity from the environment to the delegate.
  LiteRtEnvironmentOptions env_options;
  auto env_options_status =
      runtime_context->get_environment_options(env, &env_options);
  if (env_options_status == kLiteRtStatusOk) {
    LiteRtPropagateMinLoggerSeverityWithRuntimeContext(runtime_context,
                                                       env_options);
  }

  if (gpu_delegate_options != nullptr && gpu_options_payload != nullptr) {
    LITERT_LOG(LITERT_VERBOSE, "User provided gpu options found.");
    // We only apply this option when the weight loader is not provided. If the
    // weight loader is provided, constant tensor sharing is always needed.
    if (gpu_delegate_options->weight_loader == nullptr) {
      LrtGetGpuOptionsConstantTensorsSharing(
          &gpu_delegate_options->enable_constant_tensors_sharing,
          gpu_options_payload);
    } else {
      gpu_delegate_options->enable_constant_tensors_sharing = true;
    }

    LrtGetGpuOptionsInfiniteFloatCapping(
        &gpu_delegate_options->enable_infinite_float_capping,
        gpu_options_payload);

    LrtGetGpuOptionsBenchmarkMode(&gpu_delegate_options->litert_benchmark_mode,
                                  gpu_options_payload);

    LrtGetGpuOptionsExternalTensorsMode(
        &gpu_delegate_options->litert_external_tensors_mode,
        gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
        &gpu_delegate_options->allow_src_quantized_fc_conv_ops,
        gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsHintWaitingForCompletion(
        &gpu_delegate_options->hint_waiting_for_completion,
        gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsKernelBatchSize(
        &gpu_delegate_options->kernel_batch_size, gpu_options_payload);

    LiteRtDelegatePrecision litert_delegate_precision =
        kLiteRtDelegatePrecisionDefault;

    LrtGetGpuAcceleratorCompilationOptionsPrecision(&litert_delegate_precision,
                                                    gpu_options_payload);

    gpu_delegate_options->precision =
        ::litert::ml_drift::GetMlDriftPrecision(litert_delegate_precision);

    LiteRtDelegateBufferStorageType litert_delegate_buffer_storage_type =
        kLiteRtDelegateBufferStorageTypeDefault;

    LrtGetGpuAcceleratorCompilationOptionsBufferStorageType(
        &litert_delegate_buffer_storage_type, gpu_options_payload);

    gpu_delegate_options->use_buffer_storage_type =
        litert_delegate_buffer_storage_type ==
        kLiteRtDelegateBufferStorageTypeBuffer;

    LrtGetGpuAcceleratorCompilationOptionsPreferTextureWeights(
        &gpu_delegate_options->prefer_texture_weights, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsSerializationDir(
        &gpu_delegate_options->serialization_dir, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(
        &gpu_delegate_options->model_token, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
        &gpu_delegate_options->program_cache_fd, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsWeightCacheFd(
        &gpu_delegate_options->weight_cache_fd, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
        &gpu_delegate_options->serialize_program_cache, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsCacheCompiledProgramsOnly(
        &gpu_delegate_options->cache_compiled_programs_only,
        gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
        &gpu_delegate_options->serialize_external_tensors, gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsDisableShaderOptimization(
        &gpu_delegate_options->disable_shader_optimization,
        gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToUpload(
        &gpu_delegate_options->num_threads_to_upload, gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsNumThreadsToCompile(
        &gpu_delegate_options->num_threads_to_compile, gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsConvertWeightsOnGpu(
        &gpu_delegate_options->convert_weights_on_gpu,
        gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsWaitForWeightsConversionComplete(
        &gpu_delegate_options->wait_for_weights_conversion_complete,
        gpu_options_payload);

    LrtGetGpuOptionsHintFullyDelegatedToSingleDelegate(
        &gpu_delegate_options->hint_fully_delegated_to_single_delegate,
        gpu_options_payload);

#ifdef __APPLE__
    LrtGetGpuOptionsUseMetalArgumentBuffers(
        gpu_options_payload, &gpu_delegate_options->use_metal_argument_buffers);
#endif  // __APPLE__

    int num_patterns;
    auto status =
        LrtGetNumGpuAcceleratorCompilationOptionsExternalTensorPatterns(
            &num_patterns, gpu_options_payload);
    for (int i = 0; status == kLiteRtStatusOk && i < num_patterns; ++i) {
      const char* pattern;
      status = LrtGetGpuAcceleratorCompilationOptionsExternalTensorPattern(
          &pattern, i, gpu_options_payload);
      if (status == kLiteRtStatusOk) {
        gpu_delegate_options->litert_external_tensor_patterns.insert(pattern);
      } else {
        LITERT_LOG(LITERT_ERROR,
                   "Failed to get external tensor pattern at index %d", i);
      }
    }

    int num_buffer_patterns;
    status =
        LrtGetNumGpuAcceleratorCompilationOptionsBufferStorageTensorPatterns(
            &num_buffer_patterns, gpu_options_payload);
    for (int i = 0; status == kLiteRtStatusOk && i < num_buffer_patterns; ++i) {
      const char* pattern;
      status = LrtGetGpuAcceleratorCompilationOptionsBufferStorageTensorPattern(
          &pattern, i, gpu_options_payload);
      if (status == kLiteRtStatusOk) {
        gpu_delegate_options->litert_buffer_storage_tensor_patterns.insert(
            pattern);
      } else {
        LITERT_LOG(LITERT_ERROR,
                   "Failed to get buffer storage tensor pattern at index %d",
                   i);
      }
    }

    LiteRtGpuPriority gpu_priority;
    LrtGetGpuOptionsGpuPriority(&gpu_priority, gpu_options_payload);
    if (gpu_priority == kLiteRtGpuPriorityLow) {
      gpu_delegate_options->gpu_priority = kGpuLowPriority;
    } else {
      gpu_delegate_options->gpu_priority = kGpuNormalPriority;
    }

    LrtGetGpuAcceleratorCompilationOptionsMadviseOriginalSharedTensors(
        &gpu_delegate_options->madvise_original_shared_tensors,
        gpu_options_payload);

    LrtGetGpuAcceleratorCompilationOptionsSharedTensorMaps(
        &gpu_delegate_options->shared_tensor_maps_from_client,
        gpu_options_payload);

    LrtGetGpuAcceleratorRuntimeOptionsNumStepsOfCommandBufferPreparations(
        &gpu_delegate_options->num_steps_of_command_buffer_preparations,
        gpu_options_payload);

    LiteRtGpuWaitType wait_type;
    LrtGetGpuAcceleratorRuntimeOptionsWaitType(&wait_type, gpu_options_payload);
    switch (wait_type) {
      case kLiteRtGpuWaitTypeDefault:
        // Don't update delegate options.
        break;
      case kLiteRtGpuWaitTypePassive:
        gpu_delegate_options->wait_type = kGpuDelegateWaitTypePassive;
        break;
      case kLiteRtGpuWaitTypeActive:
        gpu_delegate_options->wait_type = kGpuDelegateWaitTypeActive;
        break;
      case kLiteRtGpuWaitTypeDoNotWait:
        gpu_delegate_options->wait_type = kGpuDelegateWaitTypeDoNotWait;
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unknown wait type: %d", wait_type);
        break;
    }

    const char* preferred_device_substr;
    LrtGetGpuAcceleratorRuntimeOptionsPreferredDeviceSubstr(
        &preferred_device_substr, gpu_options_payload);
    gpu_delegate_options->preferred_device_substr = preferred_device_substr;
  }

  if (gpu_delegate_options != nullptr) {
    gpu_delegate_options->runtime_context = runtime_context;
  }

  delegate = delegate_creator(std::move(gpu_delegate_options), env);

  if (gpu_options_payload != nullptr) {
    LrtDestroyGpuOptions(gpu_options_payload);
  }

  if (delegate == nullptr) {
    return kLiteRtStatusErrorRuntimeFailure;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::ml_drift
