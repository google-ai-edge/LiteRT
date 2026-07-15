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

#include "ml_drift_delegate/delegate/delegate_metal.h"

#import <Metal/Metal.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/metal/converter.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_metal_litert.h"
#include "litert/c/options/litert_gpu_options.h"
#include "ml_drift_delegate/tflite/model_builder.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"

namespace {

absl::Mutex g_metal_device_mutex(absl::kConstInit);
ml_drift::metal::MetalDevice* g_metal_device = nullptr;
int g_metal_device_ref_count = 0;

void DestroyMetalDevice(void* metal_device) {
  {
    absl::MutexLock lock(g_metal_device_mutex);
    if (metal_device != g_metal_device) {
      ABSL_LOG(ERROR) << "Metal device is not the same as the singleton.";
      ABSL_DCHECK(false);
      return;
    }

    --g_metal_device_ref_count;
    if (g_metal_device_ref_count > 0) {
      return;
    }

    g_metal_device = nullptr;
  }

  delete reinterpret_cast<ml_drift::metal::MetalDevice*>(metal_device);
  ABSL_LOG(INFO) << "Destroyed the Metal environment.";
}

litert::Expected<ml_drift::metal::MetalDevice*> GetSingletonDevice(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env) {
  {
    absl::MutexLock lock(g_metal_device_mutex);
    if (g_metal_device == nullptr) {
      g_metal_device = new ml_drift::metal::MetalDevice();
      if (!g_metal_device) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create Metal device.");
      }
      ABSL_LOG(INFO) << "Created a Metal device.";
    }
    ++g_metal_device_ref_count;
  }

  // Update the LiteRtEnvironment with the Metal device.
  // So LiteRT runtime can use the Metal device.
  const void* device_ptr = (__bridge void*)g_metal_device->device();
  LITERT_ASSIGN_OR_ABORT(LiteRtAny device_id,
                         litert::ToLiteRtAny(litert::LiteRtVariant(device_ptr)));

  id<MTLCommandQueue> device_command_queue = [g_metal_device->device() newCommandQueue];
  const void* device_command_queue_ptr = (__bridge void*)device_command_queue;
  LITERT_ASSIGN_OR_ABORT(LiteRtAny command_queue_id,
                         litert::ToLiteRtAny(litert::LiteRtVariant(device_command_queue_ptr)));

  LITERT_ASSIGN_OR_RETURN(LiteRtAny callback,
                          litert::ToLiteRtAny(reinterpret_cast<const void*>(&DestroyMetalDevice)));
  LITERT_ASSIGN_OR_RETURN(LiteRtAny user_data,
                          litert::ToLiteRtAny(reinterpret_cast<const void*>(g_metal_device)));

  const std::array<LiteRtEnvOption, 4> environment_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagMetalDevice, .value = device_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagMetalCommandQueue, .value = command_queue_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy, .value = callback},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
                      .value = user_data},
  };
  runtime_context->gpu_environment_create(env, environment_options.size(),
                                          environment_options.data());
  return g_metal_device;
}

void* Init(TfLiteContext* context, const char* buffer, size_t) {
  auto kernel = litert::ml_drift::DelegateKernelLiteRt::Create(
      context, reinterpret_cast<const TfLiteDelegateParams*>(buffer));
  if (!kernel.ok()) {
    ABSL_LOG(ERROR) << "Failed to create DelegateKernelLiteRtMetal: " << kernel.status();
    return TfLiteKernelInitFailed();
  }
  return *kernel;
}

void Free(TfLiteContext*, void* buffer) {
  delete reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(node->user_data);
  if (delegate_kernel == nullptr || delegate_kernel == TfLiteKernelInitFailed()) {
    ABSL_LOG(ERROR) << "Metal delegate kernel initialization failed.";
    return kTfLiteError;
  }
  if (absl::Status s = delegate_kernel->GetRequiredTemporaries(context, node, &node->temporaries);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  @autoreleasepool {
    auto* delegate_kernel =
        reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(node->user_data);

    if (delegate_kernel->HasQuantizedTensors()) {
      if (absl::Status s = delegate_kernel->DequantizeInputs(context); !s.ok()) {
        ABSL_LOG(ERROR) << s;
        return kTfLiteError;
      }
    }

    if (delegate_kernel->NoExternalTensorsMode()) {
      if (absl::Status s = delegate_kernel->UploadOrBindTensorBuffer(context); !s.ok()) {
        ABSL_LOG(ERROR) << s;
        return kTfLiteError;
      }
    } else {
      if (absl::Status s = delegate_kernel->BindTensorBuffers(context); !s.ok()) {
        ABSL_LOG(ERROR) << s;
        return kTfLiteError;
      }
    }

    auto* buffer_context = reinterpret_cast<LiteRtExternalLiteRtBufferContext>(
        context->GetExternalContext(context, kTfLiteLiteRtBufferContext));
    if (buffer_context != nullptr) {
      LiteRtOptions run_options = nullptr;
      delegate_kernel->runtime_context()->external_litert_buffer_context_get_run_options(
          buffer_context, &run_options);
      if (run_options != nullptr) {
        LiteRtOpaqueOptions opaque_gpu_options;
        if (delegate_kernel->runtime_context()->get_opaque_options(
                run_options, &opaque_gpu_options) == kLiteRtStatusOk) {
          void* gpu_payload_data = nullptr;
          const char* identifier = LrtGetGpuOptionsIdentifier();
          if (delegate_kernel->runtime_context()->find_opaque_options_data(
                  opaque_gpu_options, identifier, &gpu_payload_data) ==
              kLiteRtStatusOk) {
            LrtGpuOptions* gpu_opts = nullptr;
            if (LrtCreateGpuOptionsFromToml(
                    reinterpret_cast<const char*>(gpu_payload_data),
                    &gpu_opts) == kLiteRtStatusOk) {
              bool enable_residency = true;
              if (LrtGetGpuOptionsMetalResidencySet(
                      gpu_opts, &enable_residency) == kLiteRtStatusOk) {
                auto* metal_backend = static_cast<litert::ml_drift::GpuBackendMetal*>(
                    delegate_kernel->backend());
                metal_backend->SetResidencyRuntimeEnabled(enable_residency);
              }
              LrtDestroyGpuOptions(gpu_opts);
            }
          }
        }
      }
    }

    if (absl::Status s = delegate_kernel->HandleInputEvents(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }

    if (absl::Status s = delegate_kernel->Dispatch(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }

    if (absl::Status s = delegate_kernel->HandleOutputEvents(
            context,
            litert::ml_drift::IsAsyncExecutionMode(context, delegate_kernel->runtime_context()));
        !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }

    if (delegate_kernel->NoExternalTensorsMode()) {
      // Download internal output GPU memory to output TensorBufferGPU memory.
      if (absl::Status s = delegate_kernel->DownloadGpuMemoryToTensorBufferGpuMemory(context);
          !s.ok()) {
        ABSL_LOG(ERROR) << s;
        return kTfLiteError;
      }
    }

    if (delegate_kernel->IsBenchmarkMode()) {
      // In benchmark mode, call WaitForCompletion() to wait for all the
      // enqueued commands to be completed.
      auto status = delegate_kernel->backend()->WaitForCompletion();
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to wait for completion: " << status;
        return kTfLiteError;
      }
    }

    if (absl::Status s = delegate_kernel->DownloadGpuMemoryToCpuMemory(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }

    if (delegate_kernel->HasQuantizedTensors()) {
      if (absl::Status s = delegate_kernel->QuantizeOutputs(context); !s.ok()) {
        ABSL_LOG(ERROR) << s;
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Check ML Drift op compatibility.
  const absl::flat_hash_set<TfLiteBuiltinOperator> kExcludedOps = {};
  const auto& delegate_options =
      reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(delegate->data_)->options;

  int start_node_index = 0;
  int end_node_index = std::numeric_limits<int>::max();
  if (delegate_options->debug_delegate_partition) {
    start_node_index = delegate_options->debug_first_delegate_node_index;
    end_node_index = delegate_options->debug_last_delegate_node_index;
  }
  litert::ml_drift::CustomOperationParserFactory custom_parser_factory;
  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, /*allow_quant_ops=*/true, /*max_delegated_partitions=*/1,
                      &kExcludedOps, start_node_index, end_node_index, &custom_parser_factory);

  // Replace the ops with delegate kernel.
  const TfLiteRegistration kRegistration = {
      .init = Init,
      .free = Free,
      .prepare = Prepare,
      .invoke = Invoke,
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "LITERT_METAL",
      .version = 1,
      .registration_external = nullptr,
      .async_kernel = nullptr,
  };
  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}
}  // namespace

extern "C" {

void LiteRtDeleteMlDriftMetalDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(delegate->data_);
  delete delegate;
}

}  // extern "C"

namespace litert {
namespace ml_drift {

// Returns default options for ML Drift Metal delegate.
//
// This calls `MlDriftMetalDelegateDefaultOptions()` add return the result in an
// RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftMetalDelegateDefaultOptionsPtr() {
  return std::make_unique<MlDriftDelegateOptions>(MlDriftDelegateOptions{
      .precision = MlDriftDelegatePrecision::kDefault,
      .debug_last_delegate_node_index = std::numeric_limits<int>::max(),
      .enable_fast_tuning = true,
      // Note that the program cache is not serialized unless serialization_dir
      // and model_token are also set.
      .serialize_program_cache = true,
      .madvise_original_shared_tensors = true,
      .wait_type = kGpuDelegateWaitTypeActive,
  });
}

// Creates a new ML Drift Metal delegate object.
TfLiteDelegatePtr CreateMlDriftMetalDelegate(MlDriftDelegateOptionsPtr options,
                                             LiteRtEnvironment litert_env) {
  if (!options) {
    ABSL_LOG(ERROR) << "Missing MLDrift delegate options";
    return {nullptr, LiteRtDeleteMlDriftMetalDelegate};
  }

  if (options->litert_benchmark_mode) {
    ABSL_LOG(INFO) << "Benchmark mode is enabled.";
  }
  if (options->litert_external_tensors_mode) {
    ABSL_LOG(INFO) << "External tensors mode is enabled.";
  }

  // Get the flag of enable op profiling and use it to set the delegate flags.
  const bool enable_op_profiling = options->enable_op_profiling;

  // Initialize delegate_data.
  auto delegate_data = std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->options = std::move(options);
  delegate_data->weight_loader = delegate_data->options->weight_loader;

  const LiteRtRuntimeContext* runtime_context = delegate_data->options->runtime_context;
  if (runtime_context == nullptr) {
    ABSL_LOG(ERROR) << "Missing LiteRT runtime context.";
    return {nullptr, LiteRtDeleteMlDriftMetalDelegate};
  }

  // Copy serialization options since they are not owned by the delegate.
  if (delegate_data->options->serialization_dir) {
    delegate_data->serialization_dir = delegate_data->options->serialization_dir;
  }
  if (delegate_data->options->model_token) {
    delegate_data->model_token = delegate_data->options->model_token;
  }

  auto metal_device = GetSingletonDevice(runtime_context, litert_env);
  if (!metal_device) {
    ABSL_LOG(ERROR) << "Failed to get Metal device: " << metal_device.Error();
    return {nullptr, LiteRtDeleteMlDriftMetalDelegate};
  }

#ifdef __APPLE__
  if ((*metal_device)->device().argumentBuffersSupport == MTLArgumentBuffersTier1) {
    if (delegate_data->options->use_metal_argument_buffers) {
      ABSL_LOG(INFO) << "Metal argument buffers are not supported on this device, disabling "
                        "use_metal_argument_buffers option.";
      delegate_data->options->use_metal_argument_buffers = false;
    }
  }
#endif

  LiteRtEnvironmentOptions gpu_env_options;
  runtime_context->get_environment_options(litert_env, &gpu_env_options);
  LiteRtAny option_value;

  auto status = runtime_context->get_environment_options_value(
      gpu_env_options, kLiteRtEnvOptionTagMetalCommandQueue, &option_value);
  if (status != kLiteRtStatusOk) {
    ABSL_LOG(ERROR) << "Failed to get command queue from LiteRt environment options.";
    return {nullptr, LiteRtDeleteMlDriftMetalDelegate};
  }
  id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)(option_value.ptr_value);
  delegate_data->shared_backend = std::make_shared<GpuBackendMetalLitert>(
      *metal_device, delegate_data->options->wait_type, command_queue, runtime_context,
      delegate_data->options->enable_metal_residency_set);

  switch (delegate_data->options->precision) {
    case kDefault:
      delegate_data->calculation_precision =
          delegate_data->shared_backend->GetInfo()->IsRoundToNearestSupported()
              ? ::ml_drift::CalculationsPrecision::F16
              : ::ml_drift::CalculationsPrecision::F32_F16;
      break;
    case kFp16:
      delegate_data->calculation_precision = ::ml_drift::CalculationsPrecision::F16;
      break;
    case kFp32:
      delegate_data->calculation_precision = ::ml_drift::CalculationsPrecision::F32;
      break;
  }
  const bool hint_fully_delegated_to_single_delegate =
      delegate_data->options->hint_fully_delegated_to_single_delegate;

  // Initialize the ml_drift Metal delegate.
  TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                             LiteRtDeleteMlDriftMetalDelegate);
  delegate->data_ = delegate_data.release();
  delegate->Prepare = DelegatePrepare;
  if (enable_op_profiling) {
    delegate->flags = kTfLiteDelegateFlagsPerOperatorProfiling;
  }
  if (hint_fully_delegated_to_single_delegate) {
    delegate->flags |= kTfLiteDelegateFlagsHintFullyDelegatedToSingleDelegate;
  }
  return delegate;
}

}  // namespace ml_drift
}  // namespace litert
