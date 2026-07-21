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

#include "ml_drift_delegate/delegate/delegate_vulkan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_vulkan.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "ml_drift_delegate/tflite/model_builder.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/core/c/common.h"

using ::litert::ml_drift::DelegateKernelLiteRt;
using ::litert::ml_drift::MlDriftDelegateData;

namespace {

absl::Mutex g_vulkan_env_mutex(absl::kConstInit);
litert::ml_drift::SharedVulkanEnv* g_vulkan_env = nullptr;
int g_vulkan_env_ref_count = 0;

std::unique_ptr<litert::ml_drift::SharedVulkanEnv> CreateSharedVulkanEnv() {
  if (ml_drift::syrtis::InitVulkan() == 0) {
    ABSL_LOG(ERROR) << "Failed to locate Vulkan library.";
    return nullptr;
  }

  auto env = std::make_unique<litert::ml_drift::SharedVulkanEnv>();
  if (auto status = ml_drift::syrtis::CreateEnvironment(&env->vulkan_env());
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Vulkan environment: " << status;
    return nullptr;
  }

  const VkCommandPoolCreateInfo command_pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = env->vulkan_env().GetQueue().QueueFamilyIndex()};
  if (auto result = ml_drift::syrtis::vkCreateCommandPool(
          env->vulkan_env().GetDevice(), &command_pool_create_info, nullptr,
          &env->command_pool());
      result != VK_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to create Vulkan command pool: " << result;
    return nullptr;
  }

  ABSL_LOG(INFO) << "Created a Vulkan environment.";
  return env;
}

void DestroySharedVulkanEnv(void* vulkan_env) {
  {
    absl::MutexLock lock(g_vulkan_env_mutex);
    if (vulkan_env != g_vulkan_env) {
      ABSL_LOG(ERROR) << "Vulkan environment is not the same as the singleton.";
      ABSL_DCHECK(false);
      return;
    }

    --g_vulkan_env_ref_count;
    if (g_vulkan_env_ref_count > 0) {
      return;
    }

    g_vulkan_env = nullptr;
  }

  delete reinterpret_cast<litert::ml_drift::SharedVulkanEnv*>(vulkan_env);
  ABSL_LOG(INFO) << "Destroyed the Vulkan environment.";
}

litert::Expected<litert::ml_drift::SharedVulkanEnv*>
GetSingletonSharedVulkanEnv(LiteRtEnvironment env,
                            const LiteRtRuntimeContext* runtime_context) {
  {
    absl::MutexLock lock(g_vulkan_env_mutex);
    if (g_vulkan_env == nullptr) {
      g_vulkan_env = CreateSharedVulkanEnv().release();
      if (!g_vulkan_env) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create Vulkan environment");
      }
    }
    ++g_vulkan_env_ref_count;
  }

  // Update LiteRtEnvironment with the Vulkan environment.
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny vulkan_env,
      litert::ToLiteRtAny(reinterpret_cast<int64_t>(g_vulkan_env)));
  LITERT_ASSIGN_OR_RETURN(LiteRtAny callback,
                          litert::ToLiteRtAny(reinterpret_cast<const void*>(
                              DestroySharedVulkanEnv)));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny user_data,
      litert::ToLiteRtAny(reinterpret_cast<const void*>(g_vulkan_env)));
  std::array<LiteRtEnvOption, 3> options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagVulkanEnvironment,
                      .value = vulkan_env},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
                      .value = callback},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
                      .value = user_data},
  };
  LITERT_RETURN_IF_ERROR(runtime_context->gpu_environment_create(
      env, options.size(), options.data()));

  return g_vulkan_env;
}

void* Init(TfLiteContext* context, const char* buffer, size_t) {
  auto kernel = litert::ml_drift::DelegateKernelLiteRt::Create(
      context, reinterpret_cast<const TfLiteDelegateParams*>(buffer));
  if (!kernel.ok()) {
    ABSL_LOG(ERROR)
        << "Failed to create litert::ml_drift::DelegateKernelLiteRt: "
        << kernel.status();
    return TfLiteKernelInitFailed();
  }
  return *kernel;
}

void Free(TfLiteContext*, void* buffer) {
  delete reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(
          node->user_data);
  if (delegate_kernel == nullptr ||
      delegate_kernel == TfLiteKernelInitFailed()) {
    ABSL_LOG(ERROR) << "Delegate kernel initialization failed.";
    return kTfLiteError;
  }
  if (absl::Status s = delegate_kernel->GetRequiredTemporaries(
          context, node, &node->temporaries);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(
          node->user_data);
  if (delegate_kernel->HasQuantizedTensors()) {
    if (absl::Status s = delegate_kernel->DequantizeInputs(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }
  if (delegate_kernel->NoExternalTensorsMode()) {
    if (absl::Status s = delegate_kernel->UploadOrBindTensorBuffer(context);
        !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  } else {
    if (absl::Status s = delegate_kernel->BindTensorBuffers(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
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
          context, litert::ml_drift::IsAsyncExecutionMode(
                       context, delegate_kernel->runtime_context()));
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }

  if (delegate_kernel->NoExternalTensorsMode()) {
    // Download internal output GPU memory to output TensorBufferGPU memory.
    if (absl::Status s =
            delegate_kernel->DownloadGpuMemoryToTensorBufferGpuMemory(context);
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

  if (absl::Status s = delegate_kernel->DownloadGpuMemoryToCpuMemory(context);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }
  if (delegate_kernel->HasQuantizedTensors()) {
    if (absl::Status s = delegate_kernel->QuantizeOutputs(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Check ML Drift op compatibility.
  const absl::flat_hash_set<TfLiteBuiltinOperator> kExcludedOps = {};
  const auto& delegate_options =
      reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(delegate->data_)
          ->options;

  int start_node_index = 0;
  int end_node_index = std::numeric_limits<int>::max();
  if (delegate_options->debug_delegate_partition) {
    start_node_index = delegate_options->debug_first_delegate_node_index;
    end_node_index = delegate_options->debug_last_delegate_node_index;
  }
  litert::ml_drift::CustomOperationParserFactory custom_parser_factory;
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops=*/true, /*max_delegated_partitions=*/1,
      &kExcludedOps, start_node_index, end_node_index, &custom_parser_factory);

  // Replace the ops with delegate kernel.
  const TfLiteRegistration kRegistration = {
      .init = Init,
      .free = Free,
      .prepare = Prepare,
      .invoke = Invoke,
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "LITERT_VULKAN",
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

void LiteRtDeleteMlDriftVulkanDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(
      delegate->data_);
  delete delegate;
}

}  // extern "C"

namespace litert::ml_drift {

// Returns default options for ML Drift Vulkan delegate.
//
// This calls `MlDriftClDelegateDefaultOptions()` add return the result in an
// RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftVulkanDelegateDefaultOptionsPtr() {
  return std::make_unique<MlDriftDelegateOptions>(MlDriftDelegateOptions{
    .precision = MlDriftDelegatePrecision::kDefault,
    .debug_last_delegate_node_index = std::numeric_limits<int>::max(),
    .enable_fast_tuning = true,
    // Note that the program cache is not serialized unless serialization_dir
    // and model_token are also set.
    .serialize_program_cache = true,
    .madvise_original_shared_tensors = true,
  });
}

// Creates a new ML Drift Vulkan delegate object.
TfLiteDelegatePtr CreateMlDriftVulkanDelegate(MlDriftDelegateOptionsPtr options,
                                              LiteRtEnvironment litert_env) {
  if (!options) {
    ABSL_LOG(ERROR) << "Missing MLDrift delegate options";
    return {nullptr, LiteRtDeleteMlDriftVulkanDelegate};
  }

  if (options->litert_benchmark_mode) {
    ABSL_LOG(INFO) << "Benchmark mode is enabled.";
  }
  if (options->litert_external_tensors_mode) {
    ABSL_LOG(INFO) << "External tensors mode is enabled.";
  }

  // Initialize delegate_data.
  auto delegate_data =
      std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->options = std::move(options);

  const LiteRtRuntimeContext* runtime_context =
      delegate_data->options->runtime_context;
  if (runtime_context == nullptr) {
    ABSL_LOG(ERROR) << "Missing LiteRT runtime context.";
    return {nullptr, LiteRtDeleteMlDriftVulkanDelegate};
  }

  // Copy serialization options since they are not owned by the delegate.
  if (delegate_data->options->serialization_dir) {
    delegate_data->serialization_dir =
        delegate_data->options->serialization_dir;
  }
  if (delegate_data->options->model_token) {
    delegate_data->model_token = delegate_data->options->model_token;
  }

  // Use the shared Vulkan environment in LiteRT runtime.
  auto env = GetSingletonSharedVulkanEnv(litert_env, runtime_context);
  if (!env) {
    ABSL_LOG(ERROR) << "Failed to get Vulkan environment: " << env.Error();
    return {nullptr, LiteRtDeleteMlDriftVulkanDelegate};
  }
  auto backend = std::make_shared<GpuBackendVulkan>(*env, runtime_context);
  backend->set_num_steps_of_command_buffer_preparations(
      delegate_data->options->num_steps_of_command_buffer_preparations);
  backend->set_optimize_shader_compilation(
      !delegate_data->options->disable_shader_optimization);
  delegate_data->shared_backend = std::move(backend);

  switch (delegate_data->options->precision) {
    case kDefault:
      delegate_data->calculation_precision =
          (*env)->vulkan_env().GetInfo().vulkan_info.SupportsExplicitFp16()
              ? ::ml_drift::CalculationsPrecision::F16
              : ::ml_drift::CalculationsPrecision::F32;
      break;
    case kFp16:
      delegate_data->calculation_precision =
          ::ml_drift::CalculationsPrecision::F16;
      break;
    case kFp32:
      delegate_data->calculation_precision =
          ::ml_drift::CalculationsPrecision::F32;
      break;
  }

  // Initialize the ml_drift Vulkan delegate.
  TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                             LiteRtDeleteMlDriftVulkanDelegate);
  delegate->data_ = delegate_data.release();
  delegate->Prepare = DelegatePrepare;
  return delegate;
}

}  // namespace litert::ml_drift
