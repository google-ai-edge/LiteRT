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

#include "third_party/odml/litert/ml_drift/delegate/delegate_opencl.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_device.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/util_types.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_delegate.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/file_util.h"
#include "third_party/odml/infra/ml_drift_delegate/tflite_profile.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_gl_types.h"  // IWYU pragma: keep
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_macros.h"
#include "third_party/odml/litert/ml_drift/delegate/cache/simple_cache.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/custom_parsers.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_data.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_kernel_litert.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_types.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_utils.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend_opencl_litert.h"
#include "third_party/odml/litert/ml_drift/tflite/model_builder.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/profiling/time.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "ml_drift/cl/gl_interop.h"  // from @ml_drift
#include "ml_drift/pelong/egl_environment.h"  // from @ml_drift
#endif  // LITERT_HAS_OPENGL_SUPPORT

using ::litert::ml_drift::DelegateKernelLiteRt;
using ::litert::ml_drift::MlDriftDelegateData;

namespace {

// A struct that holds the delegate-local wrapper around the LiteRT GPU
// environment. This is created when the delegate is initialized, and destroyed
// when the LiteRT GPU environment is destroyed.
struct DelegateEnvironment {
#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<ml_drift::gl::EglEnvironment> egl_env;
#endif  // LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<ml_drift::cl::Environment> cl_env;
  LiteRtEglDisplay egl_display = LITE_RT_EGL_NO_DISPLAY;
  LiteRtEglContext egl_context = LITE_RT_EGL_NO_CONTEXT;
};

#ifndef NDEBUG
// Shell environment variables to debug tflite on GPU with OpenCL delegate.
constexpr char kEnvDebugEndNode[] = "LITERT_GPU_DEBUG_END_NODE";
constexpr char kEnvDebugExcludeNodes[] = "LITERT_GPU_DEBUG_EXCLUDE_NODES";
#endif  // NDEBUG

void DestroyDelegateEnvironment(void* user_data) {
  delete reinterpret_cast<DelegateEnvironment*>(user_data);
  LITERT_LOG(LITERT_DEBUG, "Destroyed delegate environment.");
}

absl::Status CreateClContext(const ml_drift::cl::CLDevice& device,
                             const ml_drift::cl::CLContextOptions& options,
                             LiteRtEglContext egl_context,
                             LiteRtEglDisplay egl_display,
                             ml_drift::cl::CLContext* context) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (IsGlSharingSupported(device)) {
    RETURN_IF_ERROR(ml_drift::cl::CreateCLGLContext(
        device, reinterpret_cast<cl_context_properties>(egl_context),
        reinterpret_cast<cl_context_properties>(egl_display), context,
        options));
  } else {
    RETURN_IF_ERROR(ml_drift::cl::CreateCLContext(device, context, options));
  }
#else
  RETURN_IF_ERROR(ml_drift::cl::CreateCLContext(device, context, options));
#endif  // LITERT_HAS_OPENGL_SUPPORT
  return absl::OkStatus();
}

ml_drift::cl::CLContextOptions GetClContextOptions(GpuPriority gpu_priority) {
  ml_drift::cl::CLContextOptions options;
  if (gpu_priority == kGpuLowPriority) {
    LITERT_LOG(LITERT_DEBUG, "Using low priority for GPU accelerator.");
    options.performance = ml_drift::cl::PerformanceHint::kLow;
    options.priority = ml_drift::cl::PriorityHint::kLow;
  }
  return options;
}

absl::StatusOr<std::unique_ptr<ml_drift::cl::Environment>> CreateClEnvironment(
    ml_drift::cl::CLDevice device, ml_drift::cl::CLContext context,
    ml_drift::cl::CLCommandQueue queue,
    const ml_drift::cl::CLCommandQueueOptions& queue_options) {
  ml_drift::cl::ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(device, context, &profiling_queue,
                                              queue_options));

  return std::make_unique<ml_drift::cl::Environment>(
      std::move(device), std::move(context), std::move(queue),
      std::move(profiling_queue));
}

// Gets or creates the delegate-local wrapper around the LiteRT GPU environment.
absl::StatusOr<DelegateEnvironment*> GetOrCreateDelegateEnvironment(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment litert_env,
    GpuPriority gpu_priority) {
  // Use a holder to keep the environment alive.
  auto resources = std::make_unique<DelegateEnvironment>();

  // Note: CLContextOptions are not consumed when creating the CLContext if
  // has_ownership is false (i.e. if LiteRT Environment already has a GPU
  // environment).
  ml_drift::cl::CLContextOptions context_options =
      GetClContextOptions(gpu_priority);
  // Note: CLCommandQueueOptions are not consumed when creating the
  // CLCommandQueue if has_ownership is false (i.e. if LiteRT Environment
  // already has a GPU environment).
  ml_drift::cl::CLCommandQueueOptions queue_options;
  queue_options.priority = context_options.priority;

  bool has_gpu_environment = false;
  runtime_context->environment_has_gpu_environment(litert_env,
                                                   &has_gpu_environment);
  if (has_gpu_environment) {
    LiteRtEnvironmentOptions env_options;
    LITERT_RETURN_IF_ERROR(
        runtime_context->get_environment_options(litert_env, &env_options));
    LiteRtAny user_data;
    auto status = runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
        &user_data);
    if (status == kLiteRtStatusOk && user_data.ptr_value != nullptr) {
      return reinterpret_cast<DelegateEnvironment*>(
          const_cast<void*>(user_data.ptr_value));
    }

    // If we have a GPU environment but no user data, it means it was created
    // externally. We create a DelegateEnvironment from the options.
    LiteRtAny egl_display_any, egl_context_any;
    auto egl_display_status = runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagEglDisplay, &egl_display_any);
    if (egl_display_status == kLiteRtStatusOk) {
      resources->egl_display =
          reinterpret_cast<LiteRtEglDisplay>(egl_display_any.int_value);
    }
    auto egl_context_status = runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagEglContext, &egl_context_any);
    if (egl_context_status == kLiteRtStatusOk) {
      resources->egl_context =
          reinterpret_cast<LiteRtEglContext>(egl_context_any.int_value);
    }

    // Query CL environment options.
    LiteRtAny device_id, platform_id, context_id, command_queue;
    LITERT_RETURN_IF_ERROR(runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagOpenClDeviceId, &device_id));
    LITERT_RETURN_IF_ERROR(runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagOpenClPlatformId, &platform_id));
    LITERT_RETURN_IF_ERROR(runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagOpenClContext, &context_id));
    LITERT_RETURN_IF_ERROR(runtime_context->get_environment_options_value(
        env_options, kLiteRtEnvOptionTagOpenClCommandQueue, &command_queue));

    ml_drift::cl::CLDevice device(
        reinterpret_cast<cl_device_id>(device_id.int_value),
        reinterpret_cast<cl_platform_id>(platform_id.int_value));

    ml_drift::cl::CLContext context(
        reinterpret_cast<cl_context>(context_id.int_value),
        /*has_ownership=*/false, device);

    ml_drift::cl::CLCommandQueue queue(
        reinterpret_cast<cl_command_queue>(command_queue.int_value),
        /*has_ownership=*/false);

    LITERT_ASSIGN_OR_RETURN(
        resources->cl_env,
        CreateClEnvironment(std::move(device), std::move(context),
                            std::move(queue), queue_options));

    // Register callback to LiteRT Environment. This will be called when the
    // LiteRT GPU environment is destroyed.
    LITERT_ASSIGN_OR_RETURN(LiteRtAny callback,
                            litert::ToLiteRtAny(reinterpret_cast<const void*>(
                                &DestroyDelegateEnvironment)));
    LITERT_ASSIGN_OR_RETURN(
        LiteRtAny delegate_env_ptr,
        litert::ToLiteRtAny(reinterpret_cast<const void*>(resources.get())));

    const std::array<LiteRtEnvOption, 2> environment_options = {
        LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
                        .value = callback},
        LiteRtEnvOption{
            .tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
            .value = delegate_env_ptr},
    };

    LITERT_RETURN_IF_ERROR(runtime_context->add_environment_options(
        litert_env, environment_options.size(), environment_options.data(),
        /*overwrite=*/true));
    // Release ownership to LiteRT environment.
    return resources.release();
  }
  // No GPU environment found. Create a new one.

  // Handle EGL environment.
  {
#if LITERT_HAS_OPENGL_SUPPORT
    RETURN_IF_ERROR(
        ml_drift::gl::EglEnvironment::NewEglEnvironment(&resources->egl_env));
    resources->egl_context = resources->egl_env->context().context();
    resources->egl_display = resources->egl_env->display();
#endif  // LITERT_HAS_OPENGL_SUPPORT
  }

  // Handle CL environment.
  {
    ml_drift::cl::CLDevice device;
    RETURN_IF_ERROR(ml_drift::cl::CreateDefaultGPUDevice(&device));

    ml_drift::cl::CLContext context;
    RETURN_IF_ERROR(CreateClContext(device, context_options,
                                    resources->egl_context,
                                    resources->egl_display, &context));

    ml_drift::cl::CLCommandQueue queue;
    RETURN_IF_ERROR(
        CreateCLCommandQueue(device, context, &queue, queue_options));

    LITERT_ASSIGN_OR_RETURN(
        resources->cl_env,
        CreateClEnvironment(std::move(device), std::move(context),
                            std::move(queue), queue_options));
  }

  // Register callback to LiteRT Environment. This will be called when the
  // LiteRT GPU environment is destroyed.
  LITERT_ASSIGN_OR_RETURN(LiteRtAny callback,
                          litert::ToLiteRtAny(reinterpret_cast<const void*>(
                              &DestroyDelegateEnvironment)));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny user_data,
      litert::ToLiteRtAny(reinterpret_cast<const void*>(resources.get())));

  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny device_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->cl_env->device().id()))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny platform_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->cl_env->device().platform()))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny context_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->cl_env->context().context()))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny command_queue,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->cl_env->queue()->queue()))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny litert_egl_display,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->egl_display))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny litert_egl_context,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->egl_context))));

  const std::array<LiteRtEnvOption, 8> environment_options = {
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagOpenClDeviceId,
                      .value = device_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagOpenClPlatformId,
                      .value = platform_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagOpenClContext,
                      .value = context_id},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagOpenClCommandQueue,
                      .value = command_queue},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagEglDisplay,
                      .value = litert_egl_display},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagEglContext,
                      .value = litert_egl_context},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
                      .value = callback},
      LiteRtEnvOption{.tag = kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
                      .value = user_data},
  };

  LITERT_RETURN_IF_ERROR(runtime_context->gpu_environment_create(
      litert_env, environment_options.size(), environment_options.data()));
  // Release ownership to LiteRT environment.
  return resources.release();
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
  bool is_profiling = ml_drift::IsTfLiteProfilerActive(context);

  auto* delegate_kernel =
      reinterpret_cast<litert::ml_drift::DelegateKernelLiteRt*>(
          node->user_data);

  uint64_t upload_start;
  if (is_profiling) {
    upload_start = tflite::profiling::time::NowMicros();
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

  if (is_profiling) {
    auto status = delegate_kernel->backend()->WaitForCompletion();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for completion: " << status;
      return kTfLiteError;
    }
    uint64_t dispatch_start = tflite::profiling::time::NowMicros();
    ml_drift::AddTfLiteProfilerEvent(context, "UploadOrBindTensorBuffer",
                                     dispatch_start - upload_start);
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

  uint64_t download_start;
  if (is_profiling) {
    auto status = delegate_kernel->backend()->WaitForCompletion();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for completion: " << status;
      return kTfLiteError;
    }
    download_start = tflite::profiling::time::NowMicros();
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

  // Wait for GPU completion on specific GPUs when IsHintWaitingForCompletion()
  // is enabled.
  auto gpu_info = delegate_kernel->backend()->GetInfo();
  if (!gpu_info.ok()) {
    ABSL_LOG(ERROR) << "Failed to get gpu info: " << gpu_info.status();
    return kTfLiteError;
  }
  bool waiting_for_completion =
      delegate_kernel->IsWaitingForCompletionHinted() &&
      (gpu_info.value().IsMali() || gpu_info.value().IsAMD());

  if (delegate_kernel->IsBenchmarkMode() || waiting_for_completion) {
    // In benchmark mode, call ClFinish() to wait for all the enqueued
    // commands to be completed.
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

  if (is_profiling) {
    uint64_t download_end = tflite::profiling::time::NowMicros();
    ml_drift::AddTfLiteProfilerEvent(context,
                                     "DownloadGpuMemoryToTensorBufferGpuMemory",
                                     download_end - download_start);
  }

  if (absl::Status s = delegate_kernel->FlushBufferCacheIfNeeded(context);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
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
#ifndef NDEBUG
  } else if (auto* env_debug_end_node = std::getenv(kEnvDebugEndNode)) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* reg = nullptr;
    int end_node_index_from_env = 0;
    if (absl::SimpleAtoi(env_debug_end_node, &end_node_index_from_env) &&
        context->GetNodeAndRegistration(context, end_node_index_from_env, &node,
                                        &reg) == kTfLiteOk &&
        reg != nullptr) {
      end_node_index = end_node_index_from_env;
      ABSL_LOG(INFO) << kEnvDebugEndNode << " set to " << end_node_index
                     << ". Restricting OpenCL delegation from node 0 to node "
                     << end_node_index << ": code=" << reg->builtin_code;
    }
#endif  // NDEBUG
  }
  litert::ml_drift::CustomOperationParserFactory custom_parser_factory;
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops=*/true, /*max_delegated_partitions=*/1,
      &kExcludedOps, start_node_index, end_node_index, &custom_parser_factory);

#ifndef NDEBUG
  if (auto* env_debug_exclude_nodes = std::getenv(kEnvDebugExcludeNodes)) {
    absl::flat_hash_set<int> excluded_nodes;
    for (absl::string_view s : absl::StrSplit(env_debug_exclude_nodes, ',')) {
      int node_idx;
      if (absl::SimpleAtoi(s, &node_idx)) {
        excluded_nodes.insert(node_idx);
      }
    }
    ABSL_LOG(INFO) << kEnvDebugExcludeNodes << " set to "
                   << absl::StrJoin(excluded_nodes, ",");

    int new_size = 0;
    for (int i = 0; i < ops_to_replace->size; ++i) {
      int node_idx = ops_to_replace->data[i];
      if (excluded_nodes.contains(node_idx)) {
        ABSL_LOG(INFO) << "Excluding node " << node_idx << " (" << i
                       << " in ops_to_replace) from OpenCL delegation.";
      } else {
        ops_to_replace->data[new_size++] = node_idx;
      }
    }
    ops_to_replace->size = new_size;
  }
  for (int i = 0; i < ops_to_replace->size; ++i) {
    int node_idx = ops_to_replace->data[i];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* reg = nullptr;
    if (context->GetNodeAndRegistration(context, node_idx, &node, &reg) ==
            kTfLiteOk &&
        reg != nullptr) {
      ABSL_LOG(INFO) << "- Delegating node " << node_idx
                     << ",  code=" << reg->builtin_code;
    }
  }
#endif  // NDEBUG

  // Replace the ops with delegate kernel.
  const TfLiteRegistration kRegistration = {
      .init = Init,
      .free = Free,
      .prepare = Prepare,
      .invoke = Invoke,
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "LITERT_CL",
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

void LiteRtDeleteMlDriftClDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(
      delegate->data_);
  delete delegate;
}

}  // extern "C"

namespace litert::ml_drift {

// Returns default options for ML Drift OpenCL delegate.
//
// This calls `MlDriftClDelegateDefaultOptions()` add return the result in an
// RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftClDelegateDefaultOptionsPtr() {
  return std::make_unique<MlDriftDelegateOptions>(MlDriftDelegateOptions{
      .precision = MlDriftDelegatePrecision::kDefault,
      .debug_last_delegate_node_index = std::numeric_limits<int>::max(),
      .enable_fast_tuning = true,
      // Note that the program cache is not serialized unless serialization_dir
      // and model_token are also set.
      .serialize_program_cache = true,
      .madvise_original_shared_tensors = true,
      .weight_loader = nullptr,
      // Don't wait for GPU completion on synchronous execution mode for
      // backward compatibility.
      .wait_type = kGpuDelegateWaitTypeDoNotWait,
  });
}

// Creates a new ML Drift OpenCL delegate object.
TfLiteDelegatePtr CreateMlDriftClDelegate(MlDriftDelegateOptionsPtr options,
                                          LiteRtEnvironment litert_env) {
  if (!options) {
    ABSL_LOG(ERROR) << "Missing MLDrift delegate options";
    return {nullptr, LiteRtDeleteMlDriftClDelegate};
  }

  if (options->litert_benchmark_mode) {
    ABSL_LOG(INFO) << "Benchmark mode is enabled.";
  }
  if (options->litert_external_tensors_mode) {
    ABSL_LOG(INFO) << "External tensors mode is enabled.";
  }
  ABSL_LOG(INFO) << "options->hint_waiting_for_completion: "
                 << options->hint_waiting_for_completion;
  ABSL_LOG(INFO) << "options->kernel_batch_size: "
                 << options->kernel_batch_size;

  // Initialize delegate_data.
  auto delegate_data =
      std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->options = std::move(options);
  delegate_data->weight_loader = delegate_data->options->weight_loader;

  // Copy serialization options since they are not owned by the delegate.
  if (delegate_data->options->serialization_dir) {
    delegate_data->serialization_dir =
        delegate_data->options->serialization_dir;
  }
  if (delegate_data->options->model_token) {
    delegate_data->model_token = delegate_data->options->model_token;
  }

  litert::ml_drift::SimpleCache compiled_cache;
  if (delegate_data->options->cache_compiled_programs_only) {
    if (delegate_data->options->program_cache_fd > 0) {
      compiled_cache = litert::ml_drift::SimpleCache(
          ::ml_drift::FileDescriptor(delegate_data->options->program_cache_fd));
    } else if (delegate_data->options->serialize_program_cache) {
      compiled_cache = litert::ml_drift::SimpleCache(
          delegate_data->serialization_dir, delegate_data->model_token);
    }
  }

  if (auto result = ::ml_drift::cl::LoadOpenCL(); !result.ok()) {
    ABSL_LOG(ERROR) << "Failed to open OpenCL library: " << result;
    return {nullptr, LiteRtDeleteMlDriftClDelegate};
  }

  const LiteRtRuntimeContext* runtime_context =
      delegate_data->options->runtime_context;
  if (runtime_context == nullptr) {
    ABSL_LOG(ERROR) << "Missing LiteRT runtime context.";
    return {nullptr, LiteRtDeleteMlDriftClDelegate};
  }

  // Resulting delegate environment is owned by the LiteRT environment.
  auto delegate_env = GetOrCreateDelegateEnvironment(
      runtime_context, litert_env, delegate_data->options->gpu_priority);
  if (!delegate_env.ok()) {
    ABSL_LOG(ERROR) << "Failed to get or create delegate OpenCL environment: "
                    << delegate_env.status();
    return {nullptr, LiteRtDeleteMlDriftClDelegate};
  }

  auto backend = std::make_shared<GpuBackendOpenClLitert>(
      (*delegate_env)->cl_env.get(), (*delegate_env)->egl_display,
      std::move(compiled_cache), runtime_context);
  delegate_data->shared_backend = backend;

  switch (delegate_data->options->precision) {
    case kDefault:
      delegate_data->calculation_precision =
          backend->cl_env()->IsSupported(::ml_drift::CalculationsPrecision::F16)
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
  const bool hint_fully_delegated_to_single_delegate =
      delegate_data->options->hint_fully_delegated_to_single_delegate;
  if (delegate_data->options->kernel_batch_size > 0) {
    backend->set_kernel_batch_size(delegate_data->options->kernel_batch_size);
  }

  // Initialize the ml_drift cl delegate.
  TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                             LiteRtDeleteMlDriftClDelegate);
  delegate->data_ = delegate_data.release();
  delegate->Prepare = DelegatePrepare;
  delegate->flags = kTfLiteDelegateFlagsPerOperatorProfiling;

  if (hint_fully_delegated_to_single_delegate) {
    delegate->flags |= kTfLiteDelegateFlagsHintFullyDelegatedToSingleDelegate;
  }
  return delegate;
}

}  // namespace litert::ml_drift
