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

#include "ml_drift_delegate/delegate/delegate_opengl.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/pelong/egl_environment.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_gl_types.h"  // IWYU pragma: keep
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/composite/custom_parsers.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/gpu_backend_opengl.h"
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/tflite_profile.h"
#include "ml_drift_delegate/tflite/model_builder.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/profiling/time.h"

using ::litert::ml_drift::DelegateKernelLiteRt;
using ::litert::ml_drift::MlDriftDelegateData;

namespace {

// A struct that holds the delegate-local wrapper around the LiteRT GPU
// environment. This is created when the delegate is initialized, and destroyed
// when the LiteRT GPU environment is destroyed.
struct DelegateEnvironment {
  std::unique_ptr<ml_drift::gl::EglEnvironment> egl_env;
  LiteRtEglDisplay egl_display = LITE_RT_EGL_NO_DISPLAY;
  LiteRtEglContext egl_context = LITE_RT_EGL_NO_CONTEXT;
};

void DestroyDelegateEnvironment(void* user_data) {
  delete reinterpret_cast<DelegateEnvironment*>(user_data);
  LITERT_LOG(LITERT_DEBUG, "Destroyed delegate environment.");
}

// Gets or creates the delegate-local wrapper around the LiteRT GPU environment.
absl::StatusOr<DelegateEnvironment*> GetOrCreateDelegateEnvironment(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment litert_env) {
  // Use a holder to keep the environment alive.
  auto resources = std::make_unique<DelegateEnvironment>();

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

    // Initialize egl_env using NewEglEnvironment. If there is a current EGL
    // context on this thread, NewEglEnvironment will adopt it.
    RETURN_IF_ERROR(
        ml_drift::gl::EglEnvironment::NewEglEnvironment(&resources->egl_env));

    // Verify that the acquired EGL context and display match the ones requested
    // in LiteRT Environment options.
    if (resources->egl_display != LITE_RT_EGL_NO_DISPLAY &&
        reinterpret_cast<EGLDisplay>(resources->egl_display) !=
            resources->egl_env->display()) {
      LITERT_LOG(LITERT_ERROR,
                 "LiteRT Environment specifies a different EGL display than "
                 "the current thread's EGL display.");
      return absl::FailedPreconditionError(
          "LiteRT Environment specifies a different EGL display than the "
          "current thread's EGL display.");
    }
    if (resources->egl_context != LITE_RT_EGL_NO_CONTEXT &&
        reinterpret_cast<EGLContext>(resources->egl_context) !=
            resources->egl_env->context().context()) {
      LITERT_LOG(LITERT_ERROR,
                 "LiteRT Environment specifies a different EGL context than "
                 "the current thread's EGL context.");
      return absl::FailedPreconditionError(
          "LiteRT Environment specifies a different EGL context than the "
          "current thread's EGL context.");
    }

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
    RETURN_IF_ERROR(
        ml_drift::gl::EglEnvironment::NewEglEnvironment(&resources->egl_env));
    resources->egl_context = reinterpret_cast<LiteRtEglContext>(
        resources->egl_env->context().context());
    resources->egl_display =
        reinterpret_cast<LiteRtEglDisplay>(resources->egl_env->display());
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
      LiteRtAny litert_egl_display,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->egl_display))));
  LITERT_ASSIGN_OR_RETURN(
      LiteRtAny litert_egl_context,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(resources->egl_context))));

  const std::array<LiteRtEnvOption, 4> environment_options = {
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
  if (delegate_kernel->HasQuantizedTensors()) {
    if (absl::Status s = delegate_kernel->DequantizeInputs(context); !s.ok()) {
      ABSL_LOG(ERROR) << s;
      return kTfLiteError;
    }
  }

  uint64_t upload_start;
  if (is_profiling) {
    upload_start = tflite::profiling::time::NowMicros();
  }

  if (absl::Status s = delegate_kernel->UploadOrBindTensorBuffer(context);
      !s.ok()) {
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

  uint64_t download_start;
  if (is_profiling) {
    auto status = delegate_kernel->backend()->WaitForCompletion();
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for completion: " << status;
      return kTfLiteError;
    }
    download_start = tflite::profiling::time::NowMicros();
  }

  // Download internal output GPU memory to output TensorBufferGPU memory.
  if (absl::Status s =
          delegate_kernel->DownloadGpuMemoryToTensorBufferGpuMemory(context);
      !s.ok()) {
    ABSL_LOG(ERROR) << s;
    return kTfLiteError;
  }

  if (delegate_kernel->IsBenchmarkMode()) {
    // In benchmark mode, call WaitForCompletion() to wait for all the enqueued
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
  litert::ml_drift::ModelBuilderOptions model_builder_options;
  model_builder_options.allow_bool_tensors = true;
  litert::ml_drift::CustomOperationParserFactory custom_parser_factory;
  TfLiteIntArray* ops_to_replace = GetOpsToReplaceWithOptions(
      context, /*allow_quant_ops=*/true, /*options=*/model_builder_options,
      /*max_delegated_partitions=*/1, &kExcludedOps, start_node_index,
      end_node_index, &custom_parser_factory);

  // Replace the ops with delegate kernel.
  const TfLiteRegistration kRegistration = {
      .init = Init,
      .free = Free,
      .prepare = Prepare,
      .invoke = Invoke,
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "LITERT_OPENGL",
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

void LiteRtDeleteMlDriftOpenGlDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  delete reinterpret_cast<litert::ml_drift::MlDriftDelegateData*>(
      delegate->data_);
  delete delegate;
}

}  // extern "C"

namespace litert::ml_drift {

// Returns default options for ML Drift OpenGL delegate.
//
// This calls `MlDriftOpenGlDelegateDefaultOptions()` add return the result in
// an RAII wrapper.
MlDriftDelegateOptionsPtr MlDriftOpenGlDelegateDefaultOptionsPtr() {
  return std::make_unique<MlDriftDelegateOptions>(MlDriftDelegateOptions{
      .precision = MlDriftDelegatePrecision::kDefault,
      .debug_last_delegate_node_index = std::numeric_limits<int>::max(),
      .enable_fast_tuning = true,
  });
}

// Creates a new ML Drift OpenGL delegate object.
TfLiteDelegatePtr CreateMlDriftOpenGlDelegate(MlDriftDelegateOptionsPtr options,
                                              LiteRtEnvironment litert_env) {
  if (!options) {
    ABSL_LOG(ERROR) << "Missing MLDrift delegate options";
    return {nullptr, LiteRtDeleteMlDriftOpenGlDelegate};
  }

  if (options->litert_benchmark_mode) {
    ABSL_LOG(INFO) << "Benchmark mode is enabled.";
  }

  // Get the flag of enable op profiling and use it to set the delegate flags.
  const bool enable_op_profiling = options->enable_op_profiling;

  // Initialize delegate_data.
  auto delegate_data =
      std::make_unique<litert::ml_drift::MlDriftDelegateData>();
  delegate_data->options = std::move(options);

  const LiteRtRuntimeContext* runtime_context =
      delegate_data->options->runtime_context;
  if (runtime_context == nullptr) {
    ABSL_LOG(ERROR) << "Missing LiteRT runtime context.";
    return {nullptr, LiteRtDeleteMlDriftOpenGlDelegate};
  }

  // Copy serialization options since they are not owned by the delegate.
  if (delegate_data->options->serialization_dir) {
    delegate_data->serialization_dir =
        delegate_data->options->serialization_dir;
  }
  if (delegate_data->options->model_token) {
    delegate_data->model_token = delegate_data->options->model_token;
  }

  // Resulting delegate environment is owned by the LiteRT environment.
  auto delegate_env =
      GetOrCreateDelegateEnvironment(runtime_context, litert_env);
  if (!delegate_env.ok()) {
    ABSL_LOG(ERROR) << "Failed to get or create delegate OpenGL environment: "
                    << delegate_env.status();
    return {nullptr, LiteRtDeleteMlDriftOpenGlDelegate};
  }
  delegate_data->shared_backend = std::make_shared<GpuBackendOpenGl>(
      (*delegate_env)->egl_env.get(), runtime_context);

  switch (delegate_data->options->precision) {
    case kDefault:
      delegate_data->calculation_precision =
          (*delegate_env)->egl_env->gpu_info().SupportsFP16()
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
  if (delegate_data->options->use_f32_accum_for_fp16) {
    delegate_data->calculation_precision =
        ::ml_drift::CalculationsPrecision::F32_F16;
  }

  // Initialize the ml_drift gl delegate.
  TfLiteDelegatePtr delegate(new TfLiteDelegate(TfLiteDelegateCreate()),
                             LiteRtDeleteMlDriftOpenGlDelegate);
  delegate->data_ = delegate_data.release();
  delegate->Prepare = DelegatePrepare;
  if (enable_op_profiling) {
    delegate->flags = kTfLiteDelegateFlagsPerOperatorProfiling;
  }
  return delegate;
}

}  // namespace litert::ml_drift
