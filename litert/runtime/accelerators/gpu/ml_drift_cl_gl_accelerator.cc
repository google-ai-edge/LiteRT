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

#include <memory>

#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_opencl.h"
#if defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "litert/runtime/accelerators/gpu_static_registry.h"
#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "third_party/odml/litert/ml_drift/delegate/delegate_opencl.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_opengl.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_types.h"
#include "tflite/core/c/c_api_types.h"

// Accelerator implementation for the LiteRT GPU Accelerator.
class GpuAccelerator {
 public:
  static std::unique_ptr<GpuAccelerator> Create() {
    auto accelerator = std::make_unique<GpuAccelerator>();
    accelerator->hardware_support_ = kLiteRtHwAcceleratorGpu;
    return accelerator;
  }

  static void Destroy(void* accelerator) {
    delete reinterpret_cast<GpuAccelerator*>(accelerator);
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    static const char* lrt_name = "LiteRT GPU";
    *name = lrt_name;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    static constexpr const LiteRtApiVersion accelerator_version = {
        /*major=*/1,
        /*minor=*/0,
        /*patch=*/0,
    };
    *version = accelerator_version;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetHardwareSupport(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware) {
    static LiteRtHwAcceleratorSet hardware_support = kLiteRtHwAcceleratorGpu;
    *supported_hardware = hardware_support;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus IsTfLiteDelegateResponsibleForJitCompilation(
      LiteRtAcceleratorT* accelerator, bool* does_jit_compilation) {
    LITERT_RETURN_IF_ERROR(does_jit_compilation,
                           litert::ErrorStatusBuilder::InvalidArgument())
        << "`does_jit_compilation` pointer is null.";
    *does_jit_compilation = true;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus CreateDelegate(LiteRtRuntimeContext* runtime_context,
                                     LiteRtEnvironment env,
                                     LiteRtAccelerator accelerator,
                                     LiteRtOptions options,
                                     LiteRtDelegateWrapper* delegate_wrapper) {
    litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
    LITERT_RETURN_IF_ERROR(CreateGpuDelegateImpl(
        runtime_context, env, accelerator, options, delegate_ptr));

    auto deleter = delegate_ptr.get_deleter();
    auto* raw_delegate = delegate_ptr.release();
    auto* opaque_delegate =
        reinterpret_cast<TfLiteOpaqueDelegate*>(raw_delegate);

    runtime_context->wrap_delegate(
        opaque_delegate,
        reinterpret_cast<void (*)(TfLiteOpaqueDelegate*)>(deleter),
        delegate_wrapper);
    return kLiteRtStatusOk;
  }

 private:
  static LiteRtStatus CreateGpuDelegateImpl(
      LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env,
      LiteRtAccelerator accelerator, LiteRtOptions options,
      litert::TfLiteDelegatePtr& delegate_ptr) {
    auto gpu_options_payload =
        litert::ml_drift::GetGpuOptionsPayload(runtime_context, options);
    LiteRtGpuBackend backend = kLiteRtGpuBackendAutomatic;
    LrtGetGpuOptionsGpuBackend(&backend, gpu_options_payload);
    if (backend == kLiteRtGpuBackendAutomatic) {
      auto has_opencl = ::ml_drift::cl::LoadOpenCL();
      if (!has_opencl.ok()) {
        LITERT_LOG(
            LITERT_INFO,
            "OpenCL not supported on this platform. Using OpenGL instead.");
        backend = kLiteRtGpuBackendOpenGl;
      }
    }
    if (backend == kLiteRtGpuBackendOpenGl) {
      return litert::ml_drift::CreateDelegate(
          runtime_context, env, accelerator, gpu_options_payload,
          litert::ml_drift::MlDriftOpenGlDelegateDefaultOptionsPtr(),
          litert::ml_drift::CreateMlDriftOpenGlDelegate, delegate_ptr);
    }
    return litert::ml_drift::CreateDelegate(
        runtime_context, env, accelerator, gpu_options_payload,
        litert::ml_drift::MlDriftClDelegateDefaultOptionsPtr(),
        litert::ml_drift::CreateMlDriftClDelegate, delegate_ptr);
  }

  LiteRtHwAcceleratorSet hardware_support_;
};

// Discovery C function for the GPU OpenCL/OpenGL accelerator by LiteRT.
// This object is used by the LiteRT environment constructor and the
// object name is looked up by dlsym().
extern "C" LiteRtAcceleratorDef LiteRtAcceleratorImpl = {
    .version = 1,  // LiteRtAcceleratorDefV1
    .get_name = GpuAccelerator::GetName,
    .get_version = GpuAccelerator::GetVersion,
    .get_hardware_support = GpuAccelerator::GetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation =
        GpuAccelerator::IsTfLiteDelegateResponsibleForJitCompilation,
    .create_delegate = GpuAccelerator::CreateDelegate,
    .buffer_handlers =
        {
            .create_func = LiteRtCreateOpenClMemory,
            .destroy_func = LiteRtDestroyOpenClMemory,
            .lock_func = LiteRtLockOpenClMemory,
            .unlock_func = LiteRtUnlockOpenClMemory,
            .clear_func = LiteRtClearOpenClMemory,
            .import_func = LiteRtImportOpenClMemory,
            .device_tag = kLiteRtEnvOptionTagOpenClContext,
            .queue_tag = kLiteRtEnvOptionTagOpenClCommandQueue,
            .num_supported_buffer_types = 7,
            .supported_buffer_types =
                {
                    kLiteRtTensorBufferTypeOpenClBuffer,
                    kLiteRtTensorBufferTypeOpenClBufferFp16,
                    kLiteRtTensorBufferTypeOpenClTexture,
                    kLiteRtTensorBufferTypeOpenClTextureFp16,
                    kLiteRtTensorBufferTypeOpenClBufferPacked,
                    kLiteRtTensorBufferTypeOpenClImageBuffer,
                    kLiteRtTensorBufferTypeOpenClImageBufferFp16,
                },
        },
};

#if defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)

namespace {

class StaticGpuAcceleratorInitializer {
 public:
  StaticGpuAcceleratorInitializer() {
    LiteRtStaticLinkedAcceleratorGpuDef = &LiteRtAcceleratorImpl;
  }
};

// TODO: b/440648257 - Prevent multiple initializations of the GPU accelerator.
StaticGpuAcceleratorInitializer g_initializer;

}  // namespace

#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
