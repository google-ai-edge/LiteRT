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
#include <utility>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/options.h"
#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"
#if defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "litert/runtime/accelerators/gpu_static_registry.h"
#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "ml_drift_delegate/delegate/buffer_handler_webgpu.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/delegate_webgpu.h"
#include "tflite/core/c/c_api_types.h"

// Accelerator implementation for the LiteRT GPU WebGPU accelerator.
class GpuWebGpuAccelerator {
 public:
  static std::unique_ptr<GpuWebGpuAccelerator> Create() {
    auto accelerator = std::make_unique<GpuWebGpuAccelerator>();
    accelerator->hardware_support_ = kLiteRtHwAcceleratorGpu;
    return accelerator;
  }

  static void Destroy(void* accelerator) {
    delete reinterpret_cast<GpuWebGpuAccelerator*>(accelerator);
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    static const char* lrt_name = "GPU WebGPU";
    *name = lrt_name;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    static constexpr LiteRtApiVersion lrt_version = {
        /*major=*/1,
        /*minor=*/0,
        /*patch=*/0,
    };
    *version = lrt_version;
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
    auto delegate_options =
        litert::ml_drift::MlDriftWebGpuDelegateDefaultOptionsPtr();
    if (delegate_options && options) {
      delegate_options->weight_loader =
          reinterpret_cast<LiteRtOptionsT*>(options)->weight_loader;
      // Enable constant tensor sharing when weight_loader is provided.
      // This is required for external weight loading to work because the
      // SharedMemoryManager (which sets up the maybe_bind_data callback for
      // loading external weights) is only created when this option is enabled.
      if (delegate_options->weight_loader != nullptr) {
        delegate_options->enable_constant_tensors_sharing = true;
      }
    }
    LITERT_RETURN_IF_ERROR(litert::ml_drift::CreateDelegate(
        runtime_context, env, accelerator,
        litert::ml_drift::GetGpuOptionsPayload(runtime_context, options),
        std::move(delegate_options),
        litert::ml_drift::CreateMlDriftWebGpuDelegate, delegate_ptr));
    auto deleter = delegate_ptr.get_deleter();
    runtime_context->wrap_delegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_ptr.release()),
        reinterpret_cast<void (*)(TfLiteOpaqueDelegate*)>(deleter),
        delegate_wrapper);
    return kLiteRtStatusOk;
  }


 private:
  LiteRtHwAcceleratorSet hardware_support_;
};

// Discovery C object for the GPU WebGPU accelerator by LiteRT.
// This object is used by the LiteRT environment constructor and the
// object name is looked up by dlsym().
extern "C" LiteRtAcceleratorDef LiteRtAcceleratorImpl = {
    .version = 1,  // LiteRtAcceleratorDefV1
    .get_name = GpuWebGpuAccelerator::GetName,
    .get_version = GpuWebGpuAccelerator::GetVersion,
    .get_hardware_support = GpuWebGpuAccelerator::GetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation =
        GpuWebGpuAccelerator::IsTfLiteDelegateResponsibleForJitCompilation,
    .create_delegate = GpuWebGpuAccelerator::CreateDelegate,
    .buffer_handlers =
        {
            .create_func = LiteRtCreateWebGpuMemory,
            .destroy_func = LiteRtDestroyWebGpuMemory,
            .lock_func = LiteRtLockWebGpuMemory,
            .unlock_func = LiteRtUnlockWebGpuMemory,
            .clear_func = LiteRtClearWebGpuMemory,
            .import_func = LiteRtImportWebGpuMemory,
            .device_tag = kLiteRtEnvOptionTagWebGpuDevice,
            .queue_tag = kLiteRtEnvOptionTagWebGpuQueue,
            .num_supported_buffer_types = 7,
            .supported_buffer_types =
                {
                    kLiteRtTensorBufferTypeWebGpuBuffer,
                    kLiteRtTensorBufferTypeWebGpuBufferFp16,
                    kLiteRtTensorBufferTypeWebGpuTexture,
                    kLiteRtTensorBufferTypeWebGpuTextureFp16,
                    kLiteRtTensorBufferTypeWebGpuImageBuffer,
                    kLiteRtTensorBufferTypeWebGpuImageBufferFp16,
                    kLiteRtTensorBufferTypeWebGpuBufferPacked,
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

StaticGpuAcceleratorInitializer g_initializer;

}  // namespace

#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
