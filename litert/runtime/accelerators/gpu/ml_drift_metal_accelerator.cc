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
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_metal.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_metal.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_types.h"
#include "tflite/core/c/c_api_types.h"

// Accelerator implementation for the LiteRT GPU Metal accelerator.
class GpuMetalAccelerator {
 public:
  static std::unique_ptr<GpuMetalAccelerator> Create() {
    auto accelerator = std::make_unique<GpuMetalAccelerator>();
    accelerator->hardware_support_ = kLiteRtHwAcceleratorGpu;
    return accelerator;
  }

  static void Destroy(void* accelerator) {
    delete reinterpret_cast<GpuMetalAccelerator*>(accelerator);
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    static const char* lrt_name = "GPU Metal";
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
        litert::ml_drift::MlDriftMetalDelegateDefaultOptionsPtr();
    auto* gpu_options =
        litert::ml_drift::GetGpuOptionsPayload(runtime_context, options);
    if (delegate_options && options) {
      delegate_options->weight_loader =
          reinterpret_cast<LiteRtOptionsT*>(options)->weight_loader;
      if (delegate_options->weight_loader != nullptr) {
        delegate_options->enable_constant_tensors_sharing = true;
      }
#if defined(__APPLE__)
      if (gpu_options) {
        bool enable_residency_set = false;
        LrtGetGpuOptionsMetalResidencySet(gpu_options, &enable_residency_set);
        delegate_options->enable_metal_residency_set = enable_residency_set;
      }
#endif
    }
    LITERT_RETURN_IF_ERROR(litert::ml_drift::CreateDelegate(
        runtime_context, env, accelerator, gpu_options,
        std::move(delegate_options),
        litert::ml_drift::CreateMlDriftMetalDelegate, delegate_ptr));
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
// The symbol is exported by runtime/accelerators/gpu/macos_exported_symbols.lds
extern "C" LiteRtAcceleratorDef LiteRtAcceleratorImpl = {
    .version = 1,  // LiteRtAcceleratorDefV1
    .get_name = GpuMetalAccelerator::GetName,
    .get_version = GpuMetalAccelerator::GetVersion,
    .get_hardware_support = GpuMetalAccelerator::GetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation =
        GpuMetalAccelerator::IsTfLiteDelegateResponsibleForJitCompilation,
    .create_delegate = GpuMetalAccelerator::CreateDelegate,
    .buffer_handlers =
        {
            .create_func = LiteRtCreateMetalMemory,
            .destroy_func = LiteRtDestroyMetalMemory,
            .lock_func = LiteRtLockMetalMemory,
            .unlock_func = LiteRtUnlockMetalMemory,
            .clear_func = LiteRtClearMetalMemory,
            .import_func = LiteRtImportMetalMemory,
            .device_tag = kLiteRtEnvOptionTagMetalDevice,
            .queue_tag = kLiteRtEnvOptionTagMetalCommandQueue,
            .num_supported_buffer_types = 5,
            .supported_buffer_types =
                {
                    kLiteRtTensorBufferTypeMetalBuffer,
                    kLiteRtTensorBufferTypeMetalBufferFp16,
                    kLiteRtTensorBufferTypeMetalTexture,
                    kLiteRtTensorBufferTypeMetalTextureFp16,
                    kLiteRtTensorBufferTypeMetalBufferPacked,
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
