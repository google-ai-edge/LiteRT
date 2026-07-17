// Copyright 2026 Google LLC.
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

#include <cstddef>
#include <memory>
#include <utility>

#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/core/options.h"
#include "litert/runtime/accelerators/gpu/ml_drift_delegate_create.h"
#include "ml_drift_delegate/delegate/buffer_handler_opencl.h"
#if defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "litert/runtime/accelerators/gpu_static_registry.h"
#endif  // defined(LITERT_USE_STATIC_LINKED_GPU_ACCELERATOR)
#include "ml_drift_delegate/delegate/buffer_handler_webgpu.h"
#include "ml_drift_delegate/delegate/delegate_opencl.h"

namespace {

struct HybridMemoryInfo : public HwMemoryInfo {
  LiteRtGpuBackend backend;
  HwMemoryInfoPtr real_info;
};

}  // namespace
#include "ml_drift_delegate/delegate/delegate_opengl.h"
#include "ml_drift_delegate/delegate/delegate_types.h"
#include "ml_drift_delegate/delegate/delegate_webgpu.h"
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
    static constexpr LiteRtApiVersion accelerator_version = {
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
    active_env_ = env;
    litert::TfLiteDelegatePtr delegate_ptr{nullptr, nullptr};
    LITERT_RETURN_IF_ERROR(CreateGpuDelegateImpl(
        runtime_context, env, accelerator, options, delegate_ptr));
    auto deleter = delegate_ptr.get_deleter();
    runtime_context->wrap_delegate(
        reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_ptr.release()),
        reinterpret_cast<void (*)(TfLiteOpaqueDelegate*)>(deleter),
        delegate_wrapper);
    return kLiteRtStatusOk;
  }

  static LiteRtGpuBackend active_backend_;
  static LiteRtEnvironment active_env_;

  static LiteRtStatus CreateGpuMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info);

  static LiteRtStatus DestroyGpuMemory(HwMemoryInfoPtr hw_memory_info);

  static LiteRtStatus LockGpuMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr);

  static LiteRtStatus UnlockGpuMemory(HwMemoryInfoPtr hw_memory_info);

  static LiteRtStatus ClearGpuMemory(HwMemoryInfoPtr hw_memory_info);

  static LiteRtStatus ImportGpuMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      HwMemoryHandle hw_buffer_handle,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info);

 private:
  static LiteRtStatus CreateGpuDelegateImpl(
      LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env,
      LiteRtAccelerator accelerator, LiteRtOptions options,
      litert::TfLiteDelegatePtr& delegate_ptr) {
    auto gpu_options_payload =
        litert::ml_drift::GetGpuOptionsPayload(runtime_context, options);
#if LITERT_HAS_WEBGPU_SUPPORT
    LiteRtGpuBackend backend = kLiteRtGpuBackendAutomatic;
    LrtGetGpuOptionsGpuBackend(&backend, gpu_options_payload);
    if (backend == kLiteRtGpuBackendAutomatic) {
      auto has_opencl = ::ml_drift::cl::LoadOpenCL();
      if (!has_opencl.ok()) {
        LITERT_LOG(
            LITERT_INFO,
            "OpenCL not supported on this platform. Using WebGPU instead.");
        backend = kLiteRtGpuBackendWebGpu;
      }
    }
    if (backend == kLiteRtGpuBackendWebGpu) {
      active_backend_ = kLiteRtGpuBackendWebGpu;
      auto delegate_options =
          litert::ml_drift::MlDriftWebGpuDelegateDefaultOptionsPtr();
      if (delegate_options && options) {
        delegate_options->weight_loader =
            reinterpret_cast<LiteRtOptionsT*>(options)->weight_loader;
        if (delegate_options->weight_loader != nullptr) {
          delegate_options->enable_constant_tensors_sharing = true;
        }
      }
      return litert::ml_drift::CreateDelegate(
          runtime_context, env, accelerator, gpu_options_payload,
          std::move(delegate_options),
          litert::ml_drift::CreateMlDriftWebGpuDelegate, delegate_ptr);
    }
#endif  // LITERT_HAS_WEBGPU_SUPPORT
    if (backend == kLiteRtGpuBackendOpenGl) {
      active_backend_ = kLiteRtGpuBackendOpenGl;
      return litert::ml_drift::CreateDelegate(
          runtime_context, env, accelerator, gpu_options_payload,
          litert::ml_drift::MlDriftOpenGlDelegateDefaultOptionsPtr(),
          litert::ml_drift::CreateMlDriftOpenGlDelegate, delegate_ptr);
    }
    active_backend_ = kLiteRtGpuBackendOpenCl;
    return litert::ml_drift::CreateDelegate(
        runtime_context, env, accelerator, gpu_options_payload,
        litert::ml_drift::MlDriftClDelegateDefaultOptionsPtr(),
        litert::ml_drift::CreateMlDriftClDelegate, delegate_ptr);
  }

  LiteRtHwAcceleratorSet hardware_support_;
};

LiteRtGpuBackend GpuAccelerator::active_backend_ = kLiteRtGpuBackendAutomatic;
LiteRtEnvironment GpuAccelerator::active_env_ = nullptr;

LiteRtStatus GpuAccelerator::CreateGpuMemory(
    LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, size_t packed_bytes,
    HwMemoryInfoPtr* hw_memory_info) {
  auto wrapper = std::make_unique<HybridMemoryInfo>();
  wrapper->backend = active_backend_;

  LiteRtStatus status;
  if (active_backend_ == kLiteRtGpuBackendWebGpu) {
    status =
        LiteRtCreateWebGpuMemory(device_id, queue_id, tensor_type, buffer_type,
                                 bytes, packed_bytes, &wrapper->real_info);
  } else {
    if (active_env_ != nullptr) {
      auto option = active_env_->GetOption(kLiteRtEnvOptionTagOpenClContext);
      if (option.has_value() && option->type == kLiteRtAnyTypeInt) {
        device_id = reinterpret_cast<void*>(option->int_value);
      }
      option = active_env_->GetOption(kLiteRtEnvOptionTagOpenClCommandQueue);
      if (option.has_value() && option->type == kLiteRtAnyTypeInt) {
        queue_id = reinterpret_cast<void*>(option->int_value);
      }
    }
    status =
        LiteRtCreateOpenClMemory(device_id, queue_id, tensor_type, buffer_type,
                                 bytes, packed_bytes, &wrapper->real_info);
  }

  if (status != kLiteRtStatusOk) {
    return status;
  }

  wrapper->memory_handle = wrapper->real_info->memory_handle;
  wrapper->raw_handle = wrapper->real_info->raw_handle;

  *hw_memory_info = wrapper.release();
  return kLiteRtStatusOk;
}

LiteRtStatus GpuAccelerator::DestroyGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* wrapper = static_cast<HybridMemoryInfo*>(hw_memory_info);
  LiteRtStatus status;
  if (wrapper->backend == kLiteRtGpuBackendWebGpu) {
    status = LiteRtDestroyWebGpuMemory(wrapper->real_info);
  } else {
    status = LiteRtDestroyOpenClMemory(wrapper->real_info);
  }
  delete wrapper;
  return status;
}

LiteRtStatus GpuAccelerator::LockGpuMemory(HwMemoryInfoPtr hw_memory_info,
                                           LiteRtTensorBufferLockMode mode,
                                           void** host_memory_ptr) {
  auto* wrapper = static_cast<HybridMemoryInfo*>(hw_memory_info);
  if (wrapper->backend == kLiteRtGpuBackendWebGpu) {
    return LiteRtLockWebGpuMemory(wrapper->real_info, mode, host_memory_ptr);
  } else {
    return LiteRtLockOpenClMemory(wrapper->real_info, mode, host_memory_ptr);
  }
}

LiteRtStatus GpuAccelerator::UnlockGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* wrapper = static_cast<HybridMemoryInfo*>(hw_memory_info);
  if (wrapper->backend == kLiteRtGpuBackendWebGpu) {
    return LiteRtUnlockWebGpuMemory(wrapper->real_info);
  } else {
    return LiteRtUnlockOpenClMemory(wrapper->real_info);
  }
}

LiteRtStatus GpuAccelerator::ClearGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* wrapper = static_cast<HybridMemoryInfo*>(hw_memory_info);
  if (wrapper->backend == kLiteRtGpuBackendWebGpu) {
    return LiteRtClearWebGpuMemory(wrapper->real_info);
  } else {
    return LiteRtClearOpenClMemory(wrapper->real_info);
  }
}

LiteRtStatus GpuAccelerator::ImportGpuMemory(
    LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, HwMemoryHandle hw_buffer_handle,
    size_t bytes, size_t packed_bytes, HwMemoryInfoPtr* hw_memory_info) {
  auto wrapper = std::make_unique<HybridMemoryInfo>();
  wrapper->backend = active_backend_;

  LiteRtStatus status;
  if (active_backend_ == kLiteRtGpuBackendWebGpu) {
    status = LiteRtImportWebGpuMemory(device_id, queue_id, tensor_type,
                                      buffer_type, hw_buffer_handle, bytes,
                                      packed_bytes, &wrapper->real_info);
  } else {
    if (active_env_ != nullptr) {
      auto option = active_env_->GetOption(kLiteRtEnvOptionTagOpenClContext);
      if (option.has_value() && option->type == kLiteRtAnyTypeInt) {
        device_id = reinterpret_cast<void*>(option->int_value);
      }
      option = active_env_->GetOption(kLiteRtEnvOptionTagOpenClCommandQueue);
      if (option.has_value() && option->type == kLiteRtAnyTypeInt) {
        queue_id = reinterpret_cast<void*>(option->int_value);
      }
    }
    status = LiteRtImportOpenClMemory(device_id, queue_id, tensor_type,
                                      buffer_type, hw_buffer_handle, bytes,
                                      packed_bytes, &wrapper->real_info);
  }

  if (status != kLiteRtStatusOk) {
    return status;
  }

  wrapper->memory_handle = wrapper->real_info->memory_handle;
  wrapper->raw_handle = wrapper->real_info->raw_handle;

  *hw_memory_info = wrapper.release();
  return kLiteRtStatusOk;
}

// Discovery C object for the GPU accelerator by LiteRT.
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
            .create_func = GpuAccelerator::CreateGpuMemory,
            .destroy_func = GpuAccelerator::DestroyGpuMemory,
            .lock_func = GpuAccelerator::LockGpuMemory,
            .unlock_func = GpuAccelerator::UnlockGpuMemory,
            .clear_func = GpuAccelerator::ClearGpuMemory,
            .import_func = GpuAccelerator::ImportGpuMemory,
            // We bypass the tags for OpenCL in CreateGpuMemory and
            // ImportGpuMemory. Right now we ignore the passed device_id and
            // queue_id and directly fetch the OpenCL context and command queue
            // from the saved active_env_.
            .device_tag = kLiteRtEnvOptionTagWebGpuDevice,
            .queue_tag = kLiteRtEnvOptionTagWebGpuQueue,
            .num_supported_buffer_types = 14,
            .supported_buffer_types =
                {
                    kLiteRtTensorBufferTypeWebGpuBuffer,
                    kLiteRtTensorBufferTypeWebGpuBufferFp16,
                    kLiteRtTensorBufferTypeWebGpuTexture,
                    kLiteRtTensorBufferTypeWebGpuTextureFp16,
                    kLiteRtTensorBufferTypeWebGpuImageBuffer,
                    kLiteRtTensorBufferTypeWebGpuImageBufferFp16,
                    kLiteRtTensorBufferTypeWebGpuBufferPacked,
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
