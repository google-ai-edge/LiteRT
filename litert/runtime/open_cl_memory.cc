// Copyright 2024 Google LLC.
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

#include "litert/runtime/open_cl_memory.h"

#include <stdlib.h>

#include <cstddef>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gl_buffer.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/open_cl_sync.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#include "tflite/delegates/gpu/cl/buffer.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#include "tflite/delegates/gpu/cl/util.h"

#if LITERT_HAS_OPENCL_SUPPORT

namespace litert::internal {

template Expected<float*> OpenClMemory::Lock<float>(
    LiteRtTensorBufferLockMode mode);
template Expected<char*> OpenClMemory::Lock<char>(
    LiteRtTensorBufferLockMode mode);
template Expected<void> OpenClMemory::Unlock<float>();
template Expected<void> OpenClMemory::Unlock<char>();

template <typename T>
Expected<T*> OpenClMemory::Lock(LiteRtTensorBufferLockMode mode) {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  LITERT_RETURN_IF_ERROR(lock_state_ == LockState::kUnlocked,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "The OpenCL memory is already locked."));
  bool lock_success = false;
  LockState lock_state = ToLockState(mode);
  absl::Cleanup lock_set = [this, &lock_success, &lock_state] {
    if (lock_success) {
      lock_state_ = lock_state;
    }
  };

  if (data_ == nullptr) {
    // The current Lock() always provides a packed buffer regardless of the
    // underlying H/W buffer type. If the underlying H/W buffer has a stride,
    // the data will be converted to the packed buffer by
    // LiteRtGpuMemoryDownload().
    // TODO b/413449050 - Update behavior to return raw H/W buffer and its size.
    LITERT_ASSIGN_OR_RETURN(cpu_buffer_size_,
                            litert::internal::GetNumPackedBytes(tensor_type_));
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 cpu_buffer_size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
  }
  if (lock_state == LockState::kReadLocked ||
      lock_state == LockState::kReadWriteLocked) {
    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      LITERT_RETURN_IF_ERROR(gpu_env_->GetCommandQueue()
                                 ->EnqueueReadBuffer(GetMemoryPtr(),
                                                     cpu_buffer_size_, data_,
                                                     /*async=*/false)
                                 .ok());
    } else {
      // Use the GPU Delegate API to download the data from the OpenCL buffer
      // to the aligned memory.
      LITERT_RETURN_IF_ERROR(
          LiteRtGpuMemoryDownload(gpu_env_, &tensor_type_, buffer_type_,
                                  cpu_buffer_size_, GetMemoryPtr(), data_));
    }
  }
  lock_success = true;
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> OpenClMemory::Unlock() {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  LITERT_RETURN_IF_ERROR(lock_state_ != LockState::kUnlocked,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "The OpenCL memory is already unlocked."));
  absl::Cleanup unlock = [this] { lock_state_ = LockState::kUnlocked; };
  if (lock_state_ == LockState::kWriteLocked ||
      lock_state_ == LockState::kReadWriteLocked) {
    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      LITERT_RETURN_IF_ERROR(gpu_env_->GetCommandQueue()
                                 ->EnqueueWriteBuffer(GetMemoryPtr(),
                                                      cpu_buffer_size_, data_,
                                                      /*async=*/true)
                                 .ok());
    } else {
      // The current Unlock() translates the packed buffer (data_) if the
      // underlying H/W buffer has a stride. This conversion is done by
      // LiteRtGpuMemoryUpload().
      // TODO b/413449050 - Update behavior to upload raw H/W buffer as it is.
      LITERT_RETURN_IF_ERROR(
          LiteRtGpuMemoryUpload(gpu_env_, &tensor_type_, buffer_type_,
                                cpu_buffer_size_, data_, GetMemoryPtr()));
    }
  }
  return Expected<void>();
}

bool OpenClMemory::IsSupported() {
  static bool is_supported = ::tflite::gpu::cl::LoadOpenCL().ok();
  return is_supported;
}

Expected<OpenClMemory> OpenClMemory::Alloc(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes_size) {
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (gpu_env == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "OpenCL is not supported");
  }

  if (buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    tflite::gpu::cl::Buffer buffer;
    LITERT_RETURN_IF_ERROR(tflite::gpu::cl::CreateReadWriteBuffer(
                               bytes_size, gpu_env->GetContext(), &buffer)
                               .ok());
    return Expected<OpenClMemory>(gpu_env, tensor_type, buffer_type,
                                  std::move(buffer));
  }

  cl_mem cl_memory;
  LITERT_RETURN_IF_ERROR(LiteRtGpuMemoryCreate(
      gpu_env, &tensor_type, buffer_type, bytes_size, &cl_memory));

  tflite::gpu::cl::Buffer buffer(cl_memory, bytes_size);

  return Expected<OpenClMemory>(gpu_env, tensor_type, buffer_type,
                                std::move(buffer));
}

bool IsAhwbToClInteropSupported() {
  return ::tflite::gpu::cl::clImportMemoryARM != nullptr;
}

Expected<OpenClMemory> OpenClMemory::AllocFromAhwbBuffer(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    AhwbBuffer& ahwb_buffer) {
  cl_int error = CL_SUCCESS;
  const cl_import_properties_arm properties[] = {
      CL_IMPORT_TYPE_ARM,
      CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM,
      0,
  };

  cl_context context = gpu_env->GetContext()->context();
  LITERT_RETURN_IF_ERROR(IsAhwbToClInteropSupported(),
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "clImportMemoryARM is not supported"));
  LITERT_ASSIGN_OR_RETURN(size_t size_bytes,
                          AhwbBuffer::GetSize(ahwb_buffer.ahwb));
  cl_mem buffer =
      tflite::gpu::cl::clImportMemoryARM(context, CL_MEM_READ_WRITE, properties,
                                         ahwb_buffer.ahwb, size_bytes, &error);
  LITERT_RETURN_IF_ERROR(
      error == CL_SUCCESS,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Failed to create OpenCL buffer from "
                              "AHardwareBuffer: ",
                              tflite::gpu::cl::CLErrorCodeToString(error))));

  tflite::gpu::cl::Buffer cl_buffer(buffer, size_bytes);

  return OpenClMemory(gpu_env, tensor_type, kLiteRtTensorBufferTypeOpenClBuffer,
                      std::move(cl_buffer), ahwb_buffer.ahwb);
}

Expected<OpenClMemory> OpenClMemory::AllocFromGlBuffer(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    GlBuffer& gl_buffer) {
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  tflite::gpu::cl::CLContext* context = gpu_env->GetContext();
  cl_int error;
  cl_mem buffer = tflite::gpu::cl::clCreateFromGLBuffer(
      context->context(), CL_MEM_READ_WRITE, gl_buffer.id(), &error);
  LITERT_RETURN_IF_ERROR(
      error == CL_SUCCESS,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Failed to create OpenCL buffer from GL buffer: ",
                              tflite::gpu::cl::CLErrorCodeToString(error))));

  tflite::gpu::cl::Buffer cl_buffer(buffer, gl_buffer.size_bytes());
  return OpenClMemory(gpu_env, tensor_type, kLiteRtTensorBufferTypeOpenClBuffer,
                      std::move(cl_buffer));
}

}  // namespace litert::internal

#endif  // LITERT_HAS_OPENCL_SUPPORT
