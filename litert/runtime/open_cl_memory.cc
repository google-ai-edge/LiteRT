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

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/util.h"  // from @ml_drift
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
#include "litert/runtime/tensor_buffer_lockstate.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>

#if LITERT_HAS_OPENCL_SUPPORT

namespace litert::internal {
namespace {

size_t GetHostPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return std::max<size_t>(4096, sys_info.dwPageSize);
#elif defined(_SC_PAGESIZE)
  int64_t page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? std::max<size_t>(4096, static_cast<size_t>(page_size))
                       : 4096;
#else
  return 4096;
#endif
}

}  // namespace

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

Expected<void> OpenClMemory::Clear() {
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  const cl_int zero = 0;
  auto error_code = ::ml_drift::cl::clEnqueueFillBuffer(
      gpu_env_->GetCommandQueue()->queue(), GetMemoryPtr(), &zero, sizeof(zero),
      0, size_, 0, /*event_wait_list=*/nullptr, /*event=*/nullptr);
  if (error_code != CL_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("Failed to clear OpenCL buffer: ",
                     ::ml_drift::cl::CLErrorCodeToString(error_code)));
  }
  return Expected<void>();
}

bool OpenClMemory::IsSupported() {
  static bool is_supported = ::ml_drift::cl::LoadOpenCL().ok();
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
    ::ml_drift::cl::Buffer buffer;
    LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateReadWriteBuffer(
                               bytes_size, gpu_env->GetContext(), &buffer)
                               .ok());
    return Expected<OpenClMemory>(gpu_env, tensor_type, buffer_type,
                                  std::move(buffer));
  }

  cl_mem cl_memory;
  LITERT_RETURN_IF_ERROR(LiteRtGpuMemoryCreate(
      gpu_env, &tensor_type, buffer_type, bytes_size, &cl_memory));

  ::ml_drift::cl::Buffer buffer(cl_memory, bytes_size);

  return Expected<OpenClMemory>(gpu_env, tensor_type, buffer_type,
                                std::move(buffer));
}

bool IsAhwbToClInteropSupported() {
  return ::ml_drift::cl::clImportMemoryARM != nullptr;
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
      ::ml_drift::cl::clImportMemoryARM(context, CL_MEM_READ_WRITE, properties,
                                        ahwb_buffer.ahwb, size_bytes, &error);
  LITERT_RETURN_IF_ERROR(
      error == CL_SUCCESS,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Failed to create OpenCL buffer from "
                              "AHardwareBuffer: ",
                              ::ml_drift::cl::CLErrorCodeToString(error))));

  ::ml_drift::cl::Buffer cl_buffer(buffer, size_bytes);

  return OpenClMemory(gpu_env, tensor_type, kLiteRtTensorBufferTypeOpenClBuffer,
                      std::move(cl_buffer), ahwb_buffer.ahwb);
}

Expected<OpenClMemory> OpenClMemory::AllocFromGlBuffer(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    GlBuffer& gl_buffer) {
#ifndef CL_DELEGATE_NO_GL
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  ::ml_drift::cl::CLContext* context = gpu_env->GetContext();
  cl_int error;
  cl_mem buffer = ::ml_drift::cl::clCreateFromGLBuffer(
      context->context(), CL_MEM_READ_WRITE, gl_buffer.id(), &error);
  LITERT_RETURN_IF_ERROR(
      error == CL_SUCCESS,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Failed to create OpenCL buffer from GL buffer: ",
                              ::ml_drift::cl::CLErrorCodeToString(error))));

  ::ml_drift::cl::Buffer cl_buffer(buffer, gl_buffer.size_bytes());
  return OpenClMemory(gpu_env, tensor_type, kLiteRtTensorBufferTypeOpenClBuffer,
                      std::move(cl_buffer));
#else
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "GL interop is not supported when compiled with CL_DELEGATE_NO_GL");
#endif
}

void OpenClMemory::CleanupHostBuffer() {
  absl::MutexLock lock(mutex_);
  if (is_host_backed_ && retained_queue_ != nullptr &&
      buffer_.GetMemoryPtr() != nullptr) {
    if (host_sync_state_ == HostSyncState::kHostMapped &&
        host_mapped_ptr_ != nullptr &&
        ::ml_drift::cl::clEnqueueUnmapMemObject != nullptr) {
      if (pending_host_sync_ && ::ml_drift::cl::clFinish != nullptr) {
        ::ml_drift::cl::clFinish(retained_queue_);
        pending_host_sync_ = false;
      }
      ::ml_drift::cl::clEnqueueUnmapMemObject(
          retained_queue_, buffer_.GetMemoryPtr(), host_mapped_ptr_, 0, nullptr,
          nullptr);
      host_mapped_ptr_ = nullptr;
      host_sync_state_ = HostSyncState::kDeviceSynced;
    }
    if (::ml_drift::cl::clFinish != nullptr) {
      ::ml_drift::cl::clFinish(retained_queue_);
      pending_host_sync_ = false;
    }
    buffer_ = ::ml_drift::cl::Buffer();
  }
  if (retained_queue_ != nullptr &&
      ::ml_drift::cl::clReleaseCommandQueue != nullptr) {
    ::ml_drift::cl::clReleaseCommandQueue(retained_queue_);
    retained_queue_ = nullptr;
  }
  is_host_backed_ = false;
}

Expected<OpenClMemory> OpenClMemory::AllocFromHostBuffer(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    void* host_ptr, size_t logical_size, size_t host_alloc_size) {
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (gpu_env == nullptr || gpu_env->GetContext() == nullptr ||
      gpu_env->GetCommandQueue() == nullptr ||
      gpu_env->GetContext()->context() == nullptr ||
      gpu_env->GetCommandQueue()->queue() == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "OpenCL context or command queue is not initialized");
  }
  if (::ml_drift::cl::clRetainCommandQueue == nullptr ||
      ::ml_drift::cl::clReleaseCommandQueue == nullptr ||
      ::ml_drift::cl::clCreateBuffer == nullptr ||
      ::ml_drift::cl::clEnqueueMapBuffer == nullptr ||
      ::ml_drift::cl::clEnqueueUnmapMemObject == nullptr ||
      ::ml_drift::cl::clFinish == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Required OpenCL functions are not available");
  }
  if (host_ptr == nullptr || logical_size == 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Host buffer pointer is null or size is zero");
  }

  const size_t page_size = GetHostPageSize();
  if (reinterpret_cast<uintptr_t>(host_ptr) % page_size != 0) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("Host buffer pointer must be aligned to ", page_size,
                     " bytes for OpenCL host pointer interop"));
  }
  if (logical_size > std::numeric_limits<size_t>::max() - (page_size - 1)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Host buffer size overflows page alignment");
  }
  const size_t aligned_size =
      ((logical_size + page_size - 1) / page_size) * page_size;
  const size_t cl_size =
      (host_alloc_size >= aligned_size) ? aligned_size : logical_size;

  cl_context context = gpu_env->GetContext()->context();
  cl_command_queue retained_queue = gpu_env->GetCommandQueue()->queue();
  if (cl_int err = ::ml_drift::cl::clRetainCommandQueue(retained_queue);
      err != CL_SUCCESS) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      absl::StrCat("Failed to retain OpenCL command queue: ",
                                   ::ml_drift::cl::CLErrorCodeToString(err)));
  }
  auto queue_cleanup = absl::MakeCleanup([&retained_queue] {
    if (retained_queue != nullptr) {
      ::ml_drift::cl::clReleaseCommandQueue(retained_queue);
    }
  });

  cl_int err = CL_SUCCESS;
  cl_mem mem = ::ml_drift::cl::clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, cl_size, host_ptr,
      &err);
  if (err != CL_SUCCESS || mem == nullptr) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("Failed to create OpenCL buffer from host memory: ",
                     ::ml_drift::cl::CLErrorCodeToString(err)));
  }
  ::ml_drift::cl::Buffer cl_buffer(mem, cl_size);

  void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
      retained_queue, cl_buffer.GetMemoryPtr(), CL_TRUE,
      CL_MAP_READ | CL_MAP_WRITE, 0, logical_size, 0, nullptr, nullptr, &err);
  if (err != CL_SUCCESS || mapped_ptr == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      absl::StrCat("Failed to map OpenCL host buffer: ",
                                   ::ml_drift::cl::CLErrorCodeToString(err)));
  }
  if (mapped_ptr != host_ptr) {
    std::memcpy(mapped_ptr, host_ptr, logical_size);
  }

  cl_command_queue queue_to_own = retained_queue;
  retained_queue = nullptr;
  return OpenClMemory(gpu_env, tensor_type, std::move(cl_buffer), queue_to_own,
                      context, host_ptr, mapped_ptr, logical_size);
}

Expected<void> OpenClMemory::FinishPendingSyncLocked() {
  if (!pending_host_sync_) {
    return {};
  }
  if (host_mapped_ptr_ != nullptr && host_mapped_ptr_ != host_ptr_) {
    std::memcpy(host_ptr_, host_mapped_ptr_, host_logical_size_);
  }
  pending_host_sync_ = false;
  return {};
}

Expected<void> OpenClMemory::HostSync(OpenClHostSyncOp op) {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (!is_host_backed_ || retained_queue_ == nullptr ||
      buffer_.GetMemoryPtr() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OpenCL memory is not host-backed");
  }

  switch (op) {
    case OpenClHostSyncOp::kToDevice: {
      if (pending_host_sync_) {
        cl_int err = ::ml_drift::cl::clFinish(retained_queue_);
        if (err != CL_SUCCESS) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrCat("Failed to finish pending OpenCL host sync: ",
                           ::ml_drift::cl::CLErrorCodeToString(err)));
        }
        LITERT_RETURN_IF_ERROR(FinishPendingSyncLocked());
      }
      if (host_sync_state_ == HostSyncState::kHostMapped) {
        if (host_mapped_ptr_ != host_ptr_) {
          std::memcpy(host_mapped_ptr_, host_ptr_, host_logical_size_);
        }
        cl_int err = ::ml_drift::cl::clEnqueueUnmapMemObject(
            retained_queue_, buffer_.GetMemoryPtr(), host_mapped_ptr_, 0,
            nullptr, nullptr);
        if (err != CL_SUCCESS) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrCat("Failed to unmap OpenCL host buffer: ",
                           ::ml_drift::cl::CLErrorCodeToString(err)));
        }
        host_mapped_ptr_ = nullptr;
        host_sync_state_ = HostSyncState::kDeviceSynced;
      }
      return {};
    }
    case OpenClHostSyncOp::kPrepareForDeviceWrite: {
      if (pending_host_sync_) {
        cl_int err = ::ml_drift::cl::clFinish(retained_queue_);
        if (err != CL_SUCCESS) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrCat("Failed to finish pending OpenCL host sync: ",
                           ::ml_drift::cl::CLErrorCodeToString(err)));
        }
        pending_host_sync_ = false;
      }
      if (host_sync_state_ == HostSyncState::kHostMapped) {
        cl_int err = ::ml_drift::cl::clEnqueueUnmapMemObject(
            retained_queue_, buffer_.GetMemoryPtr(), host_mapped_ptr_, 0,
            nullptr, nullptr);
        if (err != CL_SUCCESS) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrCat("Failed to unmap OpenCL host buffer for write: ",
                           ::ml_drift::cl::CLErrorCodeToString(err)));
        }
        host_mapped_ptr_ = nullptr;
        host_sync_state_ = HostSyncState::kDeviceSynced;
      }
      return {};
    }
    case OpenClHostSyncOp::kToHostOutputAsync:
    case OpenClHostSyncOp::kToHostInputAsync: {
      if (host_sync_state_ == HostSyncState::kDeviceSynced) {
        cl_int err = CL_SUCCESS;
        void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
            retained_queue_, buffer_.GetMemoryPtr(), CL_FALSE,
            CL_MAP_READ | CL_MAP_WRITE, 0, host_logical_size_, 0, nullptr,
            nullptr, &err);
        if (err != CL_SUCCESS || mapped_ptr == nullptr) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              absl::StrCat("Failed to map OpenCL host buffer: ",
                           ::ml_drift::cl::CLErrorCodeToString(err)));
        }
        host_mapped_ptr_ = mapped_ptr;
        host_sync_state_ = HostSyncState::kHostMapped;
        pending_host_sync_ = true;
      }
      return {};
    }
    case OpenClHostSyncOp::kFinishSyncToHost:
      return FinishPendingSyncLocked();
  }
  return Unexpected(kLiteRtStatusErrorInvalidArgument,
                    "Unsupported OpenClHostSyncOp");
}

Expected<void*> OpenClMemory::HostLock(LiteRtTensorBufferLockMode mode) {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (!is_host_backed_ || retained_queue_ == nullptr ||
      buffer_.GetMemoryPtr() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OpenCL memory is not host-backed");
  }
  if (pending_host_sync_) {
    cl_int err = ::ml_drift::cl::clFinish(retained_queue_);
    if (err != CL_SUCCESS) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to finish pending OpenCL host sync: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    LITERT_RETURN_IF_ERROR(FinishPendingSyncLocked());
  }
  if (host_sync_state_ == HostSyncState::kDeviceSynced) {
    cl_int err = CL_SUCCESS;
    void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
        retained_queue_, buffer_.GetMemoryPtr(), CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE, 0, host_logical_size_, 0, nullptr, nullptr,
        &err);
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to map OpenCL host buffer on lock: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    host_mapped_ptr_ = mapped_ptr;
    host_sync_state_ = HostSyncState::kHostMapped;
    if ((mode == kLiteRtTensorBufferLockModeRead ||
         mode == kLiteRtTensorBufferLockModeReadWrite) &&
        host_mapped_ptr_ != host_ptr_) {
      std::memcpy(host_ptr_, host_mapped_ptr_, host_logical_size_);
    }
  }
  return host_ptr_;
}

Expected<void> OpenClMemory::HostUnlock(LiteRtTensorBufferLockMode mode) {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (!is_host_backed_ || retained_queue_ == nullptr ||
      buffer_.GetMemoryPtr() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OpenCL memory is not host-backed");
  }
  if ((mode == kLiteRtTensorBufferLockModeWrite ||
       mode == kLiteRtTensorBufferLockModeReadWrite) &&
      host_sync_state_ == HostSyncState::kHostMapped) {
    if (host_mapped_ptr_ != host_ptr_) {
      std::memcpy(host_mapped_ptr_, host_ptr_, host_logical_size_);
    }
    cl_int err = ::ml_drift::cl::clEnqueueUnmapMemObject(
        retained_queue_, buffer_.GetMemoryPtr(), host_mapped_ptr_, 0, nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to unmap OpenCL host buffer on unlock: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    host_mapped_ptr_ = nullptr;
    host_sync_state_ = HostSyncState::kDeviceSynced;
  }
  return {};
}

Expected<void> OpenClMemory::HostClearInPlace() {
  absl::MutexLock lock(mutex_);
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));
  if (!is_host_backed_ || retained_queue_ == nullptr ||
      buffer_.GetMemoryPtr() == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "OpenCL memory is not host-backed");
  }
  if (pending_host_sync_) {
    cl_int err = ::ml_drift::cl::clFinish(retained_queue_);
    if (err != CL_SUCCESS) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to finish pending OpenCL host sync: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    pending_host_sync_ = false;
  }
  if (host_sync_state_ == HostSyncState::kDeviceSynced) {
    cl_int err = CL_SUCCESS;
    void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
        retained_queue_, buffer_.GetMemoryPtr(), CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE, 0, host_logical_size_, 0, nullptr, nullptr,
        &err);
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to map OpenCL host buffer on clear: ",
                       ::ml_drift::cl::CLErrorCodeToString(err)));
    }
    host_mapped_ptr_ = mapped_ptr;
    host_sync_state_ = HostSyncState::kHostMapped;
  }
  std::memset(host_mapped_ptr_, 0, host_logical_size_);
  if (host_mapped_ptr_ != host_ptr_) {
    std::memset(host_ptr_, 0, host_logical_size_);
  }
  return {};
}

}  // namespace litert::internal

#endif  // LITERT_HAS_OPENCL_SUPPORT
