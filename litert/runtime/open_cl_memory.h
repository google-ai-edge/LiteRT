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

#ifndef ODML_LITERT_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
#define ODML_LITERT_LITERT_RUNTIME_OPEN_CL_BUFFER_H_

#include <cstddef>
#include <cstdlib>
#include <utility>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gl_buffer.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include <CL/cl.h>

namespace litert::internal {

enum class OpenClHostSyncOp {
  kToDevice,
  kPrepareForDeviceWrite,
  kToHostOutputAsync,
  kToHostInputAsync,
  kFinishSyncToHost,
};

/**
 * The OpenCL memory class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU OpenCL buffer.
 */
class OpenClMemory {
 public:
  enum class HostSyncState {
    kHostMapped,
    kDeviceSynced,
  };

  OpenClMemory(OpenClMemory&& other)
      : gpu_env_(other.gpu_env_),
        tensor_type_(other.tensor_type_),
        buffer_type_(other.buffer_type_),
        data_(other.data_),
        buffer_(std::move(other.buffer_)),
        deallocator_(other.deallocator_),
        size_(other.size_),
        cpu_buffer_size_(other.cpu_buffer_size_),
        ahwb_(other.ahwb_),
        lock_state_(other.lock_state_),
        retained_queue_(other.retained_queue_),
        cl_context_(other.cl_context_),
        is_host_backed_(other.is_host_backed_),
        host_ptr_(other.host_ptr_),
        host_mapped_ptr_(other.host_mapped_ptr_),
        host_logical_size_(other.host_logical_size_),
        host_sync_state_(other.host_sync_state_),
        pending_host_sync_(other.pending_host_sync_) {
    other.data_ = nullptr;
    other.deallocator_ = nullptr;
    other.size_ = 0;
    other.cpu_buffer_size_ = 0;
    other.ahwb_ = nullptr;
    other.retained_queue_ = nullptr;
    other.cl_context_ = nullptr;
    other.is_host_backed_ = false;
    other.host_ptr_ = nullptr;
    other.host_mapped_ptr_ = nullptr;
    other.host_logical_size_ = 0;
    other.pending_host_sync_ = false;
  }

  explicit OpenClMemory(GpuEnvironment* gpu_env,
                        const LiteRtRankedTensorType& tensor_type,
                        LiteRtTensorBufferType buffer_type,
                        ::ml_drift::cl::Buffer buffer,
                        AHardwareBuffer* ahwb = nullptr)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        buffer_(std::move(buffer)),
        size_(buffer_.GetMemorySizeInBytes()),
        ahwb_(ahwb) {}

  OpenClMemory(GpuEnvironment* gpu_env,
               const LiteRtRankedTensorType& tensor_type,
               LiteRtTensorBufferType buffer_type, cl_mem buffer, size_t size,
               LiteRtOpenClDeallocator deallocator)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        deallocator_(deallocator),
        size_(size) {
    // CreateBufferShared creates a buffer that is not owned by
    // ::ml_drift::cl::Buffer (OpenClMemory determines ownership). Null
    // deallocator means that the buffer is not owned by OpenClMemory.
    buffer_ = ::ml_drift::cl::CreateBufferShared(buffer);
  }

  ~OpenClMemory() {
#if LITERT_HAS_OPENCL_SUPPORT
    CleanupHostBuffer();
#endif  // LITERT_HAS_OPENCL_SUPPORT
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
    if (data_ != nullptr) {
      litert_aligned_free(data_);
    }
  }

  cl_mem GetMemoryPtr() { return buffer_.GetMemoryPtr(); }
  cl_context context() const { return cl_context_; }
  cl_command_queue queue() const { return retained_queue_; }
  // Allocates a CPU memory and conducts a copy from the OpenCL buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock(LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the OpenCL buffer.
  template <typename T>
  Expected<void> Unlock();

  // Clears the OpenCL buffer.
  Expected<void> Clear();

  Expected<void> HostSync(OpenClHostSyncOp op);
  Expected<void*> HostLock(LiteRtTensorBufferLockMode mode);
  Expected<void> HostUnlock(LiteRtTensorBufferLockMode mode);
  Expected<void> HostClearInPlace();

  // Returns true if OpenCL is supported.
  // Warning: This is only for TEST.
  static bool IsSupported();
  static Expected<OpenClMemory> Alloc(GpuEnvironment* gpu_env,
                                      const LiteRtRankedTensorType& tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes_size);
  static Expected<OpenClMemory> AllocFromHostBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      void* host_ptr, size_t logical_size, size_t host_alloc_size);
  static Expected<OpenClMemory> AllocFromAhwbBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      AhwbBuffer& ahwb_buffer);
  static Expected<OpenClMemory> AllocFromGlBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      GlBuffer& gl_buffer);
  size_t size_bytes() const { return size_; }

 private:
  explicit OpenClMemory(GpuEnvironment* gpu_env,
                        const LiteRtRankedTensorType& tensor_type,
                        ::ml_drift::cl::Buffer buffer,
                        cl_command_queue retained_queue, cl_context cl_context,
                        void* host_ptr, void* host_mapped_ptr,
                        size_t host_logical_size)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(kLiteRtTensorBufferTypeOpenClBuffer),
        buffer_(std::move(buffer)),
        size_(buffer_.GetMemorySizeInBytes()),
        retained_queue_(retained_queue),
        cl_context_(cl_context),
        is_host_backed_(true),
        host_ptr_(host_ptr),
        host_mapped_ptr_(host_mapped_ptr),
        host_logical_size_(host_logical_size),
        host_sync_state_(HostSyncState::kHostMapped),
        pending_host_sync_(false) {}

  void CleanupHostBuffer();
  Expected<void> FinishPendingSyncLocked();

  GpuEnvironment* gpu_env_ = nullptr;
  const LiteRtRankedTensorType tensor_type_;
  LiteRtTensorBufferType buffer_type_;
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  ::ml_drift::cl::Buffer buffer_;
  LiteRtOpenClDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
  // The size of the CPU memory buffer in bytes. It's doubled for fp16 buffers.
  size_t cpu_buffer_size_ = 0;
  AHardwareBuffer* ahwb_ = nullptr;
  LockState lock_state_ = LockState::kUnlocked;

  cl_command_queue retained_queue_ = nullptr;
  cl_context cl_context_ = nullptr;
  bool is_host_backed_ = false;
  void* host_ptr_ = nullptr;
  void* host_mapped_ptr_ = nullptr;
  size_t host_logical_size_ = 0;
  HostSyncState host_sync_state_ = HostSyncState::kHostMapped;
  bool pending_host_sync_ = false;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
