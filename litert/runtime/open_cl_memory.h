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
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gl_buffer.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/buffer.h"

namespace litert::internal {

/**
 * The OpenCL memory class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU OpenCL buffer.
 */
class OpenClMemory {
 public:
  OpenClMemory(OpenClMemory&& other)
      : gpu_env_(other.gpu_env_),
        tensor_type_(other.tensor_type_),
        buffer_type_(other.buffer_type_),
        data_(other.data_),
        buffer_(std::move(other.buffer_)),
        size_(other.size_),
        ahwb_(other.ahwb_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.ahwb_ = nullptr;
  }

  explicit OpenClMemory(GpuEnvironment* gpu_env,
                        const LiteRtRankedTensorType& tensor_type,
                        LiteRtTensorBufferType buffer_type,
                        tflite::gpu::cl::Buffer buffer,
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
    // tflite::gpu::cl::Buffer (OpenClMemory determines ownership). Null
    // deallocator means that the buffer is not owned by OpenClMemory.
    buffer_ = tflite::gpu::cl::CreateBufferShared(buffer);
  }

  ~OpenClMemory() {
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
    if (data_ != nullptr) {
      litert_aligned_free(data_);
    };
  }

  cl_mem GetMemoryPtr() { return buffer_.GetMemoryPtr(); }
  // Allocates a CPU memory and conducts a copy from the OpenCL buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock(LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the OpenCL buffer.
  template <typename T>
  Expected<void> Unlock();

  // Returns true if OpenCL is supported.
  // Warning: This is only for TEST.
  static bool IsSupported();
  static Expected<OpenClMemory> Alloc(GpuEnvironment* gpu_env,
                                      const LiteRtRankedTensorType& tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes_size);
  static Expected<OpenClMemory> AllocFromAhwbBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      AhwbBuffer& ahwb_buffer);
  static Expected<OpenClMemory> AllocFromGlBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      GlBuffer& gl_buffer);
  size_t size_bytes() const { return size_; }

 private:
  GpuEnvironment* gpu_env_ = nullptr;
  const LiteRtRankedTensorType tensor_type_;
  LiteRtTensorBufferType buffer_type_;
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  tflite::gpu::cl::Buffer buffer_;
  LiteRtOpenClDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
  // The size of the CPU memory buffer in bytes. It's doubled for fp16 buffers.
  size_t cpu_buffer_size_ = 0;
  AHardwareBuffer* ahwb_ = nullptr;
  LockState lock_state_ = LockState::kUnlocked;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
