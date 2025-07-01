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
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/ahwb_buffer.h"
#if LITERT_ENABLE_GPU
#include "litert/runtime/gl_buffer.h"
#include "litert/runtime/gpu_environment.h"
#endif  // LITERT_ENABLE_GPU
#include <CL/cl.h>
#if LITERT_ENABLE_GPU
#include "tflite/delegates/gpu/cl/buffer.h"
#endif  // LITERT_ENABLE_GPU

namespace litert::internal {

enum class LockState {
  kUnlocked = 0,
  kRead = 1,
  kWrite = 2,
  kReadWrite = 3,
};
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

#if LITERT_ENABLE_GPU
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
#else
  explicit OpenClMemory(void* gpu_env,
                        const LiteRtRankedTensorType& tensor_type,
                        LiteRtTensorBufferType buffer_type,
                        void* buffer,
                        AHardwareBuffer* ahwb = nullptr)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        size_(0),
        ahwb_(ahwb) {}

  OpenClMemory(void* gpu_env,
               const LiteRtRankedTensorType& tensor_type,
               LiteRtTensorBufferType buffer_type, cl_mem buffer, size_t size,
               LiteRtOpenClDeallocator deallocator)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        deallocator_(deallocator),
        size_(size) {
#endif  // LITERT_ENABLE_GPU
    // CreateBufferShared creates a buffer that is not owned by
    // tflite::gpu::cl::Buffer (OpenClMemory determines ownership). Null
    // deallocator means that the buffer is not owned by OpenClMemory.
#if LITERT_ENABLE_GPU
    buffer_ = tflite::gpu::cl::CreateBufferShared(buffer);
#endif  // LITERT_ENABLE_GPU
  }

  ~OpenClMemory() {
#if LITERT_ENABLE_GPU
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
#endif  // LITERT_ENABLE_GPU
    if (data_ != nullptr) {
      litert_aligned_free(data_);
    };
  }

#if LITERT_ENABLE_GPU
  cl_mem GetMemoryPtr() { return buffer_.GetMemoryPtr(); }
#else
  cl_mem GetMemoryPtr() { return nullptr; }
#endif  // LITERT_ENABLE_GPU
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
#if LITERT_ENABLE_GPU
  static Expected<OpenClMemory> Alloc(GpuEnvironment* gpu_env,
                                      const LiteRtRankedTensorType& tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes_size);
  static Expected<OpenClMemory> AllocFromAhwbBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      AhwbBuffer& ahwb_buffer);
#else
  static Expected<OpenClMemory> Alloc(void* gpu_env,
                                      const LiteRtRankedTensorType& tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes_size);
  static Expected<OpenClMemory> AllocFromAhwbBuffer(
      void* gpu_env, const LiteRtRankedTensorType& tensor_type,
      AhwbBuffer& ahwb_buffer);
#endif  // LITERT_ENABLE_GPU
#if LITERT_ENABLE_GPU
  static Expected<OpenClMemory> AllocFromGlBuffer(
      GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
      GlBuffer& gl_buffer);
#endif  // LITERT_ENABLE_GPU
  size_t size_bytes() const { return size_; }

 private:
#if LITERT_ENABLE_GPU
  GpuEnvironment* gpu_env_ = nullptr;
#else
  void* gpu_env_ = nullptr;
#endif  // LITERT_ENABLE_GPU
  const LiteRtRankedTensorType tensor_type_;
  LiteRtTensorBufferType buffer_type_;
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
#if LITERT_ENABLE_GPU
  tflite::gpu::cl::Buffer buffer_;
#else
  void* buffer_ = nullptr;
#endif  // LITERT_ENABLE_GPU
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
