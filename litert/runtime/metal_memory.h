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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_MEMORY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_MEMORY_H_

#import <Metal/Metal.h>

#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/litert_gpu_util.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "tflite/delegates/gpu/metal/buffer.h"
#include "tflite/delegates/gpu/metal/metal_device.h"
namespace litert::internal {

using ::tflite::gpu::metal::MetalDevice;
using MetalBuffer = ::tflite::gpu::metal::Buffer;

class MetalMemory {
 public:
  MetalMemory(MetalMemory&& other)
      : gpu_env_(other.gpu_env_),
        tensor_type_(other.tensor_type_),
        buffer_type_(other.buffer_type_),
        data_(other.data_),
        buffer_(std::move(other.buffer_)),
        size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  explicit MetalMemory(GpuEnvironment* gpu_env,
                       const LiteRtRankedTensorType& tensor_type,
                       LiteRtTensorBufferType buffer_type,
                       tflite::gpu::metal::Buffer buffer)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        buffer_(std::move(buffer)) {}

  MetalMemory(GpuEnvironment* gpu_env,
              const LiteRtRankedTensorType& tensor_type,
              LiteRtTensorBufferType buffer_type, void* buffer, size_t size,
              LiteRtMetalDeallocator deallocator)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        deallocator_(deallocator),
        size_(size) {
    // CreateBufferShared creates a buffer that is not owned by
    // tflite::gpu::metal::Buffer (MetalMemory determines ownership). Null
    // deallocator means that the buffer is not owned by MetalMemory.
    buffer_ =
        tflite::gpu::metal::CreateBufferShared((__bridge id<MTLBuffer>)buffer);
  }

  ~MetalMemory();

  void* GetMemoryPtr() { return (__bridge void*)(buffer_.GetMemoryPtr()); }
  // Allocates a CPU memory and conducts a copy from the Metal buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock(LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the Metal buffer.
  template <typename T>
  Expected<void> Unlock();

  // Returns true if Metal is supported.
  // Warning: This is only for TEST.
  static bool IsSupported();
  static Expected<MetalMemory> Alloc(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType& tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes_size);

  size_t size_bytes() const { return size_; }

 private:
  GpuEnvironment* gpu_env_ = nullptr;
  const LiteRtRankedTensorType tensor_type_;
  LiteRtTensorBufferType buffer_type_;
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  MetalBuffer buffer_;
  LiteRtMetalDeallocator deallocator_ = nullptr;
  // The command queue used in Metal delegate.
  // The size of the buffer in bytes.
  size_t size_ = 0;
  // The size of the CPU memory buffer in bytes. It's doubled for fp16 buffers.
  size_t cpu_buffer_size_ = 0;

  LockState lock_state_ = LockState::kUnlocked;
};

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_MEMORY_H_
