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

#ifndef ODML_LITERT_LITERT_RUNTIME_WEBGPU_BUFFER_H_
#define ODML_LITERT_LITERT_RUNTIME_WEBGPU_BUFFER_H_

#include <cstddef>
#include <cstdlib>
#include <utility>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/gpu_environment.h"

namespace litert::internal {

/**
 * The WebGpu buffer class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU WebGpu buffer.
 */
class WebGpuBuffer {
 public:
  WebGpuBuffer(WebGpuBuffer&& other)
      : gpu_env_(other.gpu_env_),
        tensor_type_(other.tensor_type_),
        buffer_type_(other.buffer_type_),
        data_(other.data_),
        buffer_(std::move(other.buffer_)),
        size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  WebGpuBuffer(GpuEnvironment* gpu_env,
               const LiteRtRankedTensorType& tensor_type,
               LiteRtTensorBufferType buffer_type, WGPUBuffer buffer,
               size_t size, LiteRtWebGpuBufferDeallocator deallocator)
      : gpu_env_(gpu_env),
        tensor_type_(tensor_type),
        buffer_type_(buffer_type),
        deallocator_(deallocator),
        size_(size) {
    buffer_ = buffer;
  }

  ~WebGpuBuffer() {
    if (deallocator_ != nullptr) {
      deallocator_(buffer_);
    }
    if (data_ != nullptr) {
      litert_aligned_free(data_);
    };
  }

  WGPUBuffer GetMemoryPtr() { return buffer_; }
  // Allocates a CPU memory and conducts a copy from the WebGPU buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock(LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the WebGPU buffer.
  template <typename T>
  Expected<void> Unlock();

  static Expected<WebGpuBuffer> Alloc(GpuEnvironment* gpu_env,
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
  WGPUBuffer buffer_;
  LiteRtWebGpuBufferDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_WEBGPU_BUFFER_H_
