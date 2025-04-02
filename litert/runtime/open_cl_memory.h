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
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/ahwb_buffer.h"
#include <CL/cl.h>
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"  // from @org_tensorflow

namespace litert::internal {

/**
 * The OpenCL memory class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU OpenCL buffer.
 */
class OpenClMemory {
 public:
  OpenClMemory(OpenClMemory&& other) {
    buffer_type_ = other.buffer_type_;
    data_ = other.data_;
    buffer_ = std::move(other.buffer_);
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    ahwb_ = other.ahwb_;
    other.ahwb_ = nullptr;
  }

  explicit OpenClMemory(LiteRtTensorBufferType buffer_type,
                        tflite::gpu::cl::Buffer buffer,
                        AHardwareBuffer* ahwb = nullptr)
      : buffer_type_(buffer_type),
        buffer_(std::move(buffer)),
        size_(buffer_.GetMemorySizeInBytes()),
        ahwb_(ahwb) {}

  OpenClMemory(LiteRtTensorBufferType buffer_type, cl_mem buffer, size_t size,
               LiteRtOpenClDeallocator deallocator)
      : buffer_type_(buffer_type), deallocator_(deallocator), size_(size) {
    if (deallocator_ != nullptr) {
      buffer_ = tflite::gpu::cl::CreateBufferShared(buffer);
    } else {  // The buffer will be deallocated automatically.
      buffer_ = tflite::gpu::cl::Buffer(buffer, size);
    }
  }

  ~OpenClMemory() {
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
    if (data_ != nullptr) {
      free(data_);
    };
  }

  cl_mem GetMemoryPtr() { return buffer_.GetMemoryPtr(); }
  // Allocates a CPU memory and conducts a copy from the OpenCL buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock();

  // Writes the data from the CPU memory to the OpenCL buffer.
  template <typename T>
  Expected<void> Unlock();

  static bool IsSupported();
  static Expected<OpenClMemory> Alloc(LiteRtTensorBufferType buffer_type,
                                      size_t bytes_size);
  static Expected<OpenClMemory> AllocFromAhwbBuffer(AhwbBuffer& ahwb_buffer);
  size_t size_bytes() const { return size_; }

 private:
  LiteRtTensorBufferType buffer_type_;
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
  tflite::gpu::cl::Buffer buffer_;
  LiteRtOpenClDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
  AHardwareBuffer* ahwb_ = nullptr;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_OPEN_CL_BUFFER_H_
