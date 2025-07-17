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

#ifndef ODML_LITERT_LITERT_RUNTIME_CUSTOM_BUFFER_H_
#define ODML_LITERT_LITERT_RUNTIME_CUSTOM_BUFFER_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

/**
 * The custom buffer class that provides custom H/W memory allocation and
 * two-way sync between the CPU memory and the custom H/W buffer.
 */
class CustomBuffer {
 public:
  CustomBuffer(CustomBuffer&& other)
      : buffer_type_(other.buffer_type_),
        hw_buffer_handle_(other.hw_buffer_handle_),
        hw_buffer_info_(other.hw_buffer_info_) {
    other.hw_buffer_handle_ = nullptr;
    other.hw_buffer_info_ = nullptr;
  }

  CustomBuffer(const LiteRtRankedTensorType& tensor_type,
               LiteRtTensorBufferType buffer_type, void* hw_buffer_handle,
               void* hw_buffer_info)
      : buffer_type_(buffer_type),
        hw_buffer_handle_(hw_buffer_handle),
        hw_buffer_info_(hw_buffer_info) {}

  ~CustomBuffer();

  void* GetMemoryHandle() { return hw_buffer_handle_; }
  // Allocates a CPU memory and conducts a copy from the Custom buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock(LiteRtTensorBuffer tensor_buffer,
                    LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the Custom buffer.
  template <typename T>
  Expected<void> Unlock(LiteRtTensorBuffer tensor_buffer);

 private:
  LiteRtTensorBufferType buffer_type_;
  void* hw_buffer_handle_;
  void* hw_buffer_info_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_CUSTOM_BUFFER_H_
