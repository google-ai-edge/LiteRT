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

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

/**
 * The custom buffer class that provides custom H/W memory allocation and
 * two-way sync between the CPU memory and the custom H/W buffer.
 */
class CustomBuffer {
 public:
  // Move constructor to support `memory_backed_buffers_` of
  // LiteRtTensorBufferT. The `hw_memory_info_` of the other will be
  // reset to nullptr.
  CustomBuffer(CustomBuffer&& other)
      : env_(other.env_),
        buffer_type_(other.buffer_type_),
        hw_memory_info_(other.hw_memory_info_) {
    other.hw_memory_info_ = nullptr;
  }

  // Destructor to destroy the underlying custom buffer with
  // `DestroyCustomTensorBuffer`.
  ~CustomBuffer();

  HwMemoryHandle hw_buffer_handle() { return hw_memory_info_->memory_handle; }
  // Allocates a CPU memory and conducts a copy from the Custom buffer to the
  // CPU memory.
  Expected<void*> Lock(LiteRtTensorBufferLockMode mode);

  // Writes the data from the CPU memory to the Custom buffer.
  Expected<void> Unlock();

  // Creates a custom buffer.
  static Expected<CustomBuffer> Alloc(LiteRtEnvironment env,
                                      const LiteRtRankedTensorType& tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t buffer_size,
                                      size_t packed_buffer_size);

  // Wraps an existing custom buffer. The function will not take ownership of
  // the custom buffer.
  static Expected<CustomBuffer> Wrap(LiteRtEnvironment env,
                                     const LiteRtRankedTensorType& tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     HwMemoryHandle hw_buffer_handle,
                                     size_t buffer_size,
                                     size_t packed_buffer_size);

 private:
  // Private constructor to create a custom buffer.
  CustomBuffer(LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
               LiteRtTensorBufferType buffer_type, HwMemoryInfo* hw_memory_info)
      : env_(env), buffer_type_(buffer_type), hw_memory_info_(hw_memory_info) {}

  LiteRtEnvironment env_;
  const LiteRtTensorBufferType buffer_type_;
  HwMemoryInfoPtr hw_memory_info_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_CUSTOM_BUFFER_H_
