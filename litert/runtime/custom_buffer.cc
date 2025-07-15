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

#include "litert/runtime/custom_buffer.h"

#include <stdlib.h>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_registry.h"

namespace litert {
namespace internal {

CustomBuffer::~CustomBuffer() {
  auto& registry = litert::internal::TensorBufferRegistry::GetInstance();
  LITERT_ASSIGN_OR_ABORT(auto custom_buffer_handlers,
                         registry.GetCustomHandlers(buffer_type_));
  if (hw_buffer_info_) {
    custom_buffer_handlers.destroy_func(hw_buffer_info_);
  }
}

template Expected<float*> CustomBuffer::Lock<float>(
    LiteRtTensorBuffer tensor_buffer, LiteRtTensorBufferLockMode mode);
template Expected<char*> CustomBuffer::Lock<char>(
    LiteRtTensorBuffer tensor_buffer, LiteRtTensorBufferLockMode mode);
template Expected<void> CustomBuffer::Unlock<float>(
    LiteRtTensorBuffer tensor_buffer);
template Expected<void> CustomBuffer::Unlock<char>(
    LiteRtTensorBuffer tensor_buffer);

template <typename T>
Expected<T*> CustomBuffer::Lock(LiteRtTensorBuffer tensor_buffer,
                                LiteRtTensorBufferLockMode mode) {
  auto& registry = litert::internal::TensorBufferRegistry::GetInstance();
  LITERT_ASSIGN_OR_RETURN(auto custom_buffer_handlers,
                          registry.GetCustomHandlers(buffer_type_));
  void* host_memory_ptr = nullptr;
  auto status = custom_buffer_handlers.lock_func(
      tensor_buffer, mode, hw_buffer_info_, &host_memory_ptr);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to lock custom tensor buffer.");
  }
  return reinterpret_cast<T*>(host_memory_ptr);
}

template <typename T>
Expected<void> CustomBuffer::Unlock(LiteRtTensorBuffer tensor_buffer) {
  auto& registry = litert::internal::TensorBufferRegistry::GetInstance();
  LITERT_ASSIGN_OR_RETURN(auto custom_buffer_handlers,
                          registry.GetCustomHandlers(buffer_type_));
  auto status =
      custom_buffer_handlers.unlock_func(tensor_buffer, hw_buffer_info_);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to unlock custom tensor buffer.");
  }
  return {};
}

}  // namespace internal
}  // namespace litert
