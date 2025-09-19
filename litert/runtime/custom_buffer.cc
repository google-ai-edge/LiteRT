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

#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_registry.h"

namespace litert {
namespace internal {
namespace {

Expected<litert::internal::TensorBufferRegistry*> GetTensorBufferRegistry(
    LiteRtEnvironment env) {
  litert::internal::TensorBufferRegistry* registry = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferRegistry(env, reinterpret_cast<void**>(&registry)));
  return registry;
}

}  // namespace

CustomBuffer::~CustomBuffer() {
  LITERT_ASSIGN_OR_ABORT(auto registry, GetTensorBufferRegistry(env_));
  LITERT_ASSIGN_OR_ABORT(auto custom_buffer_handlers,
                         registry->GetCustomHandlers(buffer_type_));
  if (hw_memory_info_) {
    custom_buffer_handlers.destroy_func(env_, hw_memory_info_);
  }
}

Expected<void*> CustomBuffer::Lock(LiteRtTensorBufferLockMode mode) {
  LITERT_ASSIGN_OR_RETURN(auto registry, GetTensorBufferRegistry(env_));
  LITERT_ASSIGN_OR_RETURN(auto custom_buffer_handlers,
                          registry->GetCustomHandlers(buffer_type_));
  void* host_memory_ptr = nullptr;
  auto status = custom_buffer_handlers.lock_func(env_, hw_memory_info_, mode,
                                                 &host_memory_ptr);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to lock custom tensor buffer.");
  }
  return host_memory_ptr;
}

Expected<void> CustomBuffer::Unlock() {
  LITERT_ASSIGN_OR_RETURN(auto registry, GetTensorBufferRegistry(env_));
  LITERT_ASSIGN_OR_RETURN(auto custom_buffer_handlers,
                          registry->GetCustomHandlers(buffer_type_));
  auto status = custom_buffer_handlers.unlock_func(env_, hw_memory_info_);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to unlock custom tensor buffer.");
  }
  return {};
}

Expected<CustomBuffer> CustomBuffer::Alloc(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size,
    size_t packed_buffer_size) {
  LITERT_ASSIGN_OR_RETURN(auto registry, GetTensorBufferRegistry(env));
  LITERT_ASSIGN_OR_RETURN(auto custom_buffer_handlers,
                          registry->GetCustomHandlers(buffer_type));
  HwMemoryInfoPtr hw_memory_info = nullptr;
  auto status = custom_buffer_handlers.create_func(
      env, &tensor_type, buffer_type, buffer_size, packed_buffer_size,
      &hw_memory_info);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to create custom tensor buffer.");
  }
  return CustomBuffer(env, tensor_type, buffer_type, hw_memory_info);
}

}  // namespace internal
}  // namespace litert
