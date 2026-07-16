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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_METAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_METAL_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Custom TensorBuffer handlers for Metal memory. These handlers are registered
// via TensorBufferRegistry API and used to support Metal memory.
//
// The `device_id` and `queue_id` parameters are borrowed (+0 reference count).
// The implementation internally retains the Metal device and command queue to
// keep them alive as long as the returned HwMemoryInfoPtr exists. The caller
// retains ownership of their references and is responsible for managing their
// lifetimes independent of the returned HwMemoryInfoPtr.
LiteRtStatus LiteRtCreateMetalMemory(LiteRtGpuDeviceId device_id,
                                     LiteRtGpuQueueId queue_id,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, size_t packed_bytes,
                                     HwMemoryInfoPtr* metal_memory_info);
LiteRtStatus LiteRtDestroyMetalMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtLockMetalMemory(HwMemoryInfoPtr hw_memory_info,
                                   LiteRtTensorBufferLockMode mode,
                                   void** host_memory_ptr);
LiteRtStatus LiteRtUnlockMetalMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtClearMetalMemory(HwMemoryInfoPtr hw_memory_info);

// Import an existing Metal memory buffer. This function creates the
// HwMemoryInfo wrapper but does NOT take ownership or destroy the handle. The
// implementation should store an "owns_tensor = false" flag inside its
// HwMemoryInfo-derived struct.
//
// The `device_id` and `queue_id` parameters are borrowed (+0 reference count).
// The implementation internally retains the Metal device and command queue to
// keep them alive as long as the returned HwMemoryInfoPtr exists. The caller
// retains ownership of their references and is responsible for managing their
// lifetimes independent of the returned HwMemoryInfoPtr.
//
// Returns an error if the provided `hw_buffer_handle` is not a valid Metal
// memory buffer.
LiteRtStatus LiteRtImportMetalMemory(LiteRtGpuDeviceId device_id,
                                     LiteRtGpuQueueId queue_id,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     HwMemoryHandle hw_buffer_handle,
                                     size_t bytes, size_t packed_bytes,
                                     HwMemoryInfoPtr* metal_memory_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_METAL_H_
