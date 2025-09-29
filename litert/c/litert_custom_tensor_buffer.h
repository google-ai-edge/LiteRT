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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_CUSTOM_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_CUSTOM_TENSOR_BUFFER_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Generic hardware memory handle type. This could be a specific hardware
// memory handle type such as cl_mem, WGPBuffer, etc.
// It's created by `CreateCustomTensorBuffer` and destroyed by
// `DestroyCustomTensorBuffer`.
typedef void* HwMemoryHandle;

// Custom hardware memory information data.
// Custom TensorBuffer handler can use this to keep additional information
// about the hardware memory such as CPU mapped memory pointer.
// The information should be kept in the child structure of `HwMemoryInfo`.
// It's created by `CreateCustomTensorBuffer` and destroyed by
// `DestroyCustomTensorBuffer`.
struct HwMemoryInfo {
  HwMemoryHandle memory_handle;
};

typedef struct HwMemoryInfo* HwMemoryInfoPtr;

// Custom TensorBuffer handler function to create a custom TensorBuffer.
typedef LiteRtStatus (*CreateCustomTensorBuffer)(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, size_t packed_bytes,
    HwMemoryInfoPtr* hw_memory_info);

// Custom TensorBuffer handler function to import an existing custom
// TensorBuffer.
// This function creates the HwMemoryInfo wrapper but does NOT take ownership
// or destroy the handle. The implementation should store an "owns_tensor =
// false" flag inside its HwMemoryInfo-derived struct.
typedef LiteRtStatus (*ImportCustomTensorBuffer)(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, HwMemoryHandle hw_buffer_handle,
    size_t bytes, size_t packed_bytes, HwMemoryInfoPtr* hw_memory_info);

// Custom TensorBuffer handler function to destroy a custom TensorBuffer.
typedef LiteRtStatus (*DestroyCustomTensorBuffer)(
    LiteRtEnvironment env, HwMemoryInfoPtr hw_memory_info);

// Custom TensorBuffer handler function to lock a custom TensorBuffer.
// `host_memory_ptr` is the CPU mapped memory pointer to the custom
// TensorBuffer.
typedef LiteRtStatus (*LockCustomTensorBuffer)(LiteRtEnvironment env,
                                               HwMemoryInfoPtr hw_memory_info,
                                               LiteRtTensorBufferLockMode mode,
                                               void** host_memory_ptr);

// Custom TensorBuffer handler function to unlock a custom TensorBuffer.
typedef LiteRtStatus (*UnlockCustomTensorBuffer)(
    LiteRtEnvironment env, HwMemoryInfoPtr hw_memory_info);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_CUSTOM_TENSOR_BUFFER_H_
