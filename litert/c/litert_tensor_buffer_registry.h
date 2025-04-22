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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REGISTRY_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Custom tensor buffer creation function for the given buffer type and size.
// Returns the created hardware memory handle in `hw_memory`.
typedef LiteRtStatus (*CustomTensorBufferCreate)(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void** hw_memory);

// Custom tensor buffer upload function.
typedef LiteRtStatus (*CustomTensorBufferUpload)(
    void* hw_memory, size_t bytes, const void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type);

// Custom tensor buffer download function.
typedef LiteRtStatus (*CustomTensorBufferDownload)(
    void* hw_memory, size_t bytes, void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type);

// Registers custom tensor buffer accessors for the given buffer type.
LiteRtStatus LiteRtRegisterTensorBufferAccessors(
    LiteRtTensorBufferType buffer_type, CustomTensorBufferCreate create_func,
    CustomTensorBufferUpload upload_func,
    CustomTensorBufferDownload download_func);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REGISTRY_H_
