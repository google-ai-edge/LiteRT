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
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers custom tensor buffer handlers for the given buffer type.
LiteRtStatus LiteRtRegisterTensorBufferHandlers(
    LiteRtTensorBufferType buffer_type, CreateCustomTensorBuffer create_func,
    DestroyCustomTensorBuffer destroy_func, LockCustomTensorBuffer lock_func,
    UnlockCustomTensorBuffer unlock_func);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REGISTRY_H_
