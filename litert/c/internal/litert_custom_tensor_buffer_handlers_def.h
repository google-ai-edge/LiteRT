// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_CUSTOM_TENSOR_BUFFER_HANDLERS_DEF_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_CUSTOM_TENSOR_BUFFER_HANDLERS_DEF_H_

#include <stddef.h>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES 16

/// An internal struct that holds custom tensor buffer handlers and supported
/// buffer types.
/// If a dispatch plugin wants to support custom tensor buffers, it can return
/// a pointer to this struct via LiteRtDispatchQueryInterface when queried with
/// kLiteRtInterfaceCustomTensorBufferHandlers.
///
/// @note This concrete type is shared between the runtime and the Dispatch
///     plugin and Accelerator plugin, so it must be ABI stable.
typedef struct LiteRtCustomTensorBufferHandlersDef_V1 {
  LiteRtAbiHeader abi_header;

  CreateCustomTensorBuffer create_func;
  DestroyCustomTensorBuffer destroy_func;
  LockCustomTensorBuffer lock_func;
  UnlockCustomTensorBuffer unlock_func;
  ClearCustomTensorBuffer clear_func;
  ImportCustomTensorBuffer import_func;

  LiteRtEnvOptionTag device_tag;
  LiteRtEnvOptionTag queue_tag;

  size_t num_supported_buffer_types;
  LiteRtTensorBufferType supported_buffer_types
      [LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES];
} LiteRtCustomTensorBufferHandlersDef_V1;

typedef LiteRtCustomTensorBufferHandlersDef_V1
    LiteRtCustomTensorBufferHandlersDef;

LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1, abi_header) == 0,
    "LiteRtCustomTensorBufferHandlersDef_V1 abi_header offset mismatch");

#if (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8) || \
    defined(__LP64__) || defined(_WIN64)
LITERT_ABI_STATIC_ASSERT(
    sizeof(LiteRtCustomTensorBufferHandlersDef_V1) == 136,
    "LiteRtCustomTensorBufferHandlersDef_V1 size mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1, device_tag) == 56,
    "LiteRtCustomTensorBufferHandlersDef_V1 device_tag offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1, queue_tag) == 60,
    "LiteRtCustomTensorBufferHandlersDef_V1 queue_tag offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1,
             num_supported_buffer_types) == 64,
    "LiteRtCustomTensorBufferHandlersDef_V1 num_supported_buffer_types "
    "offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1,
             supported_buffer_types) == 72,
    "LiteRtCustomTensorBufferHandlersDef_V1 supported_buffer_types "
    "offset mismatch");
#elif (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4) || \
    defined(__ILP32__) || defined(_WIN32)
LITERT_ABI_STATIC_ASSERT(
    sizeof(LiteRtCustomTensorBufferHandlersDef_V1) == 108,
    "LiteRtCustomTensorBufferHandlersDef_V1 size mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1, device_tag) == 32,
    "LiteRtCustomTensorBufferHandlersDef_V1 device_tag offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1, queue_tag) == 36,
    "LiteRtCustomTensorBufferHandlersDef_V1 queue_tag offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1,
             num_supported_buffer_types) == 40,
    "LiteRtCustomTensorBufferHandlersDef_V1 num_supported_buffer_types "
    "offset mismatch");
LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtCustomTensorBufferHandlersDef_V1,
             supported_buffer_types) == 44,
    "LiteRtCustomTensorBufferHandlersDef_V1 supported_buffer_types "
    "offset mismatch");
#endif

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_CUSTOM_TENSOR_BUFFER_HANDLERS_DEF_H_
