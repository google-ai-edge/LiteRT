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

#include <cstddef>

#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES 16

/// A internal struct that holds custom tensor buffer handlers and supported
/// buffer types.
/// If a dispatch plugin wants to support custom tensor buffers, it can
/// provide a non-null pointer to this struct in the LiteRtDispatchApi.
///
/// @note This concrete type is shared between the runtime and the Dispatch
///     plugin and Accelerator plugin, so it must be ABI stable.
typedef struct LiteRtCustomTensorBufferHandlersDef {
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
} LiteRtCustomTensorBufferHandlersDef;

#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtCustomTensorBufferHandlersDef) == 128,
              "LiteRtCustomTensorBufferHandlersDef size mismatch");
static_assert(offsetof(LiteRtCustomTensorBufferHandlersDef, device_tag) == 48,
              "LiteRtCustomTensorBufferHandlersDef device_tag offset mismatch");
static_assert(offsetof(LiteRtCustomTensorBufferHandlersDef, queue_tag) == 52,
              "LiteRtCustomTensorBufferHandlersDef queue_tag offset mismatch");
static_assert(offsetof(LiteRtCustomTensorBufferHandlersDef,
                       num_supported_buffer_types) == 56,
              "LiteRtCustomTensorBufferHandlersDef num_supported_buffer_types "
              "offset mismatch");
static_assert(offsetof(LiteRtCustomTensorBufferHandlersDef,
                       supported_buffer_types) == 64,
              "LiteRtCustomTensorBufferHandlersDef supported_buffer_types "
              "offset mismatch");
#endif  // __cplusplus

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_CUSTOM_TENSOR_BUFFER_HANDLERS_DEF_H_
