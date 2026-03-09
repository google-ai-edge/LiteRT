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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_CONTEXT_H_

#include <cstddef>
#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "tflite/c/c_api_types.h"

typedef struct TfLiteTensor TfLiteTensor;

/// A function table that contains LiteRT C APIs needed for Accelerators.
///
/// @note This concrete type is part of the public API and is ABI stable.
typedef struct LiteRtRuntimeContext {
  LiteRtStatus (*create_tensor_buffer_requirements)(
      int num_supported_tensor_buffer_types,
      const LiteRtTensorBufferType* supported_tensor_buffer_types,
      size_t buffer_size, int num_strides, const uint32_t* strides,
      LiteRtTensorBufferRequirements* requirements);

  // third_party/odml/litert/litert/c/internal/litert_external_litert_buffer_context.h
  LiteRtStatus (*get_external_litert_buffer_context_tensor_buffer)(
      LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
      LiteRtTensorBuffer* tensor_buffer);
  LiteRtStatus (*external_litert_buffer_context_create_tensor_buffer)(
      LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
      LiteRtTensorBuffer* buffer);
  LiteRtStatus (*external_litert_buffer_context_register_tensor_buffer)(
      LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
      LiteRtTensorBuffer buffer);
  LiteRtStatus (*external_litert_buffer_context_register_buffer_requirements)(
      LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
      LiteRtTensorBufferRequirements buffer_requirements);
  LiteRtStatus (*external_litert_buffer_context_get_environment)(
      LiteRtExternalLiteRtBufferContext context, LiteRtEnvironment* env);
  LiteRtStatus (*external_litert_buffer_context_is_async_execution_mode)(
      LiteRtExternalLiteRtBufferContext context, bool* is_async_execution_mode);
  void (*external_litert_buffer_context_destroy)(
      LiteRtExternalLiteRtBufferContext context);

  LiteRtStatus (*get_opaque_options)(LiteRtOptions options,
                                     LiteRtOpaqueOptions* opaque_options);
  LiteRtStatus (*find_opaque_options_data)(LiteRtOpaqueOptions options,
                                           const char* payload_identifier,
                                           void** payload_data);
  LiteRtStatus (*wrap_delegate)(TfLiteOpaqueDelegate* delegate,
                                LiteRtDelegateWrapper* wrapper);
  LiteRtStatus (*unwrap_delegate)(LiteRtDelegateWrapper wrapper,
                                  TfLiteOpaqueDelegate** delegate);
} LiteRtRuntimeContext;

LiteRtRuntimeContext* LrtGetRuntimeContext();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_CONTEXT_H_
