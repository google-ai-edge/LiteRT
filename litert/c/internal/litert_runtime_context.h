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

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_opencl_types.h"
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
  LiteRtStatus (*external_litert_buffer_context_unregister_tensor_buffer)(
      LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor);
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
  LiteRtStatus (*get_environment_options)(LiteRtEnvironment env,
                                          LiteRtEnvironmentOptions* options);
  LiteRtStatus (*get_environment_options_value)(
      LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
      LiteRtAny* option_value);
  void (*environment_has_gpu_environment)(LiteRtEnvironment environment,
                                          bool* has_gpu_environment);
  LiteRtStatus (*add_environment_options)(LiteRtEnvironment environment,
                                          int num_options,
                                          const LiteRtEnvOption* options,
                                          bool overwrite);
  LiteRtStatus (*gpu_environment_create)(LiteRtEnvironment environment,
                                         int num_options,
                                         const LiteRtEnvOption* options);

  LiteRtStatus (*wrap_delegate)(TfLiteOpaqueDelegate* delegate,
                                LiteRtDelegateWrapper* wrapper);
  LiteRtStatus (*unwrap_delegate)(LiteRtDelegateWrapper wrapper,
                                  TfLiteOpaqueDelegate** delegate);

  // third_party/odml/litert/litert/c/litert_tensor_buffer.h
  LiteRtStatus (*get_tensor_buffer_type)(LiteRtTensorBuffer tensor_buffer,
                                         LiteRtTensorBufferType* buffer_type);
  LiteRtStatus (*get_tensor_buffer_size)(LiteRtTensorBuffer tensor_buffer,
                                         size_t* buffer_size);
  LiteRtStatus (*get_tensor_buffer_offset)(LiteRtTensorBuffer tensor_buffer,
                                           size_t* offset);
  LiteRtStatus (*lock_tensor_buffer)(LiteRtTensorBuffer tensor_buffer,
                                     void** host_mem_addr,
                                     LiteRtTensorBufferLockMode mode);
  LiteRtStatus (*unlock_tensor_buffer)(LiteRtTensorBuffer tensor_buffer);
  LiteRtStatus (*get_tensor_buffer_host_memory)(
      LiteRtTensorBuffer tensor_buffer, void** host_memory_addr);
#if LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus (*get_tensor_buffer_opencl_memory)(
      LiteRtTensorBuffer tensor_buffer, LiteRtClMem* cl_mem_addr);
#endif  // LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus (*get_tensor_buffer_gl_buffer)(LiteRtTensorBuffer tensor_buffer,
                                              LiteRtGLenum* target,
                                              LiteRtGLuint* id,
                                              size_t* size_bytes,
                                              size_t* offset);
  LiteRtStatus (*get_tensor_buffer_custom_tensor_buffer_handle)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
  LiteRtStatus (*has_tensor_buffer_event)(LiteRtTensorBuffer tensor_buffer,
                                          bool* has_event);
  LiteRtStatus (*get_tensor_buffer_event)(LiteRtTensorBuffer tensor_buffer,
                                          LiteRtEvent* event);
  LiteRtStatus (*set_tensor_buffer_event)(LiteRtTensorBuffer tensor_buffer,
                                          LiteRtEvent event);

  // third_party/odml/litert/litert/c/litert_event.h
  LiteRtStatus (*create_managed_event)(LiteRtEnvironment env,
                                       LiteRtEventType event_type,
                                       LiteRtEvent* event);
  LiteRtStatus (*get_event_event_type)(LiteRtEvent event,
                                       LiteRtEventType* type);
#if LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus (*create_event_from_opencl_event)(LiteRtEnvironment env,
                                                 LiteRtClEvent cl_event,
                                                 LiteRtEvent* event);
  LiteRtStatus (*get_event_opencl_event)(LiteRtEvent event,
                                         LiteRtClEvent* cl_event);
#endif  // LITERT_HAS_OPENCL_SUPPORT
  LiteRtStatus (*set_custom_event)(LiteRtEvent event,
                                   LiteRtCustomEvent custom_event);
  LiteRtStatus (*get_custom_event)(LiteRtEvent event,
                                   LiteRtCustomEvent* custom_event);
  LiteRtStatus (*wait_event)(LiteRtEvent event, int64_t timeout_in_ms);
} LiteRtRuntimeContext;

LiteRtRuntimeContext* LrtGetRuntimeContext();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_CONTEXT_H_
