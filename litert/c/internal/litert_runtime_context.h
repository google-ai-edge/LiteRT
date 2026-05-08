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

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opencl_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "tflite/c/c_api_types.h"

typedef struct TfLiteTensor TfLiteTensor;
typedef struct AHardwareBuffer AHardwareBuffer;

/// A function table that contains LiteRT C APIs needed for Accelerators.
///
/// @note This concrete type is shared with LiteRT runtime and Accelerators and
/// Dispatch APIs. So it must be ABI stable.
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
  void (*destroy_options)(LiteRtOptions options);
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
  LiteRtStatus (*create_tensor_buffer_from_host_memory)(
      const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
      size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
      LiteRtTensorBuffer* buffer);
  LiteRtStatus (*create_managed_tensor_buffer)(
      LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
      const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
      LiteRtTensorBuffer* buffer);
  void (*destroy_tensor_buffer)(LiteRtTensorBuffer buffer);
  LiteRtStatus (*get_tensor_buffer_type)(LiteRtTensorBuffer tensor_buffer,
                                         LiteRtTensorBufferType* buffer_type);
  LiteRtStatus (*get_tensor_buffer_tensor_type)(
      LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type);
  LiteRtStatus (*get_tensor_buffer_size)(LiteRtTensorBuffer tensor_buffer,
                                         size_t* buffer_size);
  LiteRtStatus (*get_tensor_buffer_packed_size)(
      LiteRtTensorBuffer tensor_buffer, size_t* size);
  LiteRtStatus (*get_tensor_buffer_offset)(LiteRtTensorBuffer tensor_buffer,
                                           size_t* offset);
  LiteRtStatus (*lock_tensor_buffer)(LiteRtTensorBuffer tensor_buffer,
                                     void** host_mem_addr,
                                     LiteRtTensorBufferLockMode mode);
  LiteRtStatus (*unlock_tensor_buffer)(LiteRtTensorBuffer tensor_buffer);
  LiteRtStatus (*get_tensor_buffer_host_memory)(
      LiteRtTensorBuffer tensor_buffer, void** host_memory_addr);
  LiteRtStatus (*get_tensor_buffer_opencl_memory)(
      LiteRtTensorBuffer tensor_buffer, LiteRtClMem* cl_mem_addr);
  LiteRtStatus (*get_tensor_buffer_gl_buffer)(LiteRtTensorBuffer tensor_buffer,
                                              LiteRtGLenum* target,
                                              LiteRtGLuint* id,
                                              size_t* size_bytes,
                                              size_t* offset);
  LiteRtStatus (*get_tensor_buffer_custom_tensor_buffer_handle)(
      LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
  LiteRtStatus (*get_tensor_buffer_ahwb)(LiteRtTensorBuffer tensor_buffer,
                                         AHardwareBuffer** ahwb);
  LiteRtStatus (*get_tensor_buffer_dma_buf_buffer)(
      LiteRtTensorBuffer tensor_buffer, void** dmabuf_buffer_addr,
      int* dmabuf_buffer_fd);
  LiteRtStatus (*get_tensor_buffer_fast_rpc_buffer)(
      LiteRtTensorBuffer tensor_buffer, void** host_buffer_addr, int* fd);
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
  LiteRtStatus (*create_event_from_sync_fence_fd)(LiteRtEnvironment env,
                                                  int sync_fence_fd,
                                                  bool owns_fd,
                                                  LiteRtEvent* event);
  LiteRtStatus (*get_event_event_type)(LiteRtEvent event,
                                       LiteRtEventType* type);
  LiteRtStatus (*get_event_sync_fence_fd)(LiteRtEvent event,
                                          int* sync_fence_fd);
  LiteRtStatus (*create_event_from_opencl_event)(LiteRtEnvironment env,
                                                 LiteRtClEvent cl_event,
                                                 LiteRtEvent* event);
  LiteRtStatus (*get_event_opencl_event)(LiteRtEvent event,
                                         LiteRtClEvent* cl_event);
  LiteRtStatus (*set_custom_event)(LiteRtEvent event,
                                   LiteRtCustomEvent custom_event);
  LiteRtStatus (*get_custom_event)(LiteRtEvent event,
                                   LiteRtCustomEvent* custom_event);
  LiteRtStatus (*wait_event)(LiteRtEvent event, int64_t timeout_in_ms);
} LiteRtRuntimeContext;

// ABI compatibility check for LiteRtRuntimeContext.
//
// Note: Please get review from the LiteRT ABI compatibility team when you make
// changes to this struct.
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtRuntimeContext) == 384,
              "LiteRtRuntimeContext size mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, create_tensor_buffer_requirements) == 0,
    "LiteRtRuntimeContext create_tensor_buffer_requirements offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             get_external_litert_buffer_context_tensor_buffer) == 8,
    "LiteRtRuntimeContext get_external_litert_buffer_context_tensor_buffer "
    "offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             external_litert_buffer_context_create_tensor_buffer) == 16,
    "LiteRtRuntimeContext external_litert_buffer_context_create_tensor_buffer "
    "offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             external_litert_buffer_context_register_tensor_buffer) == 24,
    "LiteRtRuntimeContext "
    "external_litert_buffer_context_register_tensor_buffer offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             external_litert_buffer_context_unregister_tensor_buffer) == 32,
    "LiteRtRuntimeContext "
    "external_litert_buffer_context_unregister_tensor_buffer offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             external_litert_buffer_context_register_buffer_requirements) == 40,
    "LiteRtRuntimeContext "
    "external_litert_buffer_context_register_buffer_requirements offset "
    "mismatch");
static_assert(offsetof(LiteRtRuntimeContext,
                       external_litert_buffer_context_get_environment) == 48,
              "LiteRtRuntimeContext "
              "external_litert_buffer_context_get_environment offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext,
             external_litert_buffer_context_is_async_execution_mode) == 56,
    "LiteRtRuntimeContext "
    "external_litert_buffer_context_is_async_execution_mode offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext,
                       external_litert_buffer_context_destroy) == 64,
              "LiteRtRuntimeContext external_litert_buffer_context_destroy "
              "offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_opaque_options) == 72,
              "LiteRtRuntimeContext get_opaque_options offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, find_opaque_options_data) == 80,
              "LiteRtRuntimeContext find_opaque_options_data offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, destroy_options) == 88,
              "LiteRtRuntimeContext destroy_options offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_environment_options) == 96,
              "LiteRtRuntimeContext get_environment_options offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_environment_options_value) == 104,
    "LiteRtRuntimeContext get_environment_options_value offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, environment_has_gpu_environment) == 112,
    "LiteRtRuntimeContext environment_has_gpu_environment offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, add_environment_options) == 120,
              "LiteRtRuntimeContext add_environment_options offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, gpu_environment_create) == 128,
              "LiteRtRuntimeContext gpu_environment_create offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, wrap_delegate) == 136,
              "LiteRtRuntimeContext wrap_delegate offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, unwrap_delegate) == 144,
              "LiteRtRuntimeContext unwrap_delegate offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext,
                       create_tensor_buffer_from_host_memory) == 152,
              "LiteRtRuntimeContext create_tensor_buffer_from_host_memory "
              "offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, create_managed_tensor_buffer) == 160,
    "LiteRtRuntimeContext create_managed_tensor_buffer offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, destroy_tensor_buffer) == 168,
              "LiteRtRuntimeContext destroy_tensor_buffer offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_tensor_buffer_type) == 176,
              "LiteRtRuntimeContext get_tensor_buffer_type offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_tensor_type) == 184,
    "LiteRtRuntimeContext get_tensor_buffer_tensor_type offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_tensor_buffer_size) == 192,
              "LiteRtRuntimeContext get_tensor_buffer_size offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_packed_size) == 200,
    "LiteRtRuntimeContext get_tensor_buffer_packed_size offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_tensor_buffer_offset) == 208,
              "LiteRtRuntimeContext get_tensor_buffer_offset offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, lock_tensor_buffer) == 216,
              "LiteRtRuntimeContext lock_tensor_buffer offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, unlock_tensor_buffer) == 224,
              "LiteRtRuntimeContext unlock_tensor_buffer offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_host_memory) == 232,
    "LiteRtRuntimeContext get_tensor_buffer_host_memory offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_opencl_memory) == 240,
    "LiteRtRuntimeContext get_tensor_buffer_opencl_memory offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_gl_buffer) == 248,
    "LiteRtRuntimeContext get_tensor_buffer_gl_buffer offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext,
                       get_tensor_buffer_custom_tensor_buffer_handle) == 256,
              "LiteRtRuntimeContext "
              "get_tensor_buffer_custom_tensor_buffer_handle offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_tensor_buffer_ahwb) == 264,
              "LiteRtRuntimeContext get_tensor_buffer_ahwb offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_dma_buf_buffer) == 272,
    "LiteRtRuntimeContext get_tensor_buffer_dma_buf_buffer offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, get_tensor_buffer_fast_rpc_buffer) == 280,
    "LiteRtRuntimeContext get_tensor_buffer_fast_rpc_buffer offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, has_tensor_buffer_event) == 288,
              "LiteRtRuntimeContext has_tensor_buffer_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_tensor_buffer_event) == 296,
              "LiteRtRuntimeContext get_tensor_buffer_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, set_tensor_buffer_event) == 304,
              "LiteRtRuntimeContext set_tensor_buffer_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, create_managed_event) == 312,
              "LiteRtRuntimeContext create_managed_event offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, create_event_from_sync_fence_fd) == 320,
    "LiteRtRuntimeContext create_event_from_sync_fence_fd offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_event_event_type) == 328,
              "LiteRtRuntimeContext get_event_event_type offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_event_sync_fence_fd) == 336,
              "LiteRtRuntimeContext get_event_sync_fence_fd offset mismatch");
static_assert(
    offsetof(LiteRtRuntimeContext, create_event_from_opencl_event) == 344,
    "LiteRtRuntimeContext create_event_from_opencl_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_event_opencl_event) == 352,
              "LiteRtRuntimeContext get_event_opencl_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, set_custom_event) == 360,
              "LiteRtRuntimeContext set_custom_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, get_custom_event) == 368,
              "LiteRtRuntimeContext get_custom_event offset mismatch");
static_assert(offsetof(LiteRtRuntimeContext, wait_event) == 376,
              "LiteRtRuntimeContext wait_event offset mismatch");
#endif  // __cplusplus

LiteRtRuntimeContext* LrtGetRuntimeContext();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_RUNTIME_CONTEXT_H_
