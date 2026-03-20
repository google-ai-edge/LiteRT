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

#include "litert/c/internal/litert_runtime_context.h"

#include "litert/c/internal/litert_delegate_wrapper.h"
#include "litert/c/internal/litert_external_litert_buffer_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

LiteRtRuntimeContext* LrtGetRuntimeContext() {
  static LiteRtRuntimeContext context = {
      .create_tensor_buffer_requirements = LiteRtCreateTensorBufferRequirements,
      .get_external_litert_buffer_context_tensor_buffer =
          LiteRtGetExternalLiteRtBufferContextTensorBuffer,
      .external_litert_buffer_context_create_tensor_buffer =
          LiteRtExternalLiteRtBufferContextCreateBufferForTensor,
      .external_litert_buffer_context_register_tensor_buffer =
          LiteRtExternalLiteRtBufferContextRegisterTensorBuffer,
      .external_litert_buffer_context_unregister_tensor_buffer =
          LiteRtExternalLiteRtBufferContextUnregisterTensorBuffer,
      .external_litert_buffer_context_register_buffer_requirements =
          LiteRtExternalLiteRtBufferContextRegisterBufferRequirements,
      .external_litert_buffer_context_get_environment =
          LiteRtExternalLiteRtBufferContextGetEnvironment,
      .external_litert_buffer_context_is_async_execution_mode =
          LiteRtExternalLiteRtBufferContextIsAsyncExecutionMode,
      .external_litert_buffer_context_destroy =
          LiteRtDestroyExternalLiteRtBufferContext,
      .get_opaque_options = LiteRtGetOpaqueOptions,
      .find_opaque_options_data = LiteRtFindOpaqueOptionsData,
      .get_environment_options = LiteRtGetEnvironmentOptions,
      .get_environment_options_value = LiteRtGetEnvironmentOptionsValue,
      .environment_has_gpu_environment = LiteRtEnvironmentHasGpuEnvironment,
      .add_environment_options = LiteRtAddEnvironmentOptions,
      .gpu_environment_create = LiteRtGpuEnvironmentCreate,
      .wrap_delegate = LiteRtWrapDelegate,
      .unwrap_delegate = LiteRtUnwrapDelegate,
      .get_tensor_buffer_type = LiteRtGetTensorBufferType,
      .get_tensor_buffer_size = LiteRtGetTensorBufferSize,
      .get_tensor_buffer_offset = LiteRtGetTensorBufferOffset,
      .lock_tensor_buffer = LiteRtLockTensorBuffer,
      .unlock_tensor_buffer = LiteRtUnlockTensorBuffer,
      .get_tensor_buffer_host_memory = LiteRtGetTensorBufferHostMemory,
#if LITERT_HAS_OPENCL_SUPPORT
      .get_tensor_buffer_opencl_memory = LiteRtGetTensorBufferOpenClMemory,
#endif  // LITERT_HAS_OPENCL_SUPPORT
      .get_tensor_buffer_gl_buffer = LiteRtGetTensorBufferGlBuffer,
      .get_tensor_buffer_custom_tensor_buffer_handle =
          LiteRtGetTensorBufferCustomTensorBufferHandle,
      .has_tensor_buffer_event = LiteRtHasTensorBufferEvent,
      .get_tensor_buffer_event = LiteRtGetTensorBufferEvent,
      .set_tensor_buffer_event = LiteRtSetTensorBufferEvent,
      .create_managed_event = LiteRtCreateManagedEvent,
      .get_event_event_type = LiteRtGetEventEventType,
#if LITERT_HAS_OPENCL_SUPPORT
      .create_event_from_opencl_event = LiteRtCreateEventFromOpenClEvent,
      .get_event_opencl_event = LiteRtGetEventOpenClEvent,
#endif  // LITERT_HAS_OPENCL_SUPPORT
      .set_custom_event = LiteRtSetCustomEvent,
      .get_custom_event = LiteRtGetCustomEvent,
      .wait_event = LiteRtWaitEvent,
  };
  return &context;
}
