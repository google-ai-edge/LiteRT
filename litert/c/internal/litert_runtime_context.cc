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
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

LiteRtRuntimeContext* LiteRtGetRuntimeContext() {
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
      .wrap_delegate = LiteRtWrapDelegate,
      .unwrap_delegate = LiteRtUnwrapDelegate,
  };
  return &context;
}
