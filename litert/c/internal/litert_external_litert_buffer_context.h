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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_

#include "litert/c/litert_common.h"
#include "tflite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns a TensorBuffer object for the given tensor. The returned
// TensorBuffer object is a duplicate (reference-counted) of the buffer
// context's registered TensorBuffer. The returned object is owned by the caller
// and must be destroyed.
LiteRtStatus LiteRtGetExternalLiteRtBufferContextTensorBuffer(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer* tensor_buffer);

// Creates a TensorBuffer object for the given tensor. The returned object is
// owned by the caller and must be destroyed.
LiteRtStatus LiteRtExternalLiteRtBufferContextCreateBufferForTensor(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer* buffer);

// Registers a TensorBuffer object for the given tensor. The buffer context
// assumes ownership of the TensorBuffer object.
LiteRtStatus LiteRtExternalLiteRtBufferContextRegisterTensorBuffer(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer buffer);

// Registers the buffer requirements for the given tensor. The buffer context
// assumes ownership of the buffer requirements.
LiteRtStatus LiteRtExternalLiteRtBufferContextRegisterBufferRequirements(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBufferRequirements buffer_requirements);

// Returns the environment for the given buffer context. No ownership is
// transferred.
LiteRtStatus LiteRtExternalLiteRtBufferContextGetEnvironment(
    LiteRtExternalLiteRtBufferContext context, LiteRtEnvironment* env);

// Sets whether the async execution mode is set.
LiteRtStatus LiteRtExternalLiteRtBufferContextIsAsyncExecutionMode(
    LiteRtExternalLiteRtBufferContext context, bool* is_async_execution_mode);

// Destroys the buffer context.
void LiteRtDestroyExternalLiteRtBufferContext(
    LiteRtExternalLiteRtBufferContext context);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
