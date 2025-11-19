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

#include "litert/c/internal/litert_external_litert_buffer_context.h"

#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "tflite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtGetExternalLiteRtBufferContextTensorBuffer(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer* tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto res, context->GetTensorBuffer(tensor));
  *tensor_buffer = res.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtExternalLiteRtBufferContextCreateBufferForTensor(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer* buffer) {
  LITERT_ASSIGN_OR_RETURN(auto res, context->CreateBufferForTensor(tensor));
  *buffer = res.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtExternalLiteRtBufferContextRegisterTensorBuffer(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBuffer buffer) {
  LITERT_RETURN_IF_ERROR(
      context->RegisterTensorBuffer(tensor, LiteRtTensorBufferPtr(buffer)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtExternalLiteRtBufferContextRegisterBufferRequirements(
    LiteRtExternalLiteRtBufferContext context, const TfLiteTensor* tensor,
    LiteRtTensorBufferRequirements buffer_requirements) {
  LITERT_RETURN_IF_ERROR(context->RegisterBufferRequirements(
      tensor, LiteRtTensorBufferRequirementsPtr(buffer_requirements)));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtExternalLiteRtBufferContextGetEnvironment(
    LiteRtExternalLiteRtBufferContext context, LiteRtEnvironment* env) {
  LiteRtEnvironment res = context->GetEnvironment();
  *env = res;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtExternalLiteRtBufferContextIsAsyncExecutionMode(
    LiteRtExternalLiteRtBufferContext context, bool* is_async_execution_mode) {
  *is_async_execution_mode = context->IsAsyncExecutionMode();
  return kLiteRtStatusOk;
}

void LiteRtDestroyExternalLiteRtBufferContext(
    LiteRtExternalLiteRtBufferContext context) {
  delete context;
}

#ifdef __cplusplus
}
#endif  // __cplusplus
