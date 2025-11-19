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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_

#include "litert/c/internal/litert_external_litert_buffer_context.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "tflite/c/common.h"

namespace litert {

class ExternalLiteRtBufferContext
    : public internal::Handle<LiteRtExternalLiteRtBufferContext,
                              LiteRtDestroyExternalLiteRtBufferContext> {
 public:
  ExternalLiteRtBufferContext() = default;

  explicit ExternalLiteRtBufferContext(
      LiteRtExternalLiteRtBufferContext context, OwnHandle own_handle)
      : Handle(context, own_handle) {}

  // Returns a TensorBuffer object for the given tensor. The returned
  // TensorBuffer object is a duplicate (reference-counted) of the buffer
  // context's registered TensorBuffer.
  Expected<TensorBuffer> GetTensorBuffer(const TfLiteTensor* tensor) {
    LiteRtTensorBuffer tensor_buffer;
    // Note: Tensor buffer is duplicated by the C API.
    LITERT_RETURN_IF_ERROR(LiteRtGetExternalLiteRtBufferContextTensorBuffer(
        Get(), tensor, &tensor_buffer));
    return TensorBuffer::WrapCObject(tensor_buffer, OwnHandle::kYes);
  }

  // Creates a TensorBuffer object for the given tensor. The returned object is
  // owned by the caller.
  Expected<TensorBuffer> CreateBufferForTensor(const TfLiteTensor* tensor) {
    LiteRtTensorBuffer buffer;
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextCreateBufferForTensor(Get(), tensor,
                                                               &buffer));
    return TensorBuffer::WrapCObject(buffer, OwnHandle::kYes);
  }

  // Registers a TensorBuffer object for the given tensor. The buffer context
  // assumes ownership of the TensorBuffer object.
  Expected<void> RegisterTensorBuffer(const TfLiteTensor* tensor,
                                      TensorBuffer&& tensor_buffer) {
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextRegisterTensorBuffer(
            Get(), tensor, tensor_buffer.Release()));
    return {};
  }

  // Registers the buffer requirements for the given tensor. The buffer context
  // assumes ownership of the buffer requirements.
  Expected<void> RegisterBufferRequirements(
      const TfLiteTensor* tensor,
      TensorBufferRequirements&& buffer_requirements) {
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextRegisterBufferRequirements(
            Get(), tensor, buffer_requirements.Release()));
    return {};
  }

  // Returns the environment for the given buffer context. No ownership is
  // transferred.
  Expected<litert::Environment> GetEnvironment() {
    LiteRtEnvironment env;
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextGetEnvironment(Get(), &env));
    return Environment::WrapCObject(env, OwnHandle::kNo);
  }

  Expected<bool> IsAsyncExecutionMode() {
    bool is_async_execution_mode;
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextIsAsyncExecutionMode(
            Get(), &is_async_execution_mode));
    return is_async_execution_mode;
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
