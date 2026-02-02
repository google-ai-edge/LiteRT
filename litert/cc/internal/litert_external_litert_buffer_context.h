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

/// @file
/// @brief Defines the C++ wrapper for the external LiteRT buffer context.

namespace litert {

/// @brief Manages the external buffer context for LiteRT, providing an
/// interface to interact with tensor buffers and their requirements.
class ExternalLiteRtBufferContext
    : public internal::Handle<LiteRtExternalLiteRtBufferContext,
                              LiteRtDestroyExternalLiteRtBufferContext> {
 public:
  ExternalLiteRtBufferContext() = default;

  explicit ExternalLiteRtBufferContext(
      LiteRtExternalLiteRtBufferContext context, OwnHandle own_handle)
      : Handle(context, own_handle) {}

  /// @brief Returns a `TensorBuffer` object for the given tensor.
  ///
  /// The returned `TensorBuffer` object is a duplicate (reference-counted) of
  /// the buffer context's registered `TensorBuffer`.
  /// @param tensor The tensor for which to retrieve the buffer.
  /// @return An `Expected` object containing the `TensorBuffer`, or an error.
  Expected<TensorBuffer> GetTensorBuffer(const TfLiteTensor* tensor) {
    LiteRtTensorBuffer tensor_buffer;
    // Note: The tensor buffer is duplicated by the C API.
    LITERT_RETURN_IF_ERROR(LiteRtGetExternalLiteRtBufferContextTensorBuffer(
        Get(), tensor, &tensor_buffer));
    return TensorBuffer::WrapCObject(tensor_buffer, OwnHandle::kYes);
  }

  /// @brief Creates a `TensorBuffer` object for the given tensor.
  ///
  /// The returned object is owned by the caller.
  /// @param tensor The tensor for which to create the buffer.
  /// @return An `Expected` object containing the new `TensorBuffer`, or an
  /// error.
  Expected<TensorBuffer> CreateBufferForTensor(const TfLiteTensor* tensor) {
    LiteRtTensorBuffer buffer;
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextCreateBufferForTensor(Get(), tensor,
                                                               &buffer));
    return TensorBuffer::WrapCObject(buffer, OwnHandle::kYes);
  }

  /// @brief Registers a `TensorBuffer` object for the given tensor.
  ///
  /// The buffer context assumes ownership of the `TensorBuffer` object.
  /// @param tensor The tensor to associate with the buffer.
  /// @param tensor_buffer The `TensorBuffer` to register.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> RegisterTensorBuffer(const TfLiteTensor* tensor,
                                      TensorBuffer&& tensor_buffer) {
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextRegisterTensorBuffer(
            Get(), tensor, tensor_buffer.Release()));
    return {};
  }

  /// @brief Registers the buffer requirements for the given tensor.
  ///
  /// The buffer context assumes ownership of the buffer requirements.
  /// @param tensor The tensor to associate with the requirements.
  /// @param buffer_requirements The `TensorBufferRequirements` to register.
  /// @return An `Expected` object that is empty on success, or contains an
  /// error.
  Expected<void> RegisterBufferRequirements(
      const TfLiteTensor* tensor,
      TensorBufferRequirements&& buffer_requirements) {
    auto buffer = buffer_requirements.ToDetachedBuffer();
    LITERT_RETURN_IF_ERROR(
        LiteRtExternalLiteRtBufferContextRegisterBufferRequirements(
            Get(), tensor, buffer.data()));
    return {};
  }

  /// @brief Returns the environment for the given buffer context.
  ///
  /// No ownership is transferred.
  /// @return An `Expected` object containing the `Environment`, or an error.
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
