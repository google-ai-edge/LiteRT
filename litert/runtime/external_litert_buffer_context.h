// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
#define ODML_LITERT_LITERT_RUNTIME_EXTERNAL_LITERT_BUFFER_CONTEXT_H_

#include <memory>
#include <unordered_map>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

struct LiteRtTensorBufferRequirementsDeleter {
  void operator()(LiteRtTensorBufferRequirementsT* requirements) const {
    if (requirements) {
      LiteRtDestroyTensorBufferRequirements(requirements);
    }
  }
};

using LiteRtTensorBufferRequirementsPtr =
    std::unique_ptr<LiteRtTensorBufferRequirementsT,
                    LiteRtTensorBufferRequirementsDeleter>;

struct LiteRtTensorBufferDeleter {
  void operator()(LiteRtTensorBufferT* buffer) const {
    if (buffer) {
      LiteRtDestroyTensorBuffer(buffer);
    }
  }
};

using LiteRtTensorBufferPtr =
    std::unique_ptr<LiteRtTensorBufferT, LiteRtTensorBufferDeleter>;

class LiteRtExternalLiteRtBufferContextT : public TfLiteExternalContext {
 public:
  LiteRtExternalLiteRtBufferContextT() : env_(nullptr) {}
  explicit LiteRtExternalLiteRtBufferContextT(LiteRtEnvironment env)
      : env_(env) {}
  ~LiteRtExternalLiteRtBufferContextT() = default;

  static litert::Expected<LiteRtExternalLiteRtBufferContextT*> GetInstance(
      TfLiteOpaqueContext* context) {
    void* external_context;
    TfLiteOpaqueContextGetExternalContext(context, &external_context,
                                          kTfLiteLiteRtBufferContext);
    if (!external_context) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "External context not found");
    }
    return reinterpret_cast<LiteRtExternalLiteRtBufferContextT*>(
        external_context);
  }

  // Registers a tensor buffer requirements for the given tensor.
  // The registered TensorBufferRequirements object is owned by
  // LiteRtExternalLiteRtBufferContextT.
  // Note: Currently, the system pre-registers tensor buffer requirements before
  // they're actually used. A more efficient approach would be to query
  // DelegateKernel only when these requirements are needed.
  LiteRtStatus RegisterBufferRequirements(
      const TfLiteOpaqueTensor* tensor,
      LiteRtTensorBufferRequirementsPtr buffer_requirements);

  inline LiteRtStatus RegisterBufferRequirements(
      const TfLiteTensor* tensor,
      LiteRtTensorBufferRequirementsPtr buffer_requirements) {
    return RegisterBufferRequirements(
        reinterpret_cast<const TfLiteOpaqueTensor*>(tensor),
        std::move(buffer_requirements));
  }

  // Gets a registered tensor buffer requirements for the given tensor.
  // The returned TensorBufferRequirements object is still owned by
  // LiteRtExternalLiteRtBufferContextT.
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetBufferRequirements(const TfLiteOpaqueTensor* tensor);

  inline litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetBufferRequirements(const TfLiteTensor* tensor) {
    return GetBufferRequirements(
        reinterpret_cast<const TfLiteOpaqueTensor*>(tensor));
  }

  // Registers a tensor buffer for the given tensor.
  // The registered TensorBuffer object is owned by
  // LiteRtExternalLiteRtBufferContextT.
  LiteRtStatus RegisterTensorBuffer(const TfLiteOpaqueTensor* tensor,
                                    LiteRtTensorBufferPtr tensor_buffer);

  inline LiteRtStatus RegisterTensorBuffer(
      const TfLiteTensor* tensor, LiteRtTensorBufferPtr tensor_buffer) {
    return RegisterTensorBuffer(
        reinterpret_cast<const TfLiteOpaqueTensor*>(tensor),
        std::move(tensor_buffer));
  }

  // Gets a registered tensor buffer for the given tensor.
  // The returned TensorBuffer object is a duplicate (reference counted)
  // of registered TensorBuffer.
  litert::Expected<LiteRtTensorBufferPtr> GetTensorBuffer(
      const TfLiteOpaqueTensor* tensor);

  // Gets a registered tensor buffer for the given tensor.
  // The returned TensorBuffer object is a duplicate (reference counted)
  // of registered TensorBuffer.
  inline litert::Expected<LiteRtTensorBufferPtr> GetTensorBuffer(
      const TfLiteTensor* tensor) {
    return GetTensorBuffer(reinterpret_cast<const TfLiteOpaqueTensor*>(tensor));
  }

  // Creates a tensor buffer for the given tensor.
  // The callers takes ownership of the returned TensorBuffer object.
  litert::Expected<LiteRtTensorBufferPtr> CreateBufferForTensor(
      const TfLiteOpaqueTensor* tensor);

  inline litert::Expected<LiteRtTensorBufferPtr> CreateBufferForTensor(
      const TfLiteTensor* tensor) {
    return CreateBufferForTensor(
        reinterpret_cast<const TfLiteOpaqueTensor*>(tensor));
  }

  // Sets the async execution mode. It's set by CompiledModel and used by
  // DelegateKernel to decide whether to use async execution mode.
  inline void SetAsyncExecutionMode(bool async_execution_mode) {
    async_execution_mode_ = async_execution_mode;
  }

  // Returns true if the async execution mode is set.
  inline bool IsAsyncExecutionMode() const { return async_execution_mode_; }

  // Returns the LiteRtEnvironment used to create CompiledModel.
  inline LiteRtEnvironment GetEnvironment() const { return env_; }

 private:
  LiteRtEnvironment env_;
  std::unordered_map<const TfLiteOpaqueTensor*,
                     LiteRtTensorBufferRequirementsPtr>
      buffer_requirements_;
  std::unordered_map<const TfLiteOpaqueTensor*, LiteRtTensorBufferPtr>
      tensor_buffers_;

  LiteRtExternalLiteRtBufferContextT(
      const LiteRtExternalLiteRtBufferContextT&) = delete;
  LiteRtExternalLiteRtBufferContextT& operator=(
      const LiteRtExternalLiteRtBufferContextT&) = delete;

  bool async_execution_mode_ = false;
};

#endif  // ODML_LITERT_LITERT_RUNTIME_EXTERNAL_LITERT_BUFFER_CONTEXT_H_
