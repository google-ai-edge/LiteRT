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

#ifndef ODML_LITERT_LITERT_RUNTIME_CUSTOM_OP_DISPATCHER_H_
#define ODML_LITERT_LITERT_RUNTIME_CUSTOM_OP_DISPATCHER_H_

#include <memory>

#include "litert/c/litert_custom_op_kernel.h"
#include "litert/core/options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "tflite/c/c_api.h"
#include "tflite/c/c_api_opaque.h"

namespace litert::internal {

class CustomOpDispatcher {
 public:
  explicit CustomOpDispatcher(
      const LiteRtOptionsT::CustomOpOption& custom_op_option);
  ~CustomOpDispatcher();

  TfLiteRegistration* GetTfLiteRegistration() {
    return tfl_registration_.get();
  }

  // Make this class non-copiable because it includes a Registration
  // instance with a reference to itself.
  CustomOpDispatcher(const CustomOpDispatcher& other) = delete;
  CustomOpDispatcher(CustomOpDispatcher&& other) = delete;
  CustomOpDispatcher& operator=(const CustomOpDispatcher& other) = delete;
  CustomOpDispatcher& operator=(CustomOpDispatcher&& other) = delete;

 private:
  static void* Init(void* user_data, TfLiteOpaqueContext* context,
                    const char* buffer, size_t length);
  static void Free(void* user_data, TfLiteOpaqueContext* context, void* buffer);
  static TfLiteStatus Prepare(void* user_data, TfLiteOpaqueContext* context,
                              TfLiteOpaqueNode* node);
  static TfLiteStatus Invoke(void* user_data, TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node);

  static Expected<void> PrepareHelper(void* user_data,
                                      TfLiteOpaqueContext* context,
                                      TfLiteOpaqueNode* node);
  static Expected<void> InvokeHelper(void* user_data,
                                     TfLiteOpaqueContext* context,
                                     TfLiteOpaqueNode* node);

  Expected<LiteRtTensorBufferPtr> GetTensorBuffer(
      TfLiteOpaqueContext* context,
      const TfLiteOpaqueTensor* tfl_opaque_tensor);

  LiteRtCustomOpKernel op_kernel_;
  void* user_data_;
  TfLiteOperator* tfl_operator_;
  std::unique_ptr<TfLiteRegistration> tfl_registration_;
  LiteRtExternalLiteRtBufferContextT* buffer_context_ = nullptr;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_CUSTOM_OP_DISPATCHER_H_
