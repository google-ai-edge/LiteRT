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

#include "litert/runtime/custom_op_dispatcher.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tfl_utils.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/c_api.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/common.h"

namespace litert::internal {
namespace {

// Creates a TensorBuffer attached to the TFL tensor's data buffer.
Expected<LiteRtTensorBufferPtr> CreateHostTensorBufferFromTflTensor(
    TfLiteOpaqueContext* tfl_context,
    const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          ConvertTensorType(tfl_opaque_tensor));
  void* host_mem_addr = TfLiteOpaqueTensorData(tfl_opaque_tensor);
  size_t buffer_size = TfLiteOpaqueTensorByteSize(tfl_opaque_tensor);
  LiteRtRankedTensorType litert_tensor_type =
      static_cast<LiteRtRankedTensorType>(tensor_type);
  LiteRtTensorBufferT* tensor_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromHostMemory(
      &litert_tensor_type, host_mem_addr, buffer_size, /*deallocator=*/nullptr,
      &tensor_buffer));
  return LiteRtTensorBufferPtr(tensor_buffer);
}

}  // namespace

CustomOpDispatcher::CustomOpDispatcher(
    const LiteRtOptionsT::CustomOpOption& custom_op_option) {
  op_kernel_ = custom_op_option.op_kernel;
  user_data_ = custom_op_option.user_data;

  tfl_operator_ = TfLiteOperatorCreate(kTfLiteBuiltinCustom,
                                       custom_op_option.op_name.c_str(),
                                       custom_op_option.op_version, this);
  TfLiteOperatorSetInitWithData(tfl_operator_, Init);
  TfLiteOperatorSetFreeWithData(tfl_operator_, Free);
  TfLiteOperatorSetPrepareWithData(tfl_operator_, Prepare);
  TfLiteOperatorSetInvokeWithData(tfl_operator_, Invoke);

  tfl_registration_ = std::make_unique<TfLiteRegistration>(TfLiteRegistration{
      .registration_external = tfl_operator_,
  });
}

CustomOpDispatcher::~CustomOpDispatcher() {
  TfLiteOperatorDelete(tfl_operator_);
}

void* CustomOpDispatcher::Init(void* user_data, TfLiteOpaqueContext* context,
                               const char* buffer, size_t length) {
  if (auto buffer_context =
          LiteRtExternalLiteRtBufferContextT::GetInstance(context);
      buffer_context) {
    auto& self = *static_cast<CustomOpDispatcher*>(user_data);
    self.buffer_context_ = *buffer_context;
  }

  auto& self = *static_cast<CustomOpDispatcher*>(user_data);
  self.op_kernel_.Init(self.user_data_, buffer, length);

  // Must return a non-null pointer, otherwise Free() will not be called.
  return user_data;
}

void CustomOpDispatcher::Free(void* user_data, TfLiteOpaqueContext* context,
                              void* buffer) {
  // Parameter `buffer` contains the pointer returned by Init(), which we can
  // discard since by construction is the same as user_data.
  auto& self = *static_cast<CustomOpDispatcher*>(user_data);
  self.op_kernel_.Destroy(self.user_data_);
}

TfLiteStatus CustomOpDispatcher::Prepare(void* user_data,
                                         TfLiteOpaqueContext* context,
                                         TfLiteOpaqueNode* node) {
  if (auto status = PrepareHelper(user_data, context, node); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

Expected<void> CustomOpDispatcher::PrepareHelper(void* user_data,
                                                 TfLiteOpaqueContext* context,
                                                 TfLiteOpaqueNode* node) {
  auto num_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  std::vector<LiteRtLayout> input_layouts(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    LITERT_ASSIGN_OR_RETURN(auto layout,
                            ConvertTensorLayout(tfl_opaque_tensor));
    if (layout.has_strides) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Unexpected layout with strides for tensor %s",
                          TfLiteOpaqueTensorName(tfl_opaque_tensor)));
    }
    input_layouts[i] = static_cast<LiteRtLayout>(layout);
  }

  auto num_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  std::vector<LiteRtLayout> output_layouts;
  output_layouts.resize(num_outputs);

  auto& self = *static_cast<CustomOpDispatcher*>(user_data);
  self.op_kernel_.GetOutputLayouts(self.user_data_, input_layouts.size(),
                                   input_layouts.data(), output_layouts.size(),
                                   output_layouts.data());

  for (auto i = 0; i < output_layouts.size(); ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    LITERT_RETURN_IF_ERROR(
        ResizeTensor(output_layouts[i], context, tfl_opaque_tensor));
  }

  return {};
}

TfLiteStatus CustomOpDispatcher::Invoke(void* user_data,
                                        TfLiteOpaqueContext* context,
                                        TfLiteOpaqueNode* node) {
  if (auto status = InvokeHelper(user_data, context, node); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return kTfLiteError;
  }
  return kTfLiteOk;
}

Expected<void> CustomOpDispatcher::InvokeHelper(void* user_data,
                                                TfLiteOpaqueContext* context,
                                                TfLiteOpaqueNode* node) {
  auto& self = *static_cast<CustomOpDispatcher*>(user_data);

  auto num_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  std::vector<LiteRtTensorBuffer> inputs;
  inputs.reserve(num_inputs);

  auto num_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  std::vector<LiteRtTensorBuffer> outputs;
  outputs.reserve(num_outputs);

  absl::Cleanup tensor_buffers_cleanup = [&] {
    std::for_each(inputs.begin(), inputs.end(), LiteRtDestroyTensorBuffer);
    std::for_each(outputs.begin(), outputs.end(), LiteRtDestroyTensorBuffer);
  };

  for (auto i = 0; i < num_inputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                            self.GetTensorBuffer(context, tfl_opaque_tensor));
    inputs.push_back(tensor_buffer.release());
  }

  for (auto i = 0; i < num_outputs; ++i) {
    auto* tfl_opaque_tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                            self.GetTensorBuffer(context, tfl_opaque_tensor));
    outputs.push_back(tensor_buffer.release());
  }

  self.op_kernel_.Run(self.user_data_, inputs.size(), inputs.data(),
                      outputs.size(), outputs.data());

  return {};
}

Expected<LiteRtTensorBufferPtr> CustomOpDispatcher::GetTensorBuffer(
    TfLiteOpaqueContext* context, const TfLiteOpaqueTensor* tfl_opaque_tensor) {
  // If there is already a tensor buffer associated with the TFL tensor, then
  // return a duplicate. Otherwise create a new one attached to the TFL tensor's
  // data.
  if (buffer_context_) {
    if (auto tensor_buffer =
            buffer_context_->GetTensorBuffer(tfl_opaque_tensor);
        tensor_buffer) {
      // Duplicate the tensor buffer to avoid the lifetime issue.
      // The original tensor buffer is owned by the buffer context, and it
      // might be deallocated after the invoke.
      LiteRtTensorBuffer buffer = tensor_buffer->get();
      buffer->Duplicate();
      return LiteRtTensorBufferPtr(buffer);
    }
  }
  return CreateHostTensorBufferFromTflTensor(context, tfl_opaque_tensor);
}

}  // namespace litert::internal
