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

#include "litert/cc/litert_custom_op_kernel.h"

#include <cstddef>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {

LiteRtStatus CustomOpKernel::InitHelper(void* user_data, const void* init_data,
                                        size_t init_data_size) {
  auto* self = static_cast<CustomOpKernel*>(user_data);
  if (auto status = self->Init(init_data, init_data_size); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CustomOpKernel::GetOutputLayoutsHelper(
    void* user_data, size_t num_inputs, const LiteRtLayout* input_layouts,
    size_t num_outputs, LiteRtLayout* output_layouts) {
  auto* self = static_cast<CustomOpKernel*>(user_data);

  std::vector<Layout> input_layouts_;
  input_layouts_.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    input_layouts_.push_back(Layout(input_layouts[i]));
  }

  std::vector<Layout> output_layouts_(num_outputs);

  if (auto status = self->GetOutputLayouts(input_layouts_, output_layouts_);
      !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return status.Error().Status();
  }

  for (auto i = 0; i < num_outputs; ++i) {
    output_layouts[i] = static_cast<LiteRtLayout>(output_layouts_[i]);
  }

  return kLiteRtStatusOk;
}

LiteRtStatus CustomOpKernel::RunHelper(void* user_data, size_t num_inputs,
                                       const LiteRtTensorBuffer* inputs,
                                       size_t num_outputs,
                                       LiteRtTensorBuffer* outputs) {
  auto* self = static_cast<CustomOpKernel*>(user_data);

  std::vector<TensorBuffer> inputs_;
  inputs_.reserve(num_inputs);
  for (auto i = 0; i < num_inputs; ++i) {
    inputs_.push_back(TensorBuffer::WrapCObject(inputs[i], OwnHandle::kNo));
  }

  std::vector<TensorBuffer> outputs_;
  outputs_.reserve(num_outputs);
  for (auto i = 0; i < num_outputs; ++i) {
    outputs_.push_back(TensorBuffer::WrapCObject(outputs[i], OwnHandle::kNo));
  }

  if (auto status = self->Run(inputs_, outputs_); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return status.Error().Status();
  }

  return kLiteRtStatusOk;
}

LiteRtStatus CustomOpKernel::DestroyHelper(void* user_data) {
  auto* self = static_cast<CustomOpKernel*>(user_data);
  if (auto status = self->Destroy(); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // namespace litert
