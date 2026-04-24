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

#ifndef ODML_LITERT_LITERT_CC_LITERT_CUSTOM_OP_KERNEL_H_
#define ODML_LITERT_LITERT_CC_LITERT_CUSTOM_OP_KERNEL_H_

#include <string>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_layout.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_tensor_buffer.h"
#include <cstddef>
#include <vector>

namespace litert {

/// @brief Defines the C++ interface for custom operator kernels in LiteRT.
class CustomOpKernel {
public:
  virtual ~CustomOpKernel() = default;

  LiteRtCustomOpKernel &GetLiteRtCustomOpKernel() { return custom_op_kernel_; }

  virtual const std::string &OpName() const = 0;
  virtual int OpVersion() const = 0;

  virtual Expected<void> Init(const void *init_data, size_t init_data_size) = 0;

  virtual Expected<void>
  GetOutputLayouts(const std::vector<Layout> &input_layouts,
                   std::vector<Layout> &output_layouts) = 0;

  virtual Expected<void> Run(const std::vector<TensorBuffer> &inputs,
                             std::vector<TensorBuffer> &outputs) = 0;

  virtual Expected<void> Destroy() = 0;

protected:
  CustomOpKernel() {
    custom_op_kernel_.Init = InitHelper;
    custom_op_kernel_.GetOutputLayouts = GetOutputLayoutsHelper;
    custom_op_kernel_.Run = RunHelper;
    custom_op_kernel_.Destroy = DestroyHelper;
  }

private:
  static LiteRtStatus InitHelper(void *user_data, const void *init_data,
                                 size_t init_data_size) {
    auto *self = static_cast<CustomOpKernel *>(user_data);
    if (auto status = self->Init(init_data, init_data_size); !status) {
      LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
      return litert::ToLiteRtStatus(status.Error().StatusCC());
    }
    return kLiteRtStatusOk;
  }
  static LiteRtStatus GetOutputLayoutsHelper(void *user_data, size_t num_inputs,
                                             const LiteRtLayout *input_layouts,
                                             size_t num_outputs,
                                             LiteRtLayout *output_layouts) {
    auto *self = static_cast<CustomOpKernel *>(user_data);

    std::vector<Layout> input_layouts_vector;
    input_layouts_vector.reserve(num_inputs);
    for (auto i = 0; i < num_inputs; ++i) {
      input_layouts_vector.push_back(Layout(input_layouts[i]));
    }

    std::vector<Layout> output_layouts_vector(num_outputs);

    if (auto status =
            self->GetOutputLayouts(input_layouts_vector, output_layouts_vector);
        !status) {
      LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
      return litert::ToLiteRtStatus(status.Error().StatusCC());
    }

    for (auto i = 0; i < num_outputs; ++i) {
      output_layouts[i] = static_cast<LiteRtLayout>(output_layouts_vector[i]);
    }

    return kLiteRtStatusOk;
  }
  static LiteRtStatus RunHelper(void *user_data, size_t num_inputs,
                                const LiteRtTensorBuffer *inputs,
                                size_t num_outputs,
                                LiteRtTensorBuffer *outputs) {
    auto *self = static_cast<CustomOpKernel *>(user_data);

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
      return litert::ToLiteRtStatus(status.Error().StatusCC());
    }

    return kLiteRtStatusOk;
  }
  static LiteRtStatus DestroyHelper(void *user_data) {
    auto *self = static_cast<CustomOpKernel *>(user_data);
    if (auto status = self->Destroy(); !status) {
      LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().c_str());
      return litert::ToLiteRtStatus(status.Error().StatusCC());
    }
    return kLiteRtStatusOk;
  }

  LiteRtCustomOpKernel custom_op_kernel_;
};

} // namespace litert

#endif // ODML_LITERT_LITERT_CC_LITERT_CUSTOM_OP_KERNEL_H_
