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

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert {

/// @brief Defines the C++ interface for custom operator kernels in LiteRT.
class CustomOpKernel {
 public:
  virtual ~CustomOpKernel() = default;

  LiteRtCustomOpKernel& GetLiteRtCustomOpKernel() { return custom_op_kernel_; }

  virtual const std::string& OpName() const = 0;
  virtual int OpVersion() const = 0;

  virtual Expected<void> Init(const void* init_data, size_t init_data_size) = 0;

  virtual Expected<void> GetOutputLayouts(
      const std::vector<Layout>& input_layouts,
      std::vector<Layout>& output_layouts) = 0;

  virtual Expected<void> Run(const std::vector<TensorBuffer>& inputs,
                             std::vector<TensorBuffer>& outputs) = 0;

  virtual Expected<void> Destroy() = 0;

 protected:
  CustomOpKernel() {
    custom_op_kernel_.Init = InitHelper;
    custom_op_kernel_.GetOutputLayouts = GetOutputLayoutsHelper;
    custom_op_kernel_.Run = RunHelper;
    custom_op_kernel_.Destroy = DestroyHelper;
  }

 private:
  static LiteRtStatus InitHelper(void* user_data, const void* init_data,
                                 size_t init_data_size);
  static LiteRtStatus GetOutputLayoutsHelper(void* user_data, size_t num_inputs,
                                             const LiteRtLayout* input_layouts,
                                             size_t num_outputs,
                                             LiteRtLayout* output_layouts);
  static LiteRtStatus RunHelper(void* user_data, size_t num_inputs,
                                const LiteRtTensorBuffer* inputs,
                                size_t num_outputs,
                                LiteRtTensorBuffer* outputs);
  static LiteRtStatus DestroyHelper(void* user_data);

  LiteRtCustomOpKernel custom_op_kernel_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_CUSTOM_OP_KERNEL_H_
