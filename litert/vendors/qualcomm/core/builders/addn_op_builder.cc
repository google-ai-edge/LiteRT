// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/addn_op_builder.h"

#include <cstddef>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildAddNOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  constexpr size_t kMinNumInputs = 2;
  if (inputs.size() < kMinNumInputs || outputs.size() != 1) {
    QNN_LOG_ERROR("AddN op expects at least 2 inputs and only 1 output.");
    return res;
  }

  for (const TensorWrapper& input : inputs) {
    if (input.IsQuant()) {
      QNN_LOG_ERROR("AddN op does not support quantized inputs.");
      return res;
    }
  }

  TensorWrapper& output_tensor = outputs[0];
  if (output_tensor.IsQuant()) {
    QNN_LOG_ERROR("AddN op does not support quantized outputs.");
    return res;
  }

  if (inputs.size() == kMinNumInputs) {
    res.emplace_back(
        CreateElementWiseAddOp(inputs[0], inputs[1], output_tensor));
    return res;
  }

  TensorWrapperRef accumulator = tensor_pool.CloneNativeTensorFrom(inputs[0]);
  res.emplace_back(CreateElementWiseAddOp(inputs[0], inputs[1], accumulator));

  for (size_t i = kMinNumInputs; i < inputs.size(); ++i) {
    const bool is_last = (i == inputs.size() - 1);
    TensorWrapper& add_output =
        is_last ? output_tensor : tensor_pool.CloneNativeTensorFrom(inputs[0]);
    res.emplace_back(
        CreateElementWiseAddOp(accumulator, inputs[i], add_output));
    accumulator = add_output;
  }

  return res;
}

}  // namespace qnn
