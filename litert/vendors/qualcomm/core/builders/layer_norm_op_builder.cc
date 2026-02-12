// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/layer_norm_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

static constexpr int kAxesIndex = 3;

std::vector<OpWrapper> BuildLayerNormOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float epsilon) {
  std::vector<OpWrapper> res;

  auto& layer_norm_op = CreateOpWrapper(res, QNN_OP_LAYER_NORM);
  // inputs: input, gamma, beta, axes
  for (size_t i = 0; i < kAxesIndex && i < inputs.size(); ++i) {
    layer_norm_op.AddInputTensor(inputs[i]);
  }
  layer_norm_op.AddOutputTensor(outputs[0]);

  layer_norm_op.AddScalarParam<float>(QNN_OP_LAYER_NORM_PARAM_EPSILON, epsilon);
  if (inputs.size() > kAxesIndex) {
    layer_norm_op.AddTensorParam(QNN_OP_LAYER_NORM_PARAM_AXES,
                                 inputs[kAxesIndex]);
  }

  return res;
}

}  // namespace qnn
