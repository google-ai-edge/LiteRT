// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/log_softmax_op_builder.h"

#include <cstdint>
#include <vector>

#include "QnnOpDef.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildLogSoftmaxOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, std::uint32_t axis,
    float beta) {
  std::vector<OpWrapper> res;

  OpWrapper& log_softmax_op = CreateOpWrapper(res, QNN_OP_LOG_SOFTMAX);
  log_softmax_op.AddInputTensor(inputs[0]);
  log_softmax_op.AddScalarParam<std::uint32_t>(QNN_OP_LOG_SOFTMAX_PARAM_AXIS,
                                               axis);
  log_softmax_op.AddScalarParam<float>(QNN_OP_LOG_SOFTMAX_PARAM_BETA, beta);
  log_softmax_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
