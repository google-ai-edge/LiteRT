// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildPreluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& prelu_op = CreateOpWrapper(res, QNN_OP_PRELU);
  for (const auto& input : inputs) {
    prelu_op.AddInputTensor(input);
  }
  if (outputs.size() != 1) {
    QNN_LOG_ERROR("Prelu op must have exactly one output tensor.");
    return {};
  }
  prelu_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
