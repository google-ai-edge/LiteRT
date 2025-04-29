// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildReshapeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
  reshape_op.AddInputTensor(inputs[0]);
  reshape_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
