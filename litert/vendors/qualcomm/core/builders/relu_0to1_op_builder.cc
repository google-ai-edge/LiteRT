// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildRelu0To1Op(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& relu_0to1_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  relu_0to1_op.AddInputTensor(inputs[0]);
  relu_0to1_op.AddOutputTensor(outputs[0]);
  relu_0to1_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX);
  relu_0to1_op.AddScalarParam<float>(QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE,
                                     0);
  relu_0to1_op.AddScalarParam<float>(QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE,
                                     1);

  return res;
}

}  // namespace qnn
