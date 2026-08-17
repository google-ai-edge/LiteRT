// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/tanh_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

std::vector<OpWrapper> BuildTanhOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  // Force the u16 tanh output to (scale = 1/32768, offset = -32768). Tanh
  // outputs are in [-1, +1] and HTP's INT16 tanh LUT is locked to this
  // grid (HtpOpDefSupplement Tanh Quantization Parameters Config). Under
  // u16 storage the required zero-point is 32768 (asymmetric, not zp = 0);
  // the source-declared asymmetric zp shifts the output on device. Route
  // through the wrapper so the internal variant matches the Qnn struct.
  if (outputs[0].get().IsQuantU16()) {
    outputs[0].get().SetScaleOffsetQuantParams(1.0f / 32768.0f, 32768);
  }
  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH);

  return res;
}

}  // namespace qnn
