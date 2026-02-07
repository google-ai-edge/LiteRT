// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {

OpWrapper CreateConcatenationOp(
    const std::vector<ConstTensorWrapperRef>& inputs,
    const TensorWrapper& output, std::uint32_t axis) {
  auto name = GetUniqueOpName(QNN_OP_CONCAT);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(QNN_OP_CONCAT, QnnOpCode::kConcat);
  for (const auto& input : inputs) {
    op.AddInputTensor(input);
  }
  op.AddOutputTensor(output);
  op.AddScalarParam<std::uint32_t>(QNN_OP_CONCAT_PARAM_AXIS, axis);
  return op;
}

OpWrapper CreateConcatenationOpWithSameParam(
    const OpWrapper& src, const std::vector<ConstTensorWrapperRef>& inputs,
    const TensorWrapper& output) {
  auto name = GetUniqueOpName(QNN_OP_CONCAT);
  OpWrapper op(src);
  op.SetName(std::move(name));
  op.ClearInputOutputTensors();
  for (const auto& input : inputs) {
    op.AddInputTensor(input);
  }
  op.AddOutputTensor(output);
  return op;
}

}  // namespace qnn
