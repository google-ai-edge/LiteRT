// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
OpWrapper CreateElementWiseAddOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_ADD),
               QNN_OP_ELEMENT_WISE_ADD, QnnOpCode::kElementWiseAdd);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

std::vector<OpWrapper> BuildElementwiseSubOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SUBTRACT);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

OpWrapper CreateElementWiseMulOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_MULTIPLY),
               QNN_OP_ELEMENT_WISE_MULTIPLY,
               QnnOpCode::kElementWiseMultiply);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

std::vector<OpWrapper> BuildElementwiseDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_DIVIDE);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSinOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SIN);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCeilOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_CEIL);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCosOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_COS);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseHardSwishOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_HARD_SWISH);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRsqrtOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_RSQRT);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSqrtOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SQUARE_ROOT);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquareOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op =
      CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MULTIPLY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquaredDifferenceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SQUARED_DIFFERENCE);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_LESS);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseGreaterOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAndOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_AND);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMinimumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MINIMUM);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMaximumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MAXIMUM);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseEluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELU);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_FLOOR);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_FLOOR_DIV);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorModOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  const TensorWrapper& input_0 = inputs[0];
  const TensorWrapper& input_1 = inputs[1];
  const TensorWrapper& output_0 = outputs[0];

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MOD);
  elementwise_op.AddInputTensor(input_0);
  elementwise_op.AddInputTensor(input_1);
  elementwise_op.AddOutputTensor(output_0);

  return res;
}

OpWrapper CreateElementWiseNotEqualOp(const TensorWrapper& input_0,
                                      const TensorWrapper& input_1,
                                      const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_NOT_EQUAL),
               QNN_OP_ELEMENT_WISE_NOT_EQUAL,
               QnnOpCode::kElementWiseNotEqual);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

std::vector<OpWrapper> BuildElementwiseOrOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_OR);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwisePowerOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_POWER);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessEqualOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_LESS_EQUAL);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

OpWrapper CreateElementWiseNotOp(const TensorWrapper& input_0,
                                 const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_NOT),
               QNN_OP_ELEMENT_WISE_NOT, QnnOpCode::kElementWiseNot);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  return op;
}

std::vector<OpWrapper> BuildElementwiseGreaterEqualOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER_EQUAL);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseExpOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_EXP);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

OpWrapper CreateElementWiseEqualOp(const TensorWrapper& input_0,
                                   const TensorWrapper& input_1,
                                   const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_EQUAL),
               QNN_OP_ELEMENT_WISE_EQUAL, QnnOpCode::kElementWiseEqual);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

std::vector<OpWrapper> BuildElementwiseLogOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_LOG);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAbsOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_ABS);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNegOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEG);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRoundOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_ROUND);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSignOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SIGN);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
