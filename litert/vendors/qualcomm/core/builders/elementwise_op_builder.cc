// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {

OpWrapper CreateElementWiseUnaryOp(const TensorWrapper& input_0,
                                   const TensorWrapper& output_0,
                                   std::uint32_t param_value) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_UNARY),
               QNN_OP_ELEMENT_WISE_UNARY, QnnOpCode::kElementWiseUnary);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<std::uint32_t>(QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
                                   param_value);
  return op;
}

OpWrapper CreateElementWiseBinaryOp(const TensorWrapper& input_0,
                                    const TensorWrapper& input_1,
                                    const TensorWrapper& output_0,
                                    std::uint32_t param_value) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_BINARY),
               QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<std::uint32_t>(QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
                                   param_value);
  return op;
}

}  // namespace

OpWrapper CreateElementWiseAddOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(input_0, input_1, output_0,
                                   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD);
}

OpWrapper CreateElementWiseSubtractOp(const TensorWrapper& input_0,
                                      const TensorWrapper& input_1,
                                      const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(input_0, input_1, output_0,
                                   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT);
}

OpWrapper CreateElementWiseDivideOp(const TensorWrapper& input_0,
                                    const TensorWrapper& input_1,
                                    const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(input_0, input_1, output_0,
                                   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE);
}

OpWrapper CreateElementWiseAtanOp(const TensorWrapper& input_0,
                                  const TensorWrapper& output_0) {
  return CreateElementWiseUnaryOp(input_0, output_0,
                                  QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN);
}

std::vector<OpWrapper> BuildElementwiseSubOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  // TODO(jiunkaiy): Switch to QNN_OP_ELEMENT_WISE_BINARY (SUBTRACT) once the
  // SINT16 validation issue is fixed.
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
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY);
}

std::vector<OpWrapper> BuildElementwiseDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSinOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCeilOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCosOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseHardSwishOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRsqrtOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSqrtOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquareOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquaredDifferenceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SQUARED_DIFFERENCE);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseGreaterOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAndOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMinimumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMaximumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseEluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorModOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  const TensorWrapper& input_0 = inputs[0];
  const TensorWrapper& input_1 = inputs[1];
  const TensorWrapper& output_0 = outputs[0];

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(input_0);
  elementwise_op.AddInputTensor(input_1);
  elementwise_op.AddOutputTensor(output_0);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD);

  return res;
}

OpWrapper CreateElementWiseNotEqualOp(const TensorWrapper& input_0,
                                      const TensorWrapper& input_1,
                                      const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL);
}

OpWrapper CreateElementWiseGreaterOp(const TensorWrapper& input_0,
                                     const TensorWrapper& input_1,
                                     const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER);
}

OpWrapper CreateElementWiseLessOp(const TensorWrapper& input_0,
                                  const TensorWrapper& input_1,
                                  const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS);
}

OpWrapper CreateElementWiseGreaterEqualOp(const TensorWrapper& input_0,
                                          const TensorWrapper& input_1,
                                          const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL);
}

OpWrapper CreateElementWiseAndOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      input_0, input_1, output_0,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND);
}

std::vector<OpWrapper> BuildElementwiseOrOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR);

  return res;
}

std::vector<OpWrapper> BuildElementwisePowerOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessEqualOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL);

  return res;
}

OpWrapper CreateElementWiseNotOp(const TensorWrapper& input_0,
                                 const TensorWrapper& output_0) {
  return CreateElementWiseUnaryOp(input_0, output_0,
                                  QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT);
}

std::vector<OpWrapper> BuildElementwiseGreaterEqualOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwiseExpOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP);

  return res;
}

OpWrapper CreateElementWiseEqualOp(const TensorWrapper& input_0,
                                   const TensorWrapper& input_1,
                                   const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(input_0, input_1, output_0,
                                   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL);
}

std::vector<OpWrapper> BuildElementwiseLogOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAbsOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNegOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRoundOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSignOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAtan2Op(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  // Implements atan2(y, x) via atan(y/x) with quadrant correction:
  //   x > 0           => atan(y/x)
  //   x < 0, y >= 0   => atan(y/x) + pi
  //   x < 0, y < 0    => atan(y/x) - pi
  //   x = 0, y > 0    => +pi/2
  //   x = 0, y < 0    => -pi/2
  //   x = 0, y = 0    => 0  (matches std::atan2)
  std::vector<OpWrapper> res;
  res.reserve(19);

  TensorWrapper& const_zero =
      *tensor_pool.CreateStaticTensorWithValue(QNN_DATATYPE_FLOAT_32, {}, {1}, 0.0f);
  TensorWrapper& const_pi =
      *tensor_pool.CreateStaticTensorWithValue(QNN_DATATYPE_FLOAT_32, {}, {1}, M_PI);
  TensorWrapper& const_pos_pi_half =
      *tensor_pool.CreateStaticTensorWithValue(QNN_DATATYPE_FLOAT_32, {}, {1}, M_PI / 2);
  TensorWrapper& const_neg_pi_half =
      *tensor_pool.CreateStaticTensorWithValue(QNN_DATATYPE_FLOAT_32, {}, {1}, -M_PI / 2);

  TensorWrapper& x_greater_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[1].get().GetDimensions());
  TensorWrapper& x_less_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[1].get().GetDimensions());
  TensorWrapper& x_equal_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[1].get().GetDimensions());

  res.push_back(CreateElementWiseGreaterOp(inputs[1], const_zero, x_greater_than_zero_out));
  res.push_back(CreateElementWiseLessOp(inputs[1], const_zero, x_less_than_zero_out));
  res.push_back(CreateElementWiseEqualOp(inputs[1], const_zero, x_equal_zero_out));

  TensorWrapper& y_greater_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  TensorWrapper& y_greater_equal_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  TensorWrapper& y_less_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());

  res.push_back(CreateElementWiseGreaterOp(inputs[0], const_zero, y_greater_than_zero_out));
  res.push_back(CreateElementWiseGreaterEqualOp(inputs[0], const_zero, y_greater_equal_than_zero_out));
  res.push_back(CreateElementWiseLessOp(inputs[0], const_zero, y_less_than_zero_out));

  // atan2(y, x) = atan(y / x): inputs[0]=y, inputs[1]=x
  TensorWrapper& div_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateElementWiseDivideOp(inputs[0], inputs[1], div_out));

  TensorWrapper& atan_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateElementWiseAtanOp(div_out, atan_out));

  TensorWrapper& add_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateElementWiseAddOp(atan_out, const_pi, add_out));

  TensorWrapper& sub_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateElementWiseSubtractOp(atan_out, const_pi, sub_out));

  // case: x=0, y<0  =>  -pi/2
  TensorWrapper& case_xeq0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  res.push_back(CreateElementWiseAndOp(x_equal_zero_out, y_less_than_zero_out, case_xeq0_ylt0_out));
  TensorWrapper& select_xeq0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateSelectOp(case_xeq0_ylt0_out, const_neg_pi_half, const_zero, select_xeq0_ylt0_out));

  // case: x=0, y>0 => pi/2  (fallback from x=0,y<0)
  TensorWrapper& case_xeq0_ygt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  res.push_back(CreateElementWiseAndOp(x_equal_zero_out, y_greater_than_zero_out, case_xeq0_ygt0_out));
  TensorWrapper& select_xeq0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateSelectOp(case_xeq0_ygt0_out, const_pos_pi_half, select_xeq0_ylt0_out, select_xeq0_out));

  // case: x<0, y<0  =>  atan(y/x) - pi  (fallback from x=0 cases)
  TensorWrapper& case_xlt0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  res.push_back(CreateElementWiseAndOp(x_less_than_zero_out, y_less_than_zero_out, case_xlt0_ylt0_out));
  TensorWrapper& select_xlt0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateSelectOp(case_xlt0_ylt0_out, sub_out, select_xeq0_out, select_xlt0_ylt0_out));

  // case: x<0, y>=0  =>  atan(y/x) + pi  (fallback from x<0,y<0)
  TensorWrapper& case_xlt0_yge0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  res.push_back(CreateElementWiseAndOp(x_less_than_zero_out, y_greater_equal_than_zero_out, case_xlt0_yge0_out));
  TensorWrapper& select_xlt0_yge0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.push_back(CreateSelectOp(case_xlt0_yge0_out, add_out, select_xlt0_ylt0_out, select_xlt0_yge0_out));

  // case: x>0  =>  atan(y/x)  (final select)
  res.push_back(CreateSelectOp(x_greater_than_zero_out, atan_out, select_xlt0_yge0_out, outputs[0]));

  return res;
}

}  // namespace qnn
