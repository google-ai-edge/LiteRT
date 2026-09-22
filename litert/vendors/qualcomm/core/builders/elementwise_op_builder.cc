// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
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

  auto& x_greater_than_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER);
  x_greater_than_zero.AddSuffixToName("_atan2_x_gt_zero");
  auto& x_less_than_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_LESS);
  x_less_than_zero.AddSuffixToName("_atan2_x_lt_zero");
  auto& x_equal_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_EQUAL);
  x_equal_zero.AddSuffixToName("_atan2_x_eq_zero");

  x_greater_than_zero.AddInputTensor(inputs[1]);
  x_greater_than_zero.AddInputTensor(const_zero);
  x_greater_than_zero.AddOutputTensor(x_greater_than_zero_out);

  x_less_than_zero.AddInputTensor(inputs[1]);
  x_less_than_zero.AddInputTensor(const_zero);
  x_less_than_zero.AddOutputTensor(x_less_than_zero_out);

  x_equal_zero.AddInputTensor(inputs[1]);
  x_equal_zero.AddInputTensor(const_zero);
  x_equal_zero.AddOutputTensor(x_equal_zero_out);

  TensorWrapper& y_greater_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  TensorWrapper& y_greater_equal_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  TensorWrapper& y_less_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());

  auto& y_greater_than_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER);
  y_greater_than_zero.AddSuffixToName("_atan2_y_gt_zero");
  auto& y_greater_equal_than_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER_EQUAL);
  y_greater_equal_than_zero.AddSuffixToName("_atan2_y_ge_zero");
  auto& y_less_than_zero = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_LESS);
  y_less_than_zero.AddSuffixToName("_atan2_y_lt_zero");

  y_greater_than_zero.AddInputTensor(inputs[0]);
  y_greater_than_zero.AddInputTensor(const_zero);
  y_greater_than_zero.AddOutputTensor(y_greater_than_zero_out);

  y_greater_equal_than_zero.AddInputTensor(inputs[0]);
  y_greater_equal_than_zero.AddInputTensor(const_zero);
  y_greater_equal_than_zero.AddOutputTensor(y_greater_equal_than_zero_out);

  y_less_than_zero.AddInputTensor(inputs[0]);
  y_less_than_zero.AddInputTensor(const_zero);
  y_less_than_zero.AddOutputTensor(y_less_than_zero_out);

  // atan2(y, x) = atan(y / x): inputs[0]=y, inputs[1]=x
  TensorWrapper& div_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());

  auto& div_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  div_op.AddSuffixToName("_atan2_div_y_x");
  div_op.AddInputTensor(inputs[0]);
  div_op.AddInputTensor(inputs[1]);
  div_op.AddOutputTensor(div_out);
  div_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE);

  TensorWrapper& atan_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& atan_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_UNARY);
  atan_op.AddSuffixToName("_atan2_atan");
  atan_op.AddInputTensor(div_out);
  atan_op.AddOutputTensor(atan_out);
  atan_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN);

  TensorWrapper& add_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& add_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  add_op.AddSuffixToName("_atan2_atan_add_pi");
  add_op.AddInputTensor(atan_out);
  add_op.AddInputTensor(const_pi);
  add_op.AddOutputTensor(add_out);
  add_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD);

  TensorWrapper& sub_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& sub_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_BINARY);
  sub_op.AddSuffixToName("_atan2_atan_sub_pi");
  sub_op.AddInputTensor(atan_out);
  sub_op.AddInputTensor(const_pi);
  sub_op.AddOutputTensor(sub_out);
  sub_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT);

  // case: x=0, y<0  =>  -pi/2
  auto& case_xeq0_ylt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_AND);
  case_xeq0_ylt0.AddSuffixToName("_atan2_x_eq_zero_and_y_lt_zero");
  TensorWrapper& case_xeq0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  case_xeq0_ylt0.AddInputTensor(x_equal_zero_out);
  case_xeq0_ylt0.AddInputTensor(y_less_than_zero_out);
  case_xeq0_ylt0.AddOutputTensor(case_xeq0_ylt0_out);
  TensorWrapper& select_xeq0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& select_xeq0_ylt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_xeq0_ylt0.AddSuffixToName("_atan2_select_xeq0_ylt0");
  select_xeq0_ylt0.AddInputTensor(case_xeq0_ylt0_out);
  select_xeq0_ylt0.AddInputTensor(const_neg_pi_half);
  select_xeq0_ylt0.AddInputTensor(const_zero);
  select_xeq0_ylt0.AddOutputTensor(select_xeq0_ylt0_out);

  // case: x=0, y>0 => pi/2  (fallback from x=0,y<0)
  auto& case_xeq0_ygt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_AND);
  case_xeq0_ygt0.AddSuffixToName("_atan2_x_eq_zero_and_y_gt_zero");
  TensorWrapper& case_xeq0_ygt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  case_xeq0_ygt0.AddInputTensor(x_equal_zero_out);
  case_xeq0_ygt0.AddInputTensor(y_greater_than_zero_out);
  case_xeq0_ygt0.AddOutputTensor(case_xeq0_ygt0_out);
  TensorWrapper& select_xeq0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& select_xeq0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_xeq0.AddSuffixToName("_atan2_select_xeq0");
  select_xeq0.AddInputTensor(case_xeq0_ygt0_out);
  select_xeq0.AddInputTensor(const_pos_pi_half);
  select_xeq0.AddInputTensor(select_xeq0_ylt0_out);
  select_xeq0.AddOutputTensor(select_xeq0_out);

  // case: x<0, y<0  =>  atan(y/x) - pi  (fallback from x=0 cases)
  auto& case_xlt0_ylt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_AND);
  case_xlt0_ylt0.AddSuffixToName("_atan2_x_lt_zero_and_y_lt_zero");
  TensorWrapper& case_xlt0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  case_xlt0_ylt0.AddInputTensor(x_less_than_zero_out);
  case_xlt0_ylt0.AddInputTensor(y_less_than_zero_out);
  case_xlt0_ylt0.AddOutputTensor(case_xlt0_ylt0_out);
  TensorWrapper& select_xlt0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& select_xlt0_ylt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_xlt0_ylt0.AddSuffixToName("_atan2_select_xlt0_ylt0");
  select_xlt0_ylt0.AddInputTensor(case_xlt0_ylt0_out);
  select_xlt0_ylt0.AddInputTensor(sub_out);
  select_xlt0_ylt0.AddInputTensor(select_xeq0_out);
  select_xlt0_ylt0.AddOutputTensor(select_xlt0_ylt0_out);

  // case: x<0, y>=0  =>  atan(y/x) + pi  (fallback from x<0,y<0)
  auto& case_xlt0_yge0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_AND);
  case_xlt0_yge0.AddSuffixToName("_atan2_x_lt_zero_and_y_ge_zero");
  TensorWrapper& case_xlt0_yge0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, inputs[0].get().GetDimensions());
  case_xlt0_yge0.AddInputTensor(x_less_than_zero_out);
  case_xlt0_yge0.AddInputTensor(y_greater_equal_than_zero_out);
  case_xlt0_yge0.AddOutputTensor(case_xlt0_yge0_out);
  TensorWrapper& select_xlt0_yge0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  auto& select_xlt0_yge0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_xlt0_yge0.AddSuffixToName("_atan2_select_xlt0_yge0");
  select_xlt0_yge0.AddInputTensor(case_xlt0_yge0_out);
  select_xlt0_yge0.AddInputTensor(add_out);
  select_xlt0_yge0.AddInputTensor(select_xlt0_ylt0_out);
  select_xlt0_yge0.AddOutputTensor(select_xlt0_yge0_out);

  // case: x>0  =>  atan(y/x)  (final select)
  auto& select_xgt0 = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_xgt0.AddSuffixToName("_atan2_select_xgt0");
  select_xgt0.AddInputTensor(x_greater_than_zero_out);
  select_xgt0.AddInputTensor(atan_out);
  select_xgt0.AddInputTensor(select_xlt0_yge0_out);
  select_xgt0.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
