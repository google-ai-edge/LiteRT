// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {
OpWrapper CreateElementWiseBinaryOp(const char* op_type, QnnOpCode op_code,
                                    const char* param_name,
                                    std::uint32_t param_value,
                                    const TensorWrapper& input_0,
                                    const TensorWrapper& input_1,
                                    const TensorWrapper& output_0) {
  auto name = GetUniqueOpName(QNN_OP_ELEMENT_WISE_BINARY);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(op_type, op_code);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<std::uint32_t>(param_name, param_value);
  return op;
}

OpWrapper CreateElementWiseUnary(const char* op_type, QnnOpCode op_code,
                                 const char* param_name,
                                 std::uint32_t param_value,
                                 const TensorWrapper& input_0,
                                 const TensorWrapper& output_0) {
  auto name = GetUniqueOpName(QNN_OP_ELEMENT_WISE_UNARY);
  OpWrapper op;
  op.SetName(std::move(name));
  op.SetType(op_type, op_code);
  op.AddInputTensor(input_0);
  op.AddOutputTensor(output_0);
  op.AddScalarParam<std::uint32_t>(param_name, param_value);
  return op;
}
}  // namespace

OpWrapper CreateElementWiseAddOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary,
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, input_0, input_1, output_0);
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
      QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary,
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, input_0, input_1,
      output_0);
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

OpWrapper CreateElementWiseNotEqualOp(const TensorWrapper& input_0,
                                      const TensorWrapper& input_1,
                                      const TensorWrapper& output_0) {
  return CreateElementWiseBinaryOp(
      QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary,
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL, input_0, input_1,
      output_0);
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
  return CreateElementWiseUnary(
      QNN_OP_ELEMENT_WISE_UNARY, QnnOpCode::kElementWiseUnary,
      QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT, input_0, output_0);
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
  return CreateElementWiseBinaryOp(
      QNN_OP_ELEMENT_WISE_BINARY, QnnOpCode::kElementWiseBinary,
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL, input_0, input_1, output_0);
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

}  // namespace qnn
