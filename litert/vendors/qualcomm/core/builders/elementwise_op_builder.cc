// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

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

OpWrapper CreateElementWiseSubtractOp(const TensorWrapper& input_0,
                                      const TensorWrapper& input_1,
                                      const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_SUBTRACT),
               QNN_OP_ELEMENT_WISE_SUBTRACT, QnnOpCode::kElementWiseSubtract);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseDivideOp(const TensorWrapper& input_0,
                                    const TensorWrapper& input_1,
                                    const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_DIVIDE),
               QNN_OP_ELEMENT_WISE_DIVIDE, QnnOpCode::kElementWiseDivide);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseAtanOp(const TensorWrapper& input_0,
                                  const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_ATAN),
               QNN_OP_ELEMENT_WISE_ATAN, QnnOpCode::kElementWiseAtan);
  op.AddInputTensor(input_0);
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
               QNN_OP_ELEMENT_WISE_MULTIPLY, QnnOpCode::kElementWiseMultiply);
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
               QNN_OP_ELEMENT_WISE_NOT_EQUAL, QnnOpCode::kElementWiseNotEqual);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseGreaterOp(const TensorWrapper& input_0,
                                     const TensorWrapper& input_1,
                                     const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_GREATER),
               QNN_OP_ELEMENT_WISE_GREATER, QnnOpCode::kElementWiseGreater);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseLessOp(const TensorWrapper& input_0,
                                  const TensorWrapper& input_1,
                                  const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_LESS),
               QNN_OP_ELEMENT_WISE_LESS, QnnOpCode::kElementWiseLess);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseGreaterEqualOp(const TensorWrapper& input_0,
                                          const TensorWrapper& input_1,
                                          const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_GREATER_EQUAL),
               QNN_OP_ELEMENT_WISE_GREATER_EQUAL,
               QnnOpCode::kElementWiseGreaterEqual);
  op.AddInputTensor(input_0);
  op.AddInputTensor(input_1);
  op.AddOutputTensor(output_0);
  return op;
}

OpWrapper CreateElementWiseAndOp(const TensorWrapper& input_0,
                                 const TensorWrapper& input_1,
                                 const TensorWrapper& output_0) {
  OpWrapper op(GetUniqueOpName(QNN_OP_ELEMENT_WISE_AND),
               QNN_OP_ELEMENT_WISE_AND, QnnOpCode::kElementWiseAnd);
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
  auto& x = inputs[1].get();
  auto& y = inputs[0].get();

  std::vector<OpWrapper> res;
  res.reserve(19);

  static constexpr float kPi = 3.14159265358979323846f;
  TensorWrapper& const_zero = *tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, 0.0f);
  TensorWrapper& const_pi = *tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, kPi);
  TensorWrapper& const_pos_pi_half = *tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, kPi / 2);
  TensorWrapper& const_neg_pi_half = *tensor_pool.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, -kPi / 2);

  TensorWrapper& x_greater_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, x.GetDimensions());
  TensorWrapper& x_less_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, x.GetDimensions());
  TensorWrapper& x_equal_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, x.GetDimensions());

  res.emplace_back(
      CreateElementWiseGreaterOp(x, const_zero, x_greater_than_zero_out));
  res.emplace_back(
      CreateElementWiseLessOp(x, const_zero, x_less_than_zero_out));
  res.emplace_back(CreateElementWiseEqualOp(x, const_zero, x_equal_zero_out));

  TensorWrapper& y_greater_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  TensorWrapper& y_greater_equal_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  TensorWrapper& y_less_than_zero_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());

  res.emplace_back(
      CreateElementWiseGreaterOp(y, const_zero, y_greater_than_zero_out));
  res.emplace_back(CreateElementWiseGreaterEqualOp(
      y, const_zero, y_greater_equal_than_zero_out));
  res.emplace_back(
      CreateElementWiseLessOp(y, const_zero, y_less_than_zero_out));

  // atan2(y, x) = atan(y / x)
  TensorWrapper& div_out = tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateElementWiseDivideOp(y, x, div_out));

  TensorWrapper& atan_out = tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateElementWiseAtanOp(div_out, atan_out));

  TensorWrapper& add_out = tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateElementWiseAddOp(atan_out, const_pi, add_out));

  TensorWrapper& sub_out = tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateElementWiseSubtractOp(atan_out, const_pi, sub_out));

  // Note: The NaN results in div/atan/add/sub due to x==0 are masked by the
  // downstream selects case: x=0, y<0  =>  -pi/2
  TensorWrapper& case_xeq0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  res.emplace_back(CreateElementWiseAndOp(
      x_equal_zero_out, y_less_than_zero_out, case_xeq0_ylt0_out));
  TensorWrapper& select_xeq0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateSelectOp(case_xeq0_ylt0_out, const_neg_pi_half,
                                  const_zero, select_xeq0_ylt0_out));

  // case: x=0, y>0 => pi/2  (fallback from x=0,y<0)
  TensorWrapper& case_xeq0_ygt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  res.emplace_back(CreateElementWiseAndOp(
      x_equal_zero_out, y_greater_than_zero_out, case_xeq0_ygt0_out));
  TensorWrapper& select_xeq0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateSelectOp(case_xeq0_ygt0_out, const_pos_pi_half,
                                  select_xeq0_ylt0_out, select_xeq0_out));

  // case: x<0, y<0  =>  atan(y/x) - pi  (fallback from x=0 cases)
  TensorWrapper& case_xlt0_ylt0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  res.emplace_back(CreateElementWiseAndOp(
      x_less_than_zero_out, y_less_than_zero_out, case_xlt0_ylt0_out));
  TensorWrapper& select_xlt0_ylt0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateSelectOp(case_xlt0_ylt0_out, sub_out, select_xeq0_out,
                                  select_xlt0_ylt0_out));

  // case: x<0, y>=0  =>  atan(y/x) + pi  (fallback from x<0,y<0)
  TensorWrapper& case_xlt0_yge0_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, {}, y.GetDimensions());
  res.emplace_back(CreateElementWiseAndOp(
      x_less_than_zero_out, y_greater_equal_than_zero_out, case_xlt0_yge0_out));
  TensorWrapper& select_xlt0_yge0_out =
      tensor_pool.CloneNativeTensorFrom(outputs[0].get());
  res.emplace_back(CreateSelectOp(case_xlt0_yge0_out, add_out,
                                  select_xlt0_ylt0_out, select_xlt0_yge0_out));

  // case: x>0  =>  atan(y/x)  (final select)
  res.emplace_back(CreateSelectOp(x_greater_than_zero_out, atan_out,
                                  select_xlt0_yge0_out, outputs[0]));

  return res;
}

}  // namespace qnn
