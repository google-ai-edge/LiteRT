// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "QnnOpDef.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/onehot_fc.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

OpWrapper Binary(const TensorWrapper& lhs, const TensorWrapper& rhs,
                 const TensorWrapper& output, std::uint32_t operation) {
  OpWrapper op("binary", QNN_OP_ELEMENT_WISE_BINARY,
               QnnOpCode::kElementWiseBinary);
  op.AddInputTensor(lhs);
  op.AddInputTensor(rhs);
  op.AddOutputTensor(output);
  op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, operation);
  return op;
}

void AddBranch(TensorPool& pool, std::vector<OpWrapper>& ops,
               const TensorWrapper& positions, const TensorWrapper& scalar,
               const TensorWrapper& one_hot_value, const TensorWrapper& output,
               TensorWrapper*& indices) {
  auto& slice = pool.CreateNativeTensor(QNN_DATATYPE_INT_32, {}, {1, 2, 1});
  OpWrapper slice_op("slice", QNN_OP_STRIDED_SLICE,
                     QnnOpCode::kStridedSlice);
  slice_op.AddInputTensor(positions);
  slice_op.AddOutputTensor(slice);
  ops.emplace_back(std::move(slice_op));
  indices = &pool.CloneNativeTensorFrom(slice, {1, 2});
  ops.emplace_back(CreateReshapeOp(slice, *indices));
  auto& one_hot = pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, {1, 2, 4});
  OpWrapper one_hot_op("one_hot", QNN_OP_ONE_HOT, QnnOpCode::kOneHot);
  one_hot_op.AddInputTensor(*indices);
  one_hot_op.AddOutputTensor(one_hot);
  ops.emplace_back(std::move(one_hot_op));
  auto& less = pool.CreateNativeTensor(QNN_DATATYPE_BOOL_8, {}, {1, 2});
  auto& greater_equal = pool.CloneNativeTensorFrom(less);
  auto& logical_or = pool.CloneNativeTensorFrom(less);
  auto& not_equal = pool.CloneNativeTensorFrom(less);
  auto& logical_and = pool.CloneNativeTensorFrom(less);
  ops.emplace_back(Binary(*indices, scalar, less,
                          QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS));
  ops.emplace_back(Binary(*indices, scalar, greater_equal,
                          QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL));
  ops.emplace_back(Binary(less, greater_equal, logical_or,
                          QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR));
  ops.emplace_back(Binary(*indices, scalar, not_equal,
                          QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL));
  ops.emplace_back(Binary(logical_or, not_equal, logical_and,
                          QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND));
  auto& broadcast = pool.CloneNativeTensorFrom(logical_and, {1, 2, 1});
  ops.emplace_back(CreateReshapeOp(logical_and, broadcast));
  ops.emplace_back(CreateSelectOp(broadcast, one_hot_value, one_hot, output));
}

TEST(OneHotFcTest, ReplacesTwoOneHotFcBranchesWithGather) {
  TensorPool pool;
  std::vector<OpWrapper> ops;
  auto& positions = pool.CreateNativeTensor(QNN_DATATYPE_INT_32, {}, {1, 2, 2});
  std::array<std::int32_t, 1> scalar_data = {0};
  auto& scalar = pool.CreateStaticTensor(QNN_DATATYPE_INT_32, {}, {},
                                         sizeof(scalar_data), scalar_data.data());
  std::array<float, 1> one_hot_value_data = {0};
  auto& one_hot_value = pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {}, sizeof(one_hot_value_data),
      one_hot_value_data.data());
  auto& first_select = pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {},
                                                {1, 2, 4});
  auto& second_select = pool.CloneNativeTensorFrom(first_select);
  TensorWrapper* first_indices;
  TensorWrapper* second_indices;
  AddBranch(pool, ops, positions, scalar, one_hot_value, first_select,
            first_indices);
  AddBranch(pool, ops, positions, scalar, one_hot_value, second_select,
            second_indices);

  std::array<float, 12> weights_data =
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto& first_weight = pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {3, 4},
      weights_data.size() * sizeof(weights_data[0]), weights_data.data());
  auto& second_weight = pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, {3, 4},
      weights_data.size() * sizeof(weights_data[0]), weights_data.data());
  auto& first_fc_output =
      pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, {2, 3});
  auto& second_fc_output = pool.CloneNativeTensorFrom(first_fc_output);
  OpWrapper first_fc("fc0", QNN_OP_FULLY_CONNECTED,
                     QnnOpCode::kFullyConnected);
  first_fc.AddInputTensor(first_select);
  first_fc.AddInputTensor(first_weight);
  first_fc.AddOutputTensor(first_fc_output);
  ops.emplace_back(std::move(first_fc));
  auto& first_keep_dims_output =
      pool.CloneNativeTensorFrom(first_fc_output, {1, 2, 3});
  ops.emplace_back(CreateReshapeOp(first_fc_output, first_keep_dims_output));
  auto& first_reshape =
      pool.CloneNativeTensorFrom(first_keep_dims_output, {1, 1, 2, 3});
  ops.emplace_back(CreateReshapeOp(first_keep_dims_output, first_reshape));
  OpWrapper second_fc("fc1", QNN_OP_FULLY_CONNECTED,
                      QnnOpCode::kFullyConnected);
  second_fc.AddInputTensor(second_select);
  second_fc.AddInputTensor(second_weight);
  second_fc.AddOutputTensor(second_fc_output);
  ops.emplace_back(std::move(second_fc));
  auto& second_keep_dims_output =
      pool.CloneNativeTensorFrom(second_fc_output, {1, 2, 3});
  ops.emplace_back(CreateReshapeOp(second_fc_output, second_keep_dims_output));
  auto& second_reshape =
      pool.CloneNativeTensorFrom(second_keep_dims_output, {1, 1, 2, 3});
  ops.emplace_back(CreateReshapeOp(second_keep_dims_output, second_reshape));

  ASSERT_EQ(TransformOneHotFc([](OpWrapper&) { return true; }, ops, 0, pool,
                              26),
            8);
  ASSERT_EQ(ops.size(), 8);
  ASSERT_TRUE(ops[2].IsOpCode(QnnOpCode::kGather));
  ASSERT_TRUE(ops[3].IsOpCode(QnnOpCode::kReshape));
  ASSERT_TRUE(ops[6].IsOpCode(QnnOpCode::kGather));
  ASSERT_TRUE(ops[7].IsOpCode(QnnOpCode::kReshape));
  EXPECT_EQ(ops[2].GetInputTensor(1), *first_indices);
  EXPECT_EQ(ops[6].GetInputTensor(1), *second_indices);
  EXPECT_EQ(ops[2].GetOutputTensor(0), first_keep_dims_output);
  EXPECT_EQ(ops[6].GetOutputTensor(0), second_keep_dims_output);
  EXPECT_EQ(ops[3].GetInputTensor(0), first_keep_dims_output);
  EXPECT_EQ(ops[3].GetOutputTensor(0), first_reshape);
  EXPECT_EQ(ops[7].GetInputTensor(0), second_keep_dims_output);
  EXPECT_EQ(ops[7].GetOutputTensor(0), second_reshape);
  EXPECT_EQ(ops[2].GetInputTensor(0).GetDimensions(),
            (std::vector<std::uint32_t>{4, 3}));
  EXPECT_EQ(std::vector<float>(
                ops[2].GetInputTensor(0).GetTensorData<float>()->begin(),
                ops[2].GetInputTensor(0).GetTensorData<float>()->end()),
            (std::vector<float>{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}));
}

}  // namespace
}  // namespace qnn
