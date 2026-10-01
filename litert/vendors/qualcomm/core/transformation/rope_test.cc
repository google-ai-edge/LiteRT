// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/rope.h"

#include <cstdint>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {
namespace {

OpWrapper CreateStridedSlice(const TensorWrapper& input,
                             const TensorWrapper& output) {
  OpWrapper op("slice", QNN_OP_STRIDED_SLICE, QnnOpCode::kStridedSlice);
  op.AddInputTensor(input);
  op.AddOutputTensor(output);
  return op;
}

OpWrapper CreateSubtract(const TensorWrapper& input0,
                         const TensorWrapper& input1,
                         const TensorWrapper& output) {
  OpWrapper op("subtract", QNN_OP_ELEMENT_WISE_SUBTRACT,
               QnnOpCode::kElementWiseSubtract);
  op.AddInputTensor(input0);
  op.AddInputTensor(input1);
  op.AddOutputTensor(output);
  return op;
}

TEST(RopeTest, ReplacesFloat32RopeSubgraph) {
  TensorPool tensor_pool;
  const std::vector<std::uint32_t> kEmbeddingDims = {1, 2, 1, 32};
  const std::vector<std::uint32_t> kPairDims = {1, 2, 1, 16};
  auto& token_embedding =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kEmbeddingDims);
  auto& first_half =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& second_half =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& cos = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& sin = tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& first_cos =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& second_sin =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& first_output =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& second_cos =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& first_sin =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& second_output =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kPairDims);
  auto& output =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_32, {}, kEmbeddingDims);

  std::vector<OpWrapper> ops;
  ops.emplace_back(CreateStridedSlice(token_embedding, first_half));
  ops.emplace_back(CreateStridedSlice(token_embedding, second_half));
  ops.emplace_back(CreateElementWiseMulOp(first_half, cos, first_cos));
  ops.emplace_back(CreateElementWiseMulOp(second_half, sin, second_sin));
  ops.emplace_back(CreateSubtract(first_cos, second_sin, first_output));
  ops.emplace_back(CreateElementWiseMulOp(second_half, cos, second_cos));
  ops.emplace_back(CreateElementWiseMulOp(first_half, sin, first_sin));
  ops.emplace_back(CreateElementWiseAddOp(second_cos, first_sin, second_output));
  ops.emplace_back(CreateConcatenationOp({first_output, second_output}, output,
                                         /*axis=*/3));

  EXPECT_EQ(TransformRope([](OpWrapper&) { return true; }, ops, 0, tensor_pool,
                           ops.size()),
            1);
  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops[0].IsOpCode(QnnOpCode::kRotaryEmbedding));
  EXPECT_EQ(ops[0].GetInputCount(), 3);
  EXPECT_EQ(ops[0].GetInputTensor(0), token_embedding);
  EXPECT_EQ(ops[0].GetInputTensor(1), cos);
  EXPECT_EQ(ops[0].GetInputTensor(2), sin);
  EXPECT_EQ(ops[0].GetOutputTensor(0), output);
}

}  // namespace
}  // namespace qnn
