// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {
namespace {
using testing::ElementsAre;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Returns the value of the QNN Pack op's "axis" scalar param, or std::nullopt
// if the op is not a Pack with an axis param.
std::optional<std::uint32_t> GetPackAxisParam(const ::qnn::OpWrapper& op) {
  const auto scalar_param = op.GetScalarParam(0);
  if (!scalar_param.has_value()) return std::nullopt;
  Qnn_Param_t qnn_param = QNN_PARAM_INIT;
  scalar_param->CloneTo(qnn_param);
  if (qnn_param.name == nullptr ||
      std::string_view(qnn_param.name) != QNN_OP_PACK_PARAM_AXIS) {
    return std::nullopt;
  }
  return qnn_param.scalarParam.uint32Value;
}

// Regression test for negative-axis legalization.
// Two rank-4 inputs packed along axis=-2 produce a rank-5 output. TFLite PACK
// axis is relative to the output rank (input + 1), so axis=-2 must resolve to
// output index 3 (where the stacked dim of size 2 lives).
TEST_P(QnnModelTest, PackNegativeAxisResolvesToOutputRank) {
  const std::vector<std::uint32_t> kInputDims{1, 4, 3, 5};   // rank 4
  const std::vector<std::uint32_t> kOutputDims{1, 4, 3, 2, 5};  // rank 5
  static constexpr int32_t kAxis{-2};
  // -2 normalized against the rank-5 output: -2 + 5 = 3.
  static constexpr std::uint32_t kExpectedAxis{3};

  auto& input0_tensor = tensor_pool_.CreateInputTensorWithName(
      "input0", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& input1_tensor = tensor_pool_.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto ops = ::qnn::BuildPackOp(tensor_pool_, {input0_tensor, input1_tensor},
                                {output_tensor}, kAxis);
  ASSERT_FALSE(ops.empty());

  // The stacked dimension (size 2) lives at output index 3.
  const auto axis_param = GetPackAxisParam(ops.front());
  ASSERT_TRUE(axis_param.has_value());
  EXPECT_EQ(*axis_param, kExpectedAxis);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());
}

// A positive axis is passed through unchanged.
TEST_P(QnnModelTest, PackPositiveAxisIsUnchanged) {
  const std::vector<std::uint32_t> kInputDims{2, 3};      // rank 2
  const std::vector<std::uint32_t> kOutputDims{2, 2, 3};  // rank 3
  static constexpr int32_t kAxis{1};

  auto& input0_tensor = tensor_pool_.CreateInputTensorWithName(
      "input0", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& input1_tensor = tensor_pool_.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto ops = ::qnn::BuildPackOp(tensor_pool_, {input0_tensor, input1_tensor},
                                {output_tensor}, kAxis);
  ASSERT_FALSE(ops.empty());
  const auto axis_param = GetPackAxisParam(ops.front());
  ASSERT_TRUE(axis_param.has_value());
  EXPECT_EQ(*axis_param, static_cast<std::uint32_t>(kAxis));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input0_idx = qnn_model_.AddInputTensor(input0_tensor);
  auto input1_idx = qnn_model_.AddInputTensor(input1_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input0_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                              6.0f});
  qnn_model_.SetInputData<float>(input1_idx, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                              12.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  // Stacked along axis 1: [[in0_row0, in1_row0], [in0_row1, in1_row1]].
  ASSERT_THAT(output_data.value(),
              ElementsAre(1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 4.0f, 5.0f, 6.0f,
                          10.0f, 11.0f, 12.0f));
}

// A single input cannot form a valid QNN Pack, so the builder emits a Reshape
// instead.
TEST_P(QnnModelTest, PackSingleInputBecomesReshape) {
  const std::vector<std::uint32_t> kInputDims{2, 3};
  const std::vector<std::uint32_t> kOutputDims{1, 2, 3};
  static constexpr int32_t kAxis{0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto ops = ::qnn::BuildPackOp(tensor_pool_, {input_tensor}, {output_tensor},
                                /*axis=*/kAxis);
  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops.front().IsOpCode(::qnn::QnnOpCode::kReshape));
}

}  // namespace
}  // namespace litert::qnn
