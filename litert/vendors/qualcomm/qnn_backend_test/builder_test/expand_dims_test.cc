// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/expand_dims_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::FloatNear;
using testing::Pointwise;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, ExpandDimsAxisZeroLoweredToReshape) {
  static constexpr std::int32_t kAxis{0};
  const std::vector<std::uint32_t> kInputDims{2, 3};
  const std::vector<std::uint32_t> kOutputDims{1, 2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);
  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildExpandDimsOp(
      tensor_pool_, {input_tensor, *axis_tensor}, {output_tensor});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kReshape);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx,
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
                                 
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST_P(QnnModelTest, ExpandDimsNegativeAxis) {
  static constexpr std::int32_t kAxis{-1};
  const std::vector<std::uint32_t> kInputDims{2, 3, 4};
  const std::vector<std::uint32_t> kOutputDims{2, 3, 4, 1};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);
  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildExpandDimsOp(
      tensor_pool_, {input_tensor, *axis_tensor}, {output_tensor});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  std::vector<float> input_data(24);
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }
  qnn_model_.SetInputData<float>(input_idx, input_data);

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(), Pointwise(FloatNear(1e-3), input_data));
}

}  // namespace
}  // namespace litert::qnn
