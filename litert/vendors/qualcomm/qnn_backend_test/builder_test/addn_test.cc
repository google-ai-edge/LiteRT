// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/addn_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::FloatNear;
using testing::Pointwise;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, AddNThreeInputs) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input0_tensor = tensor_pool_.CreateInputTensorWithName(
      "input0", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& input1_tensor = tensor_pool_.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& input2_tensor = tensor_pool_.CreateInputTensorWithName(
      "input2", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kDims);

  auto ops = ::qnn::BuildAddNOp(tensor_pool_,
                                {input0_tensor, input1_tensor, input2_tensor},
                                {output_tensor});
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kElementWiseBinary);
  EXPECT_EQ(ops[1].GetOpCode(), ::qnn::QnnOpCode::kElementWiseBinary);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input0_idx = qnn_model_.AddInputTensor(input0_tensor);
  auto input1_idx = qnn_model_.AddInputTensor(input1_tensor);
  auto input2_idx = qnn_model_.AddInputTensor(input2_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input0_idx, {-2.0f, 0.2f, 0.7f, 0.8f});
  qnn_model_.SetInputData<float>(input1_idx, {0.1f, 0.2f, 0.3f, 0.5f});
  qnn_model_.SetInputData<float>(input2_idx, {0.5f, 0.1f, 0.1f, 0.2f});
  
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {-1.4f, 0.5f, 1.1f, 1.5f}));
}

TEST_P(QnnModelTest, AddNTwoInputsEmitsSingleAdd) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input0_tensor = tensor_pool_.CreateInputTensorWithName(
      "input0", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& input1_tensor = tensor_pool_.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kDims);

  auto ops = ::qnn::BuildAddNOp(tensor_pool_, {input0_tensor, input1_tensor},
                                {output_tensor});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kElementWiseBinary);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input0_idx = qnn_model_.AddInputTensor(input0_tensor);
  auto input1_idx = qnn_model_.AddInputTensor(input1_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input0_idx, {-2.0f, 0.2f, 0.7f, 0.8f});
  qnn_model_.SetInputData<float>(input1_idx, {0.1f, 0.2f, 0.3f, 0.5f});
  
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {-1.9f, 0.4f, 1.0f, 1.3f}));
}

TEST_P(QnnModelTest, AddNThreeInputsInt32) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input0_tensor = tensor_pool_.CreateInputTensorWithName(
      "input0", QNN_DATATYPE_INT_32, {}, kDims);
  auto& input1_tensor = tensor_pool_.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_INT_32, {}, kDims);
  auto& input2_tensor = tensor_pool_.CreateInputTensorWithName(
      "input2", QNN_DATATYPE_INT_32, {}, kDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_INT_32, {}, kDims);

  auto ops = ::qnn::BuildAddNOp(tensor_pool_,
                                {input0_tensor, input1_tensor, input2_tensor},
                                {output_tensor});
  ASSERT_EQ(ops.size(), 2u);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input0_idx = qnn_model_.AddInputTensor(input0_tensor);
  auto input1_idx = qnn_model_.AddInputTensor(input1_tensor);
  auto input2_idx = qnn_model_.AddInputTensor(input2_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<std::int32_t>(input0_idx, {-20, 2, 7, 8});
  qnn_model_.SetInputData<std::int32_t>(input1_idx, {1, 2, 3, 5});
  qnn_model_.SetInputData<std::int32_t>(input2_idx, {10, -5, 1, -2});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int32_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(), testing::ElementsAre(-9, -1, 11, 11));
}

}  // namespace
}  // namespace litert::qnn
