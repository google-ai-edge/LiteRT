// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnTypes.h" // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_conv3d_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::FloatNear;
using testing::Pointwise;

constexpr std::size_t kOpFilterIndex = 1;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, TransposeConv3dSimpleFloat32) {
  const std::vector<std::uint32_t> kInputDims{1, 2, 2, 4, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 2, 2, 2, 2};
  const std::vector<std::uint32_t> kOutputDims{1, 3, 3, 5, 2};
  const std::vector<float> kFilterData{-1, -1, -1, -1, -1, 1, -1, 1,  -1, 1, 1,
                                       1,  1,  1,  -1, -1, 1, -1, 1,  1,  1, 1,
                                       -1, 1,  -1, -1, -1, 1, 1,  -1, 1,  -1};
  std::vector<float> input_data(32);
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  auto &input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto &filter_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{}, kFilterDims,
      sizeof(float) * kFilterData.size(), kFilterData.data());
  auto &output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  const std::vector<std::int32_t> kShapeData{1, 3, 3, 5, 2};
  auto &output_shape_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {5},
      sizeof(std::int32_t) * kShapeData.size(), kShapeData.data());

  auto ops = ::qnn::BuildTransposeConv3dOp(
      tensor_pool_, {output_shape_tensor, filter_tensor, input_tensor},
      {output_tensor}, /*stride_d=*/1, /*stride_h=*/1, /*stride_w=*/1,
      /*dilation_d=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kTransposeConv3d);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx, input_data);

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(
      output_data.value(),
      Pointwise(FloatNear(1e-3),
                {-1,  -1,  -4,  -4,  -8,  -8,  -12, -12, 1,   1,   -16, -16,
                 -18, -16, -18, -20, -18, -24, 14,  -12, 1,   17,  18,  4,
                 22,  4,   26,  4,   29,  -29, -34, -32, -36, -30, -36, -30,
                 -36, -30, 14,  2,   -50, 2,   -8,  -26, -8,  -26, -8,  -26,
                 74,  -44, -16, 50,  28,  4,   28,  4,   28,  4,   60,  -62,
                 -1,  33,  32,  38,  36,  42,  40,  46,  45,  1,   -34, 50,
                 10,  54,  10,  58,  10,  62,  60,  0,   -49, 1,   -54, 0,
                 -58, 0,   -62, 0,   -1,  -1}));
}

TEST_P(QnnModelTest, TransposeConv3dPaddingSame) {
  const std::vector<std::uint32_t> kInputDims{1, 3, 4, 5, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 2, 2, 2, 2};
  const std::vector<std::uint32_t> kOutputDims{1, 3, 4, 5, 2};
  const std::vector<float> kFilterData{
      1,  -1, 1,  -1, 1,  -1, -1, 1, 1, -1, -1, 1, 1,  -1, -1, 1,
      -1, 1,  -1, 1,  -1, -1, -1, 1, 1, 1,  1,  1, -1, 1,  -1, 1};
  std::vector<float> input_data(120);
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  auto &input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto &filter_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{}, kFilterDims,
      sizeof(float) * kFilterData.size(), kFilterData.data());
  auto &output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  const std::vector<std::int32_t> kShapeData{1, 3, 4, 5, 2};
  auto &output_shape_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {5},
      sizeof(std::int32_t) * kShapeData.size(), kShapeData.data());

  auto ops = ::qnn::BuildTransposeConv3dOp(
      tensor_pool_, {output_shape_tensor, filter_tensor, input_tensor},
      {output_tensor}, /*stride_d=*/1, /*stride_h=*/1, /*stride_w=*/1,
      /*dilation_d=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Same);
  ASSERT_EQ(ops.size(), 1u);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx, input_data);

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(
      output_data.value(),
      Pointwise(FloatNear(1e-3),
                {-1,  -1,  -2,  0,   -2,  0,   -2,  0,   -2,  0,   -2,  0,
                 -4,  2,   -4,  2,   -4,  2,   -4,  2,   -2,  0,   -4,  2,
                 -4,  2,   -4,  2,   -4,  2,   -2,  0,   -4,  2,   -4,  2,
                 -4,  2,   -4,  2,   0,   0,   -2,  2,   -6,  2,   -10, 2,
                 -14, 2,   0,   2,   -18, 10,  -18, 14,  -18, 18,  -18, 22,
                 20,  22,  -18, 30,  -18, 34,  -18, 38,  -18, 42,  40,  42,
                 -18, 50,  -18, 54,  -18, 58,  -18, 62,  0,   0,   -82, 2,
                 -86, 2,   -90, 2,   -94, 2,   80,  82,  -18, 90,  -18, 94,
                 -18, 98,  -18, 102, 100, 102, -18, 110, -18, 114, -18, 118,
                 -18, 122, 120, 122, -18, 130, -18, 134, -18, 138, -18, 142}));
}

TEST_P(QnnModelTest, TransposeConv3dDilation) {
  const std::vector<std::uint32_t> kInputDims{1, 3, 1, 1, 1};
  const std::vector<std::uint32_t> kFilterDims{1, 2, 2, 2, 1};
  const std::vector<std::uint32_t> kOutputDims{1, 3, 3, 2, 2};
  const std::vector<float> kFilterData{1, -1, 1, 1, -1, 1, 1, -1};
  const std::vector<float> kInputData{0, 1, 2};

  auto &input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto &filter_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{}, kFilterDims,
      sizeof(float) * kFilterData.size(), kFilterData.data());
  auto &output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  const std::vector<std::int32_t> kShapeData{1, 3, 3, 2, 2};
  auto &output_shape_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {5},
      sizeof(std::int32_t) * kShapeData.size(), kShapeData.data());

  auto ops = ::qnn::BuildTransposeConv3dOp(
      tensor_pool_, {output_shape_tensor, filter_tensor, input_tensor},
      {output_tensor}, /*stride_d=*/1, /*stride_h=*/1, /*stride_w=*/1,
      /*dilation_d=*/1, /*dilation_h=*/2, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid);
  ASSERT_EQ(ops.size(), 1u);
  ASSERT_GT(ops[0].GetInputCount(), kOpFilterIndex);
  EXPECT_EQ(ops[0].GetInputTensor(kOpFilterIndex).GetDimensions(),
            std::vector<std::uint32_t>({1, 3, 2, 1, 2}));
  auto dilated_filter_data =
      ops[0].GetInputTensor(kOpFilterIndex).GetTensorData<float>();
  ASSERT_TRUE(dilated_filter_data);
  EXPECT_THAT(dilated_filter_data.value(),
              Pointwise(FloatNear(1e-3),
                        {1, -1, 1, 1, 0, 0, 0, 0, -1, 1, 1, -1}));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx, kInputData);

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(
      output_data.value(),
      Pointwise(FloatNear(1e-3),
                {0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 1, -1, 1,  1, 0, 0,
                 0, 0, -1, 1, 1, -1, 2, -2, 2, 2, 0, 0, 0, 0,  -2, 2, 2, -2}));
}

} // namespace
} // namespace litert::qnn
