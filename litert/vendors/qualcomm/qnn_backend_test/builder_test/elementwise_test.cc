// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;       // NOLINT
using testing::ElementsAreArray;  // NOLINT
using testing::FloatNear;         // NOLINT
using testing::Pointwise;         // NOLINT

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, SingleElementWiseDivide) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  ::qnn::QuantizeParamsWrapperVariant quant_param_0{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000031f,
      0};
  ::qnn::QuantizeParamsWrapperVariant quant_param_1{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000101f,
      0};
  ::qnn::QuantizeParamsWrapperVariant quant_param_2{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000030f,
      0};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param_0, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_16, quant_param_1, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param_2, kDims);
  auto ops = ::qnn::BuildElementwiseDivOp(tensor_pool_, {input_0, input_1},
                                          {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO (chunhsue-qti): Uncomment the line below when QAIRT 2.42 releases.
  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto input_idx1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx, {1, 1, 1, 1});
  qnn_model_.SetInputData<int16_t>(input_idx1, {1, 1, 1, 1});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  const float output_scale =
      std::get<::qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param_2)
          .GetScale();
  const std::int32_t output_zero_point =
      std::get<::qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param_2)
          .GetZeroPoint();
  std::vector<float> dequant_output;
  ::qnn::DequantizeInto(output_data.value(), output_scale, output_zero_point,
                        dequant_output);
  ASSERT_THAT(dequant_output,
              Pointwise(FloatNear(1e-2), {0.306f, 0.306f, 0.306f, 0.306f}));
#endif
}

TEST_P(QnnModelTest, SingleElementWiseMax) {
  ::qnn::QuantizeParamsWrapperVariant quant_param{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.00015f, 0};

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto ops = ::qnn::BuildElementwiseMaximumOp(tensor_pool_, {input_0, input_1},
                                              {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO (chunhsue-qti): Uncomment the line below when QAIRT 2.42 releases.
  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx_0, {-20000, 0, 10000, 20000});
  qnn_model_.SetInputData<int16_t>(input_idx_1,
                                   {-17204, -17204, -17204, -17204});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  // Only check quant value in this test since this op is only a data mover.
  ASSERT_THAT(output_data.value(), ElementsAre(-17204, 0, 10000, 20000));
#endif
}

TEST_P(QnnModelTest, SingleElementWiseBinaryMulAsymmetricQuant) {
  const std::vector<std::uint32_t> kDims{1, 1, 1, 4};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.005f,
          -97},
      kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.09f,
          8},
      kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.06f,
          -68},
      kDims);

  qnn_model_.MoveOpToGraph(
      ::qnn::CreateElementWiseMulOp(input_0, input_1, output_0));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int8_t>(input_idx_0, {-100, -50, 0, 50});
  qnn_model_.SetInputData<int8_t>(input_idx_1, {-100, -50, 0, 50});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ElementsAre(-66, -88, -74, -22));
#endif
}

// FloorMod follows the sign of the divisor.
TEST_P(QnnModelTest, FloorModNegativeValue) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_INT_32, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto ops = ::qnn::BuildElementwiseFloorModOp(tensor_pool_, {input_0, input_1},
                                               {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int32_t>(input_idx_0, {10, -9, -11, 7});
  qnn_model_.SetInputData<int32_t>(input_idx_1, {2, 2, -3, -4});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int32_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ElementsAre(0, 1, -2, -1));
#endif
}

// HTP only supports QNN_OP_ELEMENT_WISE_MOD on INT32
TEST_P(QnnModelTest, FloorModRejectsNonInt32) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, {}, kDims);

  auto ops = ::qnn::BuildElementwiseFloorModOp(tensor_pool_, {input_0, input_1},
                                               {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_FALSE(qnn_model_.ValidateOpConfig());
}

TEST_P(QnnModelTest, SignInt32Rank3) {
  const std::vector<std::uint32_t> kDims{1, 2520, 2};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto ops = ::qnn::BuildElementwiseSignOp(tensor_pool_, {input_0}, {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  const std::size_t kNumElements = 1 * 2520 * 2;
  std::vector<std::int32_t> in_data(kNumElements);
  std::vector<std::int32_t> expected(kNumElements);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    const std::int32_t sign = static_cast<std::int32_t>(i % 3) - 1;  // -1,0,1
    in_data[i] = sign * static_cast<std::int32_t>(i + 1);
    expected[i] = sign;
  }

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);
  qnn_model_.SetInputData<std::int32_t>(input_idx, in_data);

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int32_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), kNumElements);
  ASSERT_THAT(output_data.value(), ElementsAreArray(expected));
#endif
}

TEST_P(QnnModelTest, ElementwiseAnd5DBool) {
  const std::vector<std::uint32_t> kDims{1, 1, 17, 12, 24};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_BOOL_8, {}, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_BOOL_8, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_BOOL_8, {}, kDims);
  auto ops = ::qnn::BuildElementwiseAndOp(tensor_pool_, {input_0, input_1},
                                          {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());
}

TEST_P(QnnModelTest, ElementwiseOr5DBoolBroadcast) {
  const std::vector<std::uint32_t> kInDims{1, 1, 17, 1, 24};
  const std::vector<std::uint32_t> kOutDims{1, 1, 17, 12, 24};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_BOOL_8, {}, kInDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_BOOL_8, {}, kOutDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_BOOL_8, {}, kOutDims);
  auto ops = ::qnn::BuildElementwiseOrOp(tensor_pool_, {input_0, input_1},
                                         {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());
}
}  // namespace
}  // namespace litert::qnn
