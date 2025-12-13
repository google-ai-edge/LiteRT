// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;

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

  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param_0, kDims, "");
  auto& input_1 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param_1, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param_2, kDims, "");
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
#endif

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
}

TEST_P(QnnModelTest, SingleElementWiseMax) {
  ::qnn::QuantizeParamsWrapperVariant quant_param{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.00015f, 0};

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& input_1 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
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
#endif

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
}
}  // namespace
}  // namespace litert::qnn
