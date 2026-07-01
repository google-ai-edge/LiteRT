// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/spatial_transform_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {
namespace {
using ::testing::ElementsAreArray;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, QuantizedDepthToSpaceValidatesWithQnn) {
  const ::qnn::QuantizeParamsWrapperVariant quant_params{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.00467028f,
      87};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in", QNN_DATATYPE_SFIXED_POINT_8, quant_params, {1, 1, 1, 9});
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out", QNN_DATATYPE_SFIXED_POINT_8, quant_params, {1, 3, 3, 1});

  auto ops = ::qnn::BuildDepthToSpaceOp(tensor_pool_, {input}, {output},
                                        /*block_size=*/3);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  EXPECT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  const std::vector<std::int8_t> input_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  qnn_model_.SetInputData<std::int8_t>(input_idx, input_data);

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(), ElementsAreArray(input_data));
}

TEST_P(QnnModelTest, QuantizedSpaceToDepthValidatesWithQnn) {
  const ::qnn::QuantizeParamsWrapperVariant quant_params{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.00467028f,
      87};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in", QNN_DATATYPE_SFIXED_POINT_8, quant_params, {1, 3, 3, 1});
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out", QNN_DATATYPE_SFIXED_POINT_8, quant_params, {1, 1, 1, 9});

  auto ops = ::qnn::BuildSpaceToDepthOp(tensor_pool_, {input}, {output},
                                        /*block_size=*/3);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  EXPECT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  const std::vector<std::int8_t> input_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  qnn_model_.SetInputData<std::int8_t>(input_idx, input_data);

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(), ElementsAreArray(input_data));
}

}  // namespace
}  // namespace litert::qnn
