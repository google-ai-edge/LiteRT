// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

using testing::Pointwise;
namespace litert::qnn {
namespace {
INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, FullyConnectedInt2Sanity) {
  constexpr float kScale = 0.001;

  auto input_quant = ::qnn::ScaleOffsetQuantizeParamsWrapper(kScale, 0);
  auto output_quant = ::qnn::ScaleOffsetQuantizeParamsWrapper(kScale, 0);
  const std::vector<std::uint32_t> kInDims{1, 2};
  const std::vector<std::uint32_t> kOutDims{1, 2};

  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, input_quant, kInDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, output_quant, kOutDims, "");

  const std::vector<std::uint32_t> kFilterDims{2, 2};
  auto weight_quant_param =
      ::qnn::BwScaleOffsetQuantizeParamsWrapper(2, kScale, 0);

  std::vector<int8_t> weight_data = {1, -1, -1, 1};
  std::vector<int8_t> int2_weight_data;
  ::qnn::ConvertDataFromInt8ToInt2(weight_data, int2_weight_data);
  auto& weight_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant_param, kFilterDims,
      int2_weight_data.size(), int2_weight_data.data());

  auto ops = ::qnn::BuildFullyConnectedOp(
      tensor_pool_, {input_0, weight_tensor}, {output_0}, true);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  std::vector<int16_t> in_data = {1000, 0};
  qnn_model_.SetInputData<int16_t>(input_idx, in_data);

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 2);
  ASSERT_THAT(
      output_data.value(),
      Pointwise(testing::Eq(), {::qnn::Quantize<int16_t>(0.001, kScale, 0),
                                ::qnn::Quantize<int16_t>(-0.001, kScale, 0)}));
}

}  // namespace
}  // namespace litert::qnn
