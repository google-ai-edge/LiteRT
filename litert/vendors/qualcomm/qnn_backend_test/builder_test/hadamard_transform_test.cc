// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/hadamard_transform_op_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

using ::testing::ElementsAreArray;

namespace litert::qnn {
namespace {
INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, HadamardTransform_INT16) {
  constexpr std::uint32_t kDim = 256U;
  const std::vector<std::uint32_t> kIODims{1, 1, 1, kDim};
  const std::vector<std::uint32_t> kWeightDims{kDim, kDim};
  const ::qnn::QuantizeParamsWrapperVariant quant_param =
      ::qnn::ScaleOffsetQuantizeParamsWrapper(1.0, 0);
  constexpr std::int8_t kHadamardValue = 7;
  constexpr float kWeightScale = 1.0f / 16 / kHadamardValue;
  const ::qnn::QuantizeParamsWrapperVariant weight_quant_param =
      ::qnn::ScaleOffsetQuantizeParamsWrapper(kWeightScale, 0);
  std::array<std::int8_t, kDim * kDim> hadamard_matrix;
  // Create static weight by Sylvester's construction.
  for (std::uint32_t i = 0; i < kWeightDims[0]; ++i) {
    for (std::uint32_t j = 0; j < kWeightDims[1]; ++j) {
      int bits = absl::popcount(i & j);
      hadamard_matrix[i * 256 + j] =
          ((bits & 1) == 0) ? +kHadamardValue : -kHadamardValue;
    }
  }

  auto& input = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kIODims, "");
  auto& weight = tensor_pool_.CreateStaticTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant_param, kWeightDims, "",
      hadamard_matrix.size() * sizeof(hadamard_matrix[0]),
      hadamard_matrix.data(), false);
  auto& output = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kIODims, "");

  auto scale = ::qnn::GetSylvesterHadamardScale(weight);
  ASSERT_EQ(scale.value_or(0.0f), 1.0f);

  auto ops = ::qnn::BuildHadamardTransformOp(tensor_pool_, {input}, {output},
                                             scale.value());
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO(jiunkaiy): Uncomment the following ValidateOpConfig after HTP supports
  // SINT16 HadamardTransform. ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);

  // Set input data as a vector of 256 ones, matching the first row of the
  // Hadamard matrix.
  qnn_model_.SetInputData<int16_t>(input_idx, std::vector<int16_t>(256, 1));
  // Execute the model.
  ASSERT_TRUE(qnn_model_.Execute());
  // Get output.
  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 256);
  // All expected values are set to zero, except for the first element, as the
  // rows in a Hadamard matrix are mutually orthogonal, and the input vector is
  // the first row. The first element is n times the normalization factor
  // (1/sqrt(n)) for an n*n Hadamard matrix.
  std::vector<int16_t> expected(256, 0);
  expected[0] = 16;
  ASSERT_THAT(output_data.value(), ElementsAreArray(expected));
}

}  // namespace
}  // namespace litert::qnn
