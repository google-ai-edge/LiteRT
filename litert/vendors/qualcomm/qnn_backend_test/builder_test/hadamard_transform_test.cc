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
  const std::vector<std::uint32_t> kDims{1, 1, 1, 64};
  ::qnn::QuantizeParamsWrapperVariant quant_param =
      ::qnn::ScaleOffsetQuantizeParamsWrapper(1.0, 0);
  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto ops =
      ::qnn::BuildHadamardTransformOp(tensor_pool_, {input_0}, {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO(jiunkaiy): Uncomment the following ValidateOpConfig after HTP supports SINT16 HadamardTransform.
//   ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  // Set input data (vector of 64 ones).
  qnn_model_.SetInputData<int16_t>(input_idx, std::vector<int16_t>(64, 1));
  // Execute the model.
  ASSERT_TRUE(qnn_model_.Execute());
  // Get output.
  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 64);
  // Output is expected to be the matrix after the "normalized" HadamardTransform.
  std::vector<int16_t> expected(64, 0);
  expected[0] = 8; // Equal to 64 * 1/sqrt(64)
  ASSERT_THAT(output_data.value(), ElementsAreArray(expected));
}

}  // namespace
}  // namespace litert::qnn
