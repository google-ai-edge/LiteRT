// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/local_response_norm_op_builder.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, LocalResponseNormFloat32SameAsL2Norm) {
  static constexpr std::int32_t kRadius{20};
  static constexpr float kBias{0.0f};
  static constexpr float kAlpha{1.0f};
  static constexpr float kBeta{0.5f};
  const std::vector<std::uint32_t> kDims{1, 1, 1, 6};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kDims);

  auto ops = ::qnn::BuildLocalResponseNormOp(tensor_pool_, {input_tensor},
                                             {output_tensor}, kRadius, kBias,
                                             kAlpha, kBeta);
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
  qnn_model_.SetInputData<float>(input_idx,
                                 {-1.1f, 0.6f, 0.7f, 1.2f, -0.7f, 0.1f});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(),
              testing::Pointwise(testing::FloatNear(1e-3),
                                 {-0.55f, 0.3f, 0.35f, 0.6f, -0.35f, 0.05f}));
}

}  // namespace
}  // namespace litert::qnn
