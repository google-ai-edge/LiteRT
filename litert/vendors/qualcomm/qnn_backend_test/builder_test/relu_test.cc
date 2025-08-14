// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

TEST_F(QnnModelTest, SingleRelu) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto ops = ::qnn::BuildReluOp(tensor_pool_, {input_0}, {output_0});
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

  qnn_model_.SetInputData<float>(input_idx, {-1.f, 0.f, 1.f, 2.f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), Pointwise(FloatNear(1e-3), {0, 0, 1, 2}));
}

}  // namespace
}  // namespace litert::qnn
