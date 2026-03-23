// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/onehot_op_builder.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {
namespace {
using testing::ElementsAre;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, SingleOneHot) {
  const std::vector<std::uint32_t> kIndicesDims{1, 2};
  const std::vector<std::uint32_t> kOutputDims{1, 2, 3};

  auto& indices_tensor = tensor_pool_.CreateInputTensorWithName(
      "indices", QNN_DATATYPE_INT_32, {}, kIndicesDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto* depth_tensor =
      tensor_pool_.CreateStaticTensorWithValue(QNN_DATATYPE_INT_32, {}, {1}, 3);
  auto* on_value_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, 1.0f);
  auto* off_value_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_FLOAT_32, {}, {1}, 0.0f);

  ASSERT_NE(depth_tensor, nullptr);
  ASSERT_NE(on_value_tensor, nullptr);
  ASSERT_NE(off_value_tensor, nullptr);

  auto ops = ::qnn::BuildOneHotOp(
      tensor_pool_,
      {indices_tensor, *depth_tensor, *on_value_tensor, *off_value_tensor},
      {output_tensor}, -1);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(indices_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<std::int32_t>(input_idx, {0, 2});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 6);
  ASSERT_THAT(output_data.value(),
              ElementsAre(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f));
}

}  // namespace
}  // namespace litert::qnn
