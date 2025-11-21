// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/topk_op_builder.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnTypes.h"  // from @qairt

using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {
// TODO (chunhsue-qti): The following param tests will be exactly the same when
// running on arm64 device since qnn manager will use online SoC info
// automatically. Maybe we can find out a way to skip them.
INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, SingleTopK) {
  const std::vector<std::uint32_t> inputDims{1, 5};
  const uint32_t k_value = 3;
  const std::vector<std::uint32_t> outputDims{1, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, inputDims, "");
  auto& values_tensor = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, outputDims, "");
  auto& indices_tensor = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_UINT_32, {}, outputDims, "");

  auto ops = ::qnn::BuildTopKOp(tensor_pool_, {input_tensor},
                                {values_tensor, indices_tensor}, k_value);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto values_idx = qnn_model_.AddOutputTensor(values_tensor);
  auto indices_idx = qnn_model_.AddOutputTensor(indices_tensor);

  qnn_model_.SetInputData<float>(input_idx, {1.2f, 5.6f, 3.3f, 9.8f, 2.1f});
  ASSERT_TRUE(qnn_model_.Execute());
  auto values_data = qnn_model_.GetOutputData<float>(values_idx);
  auto indices_data = qnn_model_.GetOutputData<std::uint32_t>(indices_idx);

  ASSERT_TRUE(values_data);
  ASSERT_TRUE(indices_data);
  ASSERT_EQ(values_data->size(), 3);
  ASSERT_EQ(indices_data->size(), 3);
  ASSERT_THAT(values_data.value(),
              Pointwise(FloatNear(1e-2), {9.8f, 5.6f, 3.3f}));
  ASSERT_THAT(indices_data.value(), ElementsAre(3, 1, 2));
}

}  // namespace
}  // namespace litert::qnn
