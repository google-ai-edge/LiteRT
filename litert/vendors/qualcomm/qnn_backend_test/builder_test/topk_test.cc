// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/topk_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {
INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// QNN ScatterNd:
//  in[0] input: [1, 64, 4, 64]
//  in[1] indices: [1, 1, 2] -> data: [[[0, x]]]
//  in[2] updates: [1, 1, 4, 64]
TEST_P(QnnModelTest, ScatterNd) {
  auto input_quant = ::qnn::ScaleOffsetQuantizeParamsWrapper(1e-4, 0);

  auto& input_tensor = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_8, input_quant, {1, 64, 4, 64}, "");
  auto& indices_tensor = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_INT_32, {}, {1, 1, 2}, "");
  auto& updates_tensor = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_8, input_quant, {1, 1, 4, 64}, "");

  auto& output_tensor = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_8, input_quant, {1, 64, 4, 64}, "");

  std::vector<::qnn::OpWrapper> res;
  ::qnn::OpWrapper& scatter_nd_op = CreateOpWrapper(res, QNN_OP_SCATTER_ND);

  scatter_nd_op.AddInputTensor(input_tensor);
  scatter_nd_op.AddInputTensor(indices_tensor);
  scatter_nd_op.AddInputTensor(updates_tensor);

  scatter_nd_op.AddOutputTensor(output_tensor);

  qnn_model_.MoveOpsToGraph(std::move(res));
  ASSERT_TRUE(qnn_model_.Finalize());
}

}  // namespace
}  // namespace litert::qnn
