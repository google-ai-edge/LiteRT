// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/strided_slice_op_builder.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, StridedSliceBoolInputOutput) {
  static constexpr std::array<std::uint32_t, 2> kInputDims{1, 12};
  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_BOOL_8, {}, {kInputDims.begin(), kInputDims.end()});

  static constexpr std::array<std::uint32_t, 2> kOutputDims{1, 4};
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_BOOL_8, {},
      {kOutputDims.begin(), kOutputDims.end()});

  static constexpr std::array<std::int32_t, 2> kBeginData{0, 8};
  auto& begin_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kBeginData.size()},
      sizeof(kBeginData[0]) * kBeginData.size(), kBeginData.data());

  static constexpr std::array<std::int32_t, 2> kEndData{1, 12};
  auto& end_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kEndData.size()},
      sizeof(kEndData[0]) * kEndData.size(), kEndData.data());

  static constexpr std::array<std::int32_t, 2> kStridesData{1, 1};
  auto& strides_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kStridesData.size()},
      sizeof(kStridesData[0]) * kStridesData.size(), kStridesData.data());

  auto ops = ::qnn::BuildStridedSliceOp(
      tensor_pool_, {input_tensor, begin_tensor, end_tensor, strides_tensor},
      {output_tensor}, /*begin_mask=*/0, /*end_mask=*/0, /*ellipsis_mask=*/0,
      /*shrink_axis_mask=*/0, /*new_axis_mask=*/0, /*offset=*/false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  const auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  const auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<bool>(
      input_idx, {true, false, true, false, true, false, true, false, true,
                  false, true, false});

  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(), ElementsAre(true, false, true, false));
}

}  // namespace
}  // namespace litert::qnn
