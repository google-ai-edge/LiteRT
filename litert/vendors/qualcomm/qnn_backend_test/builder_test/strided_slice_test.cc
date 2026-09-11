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

// Input shape: {1, 12}, begin={0, 8}, size={1, 4}
// Output shape: {1, 4}
TEST_P(QnnModelTest, StridedSliceBoolInputOutput) {
  static constexpr std::array<std::uint32_t, 2> kInputDims{1, 12};
  static constexpr std::array<std::uint32_t, 2> kOutputDims{1, 4};
  static constexpr std::array<std::int32_t, 2> kBeginData{0, 8};
  static constexpr std::array<std::int32_t, 2> kSizeData{1, 4};
  static constexpr std::array<std::int32_t, 2> kStridesData{1, 1};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_BOOL_8, {}, {kInputDims.begin(), kInputDims.end()});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_BOOL_8, {},
      {kOutputDims.begin(), kOutputDims.end()});
  auto& begin_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kBeginData.size()},
      sizeof(kBeginData[0]) * kBeginData.size(), kBeginData.data());
  auto& size_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kSizeData.size()},
      sizeof(kSizeData[0]) * kSizeData.size(), kSizeData.data());
  auto& strides_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kStridesData.size()},
      sizeof(kStridesData[0]) * kStridesData.size(), kStridesData.data());

  auto ops = ::qnn::BuildStridedSliceOp(
      tensor_pool_, {input_tensor, begin_tensor, size_tensor, strides_tensor},
      {output_tensor}, /*begin_mask=*/0, /*end_mask=*/0, /*ellipsis_mask=*/0,
      /*shrink_axis_mask=*/0, /*new_axis_mask=*/0, /*offset=*/true);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<bool>(
      input_idx, {true, false, true, false, true, false, true, false, true,
                  false, true, false});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(), ElementsAre(true, false, true, false));
}

}  // namespace
}  // namespace litert::qnn
