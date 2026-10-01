// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/strided_slice_op_builder.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, StridedSliceBoolRank3) {
  const std::vector<std::uint32_t> kInDims{20, 1, 12};
  const std::vector<std::uint32_t> kOutDims{17, 1, 12};
  const std::vector<std::uint32_t> kParamDims{3};
  const std::vector<std::int32_t> kBegin{0, 0, 0};
  const std::vector<std::int32_t> kEnd{17, 1, 12};
  const std::vector<std::int32_t> kStrides{1, 1, 1};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_BOOL_8, {}, kInDims);
  auto& begin = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kParamDims,
      sizeof(std::int32_t) * kBegin.size(), kBegin.data());
  auto& end = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kParamDims, sizeof(std::int32_t) * kEnd.size(),
      kEnd.data());
  auto& strides = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kParamDims,
      sizeof(std::int32_t) * kStrides.size(), kStrides.data());
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_BOOL_8, {}, kOutDims);

  auto ops = ::qnn::BuildStridedSliceOp(
      tensor_pool_, {input_0, begin, end, strides}, {output_0},
      /*begin_mask=*/0, /*end_mask=*/0, /*ellipsis_mask=*/0,
      /*shrink_axis_mask=*/0, /*new_axis_mask=*/0, /*offset=*/false);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  constexpr std::size_t kRow = 1 * 12;
  std::array<bool, 20 * kRow> in_data{};
  std::array<bool, 17 * kRow> expected{};
  for (std::size_t i = 0; i < in_data.size(); ++i) {
    in_data[i] = (i % 2) != 0;  // alternating true/false
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = in_data[i];  // first 17 rows are a contiguous prefix
  }

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);
  qnn_model_.SetInputData<bool>(input_idx, absl::MakeConstSpan(in_data));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), expected.size());
  ASSERT_THAT(output_data.value(), ::testing::ElementsAreArray(expected));
#endif
}
}  // namespace
}  // namespace litert::qnn
