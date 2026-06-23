// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/splitv_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

// Mirrors tflite/kernels/split_v_test.cc TwoDimensional:
//   Input shape: {4, 3}, axis=0, size_splits=[1, 1, 2]
//   Outputs: {1, 3}, {1, 3}, {2, 3}
TEST_P(QnnModelTest, SplitVTwoDimensional) {
  static constexpr std::uint32_t kNumSplits{3};
  static constexpr std::int32_t kAxis{0};
  static constexpr std::array<std::int32_t, 3> kSizeSplitsData{1, 1, 2};
  const std::vector<std::uint32_t> kInputDims{4, 3};
  const std::vector<std::uint32_t> kOutput0Dims{1, 3};
  const std::vector<std::uint32_t> kOutput1Dims{1, 3};
  const std::vector<std::uint32_t> kOutput2Dims{2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output0_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output0", QNN_DATATYPE_FLOAT_32, {}, kOutput0Dims);
  auto& output1_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_FLOAT_32, {}, kOutput1Dims);
  auto& output2_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output2", QNN_DATATYPE_FLOAT_32, {}, kOutput2Dims);

  auto& size_splits_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kNumSplits},
      sizeof(std::int32_t) * kSizeSplitsData.size(), kSizeSplitsData.data());
  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);

  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildSplitVOp(
      tensor_pool_, {input_tensor, size_splits_tensor, *axis_tensor},
      {output0_tensor, output1_tensor, output2_tensor}, kNumSplits);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output0_idx = qnn_model_.AddOutputTensor(output0_tensor);
  auto output1_idx = qnn_model_.AddOutputTensor(output1_tensor);
  auto output2_idx = qnn_model_.AddOutputTensor(output2_tensor);
  qnn_model_.SetInputData<float>(
      input_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output0_data = qnn_model_.GetOutputData<float>(output0_idx);
  ASSERT_TRUE(output0_data);
  ASSERT_THAT(output0_data.value(), ElementsAre(1.0f, 2.0f, 3.0f));

  auto output1_data = qnn_model_.GetOutputData<float>(output1_idx);
  ASSERT_TRUE(output1_data);
  ASSERT_THAT(output1_data.value(), ElementsAre(4.0f, 5.0f, 6.0f));

  auto output2_data = qnn_model_.GetOutputData<float>(output2_idx);
  ASSERT_TRUE(output2_data);
  ASSERT_THAT(output2_data.value(),
              ElementsAre(7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f));
}

// Mirrors tflite/kernels/split_v_test.cc FourDimensional axis=1 with -1:
//   Input shape: {2, 2, 2, 2}, axis=1, size_splits=[1, -1]
//   The -1 entry should be inferred to 1.
//   Outputs: {2, 1, 2, 2}, {2, 1, 2, 2}
TEST_P(QnnModelTest, SplitVWithNegativeOneSizeSplit) {
  static constexpr std::uint32_t kNumSplits{2};
  static constexpr std::int32_t kAxis{1};
  static constexpr std::array<std::int32_t, 2> kSizeSplitsData{1, -1};
  const std::vector<std::uint32_t> kInputDims{2, 2, 2, 2};
  const std::vector<std::uint32_t> kOutputDims{2, 1, 2, 2};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output0_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output0", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  auto& output1_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto& size_splits_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kNumSplits},
      sizeof(std::int32_t) * kSizeSplitsData.size(), kSizeSplitsData.data());
  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);

  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildSplitVOp(
      tensor_pool_, {input_tensor, size_splits_tensor, *axis_tensor},
      {output0_tensor, output1_tensor}, kNumSplits);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output0_idx = qnn_model_.AddOutputTensor(output0_tensor);
  auto output1_idx = qnn_model_.AddOutputTensor(output1_tensor);
  qnn_model_.SetInputData<float>(
      input_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output0_data = qnn_model_.GetOutputData<float>(output0_idx);
  ASSERT_TRUE(output0_data);
  ASSERT_THAT(output0_data.value(),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f));

  auto output1_data = qnn_model_.GetOutputData<float>(output1_idx);
  ASSERT_TRUE(output1_data);
  ASSERT_THAT(output1_data.value(),
              ElementsAre(5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f));
}

// Mirrors tflite/kernels/split_v_test.cc NegativeAxis:
//   Input shape: {2, 2, 2, 2}, axis=-4 (= 0), size_splits=[1, 1]
TEST_P(QnnModelTest, SplitVNegativeAxis) {
  static constexpr std::uint32_t kNumSplits{2};
  static constexpr std::int32_t kAxis{-4};
  static constexpr std::array<std::int32_t, 2> kSizeSplitsData{1, 1};
  const std::vector<std::uint32_t> kInputDims{2, 2, 2, 2};
  const std::vector<std::uint32_t> kOutputDims{1, 2, 2, 2};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output0_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output0", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);
  auto& output1_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto& size_splits_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kNumSplits},
      sizeof(std::int32_t) * kSizeSplitsData.size(), kSizeSplitsData.data());
  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);

  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildSplitVOp(
      tensor_pool_, {input_tensor, size_splits_tensor, *axis_tensor},
      {output0_tensor, output1_tensor}, kNumSplits);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output0_idx = qnn_model_.AddOutputTensor(output0_tensor);
  auto output1_idx = qnn_model_.AddOutputTensor(output1_tensor);
  qnn_model_.SetInputData<float>(
      input_idx, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output0_data = qnn_model_.GetOutputData<float>(output0_idx);
  ASSERT_TRUE(output0_data);
  ASSERT_THAT(output0_data.value(),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f));

  auto output1_data = qnn_model_.GetOutputData<float>(output1_idx);
  ASSERT_TRUE(output1_data);
  ASSERT_THAT(output1_data.value(), ElementsAre(9.0f, 10.0f, 11.0f, 12.0f,
                                                13.0f, 14.0f, 15.0f, 16.0f));
}

// QNN_OP_SPLIT requires at least one cut, so num_splits == 1 (a single output
// identical to the input) is lowered to a no-op reshape instead. Verify the
// builder emits exactly one Reshape op and that it is an identity at runtime.
TEST_P(QnnModelTest, SplitVSingleSplitLoweredToReshape) {
  static constexpr std::uint32_t kNumSplits{1};
  static constexpr std::int32_t kAxis{0};
  static constexpr std::array<std::int32_t, 1> kSizeSplitsData{2};
  const std::vector<std::uint32_t> kInputDims{2, 3};
  const std::vector<std::uint32_t> kOutputDims{2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto& size_splits_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {kNumSplits},
      sizeof(std::int32_t) * kSizeSplitsData.size(), kSizeSplitsData.data());
  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);

  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildSplitVOp(
      tensor_pool_, {input_tensor, size_splits_tensor, *axis_tensor},
      {output_tensor}, kNumSplits);
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kReshape);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx,
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f));
}

// QNN cannot handle dynamic split_index. Verify the builder rejects a
// non-static size_splits tensor (defensive check).
TEST_P(QnnModelTest, SplitVRejectsNonStaticSizeSplits) {
  static constexpr std::uint32_t kNumSplits{3};
  static constexpr std::int32_t kAxis{0};
  const std::vector<std::uint32_t> kInputDims{4, 3};
  const std::vector<std::uint32_t> kOutputDims{1, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& size_splits_tensor = tensor_pool_.CreateInputTensorWithName(
      "size_splits", QNN_DATATYPE_INT_32, {}, {kNumSplits});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto* axis_tensor = tensor_pool_.CreateStaticTensorWithValue(
      QNN_DATATYPE_INT_32, {}, {1}, kAxis);
  ASSERT_NE(axis_tensor, nullptr);

  auto ops = ::qnn::BuildSplitVOp(
      tensor_pool_, {input_tensor, size_splits_tensor, *axis_tensor},
      {output_tensor}, kNumSplits);
  EXPECT_TRUE(ops.empty());
}

}  // namespace
}  // namespace litert::qnn
