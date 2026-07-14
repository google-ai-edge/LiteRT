// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/spatial_transform_nd_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, BatchToSpaceNdUint8) {
  const std::vector<std::uint32_t> kInputDims{2, 2, 1, 1};
  const std::vector<std::uint32_t> kOutputDims{1, 2, 2, 1};
  const std::vector<std::uint32_t> kBlockDims{2};
  const std::vector<std::uint32_t> kCropsDims{2, 2};
  constexpr std::array<std::int32_t, 2> kBlockData{1, 2};
  constexpr std::array<std::int32_t, 4> kCropsData{0, 0, 0, 0};
  const ::qnn::QuantizeParamsWrapperVariant kQuantParams{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 1.0f, 0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_UFIXED_POINT_8, kQuantParams, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kQuantParams, kOutputDims);
  auto& block_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kBlockDims,
      sizeof(std::int32_t) * kBlockData.size(), kBlockData.data());
  auto& crops_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kCropsDims,
      sizeof(std::int32_t) * kCropsData.size(), kCropsData.data());

  auto ops = ::qnn::BuildBatchToSpaceNdOp(
      tensor_pool_, {input_tensor, block_tensor, crops_tensor},
      {output_tensor});
  ASSERT_FALSE(ops.empty());
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kBatchToSpace);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<std::uint8_t>(input_idx, {1, 2, 3, 4});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::uint8_t>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(), testing::ElementsAre(1, 3, 2, 4));
}

TEST_P(QnnModelTest, SpaceToBatchNdUint8) {
  const std::vector<std::uint32_t> kInputDims{1, 2, 2, 1};
  const std::vector<std::uint32_t> kOutputDims{2, 2, 1, 1};
  const std::vector<std::uint32_t> kBlockDims{2};
  const std::vector<std::uint32_t> kPaddingsDims{2, 2};
  constexpr std::array<std::int32_t, 2> kBlockData{1, 2};
  constexpr std::array<std::int32_t, 4> kPaddingsData{0, 0, 0, 0};
  const ::qnn::QuantizeParamsWrapperVariant kQuantParams{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 1.0f, 0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_UFIXED_POINT_8, kQuantParams, kInputDims);
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_UFIXED_POINT_8, kQuantParams, kOutputDims);
  auto& block_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kBlockDims,
      sizeof(std::int32_t) * kBlockData.size(), kBlockData.data());
  auto& paddings_tensor = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, kPaddingsDims,
      sizeof(std::int32_t) * kPaddingsData.size(), kPaddingsData.data());

  auto ops = ::qnn::BuildSpaceToBatchNdOp(
      tensor_pool_, {input_tensor, block_tensor, paddings_tensor},
      {output_tensor});
  ASSERT_FALSE(ops.empty());
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kSpaceToBatch);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<std::uint8_t>(input_idx, {1, 2, 3, 4});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::uint8_t>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(), testing::ElementsAre(1, 3, 2, 4));
}

}  // namespace
}  // namespace litert::qnn
