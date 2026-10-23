// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {
namespace {

TEST(Conv2dOpBuilderTest, SharedFilterAxisNotCorruptedOnSecondBuild) {
  // Regression test: when the same static filter tensor is passed to
  // BuildConv2dOp() twice (shared weights), the second call must not
  // re-remap the already-remapped axis (3 → 2), which would cause
  // QNN HTP op validation error 3110.
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/0, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper bias_quant{0.25f, 0};
  static constexpr std::array<std::int32_t, 2> bias_data{0, 0};

  // First Conv2d — shares the filter tensor.
  auto& input1 = tensor_pool.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& bias1 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, bias_quant, {2},
      bias_data.size() * sizeof(bias_data[0]), bias_data.data());
  auto& output1 = tensor_pool.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  auto ops1 = ::qnn::BuildConv2dOp(
      tensor_pool, {input1, filter, bias1}, {output1}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid, /*use_int64_bias_as_int32=*/false);

  // Second Conv2d — reuses the same filter tensor.
  auto& input2 = tensor_pool.CreateInputTensorWithName(
      "input2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& bias2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, bias_quant, {2},
      bias_data.size() * sizeof(bias_data[0]), bias_data.data());
  auto& output2 = tensor_pool.CreateOutputTensorWithName(
      "output2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  auto ops2 = ::qnn::BuildConv2dOp(
      tensor_pool, {input2, filter, bias2}, {output2}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid, /*use_int64_bias_as_int32=*/false);

  // Both transposed filters must have axis=3, not 2.
  ASSERT_EQ(ops1.size(), 1);
  const auto& quant1 = std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
      ops1[0].GetInputTensor(1).GetQuantParams());
  EXPECT_EQ(quant1.GetAxis(), 3);

  ASSERT_EQ(ops2.size(), 1);
  const auto& quant2 = std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
      ops2[0].GetInputTensor(1).GetQuantParams());
  EXPECT_EQ(quant2.GetAxis(), 3);
}

TEST(Conv2dOpBuilderTest, RemapsBwAxisQuantizationAfterFilterTranspose) {
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  auto& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});

  const ::qnn::BwAxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*bitwidth=*/4, /*axis=*/0, /*scales=*/{0.25f, 0.5f},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper bias_quant{0.25f, 0};
  static constexpr std::array<std::int32_t, 2> bias_data{0, 0};
  auto& bias = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, bias_quant, {2},
      bias_data.size() * sizeof(bias_data[0]), bias_data.data());
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});

  auto ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 1);
  const auto& transposed_filter = ops[0].GetInputTensor(1);
  EXPECT_EQ(transposed_filter.GetDimensions(),
            (std::vector<std::uint32_t>{1, 1, 1, 2}));
  const auto& quant_params =
      std::get<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
          transposed_filter.GetQuantParams());
  EXPECT_EQ(quant_params.GetAxis(), 3);
  EXPECT_EQ(quant_params.GetBitwidth(), 4);
}

}  // namespace
}  // namespace litert::qnn
