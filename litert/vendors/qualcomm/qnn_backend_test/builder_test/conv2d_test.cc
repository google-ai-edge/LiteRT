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
  // Verify filter quant params are unchanged after calling BuildConv2dOp().
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/0, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  // First Conv2d — shares the filter tensor.
  auto& input1 = tensor_pool.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output1 = tensor_pool.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input1, filter}, {output1},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

  // Second Conv2d — reuses the same filter tensor.
  auto& input2 = tensor_pool.CreateInputTensorWithName(
      "input2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output2 = tensor_pool.CreateOutputTensorWithName(
      "output2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input2, filter}, {output2},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

  // Filter axis must remain 0 (OHWI) after both builds — not remapped in place.
  const auto& orig_quant =
      std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
          filter.GetQuantParams());
  EXPECT_EQ(orig_quant.GetAxis(), 0);
}

TEST(Conv2dOpBuilderTest,
     SharedFilterBwAxisNotCorruptedOnSecondBuild) {
  // Verify filter quant params are unchanged after calling BuildConv2dOp()
  // with BwAxisScaleOffsetQuantizeParamsWrapper.
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  const ::qnn::BwAxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*bitwidth=*/4, /*axis=*/0, /*scales=*/{0.1f, 0.2f},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  // First Conv2d — shares the filter tensor.
  auto& input1 = tensor_pool.CreateInputTensorWithName(
      "input1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output1 = tensor_pool.CreateOutputTensorWithName(
      "output1", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input1, filter}, {output1},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

  // Second Conv2d — reuses the same filter tensor.
  auto& input2 = tensor_pool.CreateInputTensorWithName(
      "input2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output2 = tensor_pool.CreateOutputTensorWithName(
      "output2", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input2, filter}, {output2},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

  // Filter axis must remain 0 (OHWI) after both builds — not remapped in place.
  const auto& orig_quant =
      std::get<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
          filter.GetQuantParams());
  EXPECT_EQ(orig_quant.GetAxis(), 0);
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

TEST(Conv2dOpBuilderTest,
     NativeFilterTransposeOpOutputAxisRemappedAndOriginalUntouched) {
  // Verify filter quant params are unchanged and Transpose op output axis is
  // remapped when BuildConv2dOp() is called with a non-static filter.
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/0, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  // Non-static filter triggers the Transpose op path.
  auto& filter = tensor_pool.CreateInputTensorWithName(
      "filter", QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1});

  const ::qnn::ScaleOffsetQuantizeParamsWrapper bias_quant{0.25f, 0};
  static constexpr std::array<std::int32_t, 2> bias_data{0, 0};

  // First Conv2d.
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

  // Second Conv2d — reuses the same non-static filter.
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

  // Each build emits [Transpose, Conv].
  ASSERT_EQ(ops1.size(), 2);
  ASSERT_EQ(ops2.size(), 2);

  // ops[0] is the Transpose op; its output tensor must carry axis=3 (HWIO).
  const auto& out1 = std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
      ops1[0].GetOutputTensor(0).GetQuantParams());
  EXPECT_EQ(out1.GetAxis(), 3);
  EXPECT_EQ(ops1[0].GetOutputTensor(0).GetDimensions(),
            (std::vector<std::uint32_t>{1, 1, 1, 2}));

  const auto& out2 = std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
      ops2[0].GetOutputTensor(0).GetQuantParams());
  EXPECT_EQ(out2.GetAxis(), 3);

  // Original filter tensor must not be mutated.
  const auto& orig = std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
      filter.GetQuantParams());
  EXPECT_EQ(orig.GetAxis(), 0);
}

}  // namespace
}  // namespace litert::qnn
