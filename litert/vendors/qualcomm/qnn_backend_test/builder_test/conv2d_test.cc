// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <tuple>  // IWYU pragma: keep
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"  // IWYU pragma: keep
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

class Conv2dOpBuilderExecutionTest : public QnnModelTest {};

INSTANTIATE_TEST_SUITE_P(, Conv2dOpBuilderExecutionTest,
                         GetHtpAndLpaiQnnModelParams(), QnnTestPrinter);

TEST_P(Conv2dOpBuilderExecutionTest, KeepsPerChannelBiasQuantization) {
  auto& tensor_pool = tensor_pool_;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  auto& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 2, 1});
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/0, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      sizeof(filter_data), filter_data.data());
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper bias_quant{
      /*axis=*/0, /*scales=*/{0.05f, 0.1f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int32_t, 2> bias_data{10, -20};
  auto& bias =
      tensor_pool.CreateStaticTensor(QNN_DATATYPE_SFIXED_POINT_32, bias_quant,
                                     {2}, sizeof(bias_data), bias_data.data());
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 2, 2});

  auto ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 1);
  ASSERT_EQ(ops[0].GetInputCount(), 3);
  const auto& op_bias = ops[0].GetInputTensor(2);
  EXPECT_EQ(op_bias.GetDataType(), QNN_DATATYPE_SFIXED_POINT_32);
  const auto& quant = op_bias.GetQnnTensor().v2.quantizeParams;
  ASSERT_EQ(quant.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  const auto& axis_quant = quant.axisScaleOffsetEncoding;
  EXPECT_EQ(axis_quant.axis, 0);
  ASSERT_EQ(axis_quant.numScaleOffsets, 2);
  EXPECT_FLOAT_EQ(axis_quant.scaleOffset[0].scale, 0.05f);
  EXPECT_FLOAT_EQ(axis_quant.scaleOffset[1].scale, 0.1f);
  EXPECT_EQ(axis_quant.scaleOffset[0].offset, 0);
  EXPECT_EQ(axis_quant.scaleOffset[1].offset, 0);
  const auto data = op_bias.GetTensorData<std::int32_t>();
  ASSERT_TRUE(data.has_value());
  ASSERT_EQ(data->size(), bias_data.size());
  EXPECT_EQ((*data)[0], bias_data[0]);
  EXPECT_EQ((*data)[1], bias_data[1]);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  const auto& options = std::get<0>(GetParam());
  if (options.GetBackendType() == ::qnn::BackendType::kLpaiBackend) {
    auto graph_configs = qnn_backend_ptr_->BuildGraphConfigs(options, "test");
    auto configs = graph_configs.GetNullTerminatedConfigs();
    ASSERT_TRUE(qnn_model_.Finalize(configs.data()));
  } else {
    ASSERT_TRUE(qnn_model_.Finalize());
  }

  const auto input_idx = qnn_model_.AddInputTensor(input);
  const auto output_idx = qnn_model_.AddOutputTensor(output);
  static constexpr std::array<std::int8_t, 2> input_data{10, -10};
  ASSERT_TRUE(qnn_model_.SetInputData<std::int8_t>(input_idx, input_data));
  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data.has_value());
  // Dequantized outputs are {1, -3, 0, -1}; output scale 0.5 and zero point 0
  // give raw INT8 values {2, -6, 0, -2}.
  static constexpr std::array<std::int8_t, 4> expected_output{2, -6, 0, -2};
  EXPECT_THAT(*output_data, ::testing::ElementsAreArray(expected_output));
}

TEST(Conv2dOpBuilderTest, FilterAxisUnchangedAfterBuildConv2dOp) {
  // Verify filter quant params are unchanged after calling BuildConv2dOp().
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/0, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  auto& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input, filter}, {output},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

  const auto& orig_quant =
      std::get<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
          filter.GetQuantParams());
  EXPECT_EQ(orig_quant.GetAxis(), 0);
}

TEST(Conv2dOpBuilderTest, FilterBwAxisUnchangedAfterBuildConv2dOp) {
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

  auto& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  ::qnn::BuildConv2dOp(tensor_pool, {input, filter}, {output},
                       /*stride_h=*/1, /*stride_w=*/1, /*dilation_h=*/1,
                       /*dilation_w=*/1, ::qnn::PaddingType::Valid,
                       /*use_int64_bias_as_int32=*/false);

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
