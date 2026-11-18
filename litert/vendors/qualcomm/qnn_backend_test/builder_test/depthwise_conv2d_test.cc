// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

class DepthwiseConv2dOpBuilderExecutionTest : public QnnModelTest {};

INSTANTIATE_TEST_SUITE_P(, DepthwiseConv2dOpBuilderExecutionTest,
                         GetHtpAndLpaiQnnModelParams(), QnnTestPrinter);

TEST_P(DepthwiseConv2dOpBuilderExecutionTest, KeepsPerChannelBiasQuantization) {
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  auto& input = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*axis=*/3, /*scales=*/{0.1f, 0.2f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, 1};
  auto& filter = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {1, 1, 1, 2},
      sizeof(filter_data), filter_data.data());
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper bias_quant{
      /*axis=*/0, /*scales=*/{0.05f, 0.1f}, /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int32_t, 2> bias_data{10, -20};
  auto& bias =
      tensor_pool_.CreateStaticTensor(QNN_DATATYPE_SFIXED_POINT_32, bias_quant,
                                      {2}, sizeof(bias_data), bias_data.data());
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid);

  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kDepthWiseConv2d));
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
  // Dequantized outputs are {1, -3}; output scale 0.5 and zero point 0
  // give raw INT8 values {2, -6}.
  static constexpr std::array<std::int8_t, 2> expected_output{2, -6};
  EXPECT_THAT(*output_data, ::testing::ElementsAreArray(expected_output));
}

TEST_P(QnnModelTest, QuantizedDepthwiseConv2dKernel4x4Stride4UsesConv2d) {
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kInputHeight = 4;
  static constexpr std::uint32_t kInputWidth = 4;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kKernelSize = 4;
  static constexpr std::uint32_t kStride = 4;
  static constexpr std::uint32_t kDilationHeight = 1;
  static constexpr std::uint32_t kDilationWidth = 1;
  static constexpr std::uint32_t kOutputHeight = 1;
  static constexpr std::uint32_t kOutputWidth = 1;
  static constexpr std::array<std::int32_t, kChannels> kBiasData{10, 20};

  static constexpr float kScale = 1.0f;
  static constexpr std::int32_t kZeroPoint = 0;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kInputHeight, kInputWidth, kChannels});

  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kOutputHeight, kOutputWidth, kChannels});

  std::vector<std::int8_t> weight_data(kBatch * kKernelSize * kKernelSize *
                                       kChannels);
  for (std::uint32_t i = 0; i < weight_data.size(); i += kChannels) {
    weight_data[i] = 2;
    weight_data[i + 1] = 3;
  }
  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kKernelSize, kKernelSize, kChannels},
      weight_data.size() * sizeof(decltype(weight_data)::value_type),
      weight_data.data());

  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilationHeight, kDilationWidth, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kConv2d));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  std::vector<std::int8_t> input_data(input.GetTensorNumElements());
  for (std::uint32_t i = 0; i < input_data.size(); i += kChannels) {
    input_data[i] = 1;
    input_data[i + 1] = 2;
  }
  ASSERT_TRUE(qnn_model_.SetInputData<std::int8_t>(input_idx, input_data));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), output.GetTensorNumElements());
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[0], kScale, kZeroPoint),
              42.0f, kScale);
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[1], kScale, kZeroPoint),
              116.0f, kScale);
#endif
}

TEST_P(QnnModelTest, QuantizedDepthwiseConv2dKernel3x3Stride3UsesDepthwise) {
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kInputHeight = 3;
  static constexpr std::uint32_t kInputWidth = 3;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kKernelSize = 3;
  static constexpr std::uint32_t kStride = 3;
  static constexpr std::uint32_t kDilationHeight = 1;
  static constexpr std::uint32_t kDilationWidth = 1;
  static constexpr std::uint32_t kOutputHeight = 1;
  static constexpr std::uint32_t kOutputWidth = 1;
  static constexpr std::array<std::int32_t, kChannels> kBiasData{10, 20};

  static constexpr float kScale = 1.0f;
  static constexpr std::int32_t kZeroPoint = 0;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kInputHeight, kInputWidth, kChannels});

  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kOutputHeight, kOutputWidth, kChannels});

  std::vector<std::int8_t> weight_data(kBatch * kKernelSize * kKernelSize *
                                       kChannels);
  for (std::uint32_t i = 0; i < weight_data.size(); i += kChannels) {
    weight_data[i] = 2;
    weight_data[i + 1] = 3;
  }
  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, kKernelSize, kKernelSize, kChannels},
      weight_data.size() * sizeof(decltype(weight_data)::value_type),
      weight_data.data());

  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, kStride, kStride,
      kDilationHeight, kDilationWidth, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kDepthWiseConv2d));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an on-device Qualcomm HTP.";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  std::vector<std::int8_t> input_data(input.GetTensorNumElements());
  for (std::uint32_t i = 0; i < input_data.size(); i += kChannels) {
    input_data[i] = 1;
    input_data[i + 1] = 2;
  }
  ASSERT_TRUE(qnn_model_.SetInputData<std::int8_t>(input_idx, input_data));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), output.GetTensorNumElements());
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[0], kScale, kZeroPoint),
              28.0f, kScale);
  EXPECT_NEAR(::qnn::Dequantize(output_data.value()[1], kScale, kZeroPoint),
              74.0f, kScale);
#endif
}

}  // namespace
}  // namespace litert::qnn
