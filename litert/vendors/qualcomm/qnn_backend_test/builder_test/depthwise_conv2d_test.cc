// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

struct DepthwiseConv2dParams {
  std::uint32_t kernel_size;
  std::uint32_t input_height;
  std::uint32_t input_width;
  std::uint32_t stride;
  std::uint32_t output_height;
  std::uint32_t output_width;
  ::qnn::QnnOpCode expected_op_code;
  // Expected dequantized values for output[0] and output[1] on-device.
  // Set to -1 to skip the execution assertion (e.g. for multi-element outputs).
  float expected_ch0;
  float expected_ch1;
};

// Flat 3-tuple: (QnnOptions, soc_model_name, DepthwiseConv2dParams)
using DepthwiseConv2dCombinedParams =
    std::tuple<::qnn::Options, const char*, DepthwiseConv2dParams>;

std::string DepthwiseConv2dPrinter(
    const ::testing::TestParamInfo<DepthwiseConv2dCombinedParams>& info) {
  const char* soc = std::get<1>(info.param);
  const DepthwiseConv2dParams& p = std::get<2>(info.param);
  return std::to_string(info.index) + "_" + (soc ? soc : "UNKNOWN_SOC") +
         "_Htp_Kernel" + std::to_string(p.kernel_size) + "x" +
         std::to_string(p.kernel_size) + "_Stride" + std::to_string(p.stride);
}

class DepthwiseConv2dTest
    : public testing::TestWithParam<DepthwiseConv2dCombinedParams>,
      public QnnModelSetupMixin {
 protected:
  void SetUp() override {
    const auto& options = std::get<0>(GetParam());
    const char* soc_model_name = std::get<1>(GetParam());
    if (!::qnn::IsTestHtpBackend()) {
      GTEST_SKIP() << "Skipping test because targeted backend is not supported";
    }
    SetUpQnnModel(options, soc_model_name);
  }
};

INSTANTIATE_TEST_SUITE_P(
    , DepthwiseConv2dTest,
    ::testing::Combine(
        ::testing::Values(GetTestingDefaultQnnOptions()),
        ::testing::ValuesIn(GetDefaultQnnSocs()),
        ::testing::Values(
            DepthwiseConv2dParams{4, 4, 4, 1, 4, 4,
                                  ::qnn::QnnOpCode::kDepthWiseConv2d, -1.0f, -1.0f},
            DepthwiseConv2dParams{4, 4, 4, 2, 2, 2,
                                  ::qnn::QnnOpCode::kConv2d, -1.0f, -1.0f},
            DepthwiseConv2dParams{4, 4, 4, 3, 2, 2,
                                  ::qnn::QnnOpCode::kConv2d, -1.0f, -1.0f},
            DepthwiseConv2dParams{4, 4, 4, 4, 1, 1,
                                  ::qnn::QnnOpCode::kConv2d, 42.0f, 116.0f},
            DepthwiseConv2dParams{3, 3, 3, 3, 1, 1,
                                  ::qnn::QnnOpCode::kDepthWiseConv2d,
                                  28.0f, 74.0f})),
    DepthwiseConv2dPrinter);

TEST_P(DepthwiseConv2dTest, QuantizedDepthwiseConv2d) {
  static constexpr std::uint32_t kBatch = 1;
  static constexpr std::uint32_t kChannels = 2;
  static constexpr std::uint32_t kDilationHeight = 1;
  static constexpr std::uint32_t kDilationWidth = 1;
  static constexpr std::array<std::int32_t, kChannels> kBiasData{10, 20};

  const DepthwiseConv2dParams& p = std::get<2>(GetParam());

  static constexpr float kScale = 1.0f;
  static constexpr std::int32_t kZeroPoint = 0;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, p.input_height, p.input_width, kChannels});

  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant,
      {kBatch, p.output_height, p.output_width, kChannels});

  std::vector<std::int8_t> weight_data(1 * p.kernel_size * p.kernel_size *
                                       kChannels);
  for (std::uint32_t i = 0; i < weight_data.size(); i += kChannels) {
    weight_data[i] = 2;
    weight_data[i + 1] = 3;
  }
  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant,
      {1, p.kernel_size, p.kernel_size, kChannels},
      weight_data.size() * sizeof(decltype(weight_data)::value_type),
      weight_data.data());

  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, {kChannels},
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildDepthwiseConv2dOp(
      tensor_pool_, {input, weight, bias}, {output}, p.stride, p.stride,
      kDilationHeight, kDilationWidth, ::qnn::PaddingType::Same);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(p.expected_op_code));

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
  if (p.expected_ch0 >= 0.0f) {
    EXPECT_NEAR(::qnn::Dequantize(output_data.value()[0], kScale, kZeroPoint),
                p.expected_ch0, kScale);
    EXPECT_NEAR(::qnn::Dequantize(output_data.value()[1], kScale, kZeroPoint),
                p.expected_ch1, kScale);
  }
#endif
}

}  // namespace
}  // namespace litert::qnn
