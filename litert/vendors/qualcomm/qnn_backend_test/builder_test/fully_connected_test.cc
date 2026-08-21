// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;
using testing::FloatNear;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, FullyConnectedFp32CastsAroundFp16FullyConnected) {
  static constexpr float kWeightScale = 1.0f;
  static constexpr std::int32_t kWeightZeroPoint = 0;
  static constexpr std::array<std::int8_t, 4> kWeightData{1, 0, 0, 1};
  static constexpr std::array<float, 2> kBiasData{0.5f, -0.5f};
  static constexpr bool kKeepNumDims = true;
  static constexpr bool kUseInt64BiasAsInt32 = false;
  static constexpr ::qnn::SdkVersion kSdkVersion{2, 47, 0};
  const std::vector<std::uint32_t> kInOutDims{1, 1, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 2};
  const std::vector<std::uint32_t> kBiasDims{2};
  const std::vector<std::uint32_t> kFullyConnectedOutputDims{1, 2};

  ::qnn::QuantizeParamsWrapperVariant weight_quant_param{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, kWeightScale,
      kWeightZeroPoint};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, {}, kInOutDims);
  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant_param, kFilterDims,
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());
  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, {}, kBiasDims,
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, {}, kInOutDims);

  auto ops = ::qnn::BuildFullyConnectedOp(tensor_pool_, {input, weight, bias},
                                          {output}, kKeepNumDims,
                                          kUseInt64BiasAsInt32, kSdkVersion);

  ASSERT_EQ(ops.size(), 5u);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[0].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[0].GetOutputTensor(0).IsF16());

  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[1].GetInputTensor(0).IsF32());
  EXPECT_TRUE(ops[1].GetOutputTensor(0).IsF16());

  ASSERT_TRUE(ops[2].IsOpCode(::qnn::QnnOpCode::kFullyConnected));
  ASSERT_EQ(ops[2].GetInputCount(), 3u);
  EXPECT_TRUE(ops[2].GetInputTensor(0).IsF16());
  EXPECT_TRUE(ops[2].GetInputTensor(1).IsQuantI8());
  EXPECT_TRUE(ops[2].GetInputTensor(1).IsPerChannelQuant());
  EXPECT_TRUE(ops[2].GetInputTensor(2).IsF16());
  EXPECT_TRUE(ops[2].GetOutputTensor(0).IsF16());
  EXPECT_EQ(ops[2].GetOutputTensor(0).GetDimensions(),
            kFullyConnectedOutputDims);

  EXPECT_TRUE(ops[3].IsOpCode(::qnn::QnnOpCode::kReshape));
  EXPECT_TRUE(ops[3].GetInputTensor(0).IsF16());
  EXPECT_TRUE(ops[3].GetOutputTensor(0).IsF16());
  EXPECT_EQ(ops[3].GetOutputTensor(0).GetDimensions(), kInOutDims);

  EXPECT_TRUE(ops[4].IsOpCode(::qnn::QnnOpCode::kCast));
  EXPECT_TRUE(ops[4].GetInputTensor(0).IsF16());
  EXPECT_TRUE(ops[4].GetOutputTensor(0).IsF32());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  if (!is_fp16_supported_) {
    GTEST_SKIP() << "The rest of this test applies only to HTP targets with "
                    "FP16 support.";
  }
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);

  qnn_model_.SetInputData<float>(input_idx, {1, 2});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 2);
  EXPECT_THAT(output_data.value(),
              ElementsAre(FloatNear(1.5f, 1e-3f), FloatNear(1.5f, 1e-3f)));
#endif
}

TEST_P(QnnModelTest, FullyConnectedInt64BiasConvertedToInt32) {
  static constexpr float kScale = 0.01f;
  static constexpr std::int32_t kZeroPoint = 0;
  static constexpr std::array<std::int8_t, 4> kWeightData{1, -1, -1, 1};
  static constexpr std::array<std::int64_t, 2> kBiasData{1, -1};
  static constexpr bool kKeepNumDims = false;
  static constexpr bool kUseInt64BiasAsInt32 = true;
  static constexpr ::qnn::SdkVersion kSdkVersion{2, 47, 0};

  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};
  const std::vector<std::uint32_t> kInOutDims{1, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 2};
  const std::vector<std::uint32_t> kBiasDims{2};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant, kInOutDims);
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant, kInOutDims);

  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant, kFilterDims,
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());

  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_64, quant, kBiasDims,
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildFullyConnectedOp(tensor_pool_, {input, weight, bias},
                                          {output}, kKeepNumDims,
                                          kUseInt64BiasAsInt32, kSdkVersion);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kFullyConnected));
  ASSERT_EQ(ops[0].GetInputCount(), 3u);
  EXPECT_EQ(ops[0].GetInputTensor(2).GetDataType(),
            QNN_DATATYPE_SFIXED_POINT_32);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);

  qnn_model_.SetInputData<std::int8_t>(input_idx, {100, 0});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 2);
  std::vector<float> dequant_output;
  ::qnn::DequantizeInto(output_data.value(), quant.GetScale(),
                        quant.GetZeroPoint(), dequant_output);
  EXPECT_THAT(dequant_output,
              ElementsAre(FloatNear(0.02f, 1e-3f), FloatNear(-0.02f, 1e-3f)));
#endif
}

TEST_P(QnnModelTest, FullyConnectedQuantizedNormalPath) {
  static constexpr float kScale = 0.01f;
  static constexpr std::int32_t kZeroPoint = 0;
  static constexpr std::array<std::int8_t, 4> kWeightData{1, -1, -1, 1};
  static constexpr std::array<std::int32_t, 2> kBiasData{1, -1};
  static constexpr bool kKeepNumDims = true;
  static constexpr bool kUseInt64BiasAsInt32 = false;
  static constexpr ::qnn::SdkVersion kSdkVersion{2, 47, 0};

  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kScale, kZeroPoint};
  const std::vector<std::uint32_t> kInOutDims{1, 1, 2};
  const std::vector<std::uint32_t> kFilterDims{2, 2};
  const std::vector<std::uint32_t> kBiasDims{2};
  const std::vector<std::uint32_t> kFullyConnectedOutputDims{1, 2};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, quant, kInOutDims);
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, quant, kInOutDims);

  auto& weight = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, quant, kFilterDims,
      kWeightData.size() * sizeof(decltype(kWeightData)::value_type),
      kWeightData.data());

  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, quant, kBiasDims,
      kBiasData.size() * sizeof(decltype(kBiasData)::value_type),
      kBiasData.data());

  auto ops = ::qnn::BuildFullyConnectedOp(tensor_pool_, {input, weight, bias},
                                          {output}, kKeepNumDims,
                                          kUseInt64BiasAsInt32, kSdkVersion);

  ASSERT_EQ(ops.size(), 2u);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kFullyConnected));
  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kReshape));
  EXPECT_EQ(ops[0].GetOutputTensor(0).GetDimensions(),
            kFullyConnectedOutputDims);
  EXPECT_EQ(ops[1].GetOutputTensor(0).GetDimensions(), kInOutDims);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);

  qnn_model_.SetInputData<std::int8_t>(input_idx, {100, 0});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 2);
  std::vector<float> dequant_output;
  ::qnn::DequantizeInto(output_data.value(), quant.GetScale(),
                        quant.GetZeroPoint(), dequant_output);
  EXPECT_THAT(dequant_output,
              ElementsAre(FloatNear(0.02f, 1e-3f), FloatNear(-0.02f, 1e-3f)));
#endif
}

}  // namespace
}  // namespace litert::qnn
