// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_conv_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

constexpr float kActScale = 0.01f;
constexpr float kFilterScale = 0.5f;
constexpr float kBiasScale = kActScale * kFilterScale;

constexpr size_t kOpBiasIndex = 2;

std::vector<::qnn::OpWrapper> BuildTransposeConvWithInt64Bias(
    ::qnn::TensorPool& tensor_pool, const std::vector<std::int64_t>& bias_data,
    bool use_int64_bias_as_int32, ::qnn::TensorWrapper** input_tensor = nullptr,
    ::qnn::TensorWrapper** output_tensor = nullptr) {
  const std::vector<std::uint32_t> kInOutDims{1, 2, 2, 1};
  const std::vector<std::uint32_t> kFilterDims{1, 1, 1, 1};  // OHWI
  const std::vector<std::uint32_t> kBiasDims{1};
  const std::vector<std::uint32_t> kShapeDims{4};

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{kActScale, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kFilterQuant{kFilterScale, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{kBiasScale, 0};

  // TFLite TransposeConv operand order: output_shape, filter, input, bias.
  const std::vector<std::int32_t> kShapeData{1, 2, 2, 1};
  auto& output_shape = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, kShapeDims,
      sizeof(std::int32_t) * kShapeData.size(), kShapeData.data());

  const std::vector<std::int8_t> kFilterData{1};
  auto& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kFilterQuant, kFilterDims,
      sizeof(std::int8_t) * kFilterData.size(), kFilterData.data());

  auto& input = tensor_pool.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, kInOutDims);

  auto& bias = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_64, kBiasQuant, kBiasDims,
      sizeof(std::int64_t) * bias_data.size(), bias_data.data());

  auto& output = tensor_pool.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, kInOutDims);

  if (input_tensor != nullptr) {
    *input_tensor = &input;
  }
  if (output_tensor != nullptr) {
    *output_tensor = &output;
  }

  return ::qnn::BuildTransposeConvOp(
      tensor_pool, {output_shape, filter, input, bias}, {output},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      use_int64_bias_as_int32);
}

TEST_P(QnnModelTest, TransposeConvInt64BiasConvertedToInt32) {
  auto ops = BuildTransposeConvWithInt64Bias(tensor_pool_, /*bias_data=*/{100},
                                             /*use_int64_bias_as_int32=*/true);

  ASSERT_EQ(ops.size(), 1);
  ASSERT_GT(ops[0].GetInputCount(), kOpBiasIndex);
  EXPECT_EQ(ops[0].GetInputTensor(kOpBiasIndex).GetDataType(),
            QNN_DATATYPE_SFIXED_POINT_32);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  EXPECT_TRUE(qnn_model_.ValidateOpConfig());
}

TEST_P(QnnModelTest, TransposeConvInt64BiasKeptWhenFlagDisabled) {
  auto ops = BuildTransposeConvWithInt64Bias(tensor_pool_, /*bias_data=*/{100},
                                             /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 1);
  ASSERT_GT(ops[0].GetInputCount(), kOpBiasIndex);
  EXPECT_EQ(ops[0].GetInputTensor(kOpBiasIndex).GetDataType(),
            QNN_DATATYPE_INT_64);
}

TEST_P(QnnModelTest, TransposeConvInt64BiasOutOfInt32RangeFails) {
  constexpr std::int64_t kBiasOverInt32Max =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  auto ops = BuildTransposeConvWithInt64Bias(tensor_pool_,
                                             /*bias_data=*/{kBiasOverInt32Max},
                                             /*use_int64_bias_as_int32=*/true);

  EXPECT_TRUE(ops.empty());
}

TEST_P(QnnModelTest, TransposeConvInt64BiasBelowInt32RangeFails) {
  constexpr std::int64_t kBiasBelowInt32Min =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::lowest()) -
      1;
  auto ops = BuildTransposeConvWithInt64Bias(tensor_pool_,
                                             /*bias_data=*/{kBiasBelowInt32Min},
                                             /*use_int64_bias_as_int32=*/true);

  EXPECT_TRUE(ops.empty());
}

TEST_P(QnnModelTest,
       TransposeConvRemapsBwAxisQuantizationAfterFilterTranspose) {
  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{kActScale, 0};
  static constexpr std::array<std::int32_t, 4> kShapeData{1, 2, 2, 2};
  auto& output_shape = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {4},
      sizeof(kShapeData[0]) * kShapeData.size(), kShapeData.data());

  const ::qnn::BwAxisScaleOffsetQuantizeParamsWrapper kFilterQuant{
      /*bitwidth=*/4, /*axis=*/0, /*scales=*/{kFilterScale, kFilterScale},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> kFilterData{1, -1};
  auto& filter = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kFilterQuant, {2, 1, 1, 1},
      sizeof(kFilterData[0]) * kFilterData.size(), kFilterData.data());
  auto& input = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{kBiasScale, 0};
  static constexpr std::array<std::int32_t, 2> kBiasData{0, 0};
  auto& bias = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});

  auto ops = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input, bias}, {output},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

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

TEST_P(QnnModelTest,
       TransposeConvSharedFilterAxisNotCorruptedOnSecondBuild) {
  // Regression: shared static filter must not have its axis double-remapped
  // (0→3→2) when passed to BuildTransposeConvOp() twice (QNN error 3110).
  static constexpr std::array<std::int32_t, 4> kShapeData{1, 2, 2, 2};
  auto& output_shape = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {4},
      sizeof(kShapeData[0]) * kShapeData.size(), kShapeData.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{kActScale, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper kFilterQuant{
      /*axis=*/0, /*scales=*/{kFilterScale, kFilterScale},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> kFilterData{1, -1};
  auto& filter = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kFilterQuant, {2, 1, 1, 1},
      sizeof(kFilterData[0]) * kFilterData.size(), kFilterData.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{kBiasScale, 0};
  static constexpr std::array<std::int32_t, 2> kBiasData{0, 0};

  // First TransposeConv — shares the filter tensor.
  auto& input1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias1 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output1 = tensor_pool_.CreateOutputTensorWithName(
      "out_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops1 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input1, bias1}, {output1},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

  // Second TransposeConv — reuses the same filter tensor.
  auto& input2 = tensor_pool_.CreateInputTensorWithName(
      "in_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias2 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output2 = tensor_pool_.CreateOutputTensorWithName(
      "out_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops2 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input2, bias2}, {output2},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

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

TEST_P(QnnModelTest,
       TransposeConvSharedFilterBwAxisNotCorruptedOnSecondBuild) {
  // Regression: same as TransposeConvSharedFilterAxisNotCorruptedOnSecondBuild
  // for the BwAxisScaleOffsetQuantizeParamsWrapper axis double-remap (0→3→2).
  static constexpr std::array<std::int32_t, 4> kShapeData{1, 2, 2, 2};
  auto& output_shape = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {4},
      sizeof(kShapeData[0]) * kShapeData.size(), kShapeData.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{kActScale, 0};
  const ::qnn::BwAxisScaleOffsetQuantizeParamsWrapper kFilterQuant{
      /*bitwidth=*/4, /*axis=*/0, /*scales=*/{kFilterScale, kFilterScale},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> kFilterData{1, -1};
  auto& filter = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, kFilterQuant, {2, 1, 1, 1},
      sizeof(kFilterData[0]) * kFilterData.size(), kFilterData.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{kBiasScale, 0};
  static constexpr std::array<std::int32_t, 2> kBiasData{0, 0};

  // First TransposeConv — shares the filter tensor.
  auto& input1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias1 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output1 = tensor_pool_.CreateOutputTensorWithName(
      "out_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops1 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input1, bias1}, {output1},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

  // Second TransposeConv — reuses the same filter tensor.
  auto& input2 = tensor_pool_.CreateInputTensorWithName(
      "in_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias2 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output2 = tensor_pool_.CreateOutputTensorWithName(
      "out_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops2 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input2, bias2}, {output2},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

  // Both transposed filters must have axis=3, not 2.
  ASSERT_EQ(ops1.size(), 1);
  const auto& quant1 =
      std::get<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
          ops1[0].GetInputTensor(1).GetQuantParams());
  EXPECT_EQ(quant1.GetAxis(), 3);

  ASSERT_EQ(ops2.size(), 1);
  const auto& quant2 =
      std::get<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
          ops2[0].GetInputTensor(1).GetQuantParams());
  EXPECT_EQ(quant2.GetAxis(), 3);
}

TEST_P(QnnModelTest,
       TransposeConvNativeFilterTransposeOpOutputAxisRemappedAndOriginalUntouched) {
  // Regression (else branch): before the fix, CloneNativeTensorFrom(src, dims)
  // read src.quantize_params_ directly, leaving the Transpose output at axis=0.
  static constexpr std::array<std::int32_t, 4> kShapeData{1, 2, 2, 2};
  auto& output_shape = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_INT_32, ::qnn::QuantizeParamsWrapperVariant{}, {4},
      sizeof(kShapeData[0]) * kShapeData.size(), kShapeData.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kActQuant{kActScale, 0};
  const ::qnn::AxisScaleOffsetQuantizeParamsWrapper kFilterQuant{
      /*axis=*/0, /*scales=*/{kFilterScale, kFilterScale},
      /*zero_points=*/{0, 0}};
  // Non-static filter triggers the Transpose op path.
  auto& filter = tensor_pool_.CreateInputTensorWithName(
      "filter", QNN_DATATYPE_SFIXED_POINT_8, kFilterQuant, {2, 1, 1, 1});

  const ::qnn::ScaleOffsetQuantizeParamsWrapper kBiasQuant{kBiasScale, 0};
  static constexpr std::array<std::int32_t, 2> kBiasData{0, 0};

  // First TransposeConv.
  auto& input1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias1 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output1 = tensor_pool_.CreateOutputTensorWithName(
      "out_1", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops1 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input1, bias1}, {output1},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

  // Second TransposeConv — reuses the same non-static filter.
  auto& input2 = tensor_pool_.CreateInputTensorWithName(
      "in_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto& bias2 = tensor_pool_.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, kBiasQuant, {2},
      sizeof(kBiasData[0]) * kBiasData.size(), kBiasData.data());
  auto& output2 = tensor_pool_.CreateOutputTensorWithName(
      "out_2", QNN_DATATYPE_SFIXED_POINT_8, kActQuant, {1, 2, 2, 2});
  auto ops2 = ::qnn::BuildTransposeConvOp(
      tensor_pool_, {output_shape, filter, input2, bias2}, {output2},
      /*stride_h=*/1, /*stride_w=*/1, ::qnn::PaddingType::Valid,
      /*use_int64_bias_as_int32=*/false);

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

TEST_P(QnnModelTest, TransposeConvInt64BiasConvertedIsNumericallyCorrect) {
  ::qnn::TensorWrapper* input = nullptr;
  ::qnn::TensorWrapper* output = nullptr;

  auto ops = BuildTransposeConvWithInt64Bias(tensor_pool_, /*bias_data=*/{100},
                                             /*use_int64_bias_as_int32=*/true,
                                             &input, &output);
  ASSERT_EQ(ops.size(), 1);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an Android device with a Qualcomm HTP.";
#else
  const auto input_idx = qnn_model_.AddInputTensor(*input);
  const auto output_idx = qnn_model_.AddOutputTensor(*output);

  // Real inputs {0.02, 0, 0.04, 0}.
  // out = in * filter + bias.
  // With filter real value {0.5} and bias real value {0.5},
  // real outputs are {0.51, 0.5, 0.52, 0.5}.
  const std::vector<std::int8_t> in_data{2, 0, 4, 0};
  qnn_model_.SetInputData<std::int8_t>(
      input_idx, absl::MakeConstSpan(in_data.data(), in_data.size()));

  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);

  const std::vector<float> kExpected{0.51f, 0.50f, 0.52f, 0.50f};
  for (size_t i = 0; i < kExpected.size(); ++i) {
    const float f_out =
        ::qnn::Dequantize<std::int8_t>(output_data.value()[i], kActScale, 0);
    EXPECT_NEAR(f_out, kExpected[i], kActScale / 2);
  }
#endif
}

}  // namespace
}  // namespace litert::qnn
