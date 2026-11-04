// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/param_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace litert::qnn {
namespace {

TEST(Conv2dOpBuilderTest, RemapsBwAxisQuantizationAfterFilterTranspose) {
  ::qnn::TensorPool tensor_pool;
  const ::qnn::ScaleOffsetQuantizeParamsWrapper activation_quant{0.5f, 0};
  ::qnn::TensorWrapper& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 1});

  const ::qnn::BwAxisScaleOffsetQuantizeParamsWrapper filter_quant{
      /*bitwidth=*/4, /*axis=*/0, /*scales=*/{0.25f, 0.5f},
      /*zero_points=*/{0, 0}};
  static constexpr std::array<std::int8_t, 2> filter_data{1, -1};
  ::qnn::TensorWrapper& filter = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, filter_quant, {2, 1, 1, 1},
      filter_data.size() * sizeof(filter_data[0]), filter_data.data());

  const ::qnn::ScaleOffsetQuantizeParamsWrapper bias_quant{0.25f, 0};
  static constexpr std::array<std::int32_t, 2> bias_data{0, 0};
  ::qnn::TensorWrapper& bias = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, bias_quant, {2},
      bias_data.size() * sizeof(bias_data[0]), bias_data.data());
  ::qnn::TensorWrapper& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, activation_quant, {1, 1, 1, 2});

  std::vector<::qnn::OpWrapper> ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Valid, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kConv2d));
  const ::qnn::TensorWrapper& transposed_filter = ops[0].GetInputTensor(1);
  EXPECT_EQ(transposed_filter.GetDimensions(),
            (std::vector<std::uint32_t>{1, 1, 1, 2}));
  const auto& quant_params =
      std::get<::qnn::BwAxisScaleOffsetQuantizeParamsWrapper>(
          transposed_filter.GetQuantParams());
  EXPECT_EQ(quant_params.GetAxis(), 3);
  EXPECT_EQ(quant_params.GetBitwidth(), 4);
}

TEST(Conv2dOpBuilderTest, GroupedDepthwiseEquivalentUsesDepthwiseConv2d) {
  ::qnn::TensorPool tensor_pool;
  ::qnn::TensorWrapper& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 1, 640, 512});
  ::qnn::TensorWrapper& filter = tensor_pool.CreateInputTensorWithName(
      "filter", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {512, 1, 5, 1});
  ::qnn::TensorWrapper& bias = tensor_pool.CreateInputTensorWithName(
      "bias", QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{},
      {512});
  ::qnn::TensorWrapper& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 1, 320, 512});

  std::vector<::qnn::OpWrapper> ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/2, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Same, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 2);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kTranspose));
  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kDepthWiseConv2d));
  EXPECT_EQ(ops[1].GetScalarParam(0), std::nullopt);
  EXPECT_EQ(ops[1].GetTensorParam(0).GetTensor().GetDimensions(),
            (std::vector<std::uint32_t>{2}));
  EXPECT_EQ(ops[1].GetTensorParam(1).GetTensor().GetDimensions(),
            (std::vector<std::uint32_t>{2, 2}));
  EXPECT_EQ(ops[1].GetTensorParam(2).GetTensor().GetDimensions(),
            (std::vector<std::uint32_t>{2}));
}

TEST(Conv2dOpBuilderTest,
     GroupedDepthwiseEquivalentKernel4x4Stride4UsesConv2d) {
  ::qnn::TensorPool tensor_pool;
  ::qnn::TensorWrapper& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 4, 4, 512});
  ::qnn::TensorWrapper& filter = tensor_pool.CreateInputTensorWithName(
      "filter", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {512, 4, 4, 1});
  ::qnn::TensorWrapper& bias = tensor_pool.CreateInputTensorWithName(
      "bias", QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{},
      {512});
  ::qnn::TensorWrapper& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 1, 1, 512});

  std::vector<::qnn::OpWrapper> ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/4,
      /*stride_w=*/4, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Same, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 2);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kTranspose));
  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kConv2d));
  EXPECT_EQ(ops[1].GetScalarParam(0),
            ::qnn::ScalarParamWrapper(QNN_OP_CONV_2D_PARAM_GROUP,
                                      std::uint32_t{512}, false));
}

TEST(Conv2dOpBuilderTest, NonDepthwiseGroupedConv2dUsesConv2dWithGroupParam) {
  ::qnn::TensorPool tensor_pool;
  ::qnn::TensorWrapper& input = tensor_pool.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 4, 4, 8});
  ::qnn::TensorWrapper& filter = tensor_pool.CreateInputTensorWithName(
      "filter", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {8, 3, 3, 2});
  ::qnn::TensorWrapper& bias = tensor_pool.CreateInputTensorWithName(
      "bias", QNN_DATATYPE_FLOAT_32, ::qnn::QuantizeParamsWrapperVariant{},
      {8});
  ::qnn::TensorWrapper& output = tensor_pool.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_16, ::qnn::QuantizeParamsWrapperVariant{},
      {1, 4, 4, 8});

  std::vector<::qnn::OpWrapper> ops = ::qnn::BuildConv2dOp(
      tensor_pool, {input, filter, bias}, {output}, /*stride_h=*/1,
      /*stride_w=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
      ::qnn::PaddingType::Same, /*use_int64_bias_as_int32=*/false);

  ASSERT_EQ(ops.size(), 2);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kTranspose));
  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kConv2d));
  EXPECT_EQ(ops[1].GetScalarParam(0),
            ::qnn::ScalarParamWrapper(QNN_OP_CONV_2D_PARAM_GROUP,
                                      std::uint32_t{4}, false));
}

}  // namespace
}  // namespace litert::qnn
