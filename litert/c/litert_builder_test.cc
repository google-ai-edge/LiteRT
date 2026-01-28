// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/c/litert_builder.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"
#include "tflite/converter/schema/schema_generated.h"

static constexpr absl::string_view kName = "GT";
static constexpr absl::string_view kData = "CircuitDeMonaco";

namespace {

using ::litert::OwningBufferRef;
using ::litert::internal::BufferManager;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(LiteRtBuilderTest, CanEraseOp) {
  LiteRtBuilderT builder;
  LiteRtOpT op;
  LITERT_ASSERT_OK(LiteRtBuilderEraseOp(&builder, &op));
  EXPECT_TRUE(builder.Erases().contains(&op));
}

TEST(LiteRtBuilderTest, EraseOpNullOpReturnsError) {
  LiteRtBuilderT builder;
  EXPECT_EQ(LiteRtBuilderEraseOp(&builder, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtBuilderTest, EraseOpNullBuilderReturnsError) {
  EXPECT_EQ(LiteRtBuilderEraseOp(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtBuilderTest, CanBuildUnrankedTensor) {
  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtUnrankedTensorType unranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32};
  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtUnrankedTensorType, LiteRtRankedTensorType(),
      unranked_tensor_type, LiteRtWeights(), kLiteRtQuantizationNone,
      LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(),
      kName.data(), &tensor));
  EXPECT_EQ(tensor->Type().first, kLiteRtUnrankedTensorType);
  EXPECT_EQ(tensor->Type().second.unranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Name(), kName);
}

TEST(LiteRtBuilderTest, BuildingUnrankedTensorWithInvalidArgumentReturnsError) {
  BufferManager manager;
  LiteRtWeightsT weights;
  {
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  weights.SetBufferId(weights.GetBufferId());
  weights.SetBufferManager(&manager);
  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtUnrankedTensorType unranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32};
  EXPECT_EQ(LiteRtBuilderBuildTensor(
                &builder, kLiteRtUnrankedTensorType, LiteRtRankedTensorType(),
                unranked_tensor_type, &weights, kLiteRtQuantizationNone,
                LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(),
                "", &tensor),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtBuilderTest, CanBuildRankedTensor) {
  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtRankedTensorType, ranked_tensor_type,
      LiteRtUnrankedTensorType(), LiteRtWeights(), kLiteRtQuantizationNone,
      LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(),
      kName.data(), &tensor));

  EXPECT_EQ(tensor->Type().first, kLiteRtRankedTensorType);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.rank, 2);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.dimensions[0], 3);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.dimensions[1], 3);
  EXPECT_EQ(tensor->Name(), kName);
}

TEST(LiteRtBuilderTest, CanBuildRankedTensorWithWeights) {
  BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};

  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtRankedTensorType, ranked_tensor_type,
      LiteRtUnrankedTensorType(), &weights, kLiteRtQuantizationNone,
      LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(), "",
      &tensor));

  EXPECT_EQ(tensor->Weights().GetBufferId(), 1);
  EXPECT_EQ(tensor->Weights().GetBufferManager(), &manager);
  EXPECT_EQ(tensor->Weights().Buffer().Size(), kData.size());
  EXPECT_EQ(tensor->Weights().Buffer().StrView(), kData);
}

TEST(LiteRtBuilderTest, CanBuildTensorWithPerTensorQuantization) {
  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LiteRtQuantizationPerTensor per_tensor_quantization = {.scale = 1.0,
                                                         .zero_point = 1};
  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtRankedTensorType, ranked_tensor_type,
      LiteRtUnrankedTensorType(), LiteRtWeights(), kLiteRtQuantizationPerTensor,
      per_tensor_quantization, LiteRtQuantizationPerChannel(), kName.data(),
      &tensor));
  EXPECT_EQ(tensor->Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(tensor->Qparams().second.per_tensor.scale, 1.0);
  EXPECT_EQ(tensor->Qparams().second.per_tensor.zero_point, 1);
}

TEST(LiteRtBuilderTest, CanBuildTensorWithPerChannelQuantization) {
  constexpr size_t kNumChannels = 2;
  constexpr size_t kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtTensorT per_channel_quantized_tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LiteRtQuantizationPerChannel per_channel_quantization =
      MakePerChannelQuantization(kScales, kZeroPoints, kQuantizedDimension,
                                 per_channel_quantized_tensor)
          .second.per_channel;

  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtRankedTensorType, ranked_tensor_type,
      LiteRtUnrankedTensorType(), LiteRtWeights(),
      kLiteRtQuantizationPerChannel, LiteRtQuantizationPerTensor(),
      per_channel_quantization, kName.data(), &tensor));

  LiteRtQuantizationPerChannel built_per_channel_quantization =
      tensor->Qparams().second.per_channel;
  EXPECT_THAT(
      absl::MakeConstSpan(built_per_channel_quantization.scales, kNumChannels),
      ElementsAreArray(kScales));
  EXPECT_THAT(absl::MakeConstSpan(built_per_channel_quantization.zero_points,
                                  kNumChannels),
              ElementsAreArray(kZeroPoints));
  EXPECT_EQ(built_per_channel_quantization.num_channels, kNumChannels);
  EXPECT_EQ(built_per_channel_quantization.quantized_dimension,
            kQuantizedDimension);
}

TEST(LiteRtBuilderTest, CanBuildOpTest) {
  LiteRtBuilderT builder;
  LiteRtTensorT input_tensor_0;
  LiteRtTensorT input_tensor_1;
  LiteRtTensorT output_tensor_0;
  LiteRtOp op;

  std::vector<LiteRtTensor> input_tensors = {&input_tensor_0, &input_tensor_1};
  std::vector<LiteRtTensor> output_tensors = {&output_tensor_0};
  LITERT_ASSERT_OK(LiteRtBuilderBuildOp(
      &builder, kLiteRtOpCodeTflAdd, input_tensors.size(), input_tensors.data(),
      output_tensors.size(), output_tensors.data(), &op));
  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(op->Inputs().size(), 2);
  EXPECT_THAT(op->Inputs(), ElementsAre(&input_tensor_0, &input_tensor_1));
  EXPECT_THAT(op->Outputs(), ElementsAre(&output_tensor_0));
}

TEST(LiteRtBuilderTest, BuildWeights) {
  static constexpr absl::string_view kData = "Nurburgring";
  const uint8_t* data = reinterpret_cast<const uint8_t*>(kData.data());

  LiteRtBuilderT builder;
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};

  LITERT_ASSERT_OK(LiteRtBuilderBuildTensor(
      &builder, kLiteRtRankedTensorType, ranked_tensor_type,
      LiteRtUnrankedTensorType(), nullptr, kLiteRtQuantizationNone,
      LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(), "",
      &tensor));

  LiteRtWeights weights;
  LITERT_ASSERT_OK(LiteRtBuilderBuildWeights(&builder, data, kData.size(),
                                             tensor, &weights));
  EXPECT_EQ(weights->Buffer().Size(), kData.size());
  EXPECT_EQ(weights->Buffer().StrView(), kData);
}

TEST(LiteRtBuilderTest, BuildAddOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflAdd, {}, {});
  uint32_t fused_activation = 6;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildAddOpOption(&builder, &op, &fused_activation));
  EXPECT_EQ(litert::internal::GetTflOptions(op)
                .AsAddOptions()
                ->fused_activation_function,
            6);
}

TEST(LiteRtBuilderTest, BuildMulOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflMul, {}, {});
  uint32_t fused_activation = 3;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildMulOpOption(&builder, &op, &fused_activation));
  EXPECT_EQ(litert::internal::GetTflOptions(op)
                .AsMulOptions()
                ->fused_activation_function,
            3);
}

TEST(LiteRtBuilderTest, BuildDivOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflDiv, {}, {});
  uint32_t fused_activation = 1;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildDivOpOption(&builder, &op, &fused_activation));
  EXPECT_EQ(litert::internal::GetTflOptions(op)
                .AsDivOptions()
                ->fused_activation_function,
            1);
}

TEST(LiteRtBuilderTest, BuildSubOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSub, {}, {});
  uint32_t fused_activation = 0;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildSubOpOption(&builder, &op, &fused_activation));
  EXPECT_EQ(litert::internal::GetTflOptions(op)
                .AsSubOptions()
                ->fused_activation_function,
            0);
}

TEST(LiteRtBuilderTest, BuildBatchMatmulOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {}, {});
  bool adj_x = true;
  bool adj_y = false;
  bool asymmetric_quantize_input = true;
  LITERT_ASSERT_OK(LiteRtBuilderBuildBatchMatmulOpOption(
      &builder, &op, &adj_x, &adj_y, &asymmetric_quantize_input));
  auto* opts = litert::internal::GetTflOptions(op).AsBatchMatMulOptions();
  EXPECT_EQ(opts->adj_x, true);
  EXPECT_EQ(opts->adj_y, false);
  EXPECT_EQ(opts->asymmetric_quantize_inputs, true);
}

TEST(LiteRtBuilderTest, BuildConcatenationOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflConcatenation, {}, {});
  uint32_t fused_activation = 2;
  int32_t axis = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildConcatenationOpOption(
      &builder, &op, &fused_activation, &axis));
  auto* opts = litert::internal::GetTflOptions(op).AsConcatenationOptions();
  EXPECT_EQ(opts->fused_activation_function, 2);
  EXPECT_EQ(opts->axis, 1);
}

TEST(LiteRtBuilderTest, BuildFullyConnectedOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflFullyConnected, {}, {});
  uint32_t fused_activation = 1;
  uint32_t weights_format = 0;
  bool keep_num_dims = true;
  uint32_t quantized_bias_type = 0;
  bool asymmetric_quantize_input = false;
  LITERT_ASSERT_OK(LiteRtBuilderBuildFullyConnectedOpOption(
      &builder, &op, &fused_activation, &weights_format, &keep_num_dims,
      &quantized_bias_type, &asymmetric_quantize_input));
  auto* opts = litert::internal::GetTflOptions(op).AsFullyConnectedOptions();
  EXPECT_EQ(opts->fused_activation_function, 1);
  EXPECT_EQ(opts->weights_format, 0);
  EXPECT_EQ(opts->keep_num_dims, true);
  EXPECT_EQ(opts->quantized_bias_type, 0);
  EXPECT_EQ(opts->asymmetric_quantize_inputs, false);
}

TEST(LiteRtBuilderTest, BuildSoftmaxOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSoftmax, {}, {});
  float beta = 1.0f;
  LITERT_ASSERT_OK(LiteRtBuilderBuildSoftmaxOpOption(&builder, &op, &beta));
  auto* opts = litert::internal::GetTflOptions(op).AsSoftmaxOptions();
  EXPECT_EQ(opts->beta, 1.0f);
}

TEST(LiteRtBuilderTest, BuildStridedSliceOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflStridedSlice, {}, {});
  int32_t begin_mask = 1;
  int32_t end_mask = 2;
  int32_t ellipsis_mask = 3;
  int32_t new_axis_mask = 4;
  int32_t shrink_axis_mask = 5;
  bool offset = true;
  LITERT_ASSERT_OK(LiteRtBuilderBuildStridedSliceOpOption(
      &builder, &op, &begin_mask, &end_mask, &ellipsis_mask, &new_axis_mask,
      &shrink_axis_mask, &offset));
  auto* opts = litert::internal::GetTflOptions(op).AsStridedSliceOptions();
  EXPECT_EQ(opts->begin_mask, 1);
  EXPECT_EQ(opts->end_mask, 2);
  EXPECT_EQ(opts->ellipsis_mask, 3);
  EXPECT_EQ(opts->new_axis_mask, 4);
  EXPECT_EQ(opts->shrink_axis_mask, 5);
  EXPECT_EQ(opts->offset, true);
}

TEST(LiteRtBuilderTest, BuildReshapeOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflReshape, {}, {});
  std::vector<int32_t> new_shape = {1, 2};
  LITERT_ASSERT_OK(LiteRtBuilderBuildReshapeOpOption(
      &builder, &op, new_shape.data(), new_shape.size()));
  auto* opts = litert::internal::GetTflOptions(op).AsReshapeOptions();
  EXPECT_THAT(opts->new_shape, ElementsAreArray(new_shape));
}

TEST(LiteRtBuilderTest, BuildSumOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSum, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(LiteRtBuilderBuildSumOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildReduceMaxOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflReduceMax, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildReduceMaxOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildReduceMinOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflReduceMin, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildReduceMinOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildReduceAnyOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflReduceAny, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildReduceAnyOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildReduceAllOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflReduceAll, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildReduceAllOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildPackOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflPack, {}, {});
  int32_t axis = 1;
  int32_t values_count = 2;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildPackOpOption(&builder, &op, &axis, &values_count));
  auto* opts = litert::internal::GetTflOptions(op).AsPackOptions();
  EXPECT_EQ(opts->axis, 1);
  EXPECT_EQ(opts->values_count, 2);
}

TEST(LiteRtBuilderTest, BuildUnpackOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflUnpack, {}, {});
  int32_t axis = 1;
  int32_t num = 2;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildUnpackOpOption(&builder, &op, &axis, &num));
  auto* opts = litert::internal::GetTflOptions(op).AsUnpackOptions();
  EXPECT_EQ(opts->axis, 1);
  EXPECT_EQ(opts->num, 2);
}

TEST(LiteRtBuilderTest, BuildGatherOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflGather, {}, {});
  int32_t axis = 1;
  int32_t batch_dims = 2;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildGatherOpOption(&builder, &op, &axis, &batch_dims));
  auto* opts = litert::internal::GetTflOptions(op).AsGatherOptions();
  EXPECT_EQ(opts->axis, 1);
  EXPECT_EQ(opts->batch_dims, 2);
}

TEST(LiteRtBuilderTest, BuildMeanOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflMean, {}, {});
  bool keepdims = true;
  LITERT_ASSERT_OK(LiteRtBuilderBuildMeanOpOption(&builder, &op, &keepdims));
  auto* opts = litert::internal::GetTflOptions(op).AsReducerOptions();
  EXPECT_EQ(opts->keep_dims, true);
}

TEST(LiteRtBuilderTest, BuildSplitOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSplit, {}, {});
  int32_t num_splits = 2;
  LITERT_ASSERT_OK(LiteRtBuilderBuildSplitOpOption(&builder, &op, &num_splits));
  auto* opts = litert::internal::GetTflOptions(op).AsSplitOptions();
  EXPECT_EQ(opts->num_splits, 2);
}

TEST(LiteRtBuilderTest, BuildConv2dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflConv2d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t dilation_w_factor = 1;
  int32_t dilation_h_factor = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildConv2dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &dilation_w_factor,
      &dilation_h_factor, &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsConv2DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->dilation_w_factor, 1);
  EXPECT_EQ(opts->dilation_h_factor, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildConv3dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflConv3d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t stride_d = 1;
  int32_t dilation_w_factor = 1;
  int32_t dilation_h_factor = 1;
  int32_t dilation_d_factor = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildConv3dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &stride_d,
      &dilation_w_factor, &dilation_h_factor, &dilation_d_factor,
      &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsConv3DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->stride_d, 1);
  EXPECT_EQ(opts->dilation_w_factor, 1);
  EXPECT_EQ(opts->dilation_h_factor, 1);
  EXPECT_EQ(opts->dilation_d_factor, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildDepthwiseConv2dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflDepthwiseConv2d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t depth_multiplier = 1;
  uint32_t fused_activation_function = 1;
  int32_t dilation_w_factor = 1;
  int32_t dilation_h_factor = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildDepthwiseConv2dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &depth_multiplier,
      &fused_activation_function, &dilation_w_factor, &dilation_h_factor));
  auto* opts = litert::internal::GetTflOptions(op).AsDepthwiseConv2DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->depth_multiplier, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
  EXPECT_EQ(opts->dilation_w_factor, 1);
  EXPECT_EQ(opts->dilation_h_factor, 1);
}

TEST(LiteRtBuilderTest, BuildTransposeConvOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflTransposeConv, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildTransposeConvOpOption(
      &builder, &op, &padding, &stride_w, &stride_h,
      &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsTransposeConvOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildAveragePool2dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflAveragePool2d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t filter_width = 1;
  int32_t filter_height = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildAveragePool2dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsPool2DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->filter_width, 1);
  EXPECT_EQ(opts->filter_height, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildMaxPool2dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflMaxPool2d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t filter_width = 1;
  int32_t filter_height = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildMaxPool2dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsPool2DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->filter_width, 1);
  EXPECT_EQ(opts->filter_height, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildL2Pool2dOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflL2Pool2d, {}, {});
  uint32_t padding = 1;
  int32_t stride_w = 1;
  int32_t stride_h = 1;
  int32_t filter_width = 1;
  int32_t filter_height = 1;
  uint32_t fused_activation_function = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildL2Pool2dOpOption(
      &builder, &op, &padding, &stride_w, &stride_h, &filter_width,
      &filter_height, &fused_activation_function));
  auto* opts = litert::internal::GetTflOptions(op).AsPool2DOptions();
  EXPECT_EQ(opts->padding, tflite::Padding_VALID);
  EXPECT_EQ(opts->stride_w, 1);
  EXPECT_EQ(opts->stride_h, 1);
  EXPECT_EQ(opts->filter_width, 1);
  EXPECT_EQ(opts->filter_height, 1);
  EXPECT_EQ(opts->fused_activation_function,
            tflite::ActivationFunctionType_RELU);
}

TEST(LiteRtBuilderTest, BuildResizeBilinearOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflResizeBilinear, {}, {});
  bool align_corners = true;
  bool half_pixel_centers = false;
  LITERT_ASSERT_OK(LiteRtBuilderBuildResizeBilinearOpOption(
      &builder, &op, &align_corners, &half_pixel_centers));
  auto* opts = litert::internal::GetTflOptions(op).AsResizeBilinearOptions();
  EXPECT_EQ(opts->align_corners, true);
  EXPECT_EQ(opts->half_pixel_centers, false);
}

TEST(LiteRtBuilderTest, BuildLeakyReluOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflLeakyRelu, {}, {});
  float alpha = 0.1f;
  LITERT_ASSERT_OK(LiteRtBuilderBuildLeakyReluOpOption(&builder, &op, &alpha));
  auto* opts = litert::internal::GetTflOptions(op).AsLeakyReluOptions();
  EXPECT_FLOAT_EQ(opts->alpha, 0.1f);
}

TEST(LiteRtBuilderTest, BuildSpaceToDepthOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSpaceToDepth, {}, {});
  int32_t block_size = 2;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildSpaceToDepthOpOption(&builder, &op, &block_size));
  auto* opts = litert::internal::GetTflOptions(op).AsSpaceToDepthOptions();
  EXPECT_EQ(opts->block_size, 2);
}

TEST(LiteRtBuilderTest, BuildDepthToSpaceOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflDepthToSpace, {}, {});
  int32_t block_size = 2;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildDepthToSpaceOpOption(&builder, &op, &block_size));
  auto* opts = litert::internal::GetTflOptions(op).AsDepthToSpaceOptions();
  EXPECT_EQ(opts->block_size, 2);
}

TEST(LiteRtBuilderTest, BuildResizeNearestNeighborOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflResizeNearestNeighbor, {}, {});
  bool align_corners = true;
  bool half_pixel_centers = false;
  LITERT_ASSERT_OK(LiteRtBuilderBuildResizeNearestNeighborOpOption(
      &builder, &op, &align_corners, &half_pixel_centers));
  auto* opts =
      litert::internal::GetTflOptions(op).AsResizeNearestNeighborOptions();
  EXPECT_EQ(opts->align_corners, true);
  EXPECT_EQ(opts->half_pixel_centers, false);
}

TEST(LiteRtBuilderTest, BuildCumsumOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflCumsum, {}, {});
  bool exclusive = true;
  bool reverse = false;
  LITERT_ASSERT_OK(
      LiteRtBuilderBuildCumsumOpOption(&builder, &op, &exclusive, &reverse));
  auto* opts = litert::internal::GetTflOptions(op).AsCumsumOptions();
  EXPECT_EQ(opts->exclusive, true);
  EXPECT_EQ(opts->reverse, false);
}

TEST(LiteRtBuilderTest, BuildGeluOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflGelu, {}, {});
  bool approximate = true;
  LITERT_ASSERT_OK(LiteRtBuilderBuildGeluOpOption(&builder, &op, &approximate));
  auto* opts = litert::internal::GetTflOptions(op).AsGeluOptions();
  EXPECT_EQ(opts->approximate, true);
}

TEST(LiteRtBuilderTest, BuildMirrorPadOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflMirrorPad, {}, {});
  uint32_t mode = 1;
  LITERT_ASSERT_OK(LiteRtBuilderBuildMirrorPadOpOption(&builder, &op, &mode));
  auto* opts = litert::internal::GetTflOptions(op).AsMirrorPadOptions();
  EXPECT_EQ(opts->mode, tflite::MirrorPadMode_SYMMETRIC);
}

TEST(LiteRtBuilderTest, BuildSqueezeOpOption) {
  LiteRtBuilderT builder;

  auto& op = builder.BuildOp(kLiteRtOpCodeTflSqueeze, {}, {});
  std::vector<int32_t> squeeze_dims = {1, 2};
  LITERT_ASSERT_OK(LiteRtBuilderBuildSqueezeOpOption(
      &builder, &op, squeeze_dims.data(), squeeze_dims.size()));
  auto* opts = litert::internal::GetTflOptions(op).AsSqueezeOptions();
  EXPECT_THAT(opts->squeeze_dims, ElementsAreArray(squeeze_dims));
}

}  // namespace
