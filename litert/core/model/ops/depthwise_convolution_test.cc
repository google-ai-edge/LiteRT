// Copyright 2026 Google LLC.
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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/convolution.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

// Helper to create options
std::unique_ptr<tflite::DepthwiseConv2DOptionsT> CreateDepthwiseConvOptions(
    tflite::Padding padding, int stride_h, int stride_w, int depth_multiplier,
    int dilation_h_factor = 1, int dilation_w_factor = 1,
    tflite::ActivationFunctionType activation =
        tflite::ActivationFunctionType_NONE) {
  auto options = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
  options->padding = padding;
  options->stride_h = stride_h;
  options->stride_w = stride_w;
  options->depth_multiplier = depth_multiplier;
  options->dilation_h_factor = dilation_h_factor;
  options->dilation_w_factor = dilation_w_factor;
  options->fused_activation_function = activation;
  return options;
}

TEST(DepthwiseConvolutionOpTest, SimpleTest) {
  LiteRtOpT op;
  // Input [1, 3, 2, 2], Filter [1, 2, 2, 4] -> OutChannels = 4. InChannels = 2.
  // Multiplier = 2. Wait, TFLite depthwise filter shape: [1, H, W,
  // OutChannels]. OutChannels = InChannels * DepthMultiplier. Here
  // InputDepth=2, OutputDepth=4 -> Multiplier=2.
  std::vector<Dims> input_shapes = {
      {1, 3, 2, 2}, {1, 2, 2, 4}, {4}};  // Input, Filter, Bias
  std::vector<Dims> output_shapes(1);

  auto options = CreateDepthwiseConvOptions(tflite::Padding_VALID, 1, 1, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Output Height: (3 - 2 + 1)/1 = 2.
  // Output Width: (2 - 2 + 1)/1 = 1.
  // Output Channels: 4.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 1, 4));
}

TEST(DepthwiseConvolutionOpTest, StrideTest) {
  LiteRtOpT op;
  // Input [1, 3, 2, 2], Filter [1, 2, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2, 2}, {1, 2, 2, 4}, {4}};
  std::vector<Dims> output_shapes(1);

  // Stride 2x2
  auto options = CreateDepthwiseConvOptions(tflite::Padding_VALID, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Output Height: ceil((3 - 2 + 1)/2) = 1.
  // Output Width: ceil((2 - 2 + 1)/2) = 1.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 4));
}

TEST(DepthwiseConvolutionOpTest, PaddingTest) {
  LiteRtOpT op;
  // Input [1, 3, 2, 2], Filter [1, 2, 2, 4]
  std::vector<Dims> input_shapes = {{1, 3, 2, 2}, {1, 2, 2, 4}, {4}};
  std::vector<Dims> output_shapes(1);

  // Padding SAME, Stride 2x2
  auto options = CreateDepthwiseConvOptions(tflite::Padding_SAME, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Output Height: ceil(3/2) = 2.
  // Output Width: ceil(2/2) = 1.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 1, 4));
}

TEST(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingValid) {
  LiteRtOpT op;
  // Input [1, 9, 9, 1], Filter [1, 3, 3, 1]
  // Depth 1, Multiplier 1.
  std::vector<Dims> input_shapes = {{1, 9, 9, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Dilation 3
  auto options =
      CreateDepthwiseConvOptions(tflite::Padding_VALID, 1, 1, 1, 3, 3);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Effective filter size: (3-1)*3 + 1 = 7.
  // Output Height: (9 - 7 + 1)/1 = 3.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 1));
}

TEST(DepthwiseConvolutionOpTest, SimpleDilatedTestPaddingSame) {
  LiteRtOpT op;
  // Input [1, 3, 3, 1], Filter [1, 2, 2, 1]
  std::vector<Dims> input_shapes = {{1, 3, 3, 1}, {1, 2, 2, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Dilation 2
  auto options =
      CreateDepthwiseConvOptions(tflite::Padding_SAME, 1, 1, 1, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Output matches input for SAME with stride 1.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 1));
}

TEST(DepthwiseConvolutionOpTest, BatchPaddingValidTest) {
  LiteRtOpT op;
  // Input [2, 3, 3, 4], Filter [1, 3, 3, 4] -> OutChannels 4. Multiplier 1.
  std::vector<Dims> input_shapes = {{2, 3, 3, 4}, {1, 3, 3, 4}, {4}};
  std::vector<Dims> output_shapes(1);

  // Stride 1, Valid
  auto options = CreateDepthwiseConvOptions(tflite::Padding_VALID, 1, 1, 1);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Output Height: (3 - 3 + 1)/1 = 1.
  // Output Width: (3 - 3 + 1)/1 = 1.
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 1, 4));
}

TEST(DepthwiseConvolutionOpTest, BatchPaddingSameTest) {
  LiteRtOpT op;
  // Input [4, 2, 2, 1], Filter [1, 3, 3, 1]
  std::vector<Dims> input_shapes = {{4, 2, 2, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Stride 1, Same
  auto options = CreateDepthwiseConvOptions(tflite::Padding_SAME, 1, 1, 1);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(4, 2, 2, 1));
}

TEST(DepthwiseConvolutionOpTest, DepthMultiplierTest) {
  LiteRtOpT op;
  // Input [1, 4, 4, 2], Filter [1, 3, 3, 6] -> OutChannels 6. Multiplier 3.
  std::vector<Dims> input_shapes = {{1, 4, 4, 2}, {1, 3, 3, 6}, {6}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateDepthwiseConvOptions(tflite::Padding_VALID, 1, 1, 3);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 6));
}

}  // namespace
}  // namespace litert::internal
