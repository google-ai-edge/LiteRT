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

#include "litert/core/model/ops/convolution.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(ConvolutionOpTest, SimpleTestFloat32) {
  LiteRtOpT op;
  // Input [2, 2, 4, 1], Filter [3, 2, 2, 1] (OHWI), Bias [3]
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {3, 2, 2, 1}, {3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 2;
  options->stride_w = 2;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  // Expected output: [2, 1, 2, 3]
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 3));
}

TEST(ConvolutionOpTest, SimpleTestFloat32SingleThreaded) {
  // Logic should be same as SimpleTestFloat32, threading doesn't affect shape.
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {3, 2, 2, 1}, {3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 2;
  options->stride_w = 2;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 3));
}

TEST(ConvolutionOpTest, SimpleTestFloat32WithChannels) {
  LiteRtOpT op;
  // Input [2, 2, 4, 2], Filter [3, 2, 2, 2] (OHWI), Bias [3]
  std::vector<Dims> input_shapes = {{2, 2, 4, 2}, {3, 2, 2, 2}, {3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 2;
  options->stride_w = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 3));
}

TEST(ConvolutionOpTest, SimpleTestFloat32WithChannelsGrouped) {
  LiteRtOpT op;
  // Input [2, 2, 2, 2], Filter [2, 2, 2, 1] (OHWI) - 2 groups implies O=2,
  // I=2/2=1 per group? TFLite kernel comment: 2 groups. Filter OHWI: O is
  // output channels. I is input channels / groups. Input channels = 2. Groups
  // = 2. So I should be 1.
  std::vector<Dims> input_shapes = {{2, 2, 2, 2}, {2, 2, 2, 1}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  // Default strides are usually 1 if not specified in test but
  // BaseConvolutionOpModel defaults to 2x2? Checking BaseConvolutionOpModel
  // constructor: stride_width=2, stride_height=2. But
  // SimpleTestFloat32WithChannelsGrouped doesn't specify strides, so it uses
  // defaults (2x2). Wait, in conv_test.cc: ConvolutionOpModel m(...,
  // {TensorType_FLOAT32, {}}, 2, 2, ...); uses default strides?
  // BaseConvolutionOpModel default args: stride=2.
  // HOWEVER, SimpleTestFloat32WithChannelsGrouped code:
  // ConvolutionOpModel m(..., {TensorType_FLOAT32, {}});
  // It calls constructor with default args.
  options->stride_h = 2;
  options->stride_w = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output height = (2 - 2 + 0)/2 + 1 = 1.
  // Output width = (2 - 2 + 0)/2 + 1 = 1.
  // Output channels = 2 (from filter dim 0).
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 1, 2));
}

TEST(ConvolutionOpTest, InputAndFilterSameWidthHeight) {
  LiteRtOpT op;
  // Input [2, 2, 4, 1], Filter [1, 2, 4, 1]
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {1, 2, 4, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Defaults: stride 2x2, valid padding.
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 2;
  options->stride_w = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: (2 - 2)/2 + 1 = 1
  // W: (4 - 4)/2 + 1 = 1
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 1, 1));
}

TEST(ConvolutionOpTest, StrideTest) {
  LiteRtOpT op;
  // Input [2, 2, 4, 1], Filter [3, 2, 2, 1]
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {3, 2, 2, 1}, {3}};
  std::vector<Dims> output_shapes(1);

  // Explicit stride 1x1.
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: (2 - 2)/1 + 1 = 1
  // W: (4 - 2)/1 + 1 = 3
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 3, 3));
}

TEST(ConvolutionOpTest, PaddingTest) {
  LiteRtOpT op;
  // Input [1, 2, 4, 1], Filter [3, 2, 2, 1]
  std::vector<Dims> input_shapes = {{1, 2, 4, 1}, {3, 2, 2, 1}, {3}};
  std::vector<Dims> output_shapes(1);

  // SAME padding, stride 1.
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output size matches input size for stride 1 SAME padding.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 4, 3));
}

TEST(ConvolutionOpTest, PointwiseFloat32) {
  LiteRtOpT op;
  // Input [2, 2, 4, 2], Filter [1, 1, 1, 2] (1x1)
  std::vector<Dims> input_shapes = {{2, 2, 4, 2}, {1, 1, 1, 2}, {1}};
  std::vector<Dims> output_shapes(1);

  // Stride 1x1, VALID padding (though 1x1 valid is same as same)
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 4, 1));
}

TEST(ConvolutionOpTest, SimpleTestFloat32WithAnisotropicStrides) {
  LiteRtOpT op;
  // Input [1, 3, 6, 1], Filter [1, 2, 2, 1]
  std::vector<Dims> input_shapes = {{1, 3, 6, 1}, {1, 2, 2, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Stride W=3, H=1
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 3;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: (3 - 2)/1 + 1 = 2
  // W: (6 - 2)/3 + 1 = 1 + 1 = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(ConvolutionOpTest, HandCalculatedFloat32) {
  LiteRtOpT op;
  // Input [1, 3, 4, 1], Filter [1, 3, 3, 1]
  std::vector<Dims> input_shapes = {{1, 3, 4, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // SAME padding, stride 1
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 4, 1));
}

TEST(ConvolutionOpTest, HandCalculatedValidFloat32) {
  LiteRtOpT op;
  // Input [1, 3, 4, 1], Filter [1, 3, 3, 1]
  std::vector<Dims> input_shapes = {{1, 3, 4, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // VALID padding, stride 1
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: 3-3+1 = 1
  // W: 4-3+1 = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 2, 1));
}

TEST(ConvolutionOpTest, SimpleTestFloatWithDilation) {
  LiteRtOpT op;
  // Input [1, 9, 9, 1], Filter [1, 3, 3, 1]
  std::vector<Dims> input_shapes = {{1, 9, 9, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Dilation 3, stride 1, VALID
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;
  options->dilation_h_factor = 3;
  options->dilation_w_factor = 3;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Effective filter size = (3-1)*3 + 1 = 7.
  // H: (9 - 7)/1 + 1 = 3
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 1));
}

TEST(ConvolutionOpTest, ActivationReluTest) {
  LiteRtOpT op;
  // Input [1, 2, 4, 1], Filter [3, 2, 2, 1]
  std::vector<Dims> input_shapes = {{1, 2, 4, 1}, {3, 2, 2, 1}, {3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 2;
  options->stride_w = 2;
  options->fused_activation_function = tflite::ActivationFunctionType_RELU;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Activation doesn't change shape
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 2, 3));
}

TEST(ConvolutionOpTest, DilationAndPaddingTest) {
  LiteRtOpT op;
  // Input [1, 6, 6, 1], Filter [1, 3, 3, 1]
  std::vector<Dims> input_shapes = {{1, 6, 6, 1}, {1, 3, 3, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  // Dilation 2, Stride 1, SAME padding
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 1;
  options->stride_w = 1;
  options->dilation_h_factor = 2;
  options->dilation_w_factor = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output size matches input size for stride 1 SAME padding, regardless of
  // dilation? TFLite logic for SAME: output_size = ceil(input_size / stride).
  // Dilation affects the padding amount required, but the output size formula
  // for SAME generally aims to preserve input size when stride is 1.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 6, 6, 1));
}

TEST(ConvolutionOpTest, Conv3D) {
  LiteRtOpT op;
  // Input [1, 4, 4, 4, 1], Filter [2, 2, 2, 1, 2] (D, H, W, In, Out)
  // Bias [2]
  std::vector<Dims> input_shapes = {{1, 4, 4, 4, 1}, {2, 2, 2, 1, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_d = 1;
  options->stride_h = 1;
  options->stride_w = 1;
  options->dilation_d_factor = 1;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // (4-2)/1 + 1 = 3.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 3, 2));
}

TEST(ConvolutionOpTest, Conv3D_PaddingSame_Stride1) {
  LiteRtOpT op;
  // Input [1, 3, 4, 5, 2] (Batch, D, H, W, In)
  // Filter [2, 2, 2, 2, 2] (D, H, W, In, Out)
  std::vector<Dims> input_shapes = {{1, 3, 4, 5, 2}, {2, 2, 2, 2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_d = 1;
  options->stride_h = 1;
  options->stride_w = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 4, 5, 2));
}

TEST(ConvolutionOpTest, Conv3D_Stride2) {
  LiteRtOpT op;
  // Input [2, 2, 3, 4, 2]
  // Filter [2, 2, 2, 2, 2]
  std::vector<Dims> input_shapes = {{2, 2, 3, 4, 2}, {2, 2, 2, 2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_d = 2;
  options->stride_h = 2;
  options->stride_w = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // D: ceil( (2-2+1)/2 ) = 1
  // H: ceil( (3-2+1)/2 ) = 1
  // W: ceil( (4-2+1)/2 ) = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 1, 2, 2));
}

TEST(ConvolutionOpTest, Conv3D_Dilation) {
  LiteRtOpT op;
  // Input [1, 5, 5, 5, 1]
  // Filter [2, 2, 2, 1, 1] (D, H, W, In, Out)
  std::vector<Dims> input_shapes = {{1, 5, 5, 5, 1}, {2, 2, 2, 1, 1}, {1}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_d = 1;
  options->stride_h = 1;
  options->stride_w = 1;
  options->dilation_d_factor = 2;
  options->dilation_h_factor = 2;
  options->dilation_w_factor = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Effective filter size = (2-1)*2 + 1 = 3
  // Out = 5 - 3 + 1 = 3
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 3, 1));
}

TEST(ConvolutionOpTest, TransposeConv) {
  LiteRtOpT op;
  // Inputs: OutputShape, Weights, InputData
  std::vector<Dims> input_shapes = {{4}, {2, 3, 3, 1}, {1, 3, 3, 1}};
  std::vector<Dims> output_shapes(1);

  // Create constant output shape tensor
  LiteRtTensorT out_shape_tensor;
  int32_t shape_data[] = {1, 5, 5, 2};
  SetWeightsFromOwnedBuffer(
      out_shape_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(shape_data),
                               sizeof(shape_data)));

  op.Inputs().push_back(&out_shape_tensor);

  ASSERT_EQ(InferTransposeConv(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 2));
}

TEST(ConvolutionOpTest, Conv3DTranspose) {
  LiteRtOpT op;
  // Input 0: OutputShape [5] -> {1, 5, 5, 5, 2}
  // Input 1: Filter [2, 3, 3, 1, 2] (D, H, W, In, Out)
  // Input 2: Data [1, 3, 3, 3, 1]

  std::vector<Dims> input_shapes = {{5}, {2, 3, 3, 1, 2}, {1, 3, 3, 3, 1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT out_shape_tensor;
  int32_t shape_data[] = {1, 5, 5, 5, 2};
  SetWeightsFromOwnedBuffer(
      out_shape_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(shape_data),
                               sizeof(shape_data)));
  op.Inputs().push_back(&out_shape_tensor);

  ASSERT_EQ(
      InferConv3DTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 5, 2));
}

TEST(ConvolutionOpTest, TransposeConv_DynamicOutputShape) {
  LiteRtOpT op;
  // Inputs: OutputShape (dynamic), Weights, InputData
  // Weights: [2, 3, 3, 1] (Out=2, H=3, W=3, In=1)
  // InputData: [4, 3, 3, 1] (Batch=4, H=3, W=3, In=1)
  std::vector<Dims> input_shapes = {{4}, {2, 3, 3, 1}, {4, 3, 3, 1}};
  std::vector<Dims> output_shapes(1);

  // OutputShape tensor has no buffer (dynamic)
  LiteRtTensorT out_shape_tensor;
  op.Inputs().push_back(&out_shape_tensor);

  ASSERT_EQ(InferTransposeConv(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  // Expect rank 4.
  // Batch (dim 0) from input -> 4.
  // Channels (dim 3) from weights (dim 0) -> 2.
  // H, W unknown (-1).
  EXPECT_THAT(output_shapes[0], ElementsAre(4, -1, -1, 2));
}

TEST(ConvolutionOpTest, Conv3DTranspose_DynamicOutputShape) {
  LiteRtOpT op;
  // Inputs: OutputShape (dynamic), Filter, InputData
  // Filter: [2, 3, 3, 1, 5] (D, H, W, In, Out) -> Out=5
  // InputData: [3, 4, 4, 4, 1] (Batch=3, D, H, W, In)
  std::vector<Dims> input_shapes = {{5}, {2, 3, 3, 1, 5}, {3, 4, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT out_shape_tensor;
  op.Inputs().push_back(&out_shape_tensor);

  ASSERT_EQ(
      InferConv3DTranspose(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Expect rank 5.
  // Batch (dim 0) from input -> 3.
  // Channels (dim 4) from filter (dim 4) -> 5.
  EXPECT_THAT(output_shapes[0], ElementsAre(3, -1, -1, -1, 5));
}

}  // namespace
}  // namespace litert::internal
