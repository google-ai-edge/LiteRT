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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

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

TEST(ConvolutionOpTest, InferConv2DInsufficientInputs) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 2, 1}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(ConvolutionOpTest, InferConv2DInvalidRanks) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 2, 1}, {1, 2, 1}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(ConvolutionOpTest, InferConv2DMissingOptions) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 2, 1}, {1, 2, 2, 1}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
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

TEST(ConvolutionOpTest, DepthwiseConv2D) {
  LiteRtOpT op;
  // Input [1, 3, 3, 2], Filter [1, 2, 2, 4] (1, H, W, Out)
  std::vector<Dims> input_shapes = {{1, 3, 3, 2}, {1, 2, 2, 4}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;
  options->depth_multiplier = 2;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Out_H = (3 - 2)/1 + 1 = 2
  // Out_W = (3 - 2)/1 + 1 = 2
  // Out_C = 4
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 4));
}

TEST(ConvolutionOpTest, ReportedIssue) {
  LiteRtOpT op;
  // Input [5, 1, 1, 1], Filter [32, 3, 3, 1], Bias [32]
  std::vector<Dims> input_shapes = {{5, 1, 1, 1}, {32, 3, 3, 1}, {32}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 2;
  options->stride_w = 2;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // TFLite's SAME padding formula: out = (in + stride - 1) / stride.
  // For in=1, stride=2, out = (1 + 2 - 1) / 2 = 1.
  ASSERT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(5, 1, 1, 32));
}

TEST(ConvolutionOpTest, Conv1DLoweredAsConv2DWithStride2) {
  LiteRtOpT op;
  // 1D convolution (e.g. Whisper encoder Conv1d) lowered to Conv2D with
  // H=1, KH=1, and stride=(2, 2).
  std::vector<Dims> input_shapes = {{1, 1, 3002, 512}, {512, 1, 3, 512}, {512}};
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
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1500, 512));
}

TEST(ConvolutionOpTest, Conv3DStrideLargerThanInputSucceeds) {
  LiteRtOpT op;
  // Input [1, 1, 2, 4, 2] (B, D, H, W, In), Filter [1, 2, 2, 2, 3] (D, H, W,
  // In, Out).
  std::vector<Dims> input_shapes = {{1, 1, 2, 4, 2}, {1, 2, 2, 2, 3}, {3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_d = 2;
  options->stride_h = 3;
  options->stride_w = 2;
  options->dilation_d_factor = 1;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 2, 3));
}

TEST(ConvolutionOpTest, ChannelMismatchFailure) {
  LiteRtOpT op;
  // Input [1, 4, 4, 3], Filter [32, 3, 3, 2] -> mismatch 3 vs 2.
  std::vector<Dims> input_shapes = {{1, 4, 4, 3}, {32, 3, 3, 2}, {32}};
  std::vector<Dims> output_shapes(1);
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 1;
  options->stride_w = 1;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ConvolutionOpTest, BiasMismatchFailure) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1], Filter [32, 3, 3, 1], Bias [16] -> mismatch 32 vs 16.
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}, {32, 3, 3, 1}, {16}};
  std::vector<Dims> output_shapes(1);
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_h = 1;
  options->stride_w = 1;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ConvolutionOpTest, InvalidOutputSizeFailure) {
  LiteRtOpT op;
  // VALID padding with filter larger than input.
  std::vector<Dims> input_shapes = {{1, 2, 2, 1}, {32, 3, 3, 1}, {32}};
  std::vector<Dims> output_shapes(1);
  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(DepthwiseConvolutionOpTest, MultiplierMismatchFailure) {
  LiteRtOpT op;
  // In=2, Out=4, Multiplier=1 -> mismatch 4 != 2*1.
  std::vector<Dims> input_shapes = {{1, 3, 3, 2}, {1, 2, 2, 4}, {4}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::DepthwiseConv2DOptionsT>();
  options->padding = tflite::Padding_VALID;
  options->stride_h = 1;
  options->stride_w = 1;
  options->depth_multiplier = 1;

  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ConvolutionOpTest, EmptyBiasSucceeds) {
  LiteRtOpT op;
  // Input [2, 2, 4, 1], Filter [3, 2, 2, 1] (OHWI), Bias empty {}
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {3, 2, 2, 1}, {}};
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

TEST(ConvolutionOpTest, NoBiasSucceeds) {
  LiteRtOpT op;
  // Input [2, 2, 4, 1], Filter [3, 2, 2, 1] (OHWI)
  std::vector<Dims> input_shapes = {{2, 2, 4, 1}, {3, 2, 2, 1}};
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

struct Conv2DOverflowCase {
  std::string test_name;
  std::vector<Dims> input_shapes;
  tflite::Padding padding;
  int32_t stride_h;
  int32_t stride_w;
};

using Conv2DOverflowTest = ::testing::TestWithParam<Conv2DOverflowCase>;

TEST_P(Conv2DOverflowTest, RejectsElementCountOverflow) {
  const auto& param = GetParam();
  LiteRtOpT op;
  std::vector<Dims> input_shapes = param.input_shapes;
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv2DOptionsT>();
  options->padding = param.padding;
  options->stride_h = param.stride_h;
  options->stride_w = param.stride_w;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    Conv2DOverflowTests, Conv2DOverflowTest,
    ::testing::Values(Conv2DOverflowCase{"PointwiseOutputElementCountOverflow",
                                         {{65536, 32768, 1, 1}, {1, 1, 1, 1}},
                                         tflite::Padding_VALID,
                                         1,
                                         1},
                      Conv2DOverflowCase{"StridedInputElementCountOverflow",
                                         {{1, 65536, 32768, 1}, {1, 2, 2, 1}},
                                         tflite::Padding_SAME,
                                         4,
                                         4}),
    [](const ::testing::TestParamInfo<Conv2DOverflowCase>& info) {
      return info.param.test_name;
    });

TEST(ConvolutionOpTest, InferConv3DRejectsInputElementCountOverflow) {
  // Strided Conv3D where output and im2col fit in int32_t, but input overflows
  // int32_t (1 * 1 * 65536 * 32768 * 1 = 2^31 > INT32_MAX).
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 1, 65536, 32768, 1}, {1, 2, 2, 1, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::Conv3DOptionsT>();
  options->padding = tflite::Padding_SAME;
  options->stride_d = 1;
  options->stride_h = 4;
  options->stride_w = 4;
  options->dilation_d_factor = 1;
  options->dilation_h_factor = 1;
  options->dilation_w_factor = 1;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Conv3DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferConv3D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

struct TransposeConvErrorCase {
  std::string test_name;
  std::vector<Dims> input_shapes;
  std::vector<int32_t> const_output_shape;
  bool set_options = false;
};

using TransposeConvErrorTest = ::testing::TestWithParam<TransposeConvErrorCase>;

TEST_P(TransposeConvErrorTest, RejectsOverflowAndInvalidShapes) {
  const auto& param = GetParam();
  LiteRtOpT op;
  std::vector<Dims> input_shapes = param.input_shapes;
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT out_shape_tensor;
  if (!param.const_output_shape.empty()) {
    SetWeightsFromOwnedBuffer(
        out_shape_tensor.Weights(),
        OwningBufferRef<uint8_t>(
            reinterpret_cast<const uint8_t*>(param.const_output_shape.data()),
            param.const_output_shape.size() * sizeof(int32_t)));
  }
  op.Inputs().push_back(&out_shape_tensor);

  if (param.set_options) {
    auto options = std::make_unique<tflite::TransposeConvOptionsT>();
    options->padding = tflite::Padding_SAME;
    options->stride_h = 1;
    options->stride_w = 1;
    TflOptions tfl_options;
    tfl_options.type = tflite::BuiltinOptions_TransposeConvOptions;
    tfl_options.value = options.release();
    SetTflOptions(op, std::move(tfl_options));
  }

  EXPECT_EQ(InferTransposeConv(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

constexpr int32_t kHugeDim = 46341;  // 46341 * 46341 > INT32_MAX

INSTANTIATE_TEST_SUITE_P(
    TransposeConvErrorTests, TransposeConvErrorTest,
    ::testing::Values(
        TransposeConvErrorCase{
            "ConstantOutputShapeExceedingInt32Max",
            {{4}, {1, 1, 1, 1}, {1, 23171, 23171, 1}},
            {1, 46342, 46342, 1},
        },
        TransposeConvErrorCase{
            "InputTensorExceedingInt32Max",
            {{4}, {1, 1, 1, 1}, {1, kHugeDim, kHugeDim, 1}},
            {},
        },
        TransposeConvErrorCase{
            "Col2ImElementCountOverflow",
            {{4}, {2048, 2, 2, 1}, {1, 512, 512, 1}},
            {},
        },
        TransposeConvErrorCase{
            "InconsistentSpatialDimensionsWithOptions",
            {{4}, {1, 1, 1, 1}, {1, 2, 2, 1}},
            {1, 3, 3, 1},
            /*set_options=*/true,
        }),
    [](const ::testing::TestParamInfo<TransposeConvErrorCase>& info) {
      return info.param.test_name;
    });

TEST(ConvolutionOpTest,
     RejectsInvalidOutputShapesCountAndMissingConv3DTransposeInputs) {
  LiteRtOpT op;
  std::vector<Dims> empty_outputs;
  std::vector<Dims> conv2d_inputs = {{1, 4, 4, 1}, {1, 2, 2, 1}};
  std::vector<Dims> conv3d_inputs = {{1, 4, 4, 4, 1}, {1, 2, 2, 1, 1}};
  std::vector<Dims> tconv_inputs = {{4}, {1, 2, 2, 1}, {1, 4, 4, 1}};
  std::vector<Dims> tconv3d_inputs = {{5}, {2, 2, 2, 1, 2}, {1, 3, 3, 3, 1}};

  EXPECT_EQ(InferConv2D(op, absl::MakeSpan(conv2d_inputs), empty_outputs),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      InferDepthwiseConv2D(op, absl::MakeSpan(conv2d_inputs), empty_outputs),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(InferTransposeConv(op, absl::MakeSpan(tconv_inputs), empty_outputs),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(InferConv3D(op, absl::MakeSpan(conv3d_inputs), empty_outputs),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      InferConv3DTranspose(op, absl::MakeSpan(tconv3d_inputs), empty_outputs),
      kLiteRtStatusErrorInvalidArgument);

  // Conv3DTranspose with empty op.Inputs() falls back safely without crashing.
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(
      InferConv3DTranspose(op, absl::MakeSpan(tconv3d_inputs), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, -1, -1, -1, 2));
}

// Aggregate with 1x1x1 unit dimensions and zero padding as defaults so tests
// only specify the TransposeConvParams fields they vary via designated
// initializers.
struct UnitTransposeConvParams {
  int32_t batch = 1;
  int32_t in_h = 1;
  int32_t in_w = 1;
  int32_t in_c = 1;
  int32_t out_h = 1;
  int32_t out_w = 1;
  int32_t out_c = 1;
  int32_t filter_h = 1;
  int32_t filter_w = 1;
  int32_t stride_h = 1;
  int32_t stride_w = 1;
  int32_t pad_t = 0;
  int32_t pad_l = 0;
};

TransposeConvParams MakeTransposeConvParams(
    const UnitTransposeConvParams& overrides = {}) {
  return TransposeConvParams{
      .batch = overrides.batch,
      .in_h = overrides.in_h,
      .in_w = overrides.in_w,
      .in_c = overrides.in_c,
      .out_h = overrides.out_h,
      .out_w = overrides.out_w,
      .out_c = overrides.out_c,
      .filter_h = overrides.filter_h,
      .filter_w = overrides.filter_w,
      .stride_h = overrides.stride_h,
      .stride_w = overrides.stride_w,
      .pad_t = overrides.pad_t,
      .pad_l = overrides.pad_l,
  };
}

TEST(ConvolutionOpTest, ReferenceTransposeConvFloatAppliesBiasAndRelu6) {
  const std::array<float, 1> input = {2.0f};
  const std::array<float, 4> filter = {-2.0f, 1.0f, 3.0f, 5.0f};
  const std::array<float, 1> bias = {1.0f};
  std::vector<float> out(4, 0.0f);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ(
      ReferenceTransposeConv(input, filter, bias, absl::MakeSpan(out), params,
                             tflite::ActivationFunctionType_RELU6),
      kLiteRtStatusOk);

  // Pre-activation: [-3, 3, 7, 11] -> RELU6: [0, 3, 6, 6].
  EXPECT_THAT(out, ElementsAre(0.0f, 3.0f, 6.0f, 6.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvFloatWithoutBiasScalesFilterByInput) {
  const std::array<float, 1> input = {2.0f};
  const std::array<float, 4> filter = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4, -5.0f);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ(
      ReferenceTransposeConv(input, filter, /*bias=*/{}, absl::MakeSpan(out),
                             params, tflite::ActivationFunctionType_NONE),
      kLiteRtStatusOk);

  EXPECT_THAT(out, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvFloatAccumulatesOverlapsAndClipsPadding) {
  // ox = ix + fx - pad_l (pad_l=1):
  // ix=0 (2.0): fx=0 -> ox=-1 (clipped), fx=1 -> ox=0 (+= 2*5 = 10).
  // ix=1 (3.0): fx=0 -> ox=0 (+= 3*10 = 30 -> 40), fx=1 -> ox=1 (+= 3*5 = 15).
  const std::array<float, 2> input = {2.0f, 3.0f};
  const std::array<float, 2> filter = {10.0f, 5.0f};
  std::vector<float> out(2, -1.0f);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.in_w = 2, .out_w = 2, .filter_w = 2, .pad_l = 1});

  EXPECT_EQ(
      ReferenceTransposeConv(input, filter, /*bias=*/{}, absl::MakeSpan(out),
                             params, tflite::ActivationFunctionType_NONE),
      kLiteRtStatusOk);

  EXPECT_THAT(out, ElementsAre(40.0f, 15.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvFloatMultiChannelStrideAddsPerChannelBias) {
  const std::array<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
  // Filter layout [out_c=2, fh=1, fw=2, in_c=2].
  const std::array<float, 8> filter = {1.0f, 2.0f, 3.0f, 4.0f,
                                       5.0f, 6.0f, 7.0f, 8.0f};
  const std::array<float, 2> bias = {0.5f, -0.5f};
  std::vector<float> out(8, 0.0f);
  const TransposeConvParams params = MakeTransposeConvParams({.in_w = 2,
                                                              .in_c = 2,
                                                              .out_w = 4,
                                                              .out_c = 2,
                                                              .filter_w = 2,
                                                              .stride_w = 2});

  EXPECT_EQ(ReferenceTransposeConv(input, filter, bias, absl::MakeSpan(out),
                                   params, tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);

  EXPECT_THAT(
      out, ElementsAre(5.5f, 16.5f, 11.5f, 22.5f, 11.5f, 38.5f, 25.5f, 52.5f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvQuantizedAppliesZeroPointReluAndRequantizes) {
  // input_q = 15, zp = 5 -> (15 - 5) = 10; scale = 0.5 -> real 5.0
  // filter_q = {-3, 2, 1, 4}, weight_scale = 0.25 -> effective_scale = 0.125
  // acc = {-30, 20, 10, 40}, bias = {2} -> {-28, 22, 12, 42}
  // real = {-3.5, 2.75, 1.5, 5.25}; RELU -> {0.0, 2.75, 1.5, 5.25}
  // output_scale = 0.5, output_zp = -10 ->
  //   round_ties_to_even(real/0.5 - 10) = {-10, -4, -7, 0}
  const std::array<int8_t, 1> input = {15};
  const std::array<int8_t, 4> filter = {-3, 2, 1, 4};
  const std::array<int32_t, 1> bias = {2};
  std::vector<int8_t> out(4, 0);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, bias, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_RELU,
                AffineQuantSpec{/*scale=*/0.5f, /*zero_point=*/5},
                /*weight_scale=*/0.25f,
                AffineQuantSpec{/*scale=*/0.5f, /*zero_point=*/-10})),
            kLiteRtStatusOk);

  EXPECT_THAT(out, ElementsAre(-10, -4, -7, 0));
}

struct TransposeConvInvalidDimsCase {
  std::string test_name;
  int32_t batch = 1;
  int32_t out_c = 1;
};

using ReferenceTransposeConvInvalidDimsTest =
    ::testing::TestWithParam<TransposeConvInvalidDimsCase>;

TEST_P(ReferenceTransposeConvInvalidDimsTest,
       LeavesOutputUntouchedOnInvalidArgs) {
  const TransposeConvInvalidDimsCase& test_case = GetParam();
  const std::array<float, 1> input = {2.0f};
  const std::array<float, 4> filter = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4, -7.0f);
  const TransposeConvParams params =
      MakeTransposeConvParams({.batch = test_case.batch,
                               .out_h = 2,
                               .out_w = 2,
                               .out_c = test_case.out_c,
                               .filter_h = 2,
                               .filter_w = 2});

  EXPECT_EQ(
      ReferenceTransposeConv(input, filter, /*bias=*/{}, absl::MakeSpan(out),
                             params, tflite::ActivationFunctionType_NONE),
      kLiteRtStatusErrorInvalidArgument);

  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f, -7.0f, -7.0f));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceTransposeConvInvalidDimsTests,
    ReferenceTransposeConvInvalidDimsTest,
    ::testing::Values(TransposeConvInvalidDimsCase{.test_name = "NegativeBatch",
                                                   .batch = -1},
                      TransposeConvInvalidDimsCase{
                          .test_name = "ZeroOutputChannels", .out_c = 0}),
    [](const ::testing::TestParamInfo<TransposeConvInvalidDimsCase>& info) {
      return info.param.test_name;
    });

TEST(ConvolutionOpTest,
     ReferenceTransposeConvUndersizedSpansLeaveOutputUntouched) {
  const std::array<float, 1> input = {2.0f};
  const std::array<float, 4> filter = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4, -7.0f);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ(ReferenceTransposeConv(/*input=*/{}, filter, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ReferenceTransposeConv(input, absl::MakeConstSpan(filter).first(2),
                                   /*bias=*/{}, absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ReferenceTransposeConv(input, filter, /*bias=*/{},
                                   absl::MakeSpan(out).first(3), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusErrorInvalidArgument);
  // Undersized non-empty bias span is also rejected.
  const TransposeConvParams two_ch_params =
      MakeTransposeConvParams({.out_c = 2});
  const std::array<float, 2> two_ch_filter = {1.0f, 2.0f};
  const std::array<float, 1> short_bias = {0.5f};
  EXPECT_EQ(ReferenceTransposeConv(input, two_ch_filter, short_bias,
                                   absl::MakeSpan(out), two_ch_params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f, -7.0f, -7.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvQuantizedUndersizedSpansLeaveOutputUntouched) {
  const std::array<int8_t, 1> input = {10};
  const std::array<int8_t, 2> filter = {1, 2};
  const std::array<int32_t, 1> short_bias = {1};
  std::vector<int8_t> out(2, -9);
  const TransposeConvParams params = MakeTransposeConvParams({.out_c = 2});

  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                /*input=*/{}, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
                /*weight_scale=*/0.5f, AffineQuantSpec{0.5f, 0})),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, absl::MakeConstSpan(filter).first(1), /*bias=*/{},
                absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
                /*weight_scale=*/0.5f, AffineQuantSpec{0.5f, 0})),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, short_bias, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
                /*weight_scale=*/0.5f, AffineQuantSpec{0.5f, 0})),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      (ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
          input, filter, /*bias=*/{}, absl::MakeSpan(out).first(1), params,
          tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
          /*weight_scale=*/0.5f, AffineQuantSpec{0.5f, 0})),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(out, ElementsAre(-9, -9));
}

TEST(ConvolutionOpTest, ReferenceTransposeConvReluN1To1AndUnsupportedFaf) {
  const std::array<float, 1> input = {1.0f};
  const std::array<float, 4> filter = {-2.5f, -0.5f, 0.5f, 2.5f};
  std::vector<float> out(4, -7.0f);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ(ReferenceTransposeConv(input, filter, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU_N1_TO_1),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(-1.0f, -0.5f, 0.5f, 1.0f));

  std::fill(out.begin(), out.end(), -7.0f);
  EXPECT_EQ(
      ReferenceTransposeConv(input, filter, /*bias=*/{}, absl::MakeSpan(out),
                             params, tflite::ActivationFunctionType_TANH),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f, -7.0f, -7.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvQuantizedReluN1To1AndUnsupportedFaf) {
  // input = 4 (zp=0, scale=0.5) -> real 2.0
  // filter = {-4, -1, 1, 4} (weight_scale=0.25) -> effective_scale = 0.125
  // acc = {-16, -4, 4, 16} -> real = {-2.0, -0.5, 0.5, 2.0}
  // RELU_N1_TO_1 -> {-1.0, -0.5, 0.5, 1.0}; output scale=0.5, zp=0 ->
  // {-2,-1,1,2}.
  const std::array<int8_t, 1> input = {4};
  const std::array<int8_t, 4> filter = {-4, -1, 1, 4};
  std::vector<int8_t> out(4, -9);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ(
      (ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
          input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
          tflite::ActivationFunctionType_RELU_N1_TO_1, AffineQuantSpec{0.5f, 0},
          /*weight_scale=*/0.25f, AffineQuantSpec{0.5f, 0})),
      kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(-2, -1, 1, 2));

  std::fill(out.begin(), out.end(), -9);
  EXPECT_EQ(
      (ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
          input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
          tflite::ActivationFunctionType_SIGN_BIT, AffineQuantSpec{0.5f, 0},
          /*weight_scale=*/0.25f, AffineQuantSpec{0.5f, 0})),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
                /*weight_scale=*/0.25f, AffineQuantSpec{0.5f, 200})),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(out, ElementsAre(-9, -9, -9, -9));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvLargeStrideAndExtremePaddingAvoidInt32Overflow) {
  const std::array<float, 2> input = {2.0f, 3.0f};
  const std::array<float, 2> filter = {5.0f, 7.0f};
  std::vector<float> out(2, -1.0f);

  // Strides > INT16_MAX (e.g., 32,768 delegated by XNNPACK) and INT32_MAX are
  // accepted in 64-bit coordinate arithmetic; only ih=0 lands in [0, out_h).
  const TransposeConvParams xnnpack_stride_params = MakeTransposeConvParams(
      {.in_h = 2, .out_h = 2, .filter_h = 2, .stride_h = 32768});
  EXPECT_EQ(ReferenceTransposeConv(input, filter, /*bias=*/{},
                                   absl::MakeSpan(out), xnnpack_stride_params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(10.0f, 14.0f));

  TransposeConvParams large_stride_params = xnnpack_stride_params;
  large_stride_params.stride_h = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(ReferenceTransposeConv(input, filter, /*bias=*/{},
                                   absl::MakeSpan(out), large_stride_params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(10.0f, 14.0f));

  // pad_t = INT32_MIN / pad_l = INT32_MIN would overflow `- p.pad_t` in
  // int32_t; in int64_t all coordinates exceed out_h/out_w so output stays 0.
  const TransposeConvParams extreme_pad_params =
      MakeTransposeConvParams({.out_h = 2,
                               .pad_t = std::numeric_limits<int32_t>::min(),
                               .pad_l = std::numeric_limits<int32_t>::min()});
  EXPECT_EQ(ReferenceTransposeConv(absl::MakeConstSpan(input).first(1),
                                   absl::MakeConstSpan(filter).first(1),
                                   /*bias=*/{}, absl::MakeSpan(out),
                                   extreme_pad_params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(0.0f, 0.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvQuantizedZeroOutChannelsLeavesOutputUntouched) {
  const std::array<int8_t, 1> input = {10};
  const std::array<int8_t, 4> filter = {1, 2, 3, 4};
  std::vector<int8_t> out(4, -9);
  const TransposeConvParams params = MakeTransposeConvParams(
      {.out_h = 2, .out_w = 2, .out_c = 0, .filter_h = 2, .filter_w = 2});

  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 0},
                /*weight_scale=*/0.5f, AffineQuantSpec{0.5f, 0})),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_THAT(out, ElementsAre(-9, -9, -9, -9));
}

TEST(ConvolutionOpTest, PrepareTransposeConvGeometryRejectsOverflow) {
  constexpr int32_t kBigDim = sizeof(size_t) >= 8 ? (1 << 30) : (1 << 15);
  const TransposeConvParams input_overflow =
      MakeTransposeConvParams({.batch = kBigDim, .in_h = kBigDim, .in_w = 4});
  EXPECT_FALSE(PrepareTransposeConvGeometry(input_overflow, sizeof(float),
                                            sizeof(float), sizeof(float))
                   .HasValue());

  const TransposeConvParams filter_overflow = MakeTransposeConvParams(
      {.in_c = 4, .out_c = kBigDim, .filter_h = kBigDim});
  EXPECT_FALSE(PrepareTransposeConvGeometry(filter_overflow, sizeof(float),
                                            sizeof(float), sizeof(float))
                   .HasValue());

  const TransposeConvParams output_overflow =
      MakeTransposeConvParams({.out_h = kBigDim, .out_w = kBigDim, .out_c = 4});
  EXPECT_FALSE(PrepareTransposeConvGeometry(output_overflow, sizeof(float),
                                            sizeof(float), sizeof(float))
                   .HasValue());

  // Output fits in int8_t bytes but overflows int64_t accumulator byte count.
  const TransposeConvParams acc_overflow =
      MakeTransposeConvParams({.out_h = kBigDim,
                               .out_w = kBigDim,
                               .out_c = sizeof(size_t) >= 8 ? 4 : 1});
  EXPECT_TRUE(PrepareTransposeConvGeometry(acc_overflow, sizeof(int8_t),
                                           sizeof(int8_t), sizeof(int8_t))
                  .HasValue());
  EXPECT_FALSE(PrepareTransposeConvGeometry(acc_overflow, sizeof(int8_t),
                                            sizeof(int8_t), sizeof(int8_t),
                                            sizeof(int64_t))
                   .HasValue());
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvFloatHandlesMultiBatchVerticalStrideOverlapAndPad) {
  // batch=2, in=[2, 2, 1, 1], filter=[1, 3, 1, 1], out=[2, 4, 1, 1],
  // stride_h=2, pad_t=1. For each batch:
  //   oh=0: in[0]*2 + bias
  //   oh=1: in[0]*3 + in[1]*1 + bias (vertical overlap)
  //   oh=2: in[1]*2 + bias
  //   oh=3: in[1]*3 + bias  (ih=0, kh=0 -> oh=-1 clipped)
  const std::array<float, 4> input = {2.0f, 3.0f, -1.0f, 4.0f};
  const std::array<float, 3> filter = {1.0f, 2.0f, 3.0f};
  const std::array<float, 1> bias = {1.0f};
  std::vector<float> out(8, 0.0f);
  const TransposeConvParams params = MakeTransposeConvParams({.batch = 2,
                                                              .in_h = 2,
                                                              .out_h = 4,
                                                              .filter_h = 3,
                                                              .stride_h = 2,
                                                              .pad_t = 1});

  EXPECT_EQ(ReferenceTransposeConv(input, filter, bias, absl::MakeSpan(out),
                                   params, tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out,
              ElementsAre(5.0f, 10.0f, 7.0f, 10.0f, -1.0f, 2.0f, 9.0f, 13.0f));
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvQuantizedRejectsInvalidInputQuantAndWeightScale) {
  const std::array<int8_t, 1> input = {10};
  const std::array<int8_t, 1> filter = {1};
  std::vector<int8_t> out(1, -9);
  const TransposeConvParams params = MakeTransposeConvParams();
  const AffineQuantSpec valid_quant{0.5f, 0};

  // Invalid input_quant scale and out-of-range input_quant zero_point.
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.0f, 0},
                /*weight_scale=*/0.5f, valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.5f, 200},
                /*weight_scale=*/0.5f, valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
  // Non-positive and non-finite weight_scale.
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, valid_quant,
                /*weight_scale=*/0.0f, valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, valid_quant,
                /*weight_scale=*/-0.5f, valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                input, filter, /*bias=*/{}, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, valid_quant,
                std::numeric_limits<float>::quiet_NaN(), valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(out, ElementsAre(-9));
}

TEST(ConvolutionOpTest, ReferenceTransposeConvQuantizedSupportsInt16Int8) {
  const std::array<int16_t, 1> input = {10};
  const std::array<int8_t, 2> filter = {2, 3};  // [out_c=2, 1, 1, in_c=1]
  const std::array<int32_t, 2> bias = {5, -4};
  std::vector<int16_t> out(2, -9);
  const TransposeConvParams params = MakeTransposeConvParams({.out_c = 2});
  const AffineQuantSpec quant{0.5f, 0};

  // Channel 0: (10 * 2 + 5) * 0.25 = 6.25 -> / 0.5 = 12.5 -> ties-to-even 12.
  // Channel 1: (10 * 3 - 4) * 0.25 = 6.5 -> / 0.5 = 13.
  EXPECT_EQ((ReferenceTransposeConvQuantized<int16_t, int8_t, int32_t>(
                input, filter, bias, absl::MakeSpan(out), params,
                tflite::ActivationFunctionType_NONE, quant,
                /*weight_scale=*/0.5f, quant)),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(12, 13));
}

TEST(ConvolutionOpTest, ReferenceTransposeConvAcceptsZeroBatch) {
  const TransposeConvParams params = MakeTransposeConvParams({.batch = 0});
  const std::array<float, 1> float_filter = {2.0f};
  const std::array<float, 1> float_bias = {1.0f};
  EXPECT_EQ(ReferenceTransposeConv(/*input=*/{}, float_filter, float_bias,
                                   /*output=*/{}, params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_EQ(ReferenceTransposeConv(/*input=*/{}, float_filter, /*bias=*/{},
                                   /*output=*/{}, params,
                                   tflite::ActivationFunctionType_RELU),
            kLiteRtStatusOk);

  // Quantized empty batch also succeeds with empty input/output spans.
  const std::array<int8_t, 1> quant_filter = {2};
  const AffineQuantSpec valid_quant{0.5f, 0};
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                /*input=*/{}, quant_filter, /*bias=*/{}, /*output=*/{}, params,
                tflite::ActivationFunctionType_NONE, valid_quant,
                /*weight_scale=*/0.5f, valid_quant)),
            kLiteRtStatusOk);

  // Empty batch still validates filter capacity, activation, and quantization.
  EXPECT_EQ(ReferenceTransposeConv(/*input=*/{}, /*filter=*/{}, /*bias=*/{},
                                   /*output=*/{}, params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ReferenceTransposeConv(/*input=*/{}, float_filter, /*bias=*/{},
                                   /*output=*/{}, params,
                                   tflite::ActivationFunctionType_SIGN_BIT),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ((ReferenceTransposeConvQuantized<int8_t, int8_t, int32_t>(
                /*input=*/{}, quant_filter, /*bias=*/{}, /*output=*/{}, params,
                tflite::ActivationFunctionType_NONE, AffineQuantSpec{0.0f, 0},
                /*weight_scale=*/0.5f, valid_quant)),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(ConvolutionOpTest,
     ReferenceTransposeConvFloatClampsOverflowToActivationBounds) {
  constexpr float kMax = std::numeric_limits<float>::max();
  constexpr float kLowest = std::numeric_limits<float>::lowest();
  const TransposeConvParams params = MakeTransposeConvParams();

  // FLT_MAX * 2.0f overflows to +inf; TFLite builtins clamp to FLT_MAX for
  // NONE and RELU, 6 for RELU6, and 1 for RELU_N1_TO_1.
  const std::array<float, 1> max_in = {kMax};
  const std::array<float, 1> weight_two = {2.0f};
  std::vector<float> out(1, 0.0f);
  EXPECT_EQ(ReferenceTransposeConv(max_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(kMax));

  EXPECT_EQ(ReferenceTransposeConv(max_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(kMax));

  EXPECT_EQ(ReferenceTransposeConv(max_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU6),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(6.0f));

  EXPECT_EQ(ReferenceTransposeConv(max_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU_N1_TO_1),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(1.0f));

  // Negative overflow (-FLT_MAX * 2.0f = -inf) clamps to -FLT_MAX for NONE,
  // 0 for RELU, and -1 for RELU_N1_TO_1.
  const std::array<float, 1> min_in = {kLowest};
  EXPECT_EQ(ReferenceTransposeConv(min_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(kLowest));

  EXPECT_EQ(ReferenceTransposeConv(min_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(0.0f));

  EXPECT_EQ(ReferenceTransposeConv(min_in, weight_two, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU_N1_TO_1),
            kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(-1.0f));

  // Overflow occurring during bias addition also clamps after bias.
  const std::array<float, 1> weight_one = {1.0f};
  const std::array<float, 1> max_bias = {kMax};
  EXPECT_EQ(
      ReferenceTransposeConv(max_in, weight_one, max_bias, absl::MakeSpan(out),
                             params, tflite::ActivationFunctionType_NONE),
      kLiteRtStatusOk);
  EXPECT_THAT(out, ElementsAre(kMax));

  // NaN propagates unchanged, matching tflite::ActivationFunctionWithMinMax.
  const std::array<float, 1> nan_in = {std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(ReferenceTransposeConv(nan_in, weight_one, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_NONE),
            kLiteRtStatusOk);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_EQ(ReferenceTransposeConv(nan_in, weight_one, /*bias=*/{},
                                   absl::MakeSpan(out), params,
                                   tflite::ActivationFunctionType_RELU),
            kLiteRtStatusOk);
  EXPECT_TRUE(std::isnan(out[0]));
}

}  // namespace
}  // namespace litert::internal
