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

#include "litert/core/model/ops/pooling.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

// Helper to create options
std::unique_ptr<tflite::Pool2DOptionsT> CreatePoolOptions(
    tflite::Padding padding, int stride_h, int stride_w, int filter_h,
    int filter_w,
    tflite::ActivationFunctionType activation =
        tflite::ActivationFunctionType_NONE) {
  auto options = std::make_unique<tflite::Pool2DOptionsT>();
  options->padding = padding;
  options->stride_h = stride_h;
  options->stride_w = stride_w;
  options->filter_height = filter_h;
  options->filter_width = filter_w;
  options->fused_activation_function = activation;
  return options;
}

// AveragePool Tests

TEST(PoolingOpTest, AveragePool_Simple) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // (4-2)/2 + 1 = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, AveragePool_Stride1Valid) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // (4-2)/1 + 1 = 3
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 1));
}

TEST(PoolingOpTest, AveragePool_PaddingSameStride1) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_SAME, 1, 1, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 1));
}

TEST(PoolingOpTest, AveragePool_PaddingSameStride2) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_SAME, 2, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // ceil(4/2) = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, AveragePool_AnisotropicStride) {
  LiteRtOpT op;
  // Input [1, 4, 6, 1]
  std::vector<Dims> input_shapes = {{1, 4, 6, 1}};
  std::vector<Dims> output_shapes(1);

  // Stride H=2, W=3. Filter 2x2. VALID.
  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 3, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: (4-2)/2 + 1 = 2
  // W: (6-2)/3 + 1 = 1.33 -> 1 + 1 = 2? No, integer div. 4/3 = 1. + 1 = 2.
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, AveragePool_AnisotropicFilter) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  // Filter 2x3. Stride 1. VALID.
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 2, 3);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // H: (4-2)/1 + 1 = 3
  // W: (4-3)/1 + 1 = 2
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 2, 1));
}

TEST(PoolingOpTest, AveragePool_Global) {
  LiteRtOpT op;
  // Input [1, 10, 10, 3]
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}};
  std::vector<Dims> output_shapes(1);

  // Filter 10x10. Stride 1. VALID.
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 10, 10);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 3));
}

TEST(PoolingOpTest, AveragePool_1x1) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  // Filter 1x1. Stride 1. VALID.
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 1, 1);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 1));
}

TEST(PoolingOpTest, AveragePool_Activation) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2,
                                   tflite::ActivationFunctionType_RELU);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, AveragePool_BatchChannel) {
  LiteRtOpT op;
  // Input [2, 4, 4, 3]
  std::vector<Dims> input_shapes = {{2, 4, 4, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 2, 3));
}

// MaxPool Tests (Same structure as AveragePool)

TEST(PoolingOpTest, MaxPool_Simple) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, MaxPool_Stride1Valid) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 2, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 3, 1));
}

TEST(PoolingOpTest, MaxPool_PaddingSameStride1) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_SAME, 1, 1, 2, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 1));
}

TEST(PoolingOpTest, MaxPool_PaddingSameStride2) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_SAME, 2, 2, 2, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, MaxPool_AnisotropicStride) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 6, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 3, 2, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, MaxPool_AnisotropicFilter) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 2, 3);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 3, 2, 1));
}

TEST(PoolingOpTest, MaxPool_Global) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 10, 10);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 3));
}

TEST(PoolingOpTest, MaxPool_1x1) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 1, 1, 1, 1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 1));
}

TEST(PoolingOpTest, MaxPool_Activation) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2,
                                   tflite::ActivationFunctionType_RELU);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(PoolingOpTest, MaxPool_BatchChannel) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 4, 4, 3}};
  std::vector<Dims> output_shapes(1);
  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2, 2, 3));
}

TEST(PoolingOpTest, L2Pool_Simple) {
  LiteRtOpT op;
  // Input [1, 4, 4, 1]
  std::vector<Dims> input_shapes = {{1, 4, 4, 1}};
  std::vector<Dims> output_shapes(1);

  // Filter 2x2. Stride 2. VALID.
  auto options = CreatePoolOptions(tflite::Padding_VALID, 2, 2, 2, 2);

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_Pool2DOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPool2D(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

}  // namespace
}  // namespace litert::internal
