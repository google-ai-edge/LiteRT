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

#include "litert/core/model/ops/concatenation.h"

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
std::unique_ptr<tflite::ConcatenationOptionsT> CreateConcatenationOptions(
    int axis) {
  auto options = std::make_unique<tflite::ConcatenationOptionsT>();
  options->axis = axis;
  return options;
}

TEST(ConcatenationOpTest, ThreeDimensionalOneInput) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2));
}

TEST(ConcatenationOpTest, FiveDimensionalOneInput) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 1, 3));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInput) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}, {2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 0: 2 + 2 = 4
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 1, 2, 1, 3));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputNegativeAxes) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}, {2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(-2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis -2 is index 3 (0-indexed: 4, 3 -> dim 1).
  // Shapes: [2, 1, 2, 1, 3]. Dim 3 is 1.
  // 1 + 1 = 2.
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 2, 3));
}

TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapes) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2}, {2, 3, 2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 1: 1 + 3 = 4
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 4, 2));
}

TEST(ConcatenationOpTest, FourInputs) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 1, 2}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 2: 2 + 2 + 2 + 2 = 8
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 8));
}

TEST(ConcatenationOpTest, TwoInputsTwoAxesNegativeAxes) {
  LiteRtOpT op_axis0;
  std::vector<Dims> input_shapes = {{2, 3}, {2, 3}};
  std::vector<Dims> output_shapes(1);

  {
    auto options = CreateConcatenationOptions(0);
    TflOptions tfl_options;
    tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
    tfl_options.value = options.release();
    SetTflOptions(op_axis0, std::move(tfl_options));
    ASSERT_EQ(InferConcatenation(op_axis0, absl::MakeSpan(input_shapes),
                                 output_shapes),
              kLiteRtStatusOk);
    EXPECT_THAT(output_shapes[0], ElementsAre(4, 3));
  }

  {
    LiteRtOpT op_axis_neg2;
    auto options = CreateConcatenationOptions(-2);
    TflOptions tfl_options;
    tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
    tfl_options.value = options.release();
    SetTflOptions(op_axis_neg2, std::move(tfl_options));
    ASSERT_EQ(InferConcatenation(op_axis_neg2, absl::MakeSpan(input_shapes),
                                 output_shapes),
              kLiteRtStatusOk);
    EXPECT_THAT(output_shapes[0], ElementsAre(4, 3));
  }

  {
    LiteRtOpT op_axis1;
    auto options = CreateConcatenationOptions(1);
    TflOptions tfl_options;
    tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
    tfl_options.value = options.release();
    SetTflOptions(op_axis1, std::move(tfl_options));
    ASSERT_EQ(InferConcatenation(op_axis1, absl::MakeSpan(input_shapes),
                                 output_shapes),
              kLiteRtStatusOk);
    EXPECT_THAT(output_shapes[0], ElementsAre(2, 6));
  }
}

TEST(ConcatenationOpTest, DynamicShapeInference) {
  LiteRtOpT op;
  // First input has dynamic dimension. Second has static 5.
  std::vector<Dims> input_shapes = {{-1, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(1);  // Axis 1
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Should infer 5 for dim 0. Axis 1 sums to 20.
  EXPECT_THAT(output_shapes[0], ElementsAre(5, 20));
}

TEST(ConcatenationOpTest, DynamicShapeMismatch) {
  LiteRtOpT op;
  // First input dynamic. Second static 5. Third static 6.
  // This should fail because 5 != 6.
  std::vector<Dims> input_shapes = {{-1, 10}, {5, 10}, {6, 10}};
  std::vector<Dims> output_shapes(1);

  auto options = CreateConcatenationOptions(1);  // Axis 1
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ConcatenationOpTest, OutputShapeSizeMismatch) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 2}, {2, 2}};
  std::vector<Dims> output_shapes;  // Empty

  auto options = CreateConcatenationOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

}  // namespace
}  // namespace litert::internal
