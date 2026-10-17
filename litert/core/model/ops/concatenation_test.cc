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
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

LiteRtOpT MakeConcatenationOp(int axis) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ConcatenationOptionsT>();
  options->axis = axis;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ConcatenationOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

TEST(ConcatenationOpTest, ThreeDimensionalOneInput) {
  LiteRtOpT op = MakeConcatenationOp(1);
  std::vector<Dims> input_shapes = {{2, 1, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2));
}

TEST(ConcatenationOpTest, FiveDimensionalOneInput) {
  LiteRtOpT op = MakeConcatenationOp(2);
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 1, 3));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInput) {
  LiteRtOpT op = MakeConcatenationOp(0);
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}, {2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 0: 2 + 2 = 4
  EXPECT_THAT(output_shapes[0], ElementsAre(4, 1, 2, 1, 3));
}

TEST(ConcatenationOpTest, FiveDimensionalTwoInputNegativeAxes) {
  LiteRtOpT op = MakeConcatenationOp(-2);
  std::vector<Dims> input_shapes = {{2, 1, 2, 1, 3}, {2, 1, 2, 1, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis -2 is index 3 (0-indexed: 4, 3 -> dim 1).
  // Shapes: [2, 1, 2, 1, 3]. Dim 3 is 1.
  // 1 + 1 = 2.
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 2, 2, 3));
}

TEST(ConcatenationOpTest, ThreeDimensionalTwoInputsDifferentShapes) {
  LiteRtOpT op = MakeConcatenationOp(1);
  std::vector<Dims> input_shapes = {{2, 1, 2}, {2, 3, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 1: 1 + 3 = 4
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 4, 2));
}

TEST(ConcatenationOpTest, FourInputs) {
  LiteRtOpT op = MakeConcatenationOp(2);
  std::vector<Dims> input_shapes = {{2, 1, 2}, {2, 1, 2}, {2, 1, 2}, {2, 1, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Axis 2: 2 + 2 + 2 + 2 = 8
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 1, 8));
}

struct ConcatenationAxisCase {
  std::string test_name;
  int axis;
  Dims expected_output;
};

using ConcatenationAxisTest = ::testing::TestWithParam<ConcatenationAxisCase>;

TEST_P(ConcatenationAxisTest, TwoInputsTwoAxesAndNegativeAxes) {
  const auto& param = GetParam();
  LiteRtOpT op = MakeConcatenationOp(param.axis);
  std::vector<Dims> input_shapes = {{2, 3}, {2, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_EQ(output_shapes[0], param.expected_output);
}

INSTANTIATE_TEST_SUITE_P(
    ConcatenationAxisTests, ConcatenationAxisTest,
    ::testing::Values(ConcatenationAxisCase{"Axis0", 0, {4, 3}},
                      ConcatenationAxisCase{"NegativeAxis2", -2, {4, 3}},
                      ConcatenationAxisCase{"Axis1", 1, {2, 6}}),
    [](const ::testing::TestParamInfo<ConcatenationAxisCase>& info) {
      return info.param.test_name;
    });

TEST(ConcatenationOpTest, DynamicShapeInference) {
  LiteRtOpT op = MakeConcatenationOp(1);
  // First input has dynamic dimension. Second has static 5.
  std::vector<Dims> input_shapes = {{-1, 10}, {5, 10}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Should infer 5 for dim 0. Axis 1 sums to 20.
  EXPECT_THAT(output_shapes[0], ElementsAre(5, 20));
}

TEST(ConcatenationOpTest, DynamicShapeMismatch) {
  LiteRtOpT op = MakeConcatenationOp(1);
  // First input dynamic. Second static 5. Third static 6.
  // This should fail because 5 != 6.
  std::vector<Dims> input_shapes = {{-1, 10}, {5, 10}, {6, 10}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ConcatenationOpTest, OutputShapeSizeMismatch) {
  LiteRtOpT op = MakeConcatenationOp(0);
  std::vector<Dims> input_shapes = {{2, 2}, {2, 2}};
  std::vector<Dims> output_shapes;  // Empty

  EXPECT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

struct ConcatenationErrorCase {
  std::string test_name;
  int axis;
  std::vector<Dims> input_shapes;
};

using ConcatenationErrorTest = ::testing::TestWithParam<ConcatenationErrorCase>;

TEST_P(ConcatenationErrorTest, RejectsAxisUnderflowInvalidDimsAndSumOverflow) {
  const auto& param = GetParam();
  LiteRtOpT op = MakeConcatenationOp(param.axis);
  std::vector<Dims> input_shapes = param.input_shapes;
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferConcatenation(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    ConcatenationErrorTests, ConcatenationErrorTest,
    ::testing::Values(
        ConcatenationErrorCase{"AxisUnderflow",
                               std::numeric_limits<int32_t>::min(),
                               {{2, 3}, {2, 3}}},
        ConcatenationErrorCase{"NegativeInputDimension", 0, {{-2, 3}, {2, 3}}},
        ConcatenationErrorCase{
            "AxisDimensionSumOverflow",
            0,
            {{std::numeric_limits<int32_t>::max(), 3}, {2, 3}}}),
    [](const ::testing::TestParamInfo<ConcatenationErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
