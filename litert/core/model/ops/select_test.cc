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

#include "litert/core/model/ops/select.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(SelectOpTest, SimpleSelectSameShape) {
  LiteRtOpT op;
  // Condition, Input1, Input2 all same shape [1, 1, 1, 4]
  std::vector<Dims> input_shapes = {{1, 1, 1, 4}, {1, 1, 1, 4}, {1, 1, 1, 4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 4));
}

TEST(SelectOpTest, SelectBroadcastCondition) {
  LiteRtOpT op;
  // Condition [1], Input1 [1, 2, 2, 1], Input2 [1, 2, 2, 1]
  std::vector<Dims> input_shapes = {{1}, {1, 2, 2, 1}, {1, 2, 2, 1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2, 1));
}

TEST(SelectOpTest, SelectBroadcastInputs) {
  LiteRtOpT op;
  // Condition [1, 2], Inputs [1, 2, 2]
  std::vector<Dims> input_shapes = {{1, 2}, {1, 2, 2}, {1, 2, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Broadcast [1, 2] vs [1, 2, 2] -> [1, 2, 2]
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2));
}

TEST(SelectOpTest, SelectBroadcastConditionToInputs5D) {
  LiteRtOpT op;
  // Condition [1] vs Inputs [1, 2, 2, 2, 1]
  std::vector<Dims> input_shapes_5d = {{1}, {1, 2, 2, 2, 1}, {1, 2, 2, 2, 1}};
  std::vector<Dims> output_shapes_5d(1);

  ASSERT_EQ(InferSelect(op, input_shapes_5d, output_shapes_5d),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes_5d[0], ElementsAre(1, 2, 2, 2, 1));
}

TEST(SelectOpTest, SelectScalarInputs) {
  LiteRtOpT op;
  // Condition [1], Input1 [1], Input2 [1]
  std::vector<Dims> input_shapes = {{1}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1));
}

TEST(SelectOpTest, SelectScalarConditionTensor) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{}, {1}, {1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1));
}

TEST(SelectOpTest, SelectV2) {
  LiteRtOpT op;
  // Broadcasting: Cond [2], True [2, 2], False [2, 2]
  std::vector<Dims> input_shapes = {{2}, {2, 2}, {2, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(SelectOpTest, SelectWithDynamic) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, -1}, {1, 5}, {1, 5}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, -1));
}

TEST(SelectOpTest, SelectWithIncompatible) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 3}, {1, 5}, {1, 5}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
