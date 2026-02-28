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

#include "litert/core/model/ops/simple_binary.h"

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

TEST(SimpleBinaryOpTest, EqualBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {2, 1}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferEqual(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, EqualDynamic) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{-1, 128}, {1, 128}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferEqual(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

TEST(SimpleBinaryOpTest, AddBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {2, 1}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  // We are testing InferAdd which is now part of simple_binary.h
  ASSERT_EQ(InferAdd(op, input_shapes, output_shapes), kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, DivBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {2, 1}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferDiv(op, input_shapes, output_shapes), kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, MulBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {2, 1}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferMul(op, input_shapes, output_shapes), kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, SubBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {2, 1}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferSub(op, input_shapes, output_shapes), kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

}  // namespace
}  // namespace litert::internal
