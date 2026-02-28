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

#include "litert/core/model/ops/simple_unary.h"

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

TEST(SimpleUnaryOpTest, AbsStaticShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, AbsDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{-1, 128}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

// Consolidating Cast tests
TEST(SimpleUnaryOpTest, CastStaticShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, CastDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{-1, 128}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

TEST(SimpleUnaryOpTest, L2Normalization) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 128}};
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferL2Normalization(op, input_shapes, output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 128));
}

TEST(SimpleUnaryOpTest, ReverseV2) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes_vec = {{1, 2, 3}, {1}};  // Input, Axis
  absl::Span<Dims> input_shapes = absl::MakeSpan(input_shapes_vec);
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReverseV2(op, input_shapes, output_shapes), kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

}  // namespace
}  // namespace litert::internal
