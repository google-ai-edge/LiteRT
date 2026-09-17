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

#include <cmath>
#include <cstdint>
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
  std::vector<Dims> input_shapes = {{1, 2, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, AbsDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

// Consolidating Cast tests
TEST(SimpleUnaryOpTest, CastStaticShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, CastDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

TEST(SimpleUnaryOpTest, L2Normalization) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferL2Normalization(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 128));
}

TEST(SimpleUnaryOpTest, ReverseV2) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1}};  // Input, Axis
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReverseV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, ReferenceTanh) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f};
  std::vector<float> output(3);

  ReferenceTanh(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 0.0f, 1e-5);
  EXPECT_NEAR(output[1], std::tanh(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::tanh(-1.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceRsqrt) {
  std::vector<float> input = {1.0f, 4.0f, 16.0f};
  std::vector<float> output(3);

  ReferenceRsqrt(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 1.0f, 1e-5);
  EXPECT_NEAR(output[1], 0.5f, 1e-5);
  EXPECT_NEAR(output[2], 0.25f, 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceSin) {
  std::vector<float> input = {0.0f, 1.0f, 2.0f};
  std::vector<float> output(3);

  ReferenceSin(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 0.0f, 1e-5);
  EXPECT_NEAR(output[1], std::sin(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::sin(2.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceCos) {
  std::vector<float> input = {0.0f, 1.0f, 2.0f};
  std::vector<float> output(3);

  ReferenceCos(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 1.0f, 1e-5);
  EXPECT_NEAR(output[1], std::cos(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::cos(2.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceCast) {
  std::vector<int32_t> in_i32 = {1, 2, -3};
  std::vector<float> out_f32(3);
  ReferenceCast(in_i32.data(), in_i32.size(), out_f32.data());
  EXPECT_THAT(out_f32, ElementsAre(1.0f, 2.0f, -3.0f));

  std::vector<float> in_f32 = {1.5f, 2.7f, -3.2f};
  std::vector<int32_t> out_i32(3);
  ReferenceCast(in_f32.data(), in_f32.size(), out_i32.data());
  EXPECT_THAT(out_i32, ElementsAre(1, 2, -3));
}

}  // namespace
}  // namespace litert::internal
