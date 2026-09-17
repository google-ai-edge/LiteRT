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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(SimpleBinaryOpTest, EqualBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferEqual(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, EqualDynamic) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1, 128}, {1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferEqual(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

TEST(SimpleBinaryOpTest, AddBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  // We are testing InferAdd which is now part of simple_binary.h
  ASSERT_EQ(InferAdd(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, DivBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferDiv(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, MulBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferMul(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, SubBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferSub(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleBinaryOpTest, ReferenceAddWithActivation) {
  std::vector<float> a = {1.0f, -5.0f, 10.0f};
  std::vector<float> b = {2.0f, 2.0f, 2.0f};
  std::vector<float> out(3);
  int32_t dims[] = {3};

  ReferenceAdd(a.data(), dims, 1, b.data(), dims, 1, out.data(), dims, 1,
               tflite::ActivationFunctionType_RELU6);

  EXPECT_THAT(out, ElementsAre(3.0f, 0.0f, 6.0f));
}

TEST(SimpleBinaryOpTest, ReferenceSubBroadcast) {
  std::vector<float> a = {10.0f, 20.0f};
  std::vector<float> b = {1.0f};
  std::vector<float> out(2);
  int32_t a_dims[] = {2};
  int32_t b_dims[] = {1};
  int32_t out_dims[] = {2};

  ReferenceSub(a.data(), a_dims, 1, b.data(), b_dims, 1, out.data(), out_dims,
               1);

  EXPECT_THAT(out, ElementsAre(9.0f, 19.0f));
}

TEST(SimpleBinaryOpTest, ReferenceMul) {
  std::vector<float> a = {2.0f, 3.0f};
  std::vector<float> b = {4.0f, 5.0f};
  std::vector<float> out(2);
  int32_t dims[] = {2};

  ReferenceMul(a.data(), dims, 1, b.data(), dims, 1, out.data(), dims, 1);

  EXPECT_THAT(out, ElementsAre(8.0f, 15.0f));
}

TEST(SimpleBinaryOpTest, ReferenceDiv) {
  std::vector<float> a = {10.0f, 20.0f};
  std::vector<float> b = {2.0f, 4.0f};
  std::vector<float> out(2);
  int32_t dims[] = {2};

  ReferenceDiv(a.data(), dims, 1, b.data(), dims, 1, out.data(), dims, 1);

  EXPECT_THAT(out, ElementsAre(5.0f, 5.0f));
}

TEST(SimpleBinaryOpTest, ReferencePow) {
  std::vector<float> a = {2.0f, 3.0f};
  std::vector<float> b = {3.0f, 2.0f};
  std::vector<float> out(2);
  int32_t dims[] = {2};

  ReferencePow(a.data(), dims, 1, b.data(), dims, 1, out.data(), dims, 1);

  EXPECT_THAT(out, ElementsAre(8.0f, 9.0f));
}

}  // namespace
}  // namespace litert::internal
