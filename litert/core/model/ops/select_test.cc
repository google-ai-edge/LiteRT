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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;

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

TEST(SelectOpTest, ReferenceSelectSameShape) {
  const bool cond[] = {true, false, true, false};
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> out(4);
  int32_t dims[] = {4};

  ReferenceSelect(cond, dims, 1, a.data(), dims, 1, b.data(), dims, 1,
                  out.data(), dims, 1);

  EXPECT_THAT(out, ElementsAre(1.0f, 20.0f, 3.0f, 40.0f));
}

TEST(SelectOpTest, ReferenceSelectBroadcastCondition) {
  const bool cond[] = {true, false};
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> out(4);
  int32_t cond_dims[] = {2, 1};
  int32_t data_dims[] = {2, 2};
  int32_t out_dims[] = {2, 2};

  ReferenceSelect(cond, cond_dims, 2, a.data(), data_dims, 2, b.data(),
                  data_dims, 2, out.data(), out_dims, 2);

  EXPECT_THAT(out, ElementsAre(1.0f, 2.0f, 30.0f, 40.0f));
}

TEST(SelectOpTest, RejectsInvalidOutputShapesCount) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 4}, {1, 4}, {1, 4}};

  std::vector<Dims> empty_outputs;
  EXPECT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), empty_outputs),
            kLiteRtStatusErrorInvalidArgument);

  std::vector<Dims> two_outputs(2);
  EXPECT_EQ(InferSelect(op, absl::MakeSpan(input_shapes), two_outputs),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(SelectOpTest, ReferenceSelectEmptyShapesWithNullBuffersSucceeds) {
  const int32_t empty_1d[] = {0};

  EXPECT_THAT(ReferenceSelect<float>(/*cond_data=*/nullptr, empty_1d, 1,
                                     /*a_data=*/nullptr, empty_1d, 1,
                                     /*b_data=*/nullptr, empty_1d, 1,
                                     /*output_data=*/nullptr, empty_1d, 1),
              IsOk());
}

TEST(SelectOpTest, ReferenceSelectRejectsIncompatibleOutputShape) {
  const bool cond[] = {true, false};
  const float a[] = {1.0f, 2.0f};
  const float b[] = {3.0f, 4.0f};
  float out[2] = {-5.0f, -5.0f};
  const int32_t dims2[] = {2};
  const int32_t dims3[] = {3};

  EXPECT_THAT(ReferenceSelect<float>(cond, dims2, 1, a, dims2, 1, b, dims3, 1,
                                     out, dims2, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f));
}

TEST(SelectOpTest, ReferenceSelectSupportsRankAboveEight) {
  const int32_t dims[9] = {1, 1, 1, 1, 1, 1, 1, 1, 2};
  const bool cond[2] = {true, false};
  const float a[2] = {1.0f, 2.0f};
  const float b[2] = {10.0f, 20.0f};
  float out[2] = {0.0f, 0.0f};

  EXPECT_THAT(ReferenceSelect<float>(cond, dims, 9, a, dims, 9, b, dims, 9, out,
                                     dims, 9),
              IsOk());
  EXPECT_THAT(out, ElementsAre(1.0f, 20.0f));
}

TEST(SelectOpTest, ReferenceSelectBroadcastsAllThreeOperandsInOnePass) {
  const int32_t cond_dims[] = {2, 1, 1};
  const int32_t a_dims[] = {1, 2, 1};
  const int32_t b_dims[] = {1, 1, 2};
  const int32_t out_dims[] = {2, 2, 2};
  const bool cond[] = {true, false};
  const int32_t a[] = {10, 20};
  const int32_t b[] = {100, 200};
  std::vector<int32_t> out(8, -1);

  EXPECT_THAT(ReferenceSelect<int32_t>(
                  ConstTensorView<bool>{cond, cond_dims},
                  ConstTensorView<int32_t>{a, a_dims},
                  ConstTensorView<int32_t>{b, b_dims},
                  TensorView<int32_t>{absl::MakeSpan(out), out_dims}),
              IsOk());
  EXPECT_THAT(out, ElementsAre(10, 10, 20, 20, 100, 200, 100, 200));
}

TEST(SelectOpTest, ReferenceSelectRejectsOverExpandedOutputAndSmallBuffers) {
  const int32_t one[] = {1};
  const int32_t expanded[] = {2, 3};
  const bool cond[] = {true};
  const float a[] = {1.0f};
  const float b[] = {2.0f};
  std::vector<float> out(6, -5.0f);

  EXPECT_THAT(
      ReferenceSelect<float>(ConstTensorView<bool>{cond, one},
                             ConstTensorView<float>{a, one},
                             ConstTensorView<float>{b, one},
                             TensorView<float>{absl::MakeSpan(out), expanded}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      ReferenceSelect<float>(ConstTensorView<bool>{cond, one},
                             ConstTensorView<float>{a, one},
                             ConstTensorView<float>{b, one},
                             TensorView<float>{absl::Span<float>(), one}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f));
}

TEST(SelectOpTest, ReferenceSelectLegacySupportsLeadingDim1DCondition) {
  const int32_t cond_dims[] = {2};
  const int32_t data_dims[] = {2, 3};
  const bool cond[] = {true, false};
  const int32_t a[] = {1, 2, 3, 4, 5, 6};
  const int32_t b[] = {10, 20, 30, 40, 50, 60};
  std::vector<int32_t> out(6, -1);

  EXPECT_THAT(ReferenceSelectLegacy<int32_t>(
                  ConstTensorView<bool>{cond, cond_dims},
                  ConstTensorView<int32_t>{a, data_dims},
                  ConstTensorView<int32_t>{b, data_dims},
                  TensorView<int32_t>{absl::MakeSpan(out), data_dims}),
              IsOk());
  EXPECT_THAT(out, ElementsAre(1, 2, 3, 40, 50, 60));

  // Trailing-dimension broadcast is rejected by legacy SELECT.
  const int32_t trailing_cond_dims[] = {1, 3};
  const bool trailing_cond[] = {true, false, true};
  EXPECT_THAT(ReferenceSelectLegacy<int32_t>(
                  ConstTensorView<bool>{trailing_cond, trailing_cond_dims},
                  ConstTensorView<int32_t>{a, data_dims},
                  ConstTensorView<int32_t>{b, data_dims},
                  TensorView<int32_t>{absl::MakeSpan(out), data_dims}),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

}  // namespace
}  // namespace litert::internal
