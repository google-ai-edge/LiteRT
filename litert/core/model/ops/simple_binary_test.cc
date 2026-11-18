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
#include <functional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"
#include "litert/test/matchers.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;

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

TEST(SimpleBinaryOpTest, FloorModSameShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 630, 2}, {1, 630, 2}};
  std::vector<Dims> output_shapes(1);
  ASSERT_EQ(InferFloorMod(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 630, 2));
}

TEST(SimpleBinaryOpTest, FloorModBroadcastScalarAndRank2) {
  LiteRtOpT op;
  {
    std::vector<Dims> input_shapes = {{1, 630, 2}, {}};
    std::vector<Dims> output_shapes(1);
    ASSERT_EQ(InferFloorMod(op, absl::MakeSpan(input_shapes), output_shapes),
              kLiteRtStatusOk);
    EXPECT_THAT(output_shapes[0], ElementsAre(1, 630, 2));
  }
  {
    std::vector<Dims> input_shapes = {{1, 630, 2}, {1, 1}};
    std::vector<Dims> output_shapes(1);
    ASSERT_EQ(InferFloorMod(op, absl::MakeSpan(input_shapes), output_shapes),
              kLiteRtStatusOk);
    EXPECT_THAT(output_shapes[0], ElementsAre(1, 630, 2));
  }
}

TEST(SimpleBinaryOpTest, FloorModIncompatibleShapesFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 630, 2}, {1, 5, 3}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferFloorMod(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
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

TEST(SimpleBinaryOpTest, ReferenceComparisonsBroadcast) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f};
  std::vector<float> b = {2.0f};
  bool out[3];
  int32_t a_dims[] = {3};
  int32_t b_dims[] = {1};
  int32_t out_dims[] = {3};

  ReferenceEqual(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims, 1);
  EXPECT_THAT(out, ElementsAre(false, true, false));

  ReferenceNotEqual(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims, 1);
  EXPECT_THAT(out, ElementsAre(true, false, true));

  ReferenceGreater(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims, 1);
  EXPECT_THAT(out, ElementsAre(false, false, true));

  ReferenceGreaterEqual(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims,
                        1);
  EXPECT_THAT(out, ElementsAre(false, true, true));

  ReferenceLess(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims, 1);
  EXPECT_THAT(out, ElementsAre(true, false, false));

  ReferenceLessEqual(a.data(), a_dims, 1, b.data(), b_dims, 1, out, out_dims,
                     1);
  EXPECT_THAT(out, ElementsAre(true, true, false));
}

TEST(SimpleBinaryOpTest, EqualInvalidOutputsFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2, 1}};
  std::vector<Dims> output_shapes(2);  // Invalid number of outputs

  EXPECT_EQ(InferEqual(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SimpleBinaryOpTest, EqualInvalidInputsFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 2, 3}};  // Invalid number of inputs (needs 2)
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferEqual(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

struct EmptyBinaryCase {
  std::string test_name;
  std::vector<int32_t> a_dims;
  std::vector<int32_t> b_dims;
  std::vector<int32_t> out_dims;
  tflite::ActivationFunctionType faf = tflite::ActivationFunctionType_NONE;
};

using ReferenceBinaryEmptyTest = ::testing::TestWithParam<EmptyBinaryCase>;

TEST_P(ReferenceBinaryEmptyTest, EmptyShapesWithNullBuffersSucceed) {
  const auto& tc = GetParam();

  EXPECT_THAT(
      ReferenceAdd(/*a_data=*/nullptr, tc.a_dims.data(),
                   static_cast<int>(tc.a_dims.size()), /*b_data=*/nullptr,
                   tc.b_dims.data(), static_cast<int>(tc.b_dims.size()),
                   /*output_data=*/nullptr, tc.out_dims.data(),
                   static_cast<int>(tc.out_dims.size()), tc.faf),
      IsOk());
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceBinaryEmptyTests, ReferenceBinaryEmptyTest,
    ::testing::Values(
        EmptyBinaryCase{.test_name = "BothEmpty1DWithRelu",
                        .a_dims = {0},
                        .b_dims = {0},
                        .out_dims = {0},
                        .faf = tflite::ActivationFunctionType_RELU},
        EmptyBinaryCase{.test_name = "EmptyAndSingleton1D",
                        .a_dims = {0},
                        .b_dims = {1},
                        .out_dims = {0}},
        EmptyBinaryCase{.test_name = "EmptyBroadcast2D",
                        .a_dims = {2, 0},
                        .b_dims = {1, 0},
                        .out_dims = {2, 0}}),
    [](const ::testing::TestParamInfo<EmptyBinaryCase>& info) {
      return info.param.test_name;
    });

TEST(SimpleBinaryOpTest, ReferenceBinaryGenericRejectsIncompatibleOutputShape) {
  const float a[] = {1.0f, 2.0f};
  const float b[] = {3.0f, 4.0f};
  float out[2] = {-7.0f, -7.0f};
  const int32_t a_dims[] = {2};
  const int32_t b_dims[] = {3};

  EXPECT_THAT(ReferenceAdd(a, a_dims, 1, b, b_dims, 1, out, a_dims, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f));
}

TEST(SimpleBinaryOpTest, ReferenceBinaryGenericSupportsHighRankAboveEight) {
  const int32_t dims[9] = {1, 1, 1, 1, 1, 1, 1, 1, 2};
  const float a[2] = {1.0f, 2.0f};
  const float b[2] = {10.0f, 20.0f};
  float out[2] = {0.0f, 0.0f};

  EXPECT_THAT(ReferenceAdd(a, dims, 9, b, dims, 9, out, dims, 9), IsOk());
  EXPECT_THAT(out, ElementsAre(11.0f, 22.0f));
}

TEST(SimpleBinaryOpTest,
     ReferenceBinaryGenericAsymmetricSubPreservesOperandOrder) {
  // [2,1,4] - [1,3,1] -> [2,3,4].
  const int32_t a_dims[] = {2, 1, 4};
  const int32_t b_dims[] = {1, 3, 1};
  const int32_t out_dims[] = {2, 3, 4};
  const float a[] = {10, 20, 30, 40, 100, 200, 300, 400};
  const float b[] = {1, 2, 3};
  std::vector<float> out(24, 0.0f);

  EXPECT_THAT(ReferenceSub(a, a_dims, 3, b, b_dims, 3, out.data(), out_dims, 3),
              IsOk());
  EXPECT_THAT(out,
              ElementsAre(9, 19, 29, 39, 8, 18, 28, 38, 7, 17, 27, 37, 99, 199,
                          299, 399, 98, 198, 298, 398, 97, 197, 297, 397));
}

TEST(SimpleBinaryOpTest, ReferenceBinaryGenericEmptyNeverInvokesCallback) {
  const int32_t a_dims[] = {2, 0};
  const int32_t b_dims[] = {1, 1};
  const int32_t out_dims[] = {2, 0};
  const float b_val[] = {1.0f};
  int call_count = 0;

  EXPECT_THAT((ReferenceBinaryGeneric<float, float, float>(
                  ConstTensorView<float>{absl::Span<const float>(), a_dims},
                  ConstTensorView<float>{b_val, b_dims},
                  TensorView<float>{absl::Span<float>(), out_dims},
                  [&](float x, float y) {
                    ++call_count;
                    return x + y;
                  })),
              IsOk());
  EXPECT_EQ(call_count, 0);
}

TEST(SimpleBinaryOpTest, ReferenceBinaryGenericRejectsOverExpandedOutput) {
  const int32_t one[] = {1};
  const int32_t target[] = {2, 3};
  const float a[] = {1.0f};
  const float b[] = {2.0f};
  std::vector<float> out(6, -3.0f);

  EXPECT_THAT(ReferenceAdd(a, one, 1, b, one, 1, out.data(), target, 2),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f));
}

TEST(SimpleBinaryOpTest, ReferenceBinaryGenericRejectsUndersizedSpans) {
  const int32_t dims[] = {2, 2};
  const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out[] = {-1.0f, -1.0f, -1.0f, -1.0f};

  EXPECT_THAT((ReferenceBinaryGeneric<float, float, float>(
                  ConstTensorView<float>{absl::MakeConstSpan(a).first(3), dims},
                  ConstTensorView<float>{b, dims}, TensorView<float>{out, dims},
                  std::plus<float>())),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      (ReferenceBinaryGeneric<float, float, float>(
          ConstTensorView<float>{a, dims}, ConstTensorView<float>{b, dims},
          TensorView<float>{absl::MakeSpan(out).first(3), dims},
          std::plus<float>())),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-1.0f, -1.0f, -1.0f, -1.0f));
}

TEST(SimpleBinaryOpTest, ReferenceBinaryRawPointersRejectInvalidRankAndNulls) {
  const float a[] = {1.0f, 2.0f};
  const float b[] = {3.0f, 4.0f};
  float out[] = {-7.0f, -7.0f};
  const int32_t dims[] = {2};
  const int32_t bad_dims[] = {3};

  EXPECT_THAT(ReferenceAdd(a, dims, /*a_rank=*/-1, b, dims, 1, out, dims, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      ReferenceAdd(/*a_data=*/nullptr, dims, 1, b, dims, 1, out, dims, 1),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ReferenceSub(a, dims, 1, b, bad_dims, 1, out, dims, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f));
}

}  // namespace
}  // namespace litert::internal
