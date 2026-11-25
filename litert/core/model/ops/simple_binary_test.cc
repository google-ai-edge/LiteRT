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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
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

TEST(SimpleBinaryOpTest, ReferencePreluFloatAppliesPerChannelAlphaToNegatives) {
  const int32_t in_dims[] = {2, 2};
  const int32_t alpha_dims[] = {1, 2};
  const std::vector<float> in = {-2.0f, -4.0f, 3.0f, -8.0f};
  const std::vector<float> alpha = {0.5f, 0.25f};
  std::vector<float> out(4, 0.0f);

  EXPECT_THAT(
      ReferencePrelu<float>(ConstTensorView<float>{in, in_dims},
                            ConstTensorView<float>{alpha, alpha_dims},
                            TensorView<float>{absl::MakeSpan(out), in_dims}),
      IsOk());

  EXPECT_THAT(out, ElementsAre(-1.0f, -1.0f, 3.0f, -2.0f));
}

TEST(SimpleBinaryOpTest,
     ReferencePreluQuantizedInt8ScalesNegativesAndRequantizes) {
  // input {-4, 4} (scale=0.5, zp=0 -> -2.0, 2.0),
  // alpha {2} (scale=0.25, zp=0 -> 0.5), output (scale=0.5, zp=1).
  // y = {-1.0, 2.0} -> q = {-2 + 1 = -1, 4 + 1 = 5}.
  const int32_t in_dims[] = {2};
  const int32_t alpha_dims[] = {1};
  const std::vector<int8_t> in = {-4, 4};
  const std::vector<int8_t> alpha = {2};
  std::vector<int8_t> out(2, 0);

  EXPECT_THAT(ReferencePrelu<int8_t>(
                  ConstTensorView<int8_t>{in, in_dims},
                  ConstTensorView<int8_t>{alpha, alpha_dims},
                  TensorView<int8_t>{absl::MakeSpan(out), in_dims},
                  AffineQuantSpec{0.5f, 0}, AffineQuantSpec{0.25f, 0},
                  AffineQuantSpec{0.5f, 1}),
              IsOk());

  EXPECT_THAT(out, ElementsAre(-1, 5));
}

TEST(SimpleBinaryOpTest, ReferenceAddNFloatSumsThreeInputsElementwise) {
  const std::vector<float> a = {1.0f, -2.0f};
  const std::vector<float> b = {0.5f, 3.0f};
  const std::vector<float> c = {-0.5f, -1.0f};
  const float* ptrs[] = {a.data(), b.data(), c.data()};
  std::vector<float> out(2, 0.0f);

  EXPECT_THAT(ReferenceAddN<float>(ptrs, out.data(), out.size()), IsOk());

  EXPECT_THAT(out, ElementsAre(1.0f, 0.0f));
}

TEST(SimpleBinaryOpTest, ReferenceAddNQuantizedInt8SumsAndRequantizes) {
  const std::vector<int8_t> a = {-2, 4};
  const std::vector<int8_t> b = {-2, 2};
  const int8_t* ptrs[] = {a.data(), b.data()};
  std::vector<int8_t> out(2, 0);

  EXPECT_THAT(
      ReferenceAddN<int8_t>(ptrs, out.data(), out.size(),
                            AffineQuantSpec{0.5f, 0}, AffineQuantSpec{1.0f, 0}),
      IsOk());

  EXPECT_THAT(out, ElementsAre(-2, 3));
}

TEST(SimpleBinaryOpTest, ReferenceAddNInt32PreservesFull32BitPrecision) {
  // Values > 2^24 lose precision if dequantized or accumulated in float32.
  const std::vector<int32_t> a = {16777216, 16777217, -16777216, 1000000000};
  const std::vector<int32_t> b = {1, 0, -1, 1000000001};
  const int32_t* ptrs[] = {a.data(), b.data()};
  std::vector<int32_t> out(4, 0);

  EXPECT_THAT(ReferenceAddN<int32_t>(ptrs, out.data(), out.size()), IsOk());

  EXPECT_THAT(out, ElementsAre(16777217, 16777217, -16777217, 2000000001));
}

TEST(SimpleBinaryOpTest,
     ReferenceFloorDivFloatFloorsTowardNegativeInfinityAcrossSigns) {
  const std::vector<float> a = {-7.0f, 7.0f, -7.0f, 7.0f};
  const std::vector<float> b = {3.0f, -3.0f, -3.0f, 3.0f};
  const int32_t dims[] = {4};
  std::vector<float> out(4, 0.0f);

  EXPECT_THAT(ReferenceFloorDiv(a.data(), dims, 1, b.data(), dims, 1,
                                out.data(), dims, 1),
              IsOk());

  EXPECT_THAT(out, ElementsAre(-3.0f, -3.0f, 2.0f, 2.0f));
}

TEST(SimpleBinaryOpTest,
     ReferenceFloorDivFloatUsesDoublePrecisionAndRejectsZeroDivisor) {
  // Exact quotient is 3 - 1/8388609 = 2.99999988... (< 3), which rounds up to
  // 3.0f in float32 division but must floor to 2.0f matching TFLite.
  const float a[] = {25165826.0f};
  const float b[] = {8388609.0f};
  const int32_t dims[] = {1};
  float out[] = {-99.0f};

  EXPECT_THAT(ReferenceFloorDiv(a, dims, 1, b, dims, 1, out, dims, 1), IsOk());
  EXPECT_THAT(out, ElementsAre(2.0f));

  // Both +0.0f and -0.0f divisors are rejected without modifying output.
  const float a2[] = {5.0f, 6.0f};
  const float b_pos_zero[] = {1.0f, 0.0f};
  const float b_neg_zero[] = {1.0f, -0.0f};
  const int32_t dims2[] = {2};
  float out2[] = {-99.0f, -99.0f};

  EXPECT_THAT(
      ReferenceFloorDiv(a2, dims2, 1, b_pos_zero, dims2, 1, out2, dims2, 1),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      ReferenceFloorDiv(a2, dims2, 1, b_neg_zero, dims2, 1, out2, dims2, 1),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out2, ElementsAre(-99.0f, -99.0f));
}

TEST(SimpleBinaryOpTest, ReferenceFloorModFloatMatchesDivisorSignAcrossSigns) {
  const std::vector<float> a = {-7.0f, 7.0f, -7.0f, 7.0f};
  const std::vector<float> b = {3.0f, -3.0f, -3.0f, 3.0f};
  const int32_t dims[] = {4};
  std::vector<float> out(4, 0.0f);

  EXPECT_THAT(ReferenceFloorMod(a.data(), dims, 1, b.data(), dims, 1,
                                out.data(), dims, 1),
              IsOk());

  EXPECT_THAT(out, ElementsAre(2.0f, -2.0f, -1.0f, 1.0f));
}

TEST(SimpleBinaryOpTest,
     ReferenceFloorModFloatAvoidsRoundingCancellationOverflowAndInfNan) {
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> a = {25165826.0f, 1e10f, 1e38f, 5.0f};
  const std::vector<float> b = {8388609.0f, 3.0f, 1e-10f, inf};
  const int32_t dims[] = {4};
  std::vector<float> out(4, -99.0f);

  EXPECT_THAT(ReferenceFloorMod(a.data(), dims, 1, b.data(), dims, 1,
                                out.data(), dims, 1),
              IsOk());

  EXPECT_EQ(out[0], 8388608.0f);
  EXPECT_EQ(out[1], 1.0f);
  EXPECT_EQ(out[2], std::fmod(1e38f, 1e-10f));
  EXPECT_EQ(out[3], 5.0f);
}

struct Int32BinaryScalarCase {
  std::string test_name;
  int32_t a;
  int32_t b;
  int32_t expected;
};

using ReferenceFloorDivInt32Test =
    ::testing::TestWithParam<Int32BinaryScalarCase>;

TEST_P(ReferenceFloorDivInt32Test, ComputesFlooredQuotientSafely) {
  const auto& tc = GetParam();
  const int32_t dims[] = {1};
  int32_t out = -99;

  EXPECT_THAT(ReferenceFloorDiv(&tc.a, dims, 1, &tc.b, dims, 1, &out, dims, 1),
              IsOk());
  EXPECT_EQ(out, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceFloorDivInt32Tests, ReferenceFloorDivInt32Test,
    ::testing::Values(
        Int32BinaryScalarCase{"PositiveDividendPositiveDivisor", 7, 3, 2},
        Int32BinaryScalarCase{"NegativeDividendNegativeDivisor", -7, -3, 2},
        Int32BinaryScalarCase{"NegativeDividendPositiveDivisor", -7, 3, -3},
        Int32BinaryScalarCase{"PositiveDividendNegativeDivisor", 7, -3, -3},
        Int32BinaryScalarCase{"ExactNegativeDivision", -6, 3, -2},
        Int32BinaryScalarCase{"ExactNegativeDivisor", 6, -3, -2},
        Int32BinaryScalarCase{"Int32MinDivNegOneSaturates",
                              std::numeric_limits<int32_t>::min(), -1,
                              std::numeric_limits<int32_t>::max()},
        Int32BinaryScalarCase{"DivisorInt32MinAvoidsOverflow", 7,
                              std::numeric_limits<int32_t>::min(), -1}),
    [](const ::testing::TestParamInfo<Int32BinaryScalarCase>& info) {
      return info.param.test_name;
    });

TEST(SimpleBinaryOpTest, ReferenceFloorDivInt32ZeroDivisorReturnsError) {
  const int32_t a[] = {5, 6};
  const int32_t b[] = {1, 0};
  const int32_t dims[] = {2};
  int32_t out[] = {-99, -99};

  EXPECT_THAT(ReferenceFloorDiv(a, dims, 1, b, dims, 1, out, dims, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-99, -99));
}

using ReferenceFloorModInt32Test =
    ::testing::TestWithParam<Int32BinaryScalarCase>;

TEST_P(ReferenceFloorModInt32Test, ComputesFlooredRemainderSafely) {
  const auto& tc = GetParam();
  const int32_t dims[] = {1};
  int32_t out = -99;

  EXPECT_THAT(ReferenceFloorMod(&tc.a, dims, 1, &tc.b, dims, 1, &out, dims, 1),
              IsOk());
  EXPECT_EQ(out, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceFloorModInt32Tests, ReferenceFloorModInt32Test,
    ::testing::Values(
        Int32BinaryScalarCase{"PositiveDividendPositiveDivisor", 7, 3, 1},
        Int32BinaryScalarCase{"NegativeDividendNegativeDivisor", -7, -3, -1},
        Int32BinaryScalarCase{"NegativeDividendPositiveDivisor", -7, 3, 2},
        Int32BinaryScalarCase{"PositiveDividendNegativeDivisor", 7, -3, -2},
        Int32BinaryScalarCase{"ExactNegativeDivision", -6, 3, 0},
        Int32BinaryScalarCase{"ExactNegativeDivisor", 6, -3, 0},
        Int32BinaryScalarCase{"Int32MinModNegOneReturnsZero",
                              std::numeric_limits<int32_t>::min(), -1, 0},
        Int32BinaryScalarCase{"DivisorInt32MinAvoidsOverflow", 7,
                              std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::min() + 7}),
    [](const ::testing::TestParamInfo<Int32BinaryScalarCase>& info) {
      return info.param.test_name;
    });

TEST(SimpleBinaryOpTest, ReferenceFloorModInt32ZeroDivisorReturnsError) {
  const int32_t a[] = {5, 6};
  const int32_t b[] = {2, 0};
  const int32_t dims[] = {2};
  int32_t out[] = {-99, -99};

  EXPECT_THAT(ReferenceFloorMod(a, dims, 1, b, dims, 1, out, dims, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-99, -99));
}

TEST(SimpleBinaryOpTest, ReferenceLogicalAndComputesTruthTable) {
  const bool a[] = {false, false, true, true};
  const bool b[] = {false, true, false, true};
  const int32_t dims[] = {4};
  bool out[4] = {};

  EXPECT_THAT(ReferenceLogicalAnd(a, dims, 1, b, dims, 1, out, dims, 1),
              IsOk());

  EXPECT_THAT(out, ElementsAre(false, false, false, true));
}

TEST(SimpleBinaryOpTest, ReferenceLogicalOrComputesTruthTable) {
  const bool a[] = {false, false, true, true};
  const bool b[] = {false, true, false, true};
  const int32_t dims[] = {4};
  bool out[4] = {};

  EXPECT_THAT(ReferenceLogicalOr(a, dims, 1, b, dims, 1, out, dims, 1), IsOk());

  EXPECT_THAT(out, ElementsAre(false, true, true, true));
}

TEST(SimpleBinaryOpTest, ReferenceBitwiseXorBroadcastsScalarOperand) {
  const std::vector<int32_t> a = {0b1010, 0b1100, 0b0011};
  const std::vector<int32_t> b = {0b0110};
  const int32_t a_dims[] = {3};
  const int32_t b_dims[] = {1};
  std::vector<int32_t> out(3, 0);

  EXPECT_THAT(ReferenceBitwiseXor(a.data(), a_dims, 1, b.data(), b_dims, 1,
                                  out.data(), a_dims, 1),
              IsOk());

  EXPECT_THAT(out, ElementsAre(0b1100, 0b1010, 0b0101));
}

TEST(SimpleBinaryOpTest, ReferenceRightShiftSignedPreservesSignBit) {
  const std::vector<int32_t> a = {16, -16};
  const std::vector<int32_t> b = {2, 2};
  const int32_t dims[] = {2};
  std::vector<int32_t> out(2, -9);

  EXPECT_THAT(ReferenceRightShift(a.data(), dims, 1, b.data(), dims, 1,
                                  out.data(), dims, 1),
              IsOk());

  EXPECT_THAT(out, ElementsAre(4, -4));
}

TEST(SimpleBinaryOpTest,
     ReferenceRightShiftSignedClampsOutOfRangeShiftCountsLikeTflite) {
  // Matches TFLite right_shift.cc clamping shift counts to [0, 31]:
  //   8 >> -1  -> 8 >> 0  = 8
  //  -8 >> 32  -> -8 >> 31 = -1
  //   8 >> 32  -> 8 >> 31  = 0
  const std::vector<int32_t> a = {8, -8, 8};
  const std::vector<int32_t> b = {-1, 32, 32};
  const int32_t dims[] = {3};
  std::vector<int32_t> out(3, -9);

  EXPECT_THAT(ReferenceRightShift(a.data(), dims, 1, b.data(), dims, 1,
                                  out.data(), dims, 1),
              IsOk());
  EXPECT_THAT(out, ElementsAre(8, -1, 0));
}

TEST(SimpleBinaryOpTest, ReferenceRightShiftUnsignedShiftsLogically) {
  const std::vector<uint8_t> a = {0x80, 0x20};
  const std::vector<uint8_t> b = {2, 1};
  const int32_t dims[] = {2};
  std::vector<uint8_t> out(2, 99);

  EXPECT_THAT(ReferenceRightShift(a.data(), dims, 1, b.data(), dims, 1,
                                  out.data(), dims, 1),
              IsOk());
  EXPECT_THAT(out, ElementsAre(0x20, 0x10));
}

TEST(SimpleBinaryOpTest,
     ReferenceRightShiftUnsignedClampsShiftCountAtBitWidth) {
  // Clamping uint8 shift count 8 to 7 shifts MSB 0x80 into bit 0 (1), while
  // 0x40 >> 7 yields 0.
  const std::vector<uint8_t> a = {0x80, 0x40};
  const std::vector<uint8_t> b = {8, 8};
  const int32_t dims[] = {2};
  std::vector<uint8_t> out(2, 99);

  EXPECT_THAT(ReferenceRightShift(a.data(), dims, 1, b.data(), dims, 1,
                                  out.data(), dims, 1),
              IsOk());
  EXPECT_THAT(out, ElementsAre(1, 0));
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

TEST(SimpleBinaryOpTest, ReferencePreluMiddleAxisBroadcastAppliesPerSlice) {
  // Input [2, 3, 2] with alpha [1, 3, 1]: each alpha covers 2 elements across
  // the innermost axis and repeats across batch.
  const int32_t in_dims[] = {2, 3, 2};
  const int32_t alpha_dims[] = {1, 3, 1};
  const std::vector<float> in = {-2, 3, -4, -6, 5, -8, -10, 1, -12, 2, -14, 4};
  const std::vector<float> alpha = {0.5f, 0.25f, 0.125f};
  std::vector<float> out(12, 0.0f);

  EXPECT_THAT(
      ReferencePrelu<float>(ConstTensorView<float>{in, in_dims},
                            ConstTensorView<float>{alpha, alpha_dims},
                            TensorView<float>{absl::MakeSpan(out), in_dims}),
      IsOk());
  EXPECT_THAT(out, ElementsAre(-1.0f, 3.0f, -1.0f, -1.5f, 5.0f, -1.0f, -5.0f,
                               1.0f, -3.0f, 2.0f, -1.75f, 4.0f));
}

TEST(SimpleBinaryOpTest, ReferencePreluScalarAlphaAndQuantizedInt8) {
  const int32_t in_dims[] = {2, 2};
  const std::vector<int8_t> in = {-20, 0, 20, -10};
  const std::vector<int8_t> alpha = {10};  // 10 * 0.05 = 0.5
  std::vector<int8_t> out(4, 0);

  EXPECT_THAT(ReferencePrelu<int8_t>(
                  ConstTensorView<int8_t>{in, in_dims},
                  ConstTensorView<int8_t>{alpha, {}},
                  TensorView<int8_t>{absl::MakeSpan(out), in_dims},
                  AffineQuantSpec{0.1f, 0}, AffineQuantSpec{0.05f, 0},
                  AffineQuantSpec{0.1f, 0}),
              IsOk());
  EXPECT_THAT(out, ElementsAre(-10, 0, 20, -5));
}

TEST(SimpleBinaryOpTest, ReferencePreluEmptyOutputAllowsEmptySpans) {
  const int32_t in_dims[] = {0, 3};
  const int32_t alpha_dims[] = {1, 3};
  const std::vector<float> alpha = {0.5f, 0.25f, 0.125f};

  EXPECT_THAT(ReferencePrelu<float>(
                  ConstTensorView<float>{absl::Span<const float>(), in_dims},
                  ConstTensorView<float>{alpha, alpha_dims},
                  TensorView<float>{absl::Span<float>(), in_dims}),
              IsOk());
}

TEST(SimpleBinaryOpTest, ReferencePreluRejectsMismatchedOutputOrBadAlpha) {
  const int32_t in_dims[] = {1, 3};
  const int32_t alpha_dims[] = {2, 3};
  const int32_t bad_alpha_dims[] = {1, 2};
  const std::vector<float> in = {-1.0f, 2.0f, -3.0f};
  const std::vector<float> alpha = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  std::vector<float> out(6, -7.0f);

  // TFLite PReLU requires output shape to match input shape (alpha broadcasts
  // into input, not vice versa).
  EXPECT_THAT(
      ReferencePrelu<float>(ConstTensorView<float>{in, in_dims},
                            ConstTensorView<float>{alpha, alpha_dims},
                            TensorView<float>{absl::MakeSpan(out), alpha_dims}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  // Incompatible alpha shape is rejected.
  EXPECT_THAT(ReferencePrelu<float>(
                  ConstTensorView<float>{in, in_dims},
                  ConstTensorView<float>{absl::MakeConstSpan(alpha).first(2),
                                         bad_alpha_dims},
                  TensorView<float>{absl::MakeSpan(out).first(3), in_dims}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  // Undersized input buffer is rejected.
  EXPECT_THAT(
      ReferencePrelu<float>(
          ConstTensorView<float>{absl::MakeConstSpan(in).first(2), in_dims},
          ConstTensorView<float>{absl::MakeConstSpan(alpha).first(3), in_dims},
          TensorView<float>{absl::MakeSpan(out).first(3), in_dims}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f));
}

TEST(SimpleBinaryOpTest, ReferenceAddNNullInputElementLeavesOutputUntouched) {
  const std::vector<float> in = {1.0f, 2.0f};
  const std::vector<const float*> ptrs = {in.data(), nullptr};
  std::vector<float> out = {-7.0f, -7.0f};

  EXPECT_THAT(ReferenceAddN<float>(ptrs, out.data(), /*num_elements=*/2),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f));
}

TEST(SimpleBinaryOpTest, ReferenceAddNZeroElementsLeavesOutputUntouched) {
  const std::vector<float> in = {1.0f, 2.0f};
  const std::vector<const float*> ptrs = {in.data(), in.data()};
  std::vector<float> out = {-7.0f, -7.0f};

  EXPECT_THAT(ReferenceAddN<float>(ptrs, out.data(), /*num_elements=*/0),
              IsOk());
  EXPECT_THAT(out, ElementsAre(-7.0f, -7.0f));
}

TEST(SimpleBinaryOpTest, ReferenceAddNNullOutputReturnsInvalidArgument) {
  const std::vector<float> in = {1.0f, 2.0f};
  const std::vector<const float*> ptrs = {in.data(), in.data()};

  EXPECT_THAT(
      ReferenceAddN<float>(ptrs, /*output_data=*/nullptr, /*num_elements=*/2),
      IsError(kLiteRtStatusErrorInvalidArgument));
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

TEST(SimpleBinaryOpTest,
     ReferenceFloorDivAndFloorModTensorViewHandleEmptyAndUndersizedSpans) {
  const int32_t empty_dims[] = {0, 2};
  const int32_t b_dims[] = {1, 2};
  const int32_t b_zeros[] = {0, 0};

  // Empty output with zero divisors succeeds without division.
  EXPECT_THAT(
      ReferenceFloorDiv<int32_t>(
          ConstTensorView<int32_t>{absl::Span<const int32_t>(), empty_dims},
          ConstTensorView<int32_t>{b_zeros, b_dims},
          TensorView<int32_t>{absl::Span<int32_t>(), empty_dims}),
      IsOk());
  EXPECT_THAT(
      ReferenceFloorMod<int32_t>(
          ConstTensorView<int32_t>{absl::Span<const int32_t>(), empty_dims},
          ConstTensorView<int32_t>{b_zeros, b_dims},
          TensorView<int32_t>{absl::Span<int32_t>(), empty_dims}),
      IsOk());

  // Undersized span is rejected.
  const int32_t a[] = {7, 8};
  const int32_t b[] = {3, 3};
  int32_t out[] = {-9, -9};
  EXPECT_THAT(
      ReferenceFloorDiv<int32_t>(
          ConstTensorView<int32_t>{absl::MakeConstSpan(a).first(1), b_dims},
          ConstTensorView<int32_t>{b, b_dims},
          TensorView<int32_t>{out, b_dims}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-9, -9));
}

TEST(SimpleBinaryOpTest,
     ReferencePreluAndAddNRejectInvalidOutputQuantizationSpec) {
  const int32_t dims[] = {2};
  const int8_t in[] = {1, 2};
  const int8_t alpha[] = {1, 1};
  const int8_t* ptrs[] = {in, alpha};
  int8_t out[] = {-9, -9};

  EXPECT_THAT(ReferencePrelu<int8_t>(
                  ConstTensorView<int8_t>{in, dims},
                  ConstTensorView<int8_t>{alpha, dims},
                  TensorView<int8_t>{out, dims}, AffineQuantSpec{0.5f, 0},
                  AffineQuantSpec{0.5f, 0}, AffineQuantSpec{0.5f, 200}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ReferenceAddN<int8_t>(ptrs, out, 2, AffineQuantSpec{0.5f, 0},
                                    AffineQuantSpec{0.0f, 0}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-9, -9));
}

TEST(SimpleBinaryOpTest,
     ReferenceLogicalAndBitwiseTensorViewOverloadsComputeAndRejectBadShapes) {
  const int32_t dims[] = {4};
  const int32_t bad_dims[] = {3};
  const bool ba[] = {false, false, true, true};
  const bool bb[] = {false, true, false, true};
  bool bout[4] = {};

  EXPECT_THAT(ReferenceLogicalAnd(ConstTensorView<bool>{ba, dims},
                                  ConstTensorView<bool>{bb, dims},
                                  TensorView<bool>{bout, dims}),
              IsOk());
  EXPECT_THAT(bout, ElementsAre(false, false, false, true));

  EXPECT_THAT(ReferenceLogicalOr(ConstTensorView<bool>{ba, dims},
                                 ConstTensorView<bool>{bb, dims},
                                 TensorView<bool>{bout, dims}),
              IsOk());
  EXPECT_THAT(bout, ElementsAre(false, true, true, true));

  const int32_t ia[] = {0b1010, 0b1100, 16, -16};
  const int32_t ib[] = {0b0110, 0b1010, 2, 1};
  int32_t iout[4] = {};

  EXPECT_THAT(ReferenceBitwiseXor<int32_t>(ConstTensorView<int32_t>{ia, dims},
                                           ConstTensorView<int32_t>{ib, dims},
                                           TensorView<int32_t>{iout, dims}),
              IsOk());
  EXPECT_THAT(iout, ElementsAre(0b1100, 0b0110, 18, -15));

  EXPECT_THAT(ReferenceRightShift<int32_t>(ConstTensorView<int32_t>{ia, dims},
                                           ConstTensorView<int32_t>{ib, dims},
                                           TensorView<int32_t>{iout, dims}),
              IsOk());
  EXPECT_THAT(iout, ElementsAre(0, 0, 4, -8));

  // Incompatible TensorView shapes fail cleanly for generic and checked ops.
  EXPECT_THAT(ReferenceBitwiseXor<int32_t>(ConstTensorView<int32_t>{ia, dims},
                                           ConstTensorView<int32_t>{ib, dims},
                                           TensorView<int32_t>{iout, bad_dims}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ReferenceFloorDiv<int32_t>(ConstTensorView<int32_t>{ia, dims},
                                         ConstTensorView<int32_t>{ib, dims},
                                         TensorView<int32_t>{iout, bad_dims}),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(SimpleBinaryOpTest,
     ReferenceAddNAndPreluRejectByteOverflowAndNanQuantization) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const int32_t dims[] = {2};
  const int8_t in[] = {1, 2};
  const int8_t alpha[] = {1, 1};
  const int8_t* ptrs[] = {in, alpha};
  int8_t out[] = {-9, -9};

  // CheckedByteCount overflow on num_elements.
  EXPECT_THAT(ReferenceAddN<int32_t>(
                  {}, /*output_data=*/nullptr,
                  /*num_elements=*/std::numeric_limits<size_t>::max()),
              IsError(kLiteRtStatusErrorInvalidArgument));
  // NaN input scale causes QuantizeValueFromDouble to reject NaN into int8_t.
  EXPECT_THAT(ReferencePrelu<int8_t>(
                  ConstTensorView<int8_t>{in, dims},
                  ConstTensorView<int8_t>{alpha, dims},
                  TensorView<int8_t>{out, dims}, AffineQuantSpec{nan, 0},
                  AffineQuantSpec{0.5f, 0}, AffineQuantSpec{0.5f, 0}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ReferenceAddN<int8_t>(ptrs, out, 2, AffineQuantSpec{nan, 0},
                                    AffineQuantSpec{0.5f, 0}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-9, -9));
}

}  // namespace
}  // namespace litert::internal
