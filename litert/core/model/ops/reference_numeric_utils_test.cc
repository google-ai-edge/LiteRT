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

#include "litert/core/model/ops/reference_numeric_utils.h"

#include <cfenv>  // NOLINT(build/c++11): FPU rounding-mode invariance test.
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;
using ::testing::litert::IsOkAndHolds;

class ScopedRoundingMode {
 public:
  explicit ScopedRoundingMode(int mode) : original_mode_(std::fegetround()) {
    applied_ = (std::fesetround(mode) == 0);
  }
  ~ScopedRoundingMode() { std::fesetround(original_mode_); }

  bool applied() const { return applied_; }

 private:
  int original_mode_;
  bool applied_ = false;
};

// Prevents compile-time constant folding so floating-point operations execute
// under the runtime rounding mode set by `std::fesetround`.
double RuntimeValue(double value) {
  volatile double v = value;
  return v;
}

TEST(ReferenceNumericUtilsTest, DequantizeFloatReturnsInputUnchanged) {
  EXPECT_THAT(DequantizeValue<float>(1.75f, AffineQuantSpec{0.05f, 10}),
              FloatEq(1.75f));
}

TEST(ReferenceNumericUtilsTest, DequantizeInt8AppliesZeroPointAndScale) {
  EXPECT_THAT(DequantizeValue<int8_t>(15, AffineQuantSpec{0.25f, 5}),
              FloatEq(2.5f));
  EXPECT_THAT(DequantizeValue<int8_t>(-10, 0.5f, -6), FloatEq(-2.0f));
}

TEST(ReferenceNumericUtilsTest,
     DequantizeCoversUnsignedAndWiderIntegerTypesWithThreeArgOverload) {
  EXPECT_THAT(DequantizeValue<uint8_t>(200, 0.25f, 128), FloatEq(18.0f));
  EXPECT_THAT(DequantizeValue<int16_t>(-1000, 0.125f, -200), FloatEq(-100.0f));
  EXPECT_THAT(DequantizeValue<uint16_t>(40000, 0.5f, 32768), FloatEq(3616.0f));
  EXPECT_THAT(DequantizeValue<int32_t>(100000, 0.25f, -20000),
              FloatEq(30000.0f));
  EXPECT_THAT(DequantizeValue<uint32_t>(4000000000u, 0.5f, 3999999990LL),
              FloatEq(5.0f));
}

TEST(ReferenceNumericUtilsTest, QuantizeFloatPassthroughIgnoresIntMetadata) {
  EXPECT_THAT(QuantizeValue<float>(3.25f, AffineQuantSpec{0.1f, 7}),
              IsOkAndHolds(FloatEq(3.25f)));
  EXPECT_THAT(QuantizeValue<float>(3.25f, AffineQuantSpec{-1.0f, 1000}),
              IsOkAndHolds(FloatEq(3.25f)));
  Expected<float> nan_res = QuantizeValueFromDouble<float>(
      std::numeric_limits<double>::quiet_NaN(), AffineQuantSpec{0.0f, 999});
  ASSERT_THAT(nan_res, IsOk());
  EXPECT_TRUE(std::isnan(*nan_res));
}

struct InvalidInt8ZeroPointCase {
  std::string name;
  int64_t zero_point;
  double value;
};

using QuantizeInvalidInt8ZeroPointTest =
    ::testing::TestWithParam<InvalidInt8ZeroPointCase>;

TEST_P(QuantizeInvalidInt8ZeroPointTest, RejectsOutOfRangeZeroPoint) {
  const auto& tc = GetParam();
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(tc.value, 1.0f, tc.zero_point),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(QuantizeValue<int8_t>(static_cast<float>(tc.value),
                                    AffineQuantSpec{1.0f, tc.zero_point}),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    QuantizeInvalidInt8ZeroPointTests, QuantizeInvalidInt8ZeroPointTest,
    ::testing::Values(InvalidInt8ZeroPointCase{"JustBelowMin", -129, 0.0},
                      InvalidInt8ZeroPointCase{"JustAboveMax", 128, 0.0},
                      InvalidInt8ZeroPointCase{"OutOfRange200", 200, 1.0},
                      InvalidInt8ZeroPointCase{"OutOfRange200NegativeVal", 200,
                                               -1.0},
                      InvalidInt8ZeroPointCase{
                          "LargeNegativeCancellingValue",
                          -(int64_t{1} << 55) + 1,
                          static_cast<double>(int64_t{1} << 55),
                      },
                      InvalidInt8ZeroPointCase{
                          "Int64Min",
                          std::numeric_limits<int64_t>::min(),
                          0.0,
                      },
                      InvalidInt8ZeroPointCase{
                          "Int64MaxCancellingMin",
                          std::numeric_limits<int64_t>::max(),
                          -static_cast<double>(uint64_t{1} << 63),
                      }),
    [](const ::testing::TestParamInfo<InvalidInt8ZeroPointCase>& info) {
      return info.param.name;
    });

template <typename T>
class QuantizeSupportedIntTypesTest : public ::testing::Test {};

using SupportedQuantizedIntTypes =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t>;
TYPED_TEST_SUITE(QuantizeSupportedIntTypesTest, SupportedQuantizedIntTypes);

TYPED_TEST(QuantizeSupportedIntTypesTest,
           AcceptsEndpointZeroPointsAndRejectsBeyondEndpoints) {
  using T = TypeParam;
  constexpr int64_t kMin = static_cast<int64_t>(std::numeric_limits<T>::min());
  constexpr int64_t kMax = static_cast<int64_t>(std::numeric_limits<T>::max());

  EXPECT_THAT(QuantizeValueFromDouble<T>(0.0, 0.5f, kMin),
              IsOkAndHolds(std::numeric_limits<T>::min()));
  EXPECT_THAT(QuantizeValueFromDouble<T>(0.0, 0.5f, kMax),
              IsOkAndHolds(std::numeric_limits<T>::max()));
  EXPECT_THAT(QuantizeValue<T>(0.0f, 0.5f, kMin),
              IsOkAndHolds(std::numeric_limits<T>::min()));
  EXPECT_THAT(QuantizeValue<T>(0.0f, 0.5f, kMax),
              IsOkAndHolds(std::numeric_limits<T>::max()));

  EXPECT_THAT(QuantizeValueFromDouble<T>(0.0, 0.5f, kMin - 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(QuantizeValueFromDouble<T>(0.0, 0.5f, kMax + 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

struct InvalidScaleCase {
  std::string name;
  float scale;
};

using QuantizeInvalidScaleTest = ::testing::TestWithParam<InvalidScaleCase>;

TEST_P(QuantizeInvalidScaleTest, RejectsNonPositiveOrNonFiniteScale) {
  const auto& tc = GetParam();
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(1.0, tc.scale, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(QuantizeValue<int8_t>(1.0f, tc.scale, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    QuantizeInvalidScaleTests, QuantizeInvalidScaleTest,
    ::testing::Values(
        InvalidScaleCase{"Zero", 0.0f}, InvalidScaleCase{"NegativeZero", -0.0f},
        InvalidScaleCase{"Negative", -0.5f},
        InvalidScaleCase{"NegativeSubnormal",
                         -std::numeric_limits<float>::denorm_min()},
        InvalidScaleCase{"QuietNaN", std::numeric_limits<float>::quiet_NaN()},
        InvalidScaleCase{"PositiveInfinity",
                         std::numeric_limits<float>::infinity()},
        InvalidScaleCase{"NegativeInfinity",
                         -std::numeric_limits<float>::infinity()}),
    [](const ::testing::TestParamInfo<InvalidScaleCase>& info) {
      return info.param.name;
    });

TEST(ReferenceNumericUtilsTest, AcceptsPositiveSubnormalScale) {
  const float subnormal = std::numeric_limits<float>::denorm_min();
  EXPECT_THAT(ValidateQuantization<int8_t>(AffineQuantSpec{subnormal, 11}),
              IsOk());
  EXPECT_THAT(ValidateQuantization<int8_t>(AffineQuantSpec{0.5f, 128}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(0.0, subnormal, 11),
              IsOkAndHolds(11));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(1.0, subnormal, 0),
              IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(-1.0, subnormal, 0),
              IsOkAndHolds(-128));
}

TEST(ReferenceNumericUtilsTest, RejectsNaNValueForIntegerOutput) {
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(qnan, 0.5f, 10),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(-qnan, 0.5f, 10),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      QuantizeValue<int8_t>(std::numeric_limits<float>::quiet_NaN(), 0.5f, 10),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(ReferenceNumericUtilsTest,
     MatchesPythonQuantizerTiesToEvenAndZeroPointOrder) {
  const std::vector<double> values = {-1.5, -0.5, 0.5, 1.5};

  std::vector<int8_t> zp0_out;
  std::vector<int8_t> zp1_out;
  std::vector<int8_t> zp_neg1_out;
  std::vector<int8_t> zp3_out;
  std::vector<int8_t> zp_neg3_out;
  for (double v : values) {
    LITERT_ASSERT_OK_AND_ASSIGN(int8_t q0,
                                QuantizeValueFromDouble<int8_t>(v, 1.0f, 0));
    LITERT_ASSERT_OK_AND_ASSIGN(int8_t q1,
                                QuantizeValueFromDouble<int8_t>(v, 1.0f, 1));
    LITERT_ASSERT_OK_AND_ASSIGN(int8_t qn1,
                                QuantizeValueFromDouble<int8_t>(v, 1.0f, -1));
    LITERT_ASSERT_OK_AND_ASSIGN(int8_t q3,
                                QuantizeValueFromDouble<int8_t>(v, 1.0f, 3));
    LITERT_ASSERT_OK_AND_ASSIGN(int8_t qn3,
                                QuantizeValueFromDouble<int8_t>(v, 1.0f, -3));
    zp0_out.push_back(q0);
    zp1_out.push_back(q1);
    zp_neg1_out.push_back(qn1);
    zp3_out.push_back(q3);
    zp_neg3_out.push_back(qn3);
  }

  // units = val / 1 + zp, then round_ties_to_even(units):
  //   zp =  0 -> [-1.5, -0.5,  0.5,  1.5] -> [-2,  0,  0,  2]
  //   zp =  1 -> [-0.5,  0.5,  1.5,  2.5] -> [ 0,  0,  2,  2]
  //   zp = -1 -> [-2.5, -1.5, -0.5,  0.5] -> [-2, -2,  0,  0]
  //   zp =  3 -> [ 1.5,  2.5,  3.5,  4.5] -> [ 2,  2,  4,  4]
  //   zp = -3 -> [-4.5, -3.5, -2.5, -1.5] -> [-4, -4, -2, -2]
  EXPECT_THAT(zp0_out, ElementsAre(-2, 0, 0, 2));
  EXPECT_THAT(zp1_out, ElementsAre(0, 0, 2, 2));
  EXPECT_THAT(zp_neg1_out, ElementsAre(-2, -2, 0, 0));
  EXPECT_THAT(zp3_out, ElementsAre(2, 2, 4, 4));
  EXPECT_THAT(zp_neg3_out, ElementsAre(-4, -4, -2, -2));
}

TEST(ReferenceNumericUtilsTest, DistinguishesNearHalfStepsFromExactTies) {
  // Around +0.5 and -0.5 with zp = 0 (ties round to 0).
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(0.5, 0.0), 1.0f, 0),
      IsOkAndHolds(0));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(0.5, 1.0), 1.0f, 0),
      IsOkAndHolds(1));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(-0.5, 0.0), 1.0f, 0),
      IsOkAndHolds(0));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(-0.5, -1.0), 1.0f, 0),
      IsOkAndHolds(-1));

  // Around +1.5 and -1.5 with zp = 0 (ties round to +/-2).
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(1.5, 1.0), 1.0f, 0),
      IsOkAndHolds(1));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(1.5, 2.0), 1.0f, 0),
      IsOkAndHolds(2));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(-1.5, -1.0), 1.0f, 0),
      IsOkAndHolds(-1));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(-1.5, -2.0), 1.0f, 0),
      IsOkAndHolds(-2));

  // Odd zero points: choose values whose `val / 1.0 + zp` sits one double ULP
  // below/above the half step in `units` (1.5 for zp = 1, -0.5 for zp = -1).
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(1.5, 1.0) - 1.0, 1.0f, 1),
      IsOkAndHolds(1));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(1.5, 2.0) - 1.0, 1.0f, 1),
      IsOkAndHolds(2));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(std::nextafter(-0.5, -1.0) + 1.0,
                                              1.0f, -1),
              IsOkAndHolds(-1));
  EXPECT_THAT(
      QuantizeValueFromDouble<int8_t>(std::nextafter(0.5, 1.0), 1.0f, -1),
      IsOkAndHolds(0));
}

TEST(ReferenceNumericUtilsTest, SaturatesLargeFiniteValuesAndInfinities) {
  const double pos_inf = std::numeric_limits<double>::infinity();
  const double neg_inf = -std::numeric_limits<double>::infinity();
  const double max_dbl = std::numeric_limits<double>::max();

  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(62.0, 0.5f, 5),
              IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(-62.0, 0.5f, -5),
              IsOkAndHolds(-128));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(1e20, 0.04f, 10),
              IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(-1e20, 0.04f, 10),
              IsOkAndHolds(-128));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(max_dbl, 0.04f, 10),
              IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(-max_dbl, 0.04f, 10),
              IsOkAndHolds(-128));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(pos_inf, 0.04f, 10),
              IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValueFromDouble<int8_t>(neg_inf, 0.04f, 10),
              IsOkAndHolds(-128));

  EXPECT_THAT(QuantizeValue<int8_t>(1.5f, AffineQuantSpec{1.0f, 1}),
              IsOkAndHolds(2));
  EXPECT_THAT(QuantizeValue<int8_t>(0.5f, AffineQuantSpec{1.0f, 0}),
              IsOkAndHolds(0));
  EXPECT_THAT(QuantizeValue<int8_t>(1e20f, 0.04f, 10), IsOkAndHolds(127));
  EXPECT_THAT(QuantizeValue<int8_t>(-1e20f, 0.04f, 10), IsOkAndHolds(-128));
  EXPECT_THAT(
      QuantizeValue<int8_t>(std::numeric_limits<float>::infinity(), 0.04f, 10),
      IsOkAndHolds(127));
  EXPECT_THAT(
      QuantizeValue<int8_t>(-std::numeric_limits<float>::infinity(), 0.04f, 10),
      IsOkAndHolds(-128));
}

TEST(ReferenceNumericUtilsTest,
     SaturatesInt32AndUint32EndpointsForDoubleAndFloatInputs) {
  constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();
  constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
  constexpr uint32_t kUint32Max = std::numeric_limits<uint32_t>::max();

  EXPECT_THAT(QuantizeValueFromDouble<int32_t>(1e20, 1.0f, 0),
              IsOkAndHolds(kInt32Max));
  EXPECT_THAT(QuantizeValueFromDouble<int32_t>(-1e20, 1.0f, 0),
              IsOkAndHolds(kInt32Min));
  EXPECT_THAT(QuantizeValue<int32_t>(1e20f, 1.0f, 0), IsOkAndHolds(kInt32Max));
  EXPECT_THAT(QuantizeValue<int32_t>(-1e20f, 1.0f, 0), IsOkAndHolds(kInt32Min));

  EXPECT_THAT(QuantizeValueFromDouble<uint32_t>(-1.0, 1.0f, 0),
              IsOkAndHolds(0u));
  EXPECT_THAT(QuantizeValueFromDouble<uint32_t>(1e20, 1.0f, 0),
              IsOkAndHolds(kUint32Max));
  EXPECT_THAT(QuantizeValue<uint32_t>(-1.0f, 1.0f, 0), IsOkAndHolds(0u));
  EXPECT_THAT(QuantizeValue<uint32_t>(1e20f, 1.0f, 0),
              IsOkAndHolds(kUint32Max));
}

TEST(ReferenceNumericUtilsTest,
     RoundTiesToEvenIsIndependentOfFloatingPointRoundingMode) {
  const std::vector<std::pair<double, double>> cases = {
      {-2.5, -2.0},
      {-1.5, -2.0},
      {-0.5, 0.0},
      {0.0, 0.0},
      {0.5, 0.0},
      {1.5, 2.0},
      {2.5, 2.0},
      {std::nextafter(0.5, 0.0), 0.0},
      {std::nextafter(0.5, 1.0), 1.0},
      {std::nextafter(-0.5, 0.0), 0.0},
      {std::nextafter(-0.5, -1.0), -1.0},
      {std::nextafter(1.5, 1.0), 1.0},
      {std::nextafter(1.5, 2.0), 2.0},
      {std::nextafter(-1.5, -1.0), -1.0},
      {std::nextafter(-1.5, -2.0), -2.0},
  };
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  for (int mode : modes) {
    ScopedRoundingMode guard(mode);
    ASSERT_TRUE(guard.applied()) << "Failed to apply rounding mode " << mode;
    for (const auto& [input, expected] : cases) {
      EXPECT_THAT(RoundTiesToEven(RuntimeValue(input)), DoubleEq(expected))
          << "mode=" << mode << ", input=" << input;
    }
    EXPECT_THAT(QuantizeValueFromDouble<int8_t>(RuntimeValue(0.5),
                                                AffineQuantSpec{1.0f, 1}),
                IsOkAndHolds(2))
        << "mode=" << mode;
    EXPECT_THAT(QuantizeValueFromDouble<int8_t>(RuntimeValue(1.5),
                                                AffineQuantSpec{1.0f, 1}),
                IsOkAndHolds(2))
        << "mode=" << mode;
  }
}

TEST(ReferenceNumericUtilsTest, ApplyActivationReluClampsNegatives) {
  std::vector<float> vals = {-2.0f, 0.0f, 3.0f, 8.0f};
  ApplyActivation(vals.data(), vals.size(),
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(vals, ElementsAre(0.0f, 0.0f, 3.0f, 8.0f));
}

TEST(ReferenceNumericUtilsTest, ApplyActivationRelu6ClampsBothBounds) {
  std::vector<float> vals = {-2.0f, 3.0f, 8.0f};
  const int32_t dims[1] = {3};
  ApplyActivation(vals.data(), dims, 1, tflite::ActivationFunctionType_RELU6);
  EXPECT_THAT(vals, ElementsAre(0.0f, 3.0f, 6.0f));
}

TEST(ReferenceNumericUtilsTest, ApplyActivationNegativeDimLeavesUntouched) {
  std::vector<float> vals = {-2.0f};
  const int32_t neg_dims[1] = {-1};
  ApplyActivation(vals.data(), neg_dims, 1,
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(vals, ElementsAre(-2.0f));
}

TEST(ReferenceNumericUtilsTest, ApplyActivationNullDimsLeavesUntouched) {
  std::vector<float> vals = {-2.0f};
  ApplyActivation(vals.data(), /*output_dims=*/nullptr, 1,
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(vals, ElementsAre(-2.0f));
}

TEST(ReferenceNumericUtilsTest, ApplyActivationNullDataIsNoOp) {
  ApplyActivation(static_cast<float*>(nullptr), 1,
                  tflite::ActivationFunctionType_RELU);
}

TEST(ReferenceNumericUtilsTest,
     ApplyActivationHandlesNoneUnsupportedScalarNegativeRankAndEmpty) {
  std::vector<float> vals = {-2.0f, 0.5f, 8.0f};
  const int32_t dims[1] = {3};
  ApplyActivation(vals.data(), dims, 1, tflite::ActivationFunctionType_NONE);
  EXPECT_THAT(vals, ElementsAre(-2.0f, 0.5f, 8.0f));

  ApplyActivation(vals.data(), vals.size(),
                  tflite::ActivationFunctionType_RELU_N1_TO_1);
  ApplyActivation(vals.data(), vals.size(),
                  tflite::ActivationFunctionType_TANH);
  EXPECT_THAT(vals, ElementsAre(-2.0f, 0.5f, 8.0f));

  // Rank-0 scalar with null dims applies activation to the single element.
  float scalar = -3.0f;
  ApplyActivation(&scalar, /*output_dims=*/nullptr, /*rank=*/0,
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(scalar, FloatEq(0.0f));

  // Negative rank leaves buffer untouched.
  ApplyActivation(vals.data(), dims, /*rank=*/-1,
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(vals, ElementsAre(-2.0f, 0.5f, 8.0f));

  // Empty tensors (zero elements or zero dimension) are no-ops.
  ApplyActivation(vals.data(), /*num_elements=*/0,
                  tflite::ActivationFunctionType_RELU);
  const int32_t empty_dims[2] = {0, 3};
  ApplyActivation(vals.data(), empty_dims, /*rank=*/2,
                  tflite::ActivationFunctionType_RELU);
  EXPECT_THAT(vals, ElementsAre(-2.0f, 0.5f, 8.0f));
}

}  // namespace
}  // namespace litert::internal
