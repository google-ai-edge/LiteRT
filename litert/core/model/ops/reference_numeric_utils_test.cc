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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(ReferenceNumericUtilsTest, DequantizeFloatReturnsInputUnchanged) {
  EXPECT_THAT(DequantizeValue<float>(1.75f, AffineQuantSpec{0.05f, 10}),
              FloatEq(1.75f));
}

TEST(ReferenceNumericUtilsTest, DequantizeInt8AppliesZeroPointAndScale) {
  EXPECT_THAT(DequantizeValue<int8_t>(15, AffineQuantSpec{0.25f, 5}),
              FloatEq(2.5f));
}

TEST(ReferenceNumericUtilsTest, QuantizeFloatReturnsInputUnchanged) {
  EXPECT_THAT(QuantizeValue<float>(3.25f, AffineQuantSpec{0.1f, 7}),
              FloatEq(3.25f));
}

struct QuantizeInt8Case {
  std::string name;
  double value;
  AffineQuantSpec spec;
  int8_t expected;
};

using QuantizeInt8Test = ::testing::TestWithParam<QuantizeInt8Case>;

TEST_P(QuantizeInt8Test, QuantizesAndSaturatesWithoutOverflow) {
  const auto& tc = GetParam();
  EXPECT_EQ(QuantizeValueFromDouble<int8_t>(tc.value, tc.spec), tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    QuantizeInt8Tests, QuantizeInt8Test,
    ::testing::Values(
        QuantizeInt8Case{
            .name = "AppliesZeroPoint",
            .value = 1.0,
            .spec = {0.5f, 5},
            .expected = 7,
        },
        QuantizeInt8Case{
            .name = "PositiveHalfRoundsAwayFromZero",
            .value = 0.25,
            .spec = {0.5f, 0},
            .expected = 1,
        },
        QuantizeInt8Case{
            .name = "NegativeHalfRoundsAwayFromZero",
            .value = -0.25,
            .spec = {0.5f, 0},
            .expected = -1,
        },
        QuantizeInt8Case{
            .name = "SaturatesHighWithNonZeroZeroPoint",
            .value = 62.0,
            .spec = {0.5f, 5},
            .expected = 127,
        },
        QuantizeInt8Case{
            .name = "SaturatesLowWithNonZeroZeroPoint",
            .value = -62.0,
            .spec = {0.5f, -5},
            .expected = -128,
        },
        QuantizeInt8Case{
            .name = "LargeFinitePositiveSaturatesToMaxWithoutInt64Overflow",
            .value = 1e20,
            .spec = {0.04f, 10},
            .expected = 127,
        },
        QuantizeInt8Case{
            .name = "LargeFiniteNegativeSaturatesToMinWithoutInt64Overflow",
            .value = -1e20,
            .spec = {0.04f, 10},
            .expected = -128,
        },
        QuantizeInt8Case{
            .name = "PositiveInfinitySaturatesToMax",
            .value = std::numeric_limits<double>::infinity(),
            .spec = {0.04f, 10},
            .expected = 127,
        },
        QuantizeInt8Case{
            .name = "NegativeInfinitySaturatesToMin",
            .value = -std::numeric_limits<double>::infinity(),
            .spec = {0.04f, 10},
            .expected = -128,
        },
        QuantizeInt8Case{
            .name = "NaNReturnsClampedZeroPoint",
            .value = std::numeric_limits<double>::quiet_NaN(),
            .spec = {0.04f, 10},
            .expected = 10,
        },
        QuantizeInt8Case{
            .name = "ZeroScaleReturnsClampedZeroPoint",
            .value = 5.0,
            .spec = {0.0f, 10},
            .expected = 10,
        },
        QuantizeInt8Case{
            .name = "NegativeScaleReturnsClampedZeroPoint",
            .value = 5.0,
            .spec = {-0.5f, 200},
            .expected = 127,
        },
        QuantizeInt8Case{
            .name = "InfiniteScaleReturnsClampedZeroPoint",
            .value = 5.0,
            .spec = {std::numeric_limits<float>::infinity(), -200},
            .expected = -128,
        },
        QuantizeInt8Case{
            .name = "ExtremePositiveZeroPointSaturatesHigh",
            .value = 1.0,
            .spec = {1.0f, std::numeric_limits<int64_t>::max()},
            .expected = 127,
        },
        QuantizeInt8Case{
            .name = "ExtremeNegativeZeroPointSaturatesLow",
            .value = -1.0,
            .spec = {1.0f, std::numeric_limits<int64_t>::min()},
            .expected = -128,
        },
        QuantizeInt8Case{
            .name = "ExtremeZeroPointCancellingLargeValue",
            // At 2^55 IEEE-754 double ULP is 4, so use an exact offset of 4
            // to exercise the |zero_point| > 2^50 fallback path.
            .value = static_cast<double>(int64_t{1} << 55),
            .spec = {1.0f, -(int64_t{1} << 55) + 4},
            .expected = 4,
        }),
    [](const ::testing::TestParamInfo<QuantizeInt8Case>& info) {
      return info.param.name;
    });

TEST(ReferenceNumericUtilsTest,
     QuantizeFloatLargePositiveSaturatesInt8WithoutOverflow) {
  EXPECT_EQ(QuantizeValue<int8_t>(1e20f, 0.04f, 10), 127);
}

TEST(ReferenceNumericUtilsTest,
     QuantizeFloatLargeNegativeSaturatesInt8WithoutOverflow) {
  EXPECT_EQ(QuantizeValue<int8_t>(-1e20f, 0.04f, 10), -128);
}

TEST(ReferenceNumericUtilsTest, QuantizeUint32NegativeSaturatesToZero) {
  EXPECT_EQ(QuantizeValueFromDouble<uint32_t>(-1.0, 1.0f, 0), 0u);
}

TEST(ReferenceNumericUtilsTest, QuantizeUint32LargePositiveSaturatesToMax) {
  EXPECT_EQ(QuantizeValueFromDouble<uint32_t>(1e20, 1.0f, 0),
            std::numeric_limits<uint32_t>::max());
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

}  // namespace
}  // namespace litert::internal
