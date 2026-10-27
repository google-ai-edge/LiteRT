// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>

namespace litert::arm_vulkan_ml {
namespace {

TEST(Quantization, ScaleEncodingAndTinyMultipliers) {
  for (double scale : {1.0, 0.5, 0.0123456789, 1e-12, 0x1p-62}) {
    auto encoded = MakeScale32(scale);
    ASSERT_TRUE(encoded);
    const double quantum = std::ldexp(1.0, -encoded->shift);
    EXPECT_NEAR(std::ldexp(double(encoded->multiplier), -encoded->shift),
                scale, quantum * .50001);
  }
  EXPECT_FALSE(MakeScale32(0x1p-64));
  EXPECT_FALSE(MakeScale32(0));
  EXPECT_FALSE(MakeScale32(-1));
  EXPECT_FALSE(MakeScale32(std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(MakeScale32(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(MakeScale32(0x1p30));
}

TEST(Quantization, SingleRoundOracleAndRangeBoundaries) {
  std::mt19937 random(12345);
  for (int shift = 2; shift <= 62; ++shift) {
    const int64_t limit = int64_t{1} << (shift - 1);
    for (int i = 0; i < 1000; ++i) {
      const int64_t magnitude = random() % std::min(limit, int64_t{1} << 31);
      const int32_t input = (i & 1) ? -magnitude : magnitude;
      Scale32 scale{static_cast<int32_t>(random() & 0x7fffffff),
                    static_cast<int8_t>(shift)};
      // Independent exact rational oracle: Euclidean quotient and remainder,
      // followed by nearest rounding with upward ties. Use 128-bit arithmetic
      // so the oracle does not share the implementation's overflow assumptions.
      const __int128 product = __int128{input} * scale.multiplier;
      const __int128 divisor = __int128{1} << shift;
      __int128 quotient = product / divisor;
      __int128 remainder = product % divisor;
      if (remainder < 0) { --quotient; remainder += divisor; }
      if (remainder * 2 >= divisor) ++quotient;
      ASSERT_TRUE(ApplyScale32(input, scale));
      EXPECT_EQ(*ApplyScale32(input, scale), static_cast<int32_t>(quotient));
    }
    if (shift < 32) {
      EXPECT_TRUE(IsScale32RangeValid({1, static_cast<int8_t>(shift)},
                                      -limit, limit - 1));
      EXPECT_FALSE(IsScale32RangeValid({1, static_cast<int8_t>(shift)},
                                       -limit, limit));
    }
  }
  const auto half = *MakeScale32(.5);
  EXPECT_EQ(ApplyScale32(-3, half), -1);
  EXPECT_EQ(ApplyScale32(-1, half), 0);
  EXPECT_EQ(ApplyScale32(1, half), 1);
  EXPECT_EQ(ApplyScale32(3, half), 2);
  EXPECT_FALSE(ApplyScale32(1, {1, 63}));
}

TEST(Quantization, ActivationClamping) {
  auto range = ActivationRange(Activation::kRelu6, .1, -14);
  ASSERT_TRUE(range);
  EXPECT_EQ(range->minimum, -14);
  EXPECT_EQ(range->maximum, 46);
  EXPECT_EQ(ActivationRange(Activation::kRelu6, 1e-300, -14)->maximum, 127);
  EXPECT_EQ(ActivationRange(Activation::kNone, .1, -14)->minimum, -128);
  EXPECT_FALSE(ActivationRange(Activation::kRelu, .1, 128));
}

TEST(Quantization, AccumulatorBoundsContainAllInputs) {
  const int8_t weights[] = {-128, 127};
  auto range = DotProductRange(weights, 2, -14, 0, 23);
  ASSERT_TRUE(range);
  int32_t minimum = std::numeric_limits<int32_t>::max();
  int32_t maximum = std::numeric_limits<int32_t>::min();
  for (int a = -128; a <= 127; ++a) {
    for (int b = -128; b <= 127; ++b) {
      const int32_t value = (a + 14) * -128 + (b + 14) * 127 + 23;
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
  }
  EXPECT_EQ(range->minimum, minimum);
  EXPECT_EQ(range->maximum, maximum);
  EXPECT_FALSE(DotProductRange(weights, 2, -14, 0, INT32_MAX));
  EXPECT_FALSE(DotProductRange(nullptr, 1, 0, 0, 0));
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
