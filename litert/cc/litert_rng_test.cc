// Copyright 2025 Google LLC.
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

#include "litert/cc/litert_rng.h"

#include <chrono>  // NOLINT
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/cc/litert_numerics.h"
#include "litert/test/rng_fixture.h"

namespace litert {
namespace {

using ::litert::testing::RngTest;

static constexpr size_t kTestIters = 10;

using ::testing::HasSubstr;

struct DummyRng {
  using result_type = uint64_t;

  explicit DummyRng(result_type seed) : seed(seed) {}
  DummyRng() = default;

  static constexpr absl::string_view kName = "DummyRng";

  result_type operator()() { return 0; }

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return 0; }

  result_type seed;
};

TEST(LitertRngTestWithStdRng, Seed) {
  RandomDevice<std::mt19937> lite_rng(1234);
  EXPECT_THAT(absl::StrFormat("%v", lite_rng), HasSubstr("seed=1234,"));
}

TEST(LitertRngTestWithStdRng, NoSeed) {
  RandomDevice<std::mt19937> lite_rng;
  EXPECT_THAT(absl::StrFormat("%v", lite_rng), HasSubstr("seed=<default>,"));
}

TEST(LitertRngTestWithCustomRng, Seed) {
  RandomDevice<DummyRng> lite_rng(1234);
  EXPECT_THAT(absl::StrFormat("%v", lite_rng),
              HasSubstr("DummyRng(seed=1234,"));
}

TEST(LitertRngTestWithCustomRng, NoSeed) {
  RandomDevice<DummyRng> lite_rng;
  EXPECT_THAT(absl::StrFormat("%v", lite_rng),
              HasSubstr("DummyRng(seed=<default>,"));
}

using LiteRtRngTest = RngTest;

TEST_F(LiteRtRngTest, Ints) {
  auto device = TracedDevice();
  auto gen = DefaultGenerator<int>();
  static_assert(
      std::is_same_v<decltype(gen),
                     RangedGenerator<int, std::uniform_int_distribution>>);
  for (int i = 0; i < kTestIters; ++i) {
    const auto val = gen(device);
    ASSERT_LE(val, gen.Max());
    ASSERT_GE(val, gen.Min());
  }
}

TEST_F(LiteRtRngTest, IntsWithRange) {
  static constexpr auto kMin = 10;
  static constexpr auto kMax = 20;
  auto device = TracedDevice();
  auto gen = DefaultGenerator<int>(kMin, kMax);
  static_assert(
      std::is_same_v<decltype(gen),
                     RangedGenerator<int, std::uniform_int_distribution>>);
  EXPECT_EQ(gen.Max(), kMax);
  EXPECT_EQ(gen.Min(), kMin);
  for (int i = 0; i < kTestIters; ++i) {
    const auto val = gen(device);
    ASSERT_LE(val, kMax);
    ASSERT_GE(val, kMin);
  }
}

TEST_F(LiteRtRngTest, FloatsWithRange) {
  static constexpr auto kMin = 10.0f;
  static constexpr auto kMax = 20.0f;
  auto device = TracedDevice();
  auto gen = DefaultRangedGenerator<float>(kMin, kMax);
  static_assert(
      std::is_same_v<decltype(gen),
                     RangedGenerator<float, std::uniform_real_distribution>>);
  EXPECT_EQ(gen.Max(), kMax);
  EXPECT_EQ(gen.Min(), kMin);
  for (int i = 0; i < kTestIters; ++i) {
    const auto val = gen(device);
    ASSERT_LE(val, kMax);
    ASSERT_GE(val, kMin);
  }
}

TEST_F(LiteRtRngTest, ReinterpretFloat) {
  auto device = TracedDevice();
  auto gen = DefaultGenerator<float>();
  static_assert(std::is_same_v<
                decltype(gen),
                ReinterpretGenerator<float, std::uniform_real_distribution>>);
  for (int i = 0; i < kTestIters; ++i) {
    const auto val = gen(device);
    ASSERT_FALSE(std::isnan(val));
    ASSERT_TRUE(val == 0.0f || std::abs(val) > NumericLimits<float>::Min());
    ASSERT_LE(val, gen.Max());
    ASSERT_GE(val, gen.Min());
  }
}

TEST_F(LiteRtRngTest, TestWithFuzz) {
  auto device = TracedDevice();
  auto gen = DefaultGenerator<int>();
  for (auto _ :
       FuzzBlock(std::chrono::milliseconds(50), kTestIters, kTestIters)) {
    const auto val = gen(device);
    ASSERT_LE(val, gen.Max());
    ASSERT_GE(val, gen.Min());
  }
}

TEST_F(LiteRtRngTest, FullySpecifiedRandomTensorType) {
  auto device = TracedDevice();
  RandomTensorType type;
  auto tensor_type = type.Generate(
      device, {RandomTensorType::DimSpec(2u), RandomTensorType::DimSpec(2u)},
      {kLiteRtElementTypeFloat32});
  ASSERT_TRUE(tensor_type);
  EXPECT_EQ(tensor_type->element_type, kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor_type->layout.dimensions[0], 2);
  EXPECT_EQ(tensor_type->layout.dimensions[1], 2);
}

TEST_F(LiteRtRngTest, RandomElementType) {
  auto device = TracedDevice();
  RandomTensorType type;
  auto tensor_type = type.Generate(
      device, {}, {kLiteRtElementTypeFloat32, kLiteRtElementTypeInt32});
  ASSERT_TRUE(tensor_type);
  EXPECT_TRUE(tensor_type->element_type == kLiteRtElementTypeFloat32 ||
              tensor_type->element_type == kLiteRtElementTypeInt32);
}

TEST_F(LiteRtRngTest, RandomTensorShape) {
  auto device = TracedDevice();
  RandomTensorType type;
  auto tensor_type =
      type.Generate(device, {RandomTensorType::DimRange(1u, 3u), std::nullopt},
                    {kLiteRtElementTypeFloat32});
  ASSERT_TRUE(tensor_type);
  EXPECT_EQ(tensor_type->element_type, kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor_type->layout.rank, 2);
  const auto dim1 = tensor_type->layout.dimensions[0];
  EXPECT_GE(dim1, 1u);
  EXPECT_LE(dim1, 3u);
  const auto dim2 = tensor_type->layout.dimensions[1];
  EXPECT_GE(dim2, 0u);
  EXPECT_LE(dim2, RandomTensorType::kMaxDim);
}

TEST_F(LiteRtRngTest, RandomTensorShapeWithRandomRank) {
  auto device = TracedDevice();
  RandomTensorType type;
  auto tensor_type = type.Generate(device, /*max_rank=*/4);
  ASSERT_TRUE(tensor_type);
  EXPECT_LE(tensor_type->layout.rank, 4);
}

}  // namespace
}  // namespace litert
