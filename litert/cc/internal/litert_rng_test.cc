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

#include "litert/cc/internal/litert_rng.h"

#include <bitset>
#include <chrono>  // NOLINT
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_numerics.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"

namespace litert {
namespace {

using ::litert::testing::RngTest;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::Le;
using ::testing::Types;
using ::testing::UnorderedElementsAre;

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
  for (auto _ : FuzzBlock(kTestIters, std::chrono::milliseconds(50))) {
    const auto val = gen(device);
    ASSERT_LE(val, gen.Max());
    ASSERT_GE(val, gen.Min());
  }
}

using RandomTensorTypeTest = RngTest;

TEST_F(RandomTensorTypeTest, FullySpecifiedRandomTensorType) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 1024, kLiteRtElementTypeFloat32>;
  R type;
  EXPECT_THAT(R::kMaxDimSize, Le(std::pow(static_cast<double>(R::kMaxFlatSize),
                                          1.0 / static_cast<double>(2))));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, type(device, {2, 2}));
  EXPECT_EQ(res.element_type, kLiteRtElementTypeFloat32);
  EXPECT_EQ(res.layout.rank, 2);
  EXPECT_THAT(absl::MakeSpan(res.layout.dimensions, 2), ElementsAre(2, 2));
  size_t num_elements;
  LITERT_ASSERT_OK(LiteRtGetNumLayoutElements(&res.layout, &num_elements));
  EXPECT_LE(num_elements, R::kMaxFlatSize);
}

TEST_F(RandomTensorTypeTest, RandomElementType) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 1024, kLiteRtElementTypeFloat32,
                       kLiteRtElementTypeInt32>;
  R type;
  EXPECT_THAT(R::kMaxDimSize, Le(std::pow(static_cast<double>(R::kMaxFlatSize),
                                          1.0 / static_cast<double>(2))));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, type(device, {2, 2}));
  EXPECT_TRUE(res.element_type == kLiteRtElementTypeFloat32 ||
              res.element_type == kLiteRtElementTypeInt32);
  EXPECT_EQ(res.layout.rank, 2);
  EXPECT_THAT(absl::MakeSpan(res.layout.dimensions, 2), ElementsAre(2, 2));
  size_t num_elements;
  LITERT_ASSERT_OK(LiteRtGetNumLayoutElements(&res.layout, &num_elements));
  EXPECT_LE(num_elements, R::kMaxFlatSize);
}

TEST_F(RandomTensorTypeTest, RandomDimFromRange) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 1024, kLiteRtElementTypeFloat32>;
  R type;
  EXPECT_THAT(R::kMaxDimSize, Le(std::pow(static_cast<double>(R::kMaxFlatSize),
                                          1.0 / static_cast<double>(2))));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, type(device, {R::DimRange(1, 10), 2}));
  EXPECT_EQ(res.layout.rank, 2);
  EXPECT_THAT(absl::MakeSpan(res.layout.dimensions, 2),
              ElementsAre(AllOf(Ge(1), Le(10)), 2));
  size_t num_elements;
  LITERT_ASSERT_OK(LiteRtGetNumLayoutElements(&res.layout, &num_elements));
  EXPECT_LE(num_elements, R::kMaxFlatSize);
}

TEST_F(RandomTensorTypeTest, RandomDim) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 5, kLiteRtElementTypeFloat32>;
  R type;
  EXPECT_THAT(R::kMaxDimSize, Le(std::pow(static_cast<double>(R::kMaxFlatSize),
                                          1.0 / static_cast<double>(2))));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, type(device, {2, R::RandDim()}));
  EXPECT_EQ(res.layout.rank, 2);
  EXPECT_THAT(absl::MakeSpan(res.layout.dimensions, 2),
              ElementsAre(2, AllOf(Ge(1), Le(2))));
  size_t num_elements;
  LITERT_ASSERT_OK(LiteRtGetNumLayoutElements(&res.layout, &num_elements));
  EXPECT_LE(num_elements, R::kMaxFlatSize);
}

TEST_F(RandomTensorTypeTest, Shuffle) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 4, /*MaxSize*/ 1024, kLiteRtElementTypeFloat32>;
  R type;
  EXPECT_THAT(R::kMaxDimSize, Le(std::pow(static_cast<double>(R::kMaxFlatSize),
                                          1.0 / static_cast<double>(4))));
  LITERT_ASSERT_OK_AND_ASSIGN(auto res,
                              type(device, {1, 2, 3, 4}, /*shuffle=*/true));
  EXPECT_EQ(res.layout.rank, 4);
  EXPECT_THAT(absl::MakeSpan(res.layout.dimensions, 4),
              UnorderedElementsAre(1, 2, 3, 4));
  size_t num_elements;
  LITERT_ASSERT_OK(LiteRtGetNumLayoutElements(&res.layout, &num_elements));
  EXPECT_LE(num_elements, R::kMaxFlatSize);
}

TEST_F(RandomTensorTypeTest, DimSizeTooLarge) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 5, kLiteRtElementTypeFloat32>;
  R type;
  ASSERT_FALSE(type(device, {3, 2}));
}

TEST_F(RandomTensorTypeTest, DimSizeTooSmall) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 4, kLiteRtElementTypeFloat32>;
  R type;
  ASSERT_FALSE(type(device, {0, 2}));
}

TEST_F(RandomTensorTypeTest, DimRangeTooLarge) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 4, kLiteRtElementTypeFloat32>;
  R type;
  ASSERT_FALSE(type(device, {R::DimRange(1, 3), 2}));
}

TEST_F(RandomTensorTypeTest, DimRangeTooSmall) {
  auto device = TracedDevice();
  using R =
      RandomTensorType</*Rank*/ 2, /*MaxSize*/ 4, kLiteRtElementTypeFloat32>;
  R type;
  ASSERT_FALSE(type(device, {R::DimRange(0, 1), 2}));
}

template <typename D>
class RandomTensorDataTest : public RngTest {};

using RandomTensorDataTestTypes = Types<int, float>;
TYPED_TEST_SUITE(RandomTensorDataTest, RandomTensorDataTestTypes);

TYPED_TEST(RandomTensorDataTest, NoRange) {
  auto device = this->TracedDevice();
  RandomTensorData<TypeParam, DefaultGenerator> data;
  std::vector<TypeParam> buf(10);
  LITERT_ASSERT_OK(data(device, absl::MakeSpan(buf)));
  EXPECT_EQ(data.High(), NumericLimits<TypeParam>::Max());
  EXPECT_EQ(data.Low(), NumericLimits<TypeParam>::Lowest());
  EXPECT_THAT(absl::MakeConstSpan(absl::MakeSpan(buf)),
              Each(AllOf(Le(data.High()), Ge(data.Low()))));
}

TYPED_TEST(RandomTensorDataTest, NoRangeFromSize) {
  auto device = this->TracedDevice();
  RandomTensorData<TypeParam, DefaultGenerator> data;
  LITERT_ASSERT_OK_AND_ASSIGN(const auto buf, data(device, 10));
  EXPECT_EQ(data.High(), NumericLimits<TypeParam>::Max());
  EXPECT_EQ(data.Low(), NumericLimits<TypeParam>::Lowest());
  EXPECT_THAT(absl::MakeConstSpan(buf),
              Each(AllOf(Le(data.High()), Ge(data.Low()))));
}

TYPED_TEST(RandomTensorDataTest, NoRangeFromLayout) {
  auto device = this->TracedDevice();
  RandomTensorData<TypeParam, DefaultGenerator> data;
  LiteRtLayout layout;
  layout.rank = 2;
  layout.dimensions[0] = 3;
  layout.dimensions[1] = 2;
  LITERT_ASSERT_OK_AND_ASSIGN(const auto buf, data(device, layout));
  ASSERT_EQ(buf.size(), 6);
  EXPECT_EQ(data.High(), NumericLimits<TypeParam>::Max());
  EXPECT_EQ(data.Low(), NumericLimits<TypeParam>::Lowest());
  EXPECT_THAT(absl::MakeConstSpan(buf),
              Each(AllOf(Le(data.High()), Ge(data.Low()))));
}

TYPED_TEST(RandomTensorDataTest, ExplicitRange) {
  auto device = this->TracedDevice();
  static constexpr TypeParam kMin = -10;
  static constexpr TypeParam kMax = 10;
  RandomTensorData<TypeParam, DefaultRangedGenerator> data(kMin, kMax);
  std::vector<TypeParam> buf(10);
  LITERT_ASSERT_OK(data(device, absl::MakeSpan(buf)));
  EXPECT_EQ(data.High(), 10);
  EXPECT_EQ(data.Low(), -10);
  EXPECT_THAT(absl::MakeConstSpan(buf),
              Each(AllOf(Le(data.High()), Ge(data.Low()))));
}

struct Functor {
  template <typename RandomTensor, typename Rng>
  auto operator()(RandomTensor& gen, Rng& rng) {
    return gen(rng, 10);
  }
};

TYPED_TEST(RandomTensorDataTest, BuilderWithRange) {
  auto device = this->TracedDevice();
  RandomTensorDataBuilder b;
  static constexpr TypeParam min = -1;
  static constexpr TypeParam max = 1;
  if constexpr (std::is_same_v<TypeParam, int32_t>) {
    b.SetIntRange(min, max);
  } else {
    b.SetFloatRange(min, max);
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto buf, (b.Call<TypeParam, Functor>(device)));
  EXPECT_EQ(buf.size(), 10);
  EXPECT_THAT(buf, Each(AllOf(Le(max), Ge(min))));
}

TYPED_TEST(RandomTensorDataTest, BuilderWithDummy) {
  auto device = this->TracedDevice();
  RandomTensorDataBuilder b;
  if constexpr (std::is_same_v<TypeParam, int32_t>) {
    b.SetIntDummy();
  } else {
    b.SetFloatDummy();
  }
  LITERT_ASSERT_OK_AND_ASSIGN(auto buf, (b.Call<TypeParam, Functor>(device)));
  EXPECT_EQ(buf.size(), 10);

  std::vector<TypeParam> expected(10);
  std::iota(expected.begin(), expected.end(), 0);

  EXPECT_EQ(buf, expected);
}

struct FloatGenerator {
  template <typename... Floats>
  explicit FloatGenerator(Floats&&... vals)
      : vals({std::forward<Floats>(vals)...}) {}

  uint32_t operator()() {
    auto res = vals[cur];
    cur = (cur + 1) % vals.size();
    return (*reinterpret_cast<uint32_t*>(&res));
  }

  size_t NumValues() const { return vals.size(); }

 private:
  std::vector<float> vals;
  size_t cur = 0;
};

TEST(F16InF23Test, Works) {
  FloatGenerator device(
      3.14f, .314f, -3.14f, 0.0061328127f, std::numeric_limits<float>::max(),
      std::numeric_limits<float>::lowest(), std::numeric_limits<float>::min());

  F16InF32Generator<float> f16_gen;
  // Leaking a bit of the implementation here but the generator algorithm
  // will emit (f16) subnormals but not infinities.
  static constexpr auto kSmallestF16SNormal = 0.00006103515625f;
  static constexpr auto kLargestF16Normal = 65504.0f;

  for (int i = 0; i < device.NumValues(); ++i) {
    const float f = std::abs(f16_gen(device));
    EXPECT_GE(f, kSmallestF16SNormal);
    EXPECT_LE(f, kLargestF16Normal);
    const auto mant_bits = std::bitset<32>(f);
    for (int i = 0; i < 13; ++i) {
      EXPECT_FALSE(mant_bits.test(mant_bits.size() - 1 - i));
    }
  }
}

}  // namespace
}  // namespace litert
