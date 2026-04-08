// Copyright 2024 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_TFLITE_HALF_TRAITS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_TFLITE_HALF_TRAITS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>

#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_numerics.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "tflite/types/half.h"

namespace litert {

// So as to avoid pulling in tflite half.h dependency into public SDK (e.g.
// litert/cc) due to CMake support, all extensions to litert::NumericLimits,
// litert::ElementType, and litert::DataGenerator are defined in this file.

// -----------------------------------------------------------------------------
// NumericLimits
// -----------------------------------------------------------------------------

template <>
struct NumericLimits<tflite::half> {
 public:
  using DataType = tflite::half;

  static constexpr DataType Min() { return DataType::smallest_normal(); }
  static constexpr DataType Max() { return DataType::max(); }
  static constexpr DataType Lowest() { return DataType::min(); }
  static constexpr DataType Epsilon() { return DataType::epsilon(); }
};

template <typename T,
          std::enable_if_t<std::is_same_v<T, tflite::half>, bool> = true>
static constexpr T GetOrFlush(T val) {
  if (std::abs(static_cast<float>(val)) >
      static_cast<float>(NumericLimits<tflite::half>::Min())) {
    return val;
  }
  return static_cast<tflite::half>(0.0f);
}

// -----------------------------------------------------------------------------
// ElementType
// -----------------------------------------------------------------------------

template <>
inline constexpr ElementType GetElementType<tflite::half>() {
  return ElementType::Float16;
}

// -----------------------------------------------------------------------------
// Random Generators
// -----------------------------------------------------------------------------

template <template <typename> typename Dist>
class RangedGenerator<tflite::half, Dist> final
    : public DataGeneratorBase<tflite::half> {
 public:
  using DataType = tflite::half;
  using Wide = float;

  explicit RangedGenerator(Wide min = NumericLimits<DataType>::Lowest(),
                           Wide max = NumericLimits<DataType>::Max())
      : dist_(static_cast<float>(min), static_cast<float>(max)) {}

  template <typename Rng>
  DataType operator()(Rng& rng) {
    return static_cast<DataType>(dist_(rng));
  }

  DataType Max() const override { return static_cast<DataType>(dist_.max()); }
  DataType Min() const override { return static_cast<DataType>(dist_.min()); }

 private:
  Dist<float> dist_;
};

template <template <typename> typename Dist>
class ReinterpretGenerator<tflite::half, Dist> final
    : public DataGeneratorBase<tflite::half> {
 public:
  using DataType = tflite::half;
  using Wide = float;

  template <typename Rng>
  DataType operator()(Rng& rng) {
    DataType res;
    auto bits = dist_(rng);
    memcpy(&res, &bits, sizeof(res));
    return GetOrFlush(res);
  }

  ReinterpretGenerator() { ConstructAt(&dist_, 0, 0xFFFF); }
  ReinterpretGenerator(const ReinterpretGenerator&) = default;
  ReinterpretGenerator& operator=(const ReinterpretGenerator&) = default;
  ReinterpretGenerator(ReinterpretGenerator&&) = default;
  ReinterpretGenerator& operator=(ReinterpretGenerator&&) = default;

  DataType Max() const override { return NumericLimits<DataType>::Max(); }
  DataType Min() const override { return NumericLimits<DataType>::Lowest(); }

 private:
  std::uniform_int_distribution<uint16_t> dist_;
};

template <>
class DummyGenerator<tflite::half> final
    : public DataGeneratorBase<tflite::half> {
 public:
  using DataType = tflite::half;

  DummyGenerator() = default;

  template <typename Rng>
  DataType operator()(Rng& rng) {
    return static_cast<DataType>(val_++);
  }

  DataType Max() const override {
    return static_cast<DataType>(NumericLimits<DataType>::Max());
  }
  DataType Min() const override { return 0; }

 private:
  float val_ = 0;
};

// -----------------------------------------------------------------------------
// RandomTensorDataBuilder Extension
// -----------------------------------------------------------------------------

// Define the specific generator for half since DefaultRangedGenerator falls
// back to monostate
template <typename D>
using HalfGen = RangedGenerator<D, std::uniform_real_distribution>;

template <typename Functor, typename... Args>
auto CallHalf(const RandomTensorDataBuilder& b, Args&&... args) {
  if (b.IsFloatDummy()) {
    RandomTensorData<tflite::half, DummyGenerator> data;
    return Functor()(data, std::forward<Args>(args)...);
  } else {
    auto bounds = b.Bounds<float>();
    // Need to avoid overflow since Float max is much larger than half max
    float min =
        std::max(static_cast<float>(bounds.first),
                 static_cast<float>(NumericLimits<tflite::half>::Lowest()));
    float max =
        std::min(static_cast<float>(bounds.second),
                 static_cast<float>(NumericLimits<tflite::half>::Max()));

    RandomTensorData<tflite::half, HalfGen> data(
        static_cast<tflite::half>(min), static_cast<tflite::half>(max));
    return Functor()(data, std::forward<Args>(args)...);
  }
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_TFLITE_HALF_TRAITS_H_
