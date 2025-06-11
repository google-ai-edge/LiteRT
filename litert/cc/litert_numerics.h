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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_NUMERICS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_NUMERICS_H_

// Wrapper of std::numeric_limits, which needs to be extended for exotic
// datatypes (e.g. quant, half-precision, etc.).

#include <cstdint>
#include <limits>
#include <type_traits>

#include "litert/cc/litert_detail.h"

namespace litert {

template <typename T>
struct NumericLimits {
 public:
  using DataType = T;

 private:
  using StdLimits = std::numeric_limits<DataType>;

 public:
  // Smallest positive.
  static constexpr T Min() { return StdLimits::min(); }
  // Largest positive.
  static constexpr T Max() { return StdLimits::max(); }
  // Largest negative.
  static constexpr T Lowest() { return StdLimits::lowest(); }
};

// i32 -> i64, f32 -> f64.
template <typename T>
using WideType =
    SelectT<std::is_floating_point<T>, double, std::is_integral<T>, int64_t>;

// Get the 64bit version of the given type.
template <typename T>
static constexpr WideType<T> Widen(T val) {
  return static_cast<WideType<T>>(val);
}

// Create a upper boundary from a value which may exceed the intended datatypes
// limits.
template <typename T>
static constexpr T UpperBoundary(WideType<T> bound) {
  return std::min(bound, Widen(NumericLimits<T>::Max()));
}

// Create a lower boundary from a value which may exceed the intended datatypes
// limits.
template <typename T>
static constexpr T LowerBoundary(WideType<T> bound) {
  return std::max(bound, Widen(NumericLimits<T>::Lowest()));
}

// Flush denormals or NaN to zero.
template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
static constexpr T GetOrFlush(T val) {
  if (std::abs(val) > NumericLimits<T>::Min()) {
    return val;
  }
  return static_cast<T>(0.0f);
}

// Trait of number-like types that may be values within a tensor.
template <typename T>
using NumberLike =
    std::bool_constant<std::is_floating_point_v<T> || std::is_integral_v<T>>;

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_NUMERICS_H_
