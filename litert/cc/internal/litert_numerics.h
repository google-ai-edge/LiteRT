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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_NUMERICS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_NUMERICS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "litert/cc/internal/litert_detail.h"

/// @file
/// @brief Provides numeric utilities, including a wrapper for
/// `std::numeric_limits`
///        and helpers for handling exotic data types.

namespace litert {

/// @brief A wrapper for `std::numeric_limits` that can be extended for
///        exotic data types (e.g., quantized types, half-precision floats).
template <typename T>
struct NumericLimits {
 public:
  using DataType = T;

 private:
  using StdLimits = std::numeric_limits<DataType>;

 public:
  /// @brief Returns the smallest positive value.
  static constexpr T Min() { return StdLimits::min(); }
  /// @brief Returns the largest positive value.
  static constexpr T Max() { return StdLimits::max(); }
  /// @brief Returns the largest-magnitude negative value.
  static constexpr T Lowest() { return StdLimits::lowest(); }
};

/// @brief A type trait that provides a wider version of a numeric type
/// (e.g., `i32` -> `i64`, `f32` -> `f64`).
template <typename T>
using WideType =
    SelectT<std::is_floating_point<T>, double, std::is_integral<T>, int64_t>;

/// @brief Widens a given value to its 64-bit equivalent.
template <typename T>
static constexpr WideType<T> Widen(T val) {
  return static_cast<WideType<T>>(val);
}

/// @brief Creates an upper boundary from a value that may exceed the
///        intended data type's limits.
template <typename T>
static constexpr T UpperBoundary(WideType<T> bound) {
  return std::min(bound, Widen(NumericLimits<T>::Max()));
}

/// @brief Creates a lower boundary from a value that may exceed the
///        intended data type's limits.
template <typename T>
static constexpr T LowerBoundary(WideType<T> bound) {
  return std::max(bound, Widen(NumericLimits<T>::Lowest()));
}

/// @brief Flushes denormals or NaN to zero.
template <typename T,
          std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
static constexpr T GetOrFlush(T val) {
  if (std::abs(val) > NumericLimits<T>::Min()) {
    return val;
  }
  return static_cast<T>(0.0f);
}

/// @brief Represents a container element size that supports fractional byte
/// widths.
class ByteWidth {
 public:
  constexpr explicit ByteWidth(size_t numerator, size_t denominator = 1)
      : numerator_(numerator), denominator_(denominator) {}

  /// @brief Returns the number of bytes required for a buffer of a given
  /// number of elements.
  constexpr size_t NumBytes(size_t num_elements = 1) const {
    return Ceil(num_elements * numerator_, denominator_);
  }

  constexpr size_t operator*(size_t num_elements) const {
    return NumBytes(num_elements);
  }

  constexpr operator size_t() const { return NumBytes(); }  // NOLINT

 private:
  size_t numerator_;
  size_t denominator_;
};

/// @brief A trait for number-like types that can be values within a tensor.
template <typename T>
using NumberLike =
    std::bool_constant<std::is_floating_point_v<T> || std::is_integral_v<T>>;

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_NUMERICS_H_
