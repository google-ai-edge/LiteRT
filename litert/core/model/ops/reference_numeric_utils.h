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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_REFERENCE_NUMERIC_UTILS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_REFERENCE_NUMERIC_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/tensor_type_util.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

// Per-tensor affine quantization parameters shared by core reference kernels
// and test generators.
struct AffineQuantSpec {
  float scale = 1.0f;
  int64_t zero_point = 0;
};

template <typename T>
inline constexpr bool kIsSupportedQuantizedInt =
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

template <typename T>
inline constexpr bool kIsSupportedNumericType =
    std::is_floating_point_v<T> || kIsSupportedQuantizedInt<T>;

// Dequantizes a single value to `float` using `(val - zero_point) * scale` in
// `double` intermediate precision. Floating-point inputs pass through unchanged
// as `static_cast<float>(val)` without applying `scale` or `zero_point`.
template <typename T>
inline float DequantizeValue(T val, float scale, int64_t zero_point) {
  static_assert(kIsSupportedNumericType<T>,
                "DequantizeValue supports floating-point and <=32-bit integer "
                "types.");
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<float>(val);
  } else {
    return static_cast<float>(
        (static_cast<double>(val) - static_cast<double>(zero_point)) *
        static_cast<double>(scale));
  }
}

// Overload of `DequantizeValue` taking an `AffineQuantSpec`.
template <typename T>
inline float DequantizeValue(T val, const AffineQuantSpec& spec) {
  return DequantizeValue<T>(val, spec.scale, spec.zero_point);
}

// Validates affine quantization parameters for integer destination `T` (at most
// 32 bits). Requires `scale` to be finite and strictly positive (positive
// subnormals are accepted) and `zero_point` to lie within
// `[min<T>, max<T>]`.
template <typename T>
inline Expected<void> ValidateQuantization(float scale, int64_t zero_point) {
  static_assert(kIsSupportedQuantizedInt<T>,
                "ValidateQuantization supports <=32-bit integer types.");
  if (!std::isfinite(scale) || scale <= 0.0f) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Quantization scale must be finite and strictly "
                      "positive");
  }
  constexpr int64_t kLo = static_cast<int64_t>(std::numeric_limits<T>::min());
  constexpr int64_t kHi = static_cast<int64_t>(std::numeric_limits<T>::max());
  if (zero_point < kLo || zero_point > kHi) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Quantization zero_point is out of destination range");
  }
  return {};
}

template <typename T>
inline Expected<void> ValidateQuantization(const AffineQuantSpec& spec) {
  return ValidateQuantization<T>(spec.scale, spec.zero_point);
}

// Rounds a finite `double` to the nearest integer with ties to even,
// independent of the active floating-point rounding mode (`fesetround`).
inline double RoundTiesToEven(double x) {
  const double abs_x = std::fabs(x);
  const double fl = std::floor(abs_x);
  // For `abs_x < 1.0`, `fl == 0.0` so `abs_x - fl == abs_x` is exact.
  // For `abs_x >= 1.0`, `fl <= abs_x < fl + 1.0 <= 2.0 * fl`, so `abs_x - fl`
  // is exact by Sterbenz's lemma under every IEEE-754 rounding mode.
  const double frac = abs_x - fl;
  double rounded_abs = fl;
  if (frac > 0.5) {
    rounded_abs = fl + 1.0;
  } else if (frac == 0.5) {
    // `frac == 0.5` implies `fl < 2^52`, so `fl`, `fl + 1.0`, and `fl * 0.5`
    // are exact in IEEE-754 double.
    if (std::floor(fl * 0.5) * 2.0 != fl) {
      rounded_abs = fl + 1.0;
    }
  }
  return (x < 0.0 && rounded_abs != 0.0) ? -rounded_abs : rounded_abs;
}

// Quantizes a real value into `T` using the official LiteRT Python quantizer's
// affine-quantization order and ties-to-even rounding rule:
//   units = val / scale + zero_point
//   q = clip(round_ties_to_even(units), min<T>, max<T>)
//
// Policy:
//   - For floating-point `T`, passes through `static_cast<T>(val)` without
//     applying integer quantization metadata validation.
//   - For supported integer `T` (<= 32 bits), validates `scale` (finite and
//     strictly positive) and `zero_point` (within `[min<T>, max<T>]`) before
//     arithmetic, and rejects `NaN` values with
//     `kLiteRtStatusErrorInvalidArgument`.
//   - Retains `double` intermediate precision and clamps to `[min<T>, max<T>]`
//     before rounding and integer conversion so large finite values and
//     infinities saturate cleanly without UB.
//   - Uses the full storage range `[min<T>, max<T>]`; symmetric narrow ranges
//     or sub-byte ranges are not inferred from `zero_point == 0` or `T`.
template <typename T>
inline Expected<T> QuantizeValueFromDouble(double val, float scale,
                                           int64_t zero_point) {
  static_assert(kIsSupportedNumericType<T>,
                "QuantizeValueFromDouble supports floating-point and <=32-bit "
                "integer types.");
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(val);
  } else {
    if (Expected<void> valid = ValidateQuantization<T>(scale, zero_point);
        !valid) {
      return valid.Error();
    }
    if (std::isnan(val)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Cannot quantize NaN into an integer type");
    }
    const double lo = static_cast<double>(std::numeric_limits<T>::min());
    const double hi = static_cast<double>(std::numeric_limits<T>::max());
    double units = val / static_cast<double>(scale);
    units += static_cast<double>(zero_point);
    units = std::clamp(units, lo, hi);
    return static_cast<T>(RoundTiesToEven(units));
  }
}

template <typename T>
inline Expected<T> QuantizeValueFromDouble(double val,
                                           const AffineQuantSpec& spec) {
  return QuantizeValueFromDouble<T>(val, spec.scale, spec.zero_point);
}

// Thin `float`-input forwarding wrappers around `QuantizeValueFromDouble`.
template <typename T>
inline Expected<T> QuantizeValue(float val, float scale, int64_t zero_point) {
  return QuantizeValueFromDouble<T>(static_cast<double>(val), scale,
                                    zero_point);
}

template <typename T>
inline Expected<T> QuantizeValue(float val, const AffineQuantSpec& spec) {
  return QuantizeValue<T>(val, spec.scale, spec.zero_point);
}

// Applies a fused activation function in-place in unquantized floating-point
// value space (`T` must be floating-point).
//
// Only `ActivationFunctionType_NONE`, `ActivationFunctionType_RELU`, and
// `ActivationFunctionType_RELU6` are handled; unsupported activation enum
// values, empty buffers (`num_elements == 0`), and null `output_data` are
// treated as no-ops.
template <typename T>
inline void ApplyActivation(T* absl_nullable output_data, size_t num_elements,
                            tflite::ActivationFunctionType faf) {
  static_assert(
      std::is_floating_point_v<T>,
      "ApplyActivation operates in unquantized floating-point value space.");
  if (num_elements == 0 || output_data == nullptr) {
    return;
  }
  if (faf == tflite::ActivationFunctionType_RELU) {
    for (size_t i = 0; i < num_elements; ++i) {
      if (output_data[i] < static_cast<T>(0.0f)) {
        output_data[i] = static_cast<T>(0.0f);
      }
    }
  } else if (faf == tflite::ActivationFunctionType_RELU6) {
    for (size_t i = 0; i < num_elements; ++i) {
      if (output_data[i] < static_cast<T>(0.0f)) {
        output_data[i] = static_cast<T>(0.0f);
      } else if (output_data[i] > static_cast<T>(6.0f)) {
        output_data[i] = static_cast<T>(6.0f);
      }
    }
  }
}

// Shape-based overload of `ApplyActivation` operating in unquantized
// floating-point value space. Invalid shapes (`rank < 0`, null `output_dims`
// with `rank > 0`, negative dimensions, or element-count overflow), empty
// shapes, `ActivationFunctionType_NONE`, and unsupported activation enum values
// are treated as no-ops.
template <typename T>
inline void ApplyActivation(T* absl_nullable output_data,
                            const int32_t* absl_nullable output_dims, int rank,
                            tflite::ActivationFunctionType faf) {
  static_assert(
      std::is_floating_point_v<T>,
      "ApplyActivation operates in unquantized floating-point value space.");
  if (faf == tflite::ActivationFunctionType_NONE || rank < 0 ||
      (rank > 0 && output_dims == nullptr)) {
    return;
  }
  const auto dims = rank == 0 ? absl::Span<const int32_t>{}
                              : absl::MakeConstSpan(output_dims, rank);
  Expected<size_t> num_elements = GetNumElements(dims);
  if (!num_elements) {
    return;
  }
  ApplyActivation(output_data, *num_elements, faf);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_REFERENCE_NUMERIC_UTILS_H_
