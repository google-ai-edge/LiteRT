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

#include "absl/types/span.h"  // from @com_google_absl
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

template <typename T>
inline float DequantizeValue(T val, const AffineQuantSpec& spec) {
  return DequantizeValue<T>(val, spec.scale, spec.zero_point);
}

// Quantizes a real value using round-half-away-from-zero (`std::round`), then
// saturates to `[min<T>, max<T>]` BEFORE integer conversion so large finite
// inputs and infinities saturate without integer-conversion UB.
//
// Policy for degenerate inputs:
//   - If `scale` is non-finite or <= 0, or if `val` is NaN, returns
//     `zero_point` clamped to `[min<T>, max<T>]`.
template <typename T>
inline T QuantizeValueFromDouble(double val, float scale, int64_t zero_point) {
  static_assert(kIsSupportedNumericType<T>,
                "QuantizeValueFromDouble supports floating-point and <=32-bit "
                "integer types.");
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(val);
  } else {
    constexpr int64_t kLo = static_cast<int64_t>(std::numeric_limits<T>::min());
    constexpr int64_t kHi = static_cast<int64_t>(std::numeric_limits<T>::max());
    const int64_t clamped_zp = std::clamp(zero_point, kLo, kHi);
    if (!std::isfinite(scale) || scale <= 0.0f || std::isnan(val)) {
      return static_cast<T>(clamped_zp);
    }
    const double scaled = val / static_cast<double>(scale);
    const double rounded = std::round(scaled);

    // Every value in [kLo, kHi] (at most 32 bits) and every zero_point with
    // |zero_point| <= 2^50 is exact in IEEE-754 double, so subtracting
    // zero_point from the destination bounds in double is exact and bounds the
    // integer conversion strictly inside [kLo - zp, kHi - zp].
    constexpr int64_t kExactDoubleLimit = int64_t{1} << 50;
    if (zero_point >= -kExactDoubleLimit && zero_point <= kExactDoubleLimit) {
      const double lo_shifted =
          static_cast<double>(kLo) - static_cast<double>(zero_point);
      const double hi_shifted =
          static_cast<double>(kHi) - static_cast<double>(zero_point);
      if (rounded <= lo_shifted) {
        return static_cast<T>(kLo);
      }
      if (rounded >= hi_shifted) {
        return static_cast<T>(kHi);
      }
      const int64_t q = static_cast<int64_t>(rounded) + zero_point;
      return static_cast<T>(std::clamp(q, kLo, kHi));
    }

    const double sum = rounded + static_cast<double>(zero_point);
    if (sum <= static_cast<double>(kLo)) {
      return static_cast<T>(kLo);
    }
    if (sum >= static_cast<double>(kHi)) {
      return static_cast<T>(kHi);
    }
    return static_cast<T>(std::round(sum));
  }
}

template <typename T>
inline T QuantizeValueFromDouble(double val, const AffineQuantSpec& spec) {
  return QuantizeValueFromDouble<T>(val, spec.scale, spec.zero_point);
}

template <typename T>
inline T QuantizeValue(float val, float scale, int64_t zero_point) {
  return QuantizeValueFromDouble<T>(static_cast<double>(val), scale,
                                    zero_point);
}

template <typename T>
inline T QuantizeValue(float val, const AffineQuantSpec& spec) {
  return QuantizeValue<T>(val, spec.scale, spec.zero_point);
}

template <typename T>
inline void ApplyActivation(T* output_data, size_t num_elements,
                            tflite::ActivationFunctionType faf) {
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

template <typename T>
inline void ApplyActivation(T* output_data, const int32_t* output_dims,
                            int rank, tflite::ActivationFunctionType faf) {
  if (faf == tflite::ActivationFunctionType_NONE || rank < 0 ||
      (rank > 0 && output_dims == nullptr)) {
    return;
  }
  const auto dims = rank == 0 ? absl::Span<const int32_t>{}
                              : absl::MakeConstSpan(output_dims, rank);
  auto num_elements = GetNumElements(dims);
  if (!num_elements) {
    return;
  }
  ApplyActivation(output_data, *num_elements, faf);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_REFERENCE_NUMERIC_UTILS_H_
