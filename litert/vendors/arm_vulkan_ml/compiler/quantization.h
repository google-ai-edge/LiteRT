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

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_QUANTIZATION_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_QUANTIZATION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace litert::arm_vulkan_ml {

// TOSA 1.0.1 scale32=true, rounding_mode=SINGLE_ROUND. Tiny nonzero
// multipliers are retained at shift 62 when representable. Unrepresentable
// scales are rejected; they are never silently flushed to zero.
struct Scale32 {
  int32_t multiplier;
  int8_t shift;
};
std::optional<Scale32> MakeScale32(double scale);

// Both bounds are inclusive. Rejects int32 overflow and the narrower TOSA
// apply_scale_32 input interval at shifts below 32.
bool IsScale32RangeValid(Scale32 scale, int64_t minimum, int64_t maximum);
std::optional<int32_t> ApplyScale32(int32_t value, Scale32 scale);

struct Int8Range {
  int32_t minimum;
  int32_t maximum;
};
enum class Activation { kNone, kRelu, kRelu6 };
std::optional<Int8Range> ActivationRange(Activation activation, double scale,
                                         int64_t zero_point);
std::optional<Int8Range> ActivationRangeInt16(Activation activation,
                                              double scale,
                                              int64_t zero_point);

// Compile-time 513-entry INT16 TOSA TABLE population matching LiteRT's
// midpoint-compensated LUTPopulateInt16, with slope validation.
template <typename FloatT, typename Func>
std::optional<std::vector<int16_t>> PopulateInt16Table(
    FloatT input_scale, int32_t input_zero_point, FloatT output_scale,
    int32_t output_zero_point, Func&& fn) {
  if (!std::isfinite(input_scale) || input_scale <= 0 ||
      !std::isfinite(output_scale) || output_scale <= 0) {
    return std::nullopt;
  }
  const FloatT input_min =
      input_scale * static_cast<FloatT>(-32768 - input_zero_point);
  const FloatT input_max =
      input_scale * static_cast<FloatT>(32767 - input_zero_point);
  const FloatT output_min =
      output_scale * static_cast<FloatT>(-32768 - output_zero_point);
  const FloatT output_max =
      output_scale * static_cast<FloatT>(32767 - output_zero_point);
  const FloatT step = (input_max - input_min) / static_cast<FloatT>(512);
  const FloatT half_step = step / static_cast<FloatT>(2);
  const FloatT output_scaling_inv =
      static_cast<FloatT>(65536) / (output_max - output_min);
  if (!std::isfinite(step) || !std::isfinite(output_scaling_inv) ||
      output_scaling_inv <= 0) {
    return std::nullopt;
  }
  std::vector<int16_t> table(513);
  for (int i = 0; i < 512; ++i) {
    const FloatT val = fn(input_min + static_cast<FloatT>(i) * step);
    const FloatT val_mid =
        fn(input_min + static_cast<FloatT>(i) * step + half_step);
    const FloatT val_next = fn(input_min + static_cast<FloatT>(i + 1) * step);
    if (!std::isfinite(val) || !std::isfinite(val_mid) ||
        !std::isfinite(val_next)) {
      return std::nullopt;
    }
    const FloatT sample_val = std::round(val * output_scaling_inv);
    const FloatT midpoint_interp_val = std::round(
        (val_next * output_scaling_inv + sample_val) / static_cast<FloatT>(2));
    const FloatT midpoint_val = std::round(val_mid * output_scaling_inv);
    const FloatT bias = std::round((midpoint_interp_val - midpoint_val) /
                                   static_cast<FloatT>(2));
    if (!std::isfinite(sample_val) || !std::isfinite(bias)) {
      return std::nullopt;
    }
    table[i] = static_cast<int16_t>(std::clamp<double>(
        static_cast<double>(sample_val - bias), -32768.0, 32767.0));
  }
  const FloatT last = fn(input_max);
  if (!std::isfinite(last)) return std::nullopt;
  const FloatT last_scaled = std::round(last * output_scaling_inv);
  if (!std::isfinite(last_scaled)) return std::nullopt;
  table[512] = static_cast<int16_t>(
      std::clamp<double>(static_cast<double>(last_scaled), -32768.0, 32767.0));
  for (int i = 0; i < 512; ++i) {
    if (std::abs(int32_t{table[i + 1]} - int32_t{table[i]}) > 32767) {
      return std::nullopt;
    }
  }
  return table;
}

// Worst-case dot-product bounds, including actual bias and weights, for an
// arbitrary int8 activation tensor. This avoids assuming observed calibration
// ranges are runtime guarantees. A null return means int32 may overflow.
struct AccumulatorRange {
  int32_t minimum;
  int32_t maximum;
};
std::optional<AccumulatorRange> DotProductRange(const int8_t* weights,
                                                int64_t count,
                                                int32_t input_zero_point,
                                                int32_t weight_zero_point,
                                                int32_t bias);

// Compile-time int8 requantization table matching LiteRT double rounding.
// Rejects scales whose intermediate left shift cannot be proven safe.
std::optional<std::vector<int8_t>> RequantizeTable(float input_scale,
                                                   int32_t input_zero_point,
                                                   float output_scale,
                                                   int32_t output_zero_point);

// Exhaustive int8 lookup using LiteRT's documented 16-bit fixed-point
// hard-swish arithmetic, computed at compile time without a runtime kernel.
std::optional<std::vector<int8_t>> HardSwishTable(float input_scale,
                                                  int32_t input_zero_point,
                                                  float output_scale,
                                                  int32_t output_zero_point);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_QUANTIZATION_H_
