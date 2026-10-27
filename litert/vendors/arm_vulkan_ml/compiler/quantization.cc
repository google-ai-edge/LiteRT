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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace litert::arm_vulkan_ml {
namespace {
constexpr int64_t kMin = std::numeric_limits<int32_t>::min();
constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
bool IsInt8(int64_t value) { return value >= -128 && value <= 127; }
}  // namespace

std::optional<Scale32> MakeScale32(double scale) {
  if (!std::isfinite(scale) || scale <= 0) return std::nullopt;
  int exponent;
  std::frexp(scale, &exponent);
  int shift = std::min(62, 31 - exponent);
  if (shift < 2) return std::nullopt;
  int64_t multiplier = std::llround(std::ldexp(scale, shift));
  if (multiplier > kMax) {
    --shift;
    multiplier = std::llround(std::ldexp(scale, shift));
  }
  if (shift < 2 || multiplier <= 0 || multiplier > kMax) return std::nullopt;
  return Scale32{static_cast<int32_t>(multiplier), static_cast<int8_t>(shift)};
}

bool IsScale32RangeValid(Scale32 scale, int64_t minimum, int64_t maximum) {
  if (scale.multiplier < 0 || scale.shift < 2 || scale.shift > 62 ||
      minimum > maximum || minimum < kMin || maximum > kMax) return false;
  const int64_t bound = int64_t{1} << (scale.shift - 1);
  return minimum >= -bound && maximum < bound;
}

std::optional<int32_t> ApplyScale32(int32_t value, Scale32 scale) {
  if (!IsScale32RangeValid(scale, value, value)) return std::nullopt;
  const int64_t divisor = int64_t{1} << scale.shift;
  const int64_t numerator = int64_t{value} * scale.multiplier + divisor / 2;
  // C++17 signed right shift is implementation-defined; express floor division
  // explicitly, including negative half-way cases (ties toward +infinity).
  int64_t result = numerator / divisor;
  if (numerator < 0 && numerator % divisor != 0) --result;
  if (result < kMin || result > kMax) return std::nullopt;
  return static_cast<int32_t>(result);
}

std::optional<Int8Range> ActivationRange(Activation activation, double scale,
                                        int64_t zero_point) {
  if (!std::isfinite(scale) || scale <= 0 || !IsInt8(zero_point))
    return std::nullopt;
  Int8Range range{-128, 127};
  switch (activation) {
    case Activation::kNone: break;
    case Activation::kRelu: range.minimum = zero_point; break;
    case Activation::kRelu6:
      range.minimum = zero_point;
      // Clamp in floating point before conversion to avoid overflow for tiny
      // positive scales. LiteRT quantization uses round-away-from-zero here.
      range.maximum = static_cast<int32_t>(
          std::min(127.0, std::round(6.0 / scale) + zero_point));
      break;
    default: return std::nullopt;
  }
  return range;
}

std::optional<AccumulatorRange> DotProductRange(
    const int8_t* weights, int64_t count, int32_t input_zero_point,
    int32_t weight_zero_point, int32_t bias) {
  if (count < 0 || (count != 0 && weights == nullptr) ||
      !IsInt8(input_zero_point) || !IsInt8(weight_zero_point))
    return std::nullopt;
  int64_t minimum = 0;
  int64_t maximum = 0;
  for (int64_t i = 0; i < count; ++i) {
    const int64_t weight = int32_t{weights[i]} - weight_zero_point;
    const int64_t a = (-128 - input_zero_point) * weight;
    const int64_t b = (127 - input_zero_point) * weight;
    minimum += std::min(a, b);
    maximum += std::max(a, b);
    // Bounds widen monotonically, so returning early is safe and keeps the
    // analysis itself from overflowing even for invalid model dimensions.
    if (minimum < kMin || maximum > kMax) return std::nullopt;
  }
  // TOSA accumulates the products before adding bias. Prove both stages;
  // cancellation by a large bias cannot make overflowing products legal.
  minimum += bias;
  maximum += bias;
  if (minimum < kMin || maximum > kMax) return std::nullopt;
  return AccumulatorRange{static_cast<int32_t>(minimum),
                          static_cast<int32_t>(maximum)};
}
std::optional<std::vector<int8_t>> RequantizeTable(float input_scale,
    int32_t input_zero_point, float output_scale, int32_t output_zero_point) {
  if (!std::isfinite(input_scale) || !std::isfinite(output_scale) ||
      input_scale <= 0 || output_scale <= 0 ||
      !IsInt8(input_zero_point) || !IsInt8(output_zero_point)) return std::nullopt;
  int exponent;
  const double fraction = std::frexp(double(input_scale) / output_scale, &exponent);
  int64_t multiplier = std::llround(fraction * (int64_t{1} << 31));
  if (multiplier == (int64_t{1} << 31)) { multiplier /= 2; ++exponent; }
  // LiteRT QuantizeMultiplier flushes shifts below -31 to zero.
  if (exponent < -31) { multiplier = 0; exponent = 0; }
  if (exponent > 23) return std::nullopt;
  const int left = std::max(exponent, 0), right = std::max(-exponent, 0);
  std::vector<int8_t> table;
  for (int code = -128; code <= 127; ++code) {
    int64_t shifted = int64_t(code - input_zero_point) * (int64_t{1} << left);
    if (shifted < kMin || shifted > kMax) return std::nullopt;
    int64_t product = shifted * multiplier;
    // SaturatingRoundingDoublingHighMul followed by RoundingDivideByPOT.
    int64_t value = (product + (product >= 0 ? (int64_t{1} << 30)
                                                         : 1 - (int64_t{1} << 30))) /
                    (int64_t{1} << 31);
    if (right) {
      int64_t magnitude = (std::abs(value) + (int64_t{1} << (right - 1))) /
                          (int64_t{1} << right);
      value = value < 0 ? -magnitude : magnitude;
    }
    table.push_back(std::clamp(value + output_zero_point, int64_t{-128}, int64_t{127}));
  }
  return table;
}

std::optional<std::vector<int8_t>> HardSwishTable(float input_scale,
    int32_t input_zero_point, float output_scale, int32_t output_zero_point) {
  if (!std::isfinite(input_scale) || !std::isfinite(output_scale) ||
      input_scale <= 0 || output_scale <= 0 ||
      !IsInt8(input_zero_point) || !IsInt8(output_zero_point)) return std::nullopt;
  const float hires = input_scale / 128.0f;
  auto output = MakeScale32(float(hires / output_scale));
  auto relu = MakeScale32(float(hires / (3.0f / 32768.0f)));
  if (!output || !relu || output->shift < 31) return std::nullopt;
  auto multiplier16 = [](int32_t m) {
    return static_cast<int32_t>(std::min(int64_t{32767}, (int64_t{m} + 32768) / 65536));
  };
  auto high = [](int32_t a, int32_t b) {
    const int64_t product = int64_t{a} * b;
    const int64_t nudge = product >= 0 ? 16384 : 1 - 16384;
    return static_cast<int32_t>(std::min(int64_t{32767}, (product + nudge) / 32768));
  };
  auto left = [](int32_t value, int shift) {
    if (!value) return 0;
    if (shift >= 16) return value > 0 ? 32767 : -32768;
    return static_cast<int32_t>(std::clamp(int64_t{value} * (int64_t{1} << shift), int64_t{-32768}, int64_t{32767}));
  };
  auto divide = [](int32_t value, int shift) {
    if (!shift) return value;
    const int64_t magnitude = (std::abs(int64_t{value}) + (int64_t{1} << (shift - 1))) / (int64_t{1} << shift);
    return static_cast<int32_t>(value < 0 ? -magnitude : magnitude);
  };
  const int exponent = 31 - relu->shift;
  std::vector<int8_t> table;
  for (int code = -128; code <= 127; ++code) {
    const int32_t input = (code - input_zero_point) * 128;
    const int32_t preshift = high(input, multiplier16(output->multiplier));
    int32_t gate = input;
    if (exponent > 0) gate = left(gate, exponent - 1);
    gate = high(gate, multiplier16(relu->multiplier));
    if (exponent > 0) gate = left(gate, 1);
    if (exponent < 0) gate = divide(gate, -exponent);
    gate = (gate + 32768) / 2;
    const int32_t product = (gate * preshift) / 32768;
    const int32_t result = divide(product, output->shift - 31) + output_zero_point;
    table.push_back(std::clamp(result, -128, 127));
  }
  return table;
}

}  // namespace litert::arm_vulkan_ml
