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
#include <vector>

namespace litert::arm_vulkan_ml {
namespace {
constexpr int64_t kInt32Min = std::numeric_limits<int32_t>::min();
constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
bool IsInt8(int64_t value) { return value >= -128 && value <= 127; }

// Computes floor((value * multiplier + 2^(shift - 1)) / 2^shift) for
// `shift` in [1, 62]. C++17 signed right shift is implementation-defined, so
// express floor division explicitly (including negative half-way ties toward
// +infinity).
int64_t ScaleAndRoundShift(int32_t value, int32_t multiplier, int shift) {
  const int64_t divisor = int64_t{1} << shift;
  const int64_t numerator = int64_t{value} * multiplier + divisor / 2;
  int64_t result = numerator / divisor;
  if (numerator < 0 && numerator % divisor != 0) --result;
  return result;
}
}  // namespace

std::optional<Scale32> MakeScale32(double scale) {
  if (!std::isfinite(scale) || scale <= 0) return std::nullopt;
  int exponent = 0;
  std::frexp(scale, &exponent);
  int shift = std::min(62, 31 - exponent);
  if (shift < 2) return std::nullopt;
  int64_t multiplier = std::llround(std::ldexp(scale, shift));
  if (multiplier > kInt32Max) {
    --shift;
    multiplier = std::llround(std::ldexp(scale, shift));
  }
  if (shift < 2 || multiplier <= 0 || multiplier > kInt32Max) {
    return std::nullopt;
  }
  return Scale32{static_cast<int32_t>(multiplier), static_cast<int8_t>(shift)};
}

bool IsScale32RangeValid(Scale32 scale, int64_t minimum, int64_t maximum) {
  if (scale.multiplier < 0 || scale.shift < 2 || scale.shift > 62 ||
      minimum > maximum || minimum < kInt32Min || maximum > kInt32Max) {
    return false;
  }
  const int64_t bound = int64_t{1} << (scale.shift - 1);
  return minimum >= -bound && maximum < bound;
}

std::optional<int32_t> ApplyScale32(int32_t value, Scale32 scale) {
  if (!IsScale32RangeValid(scale, value, value)) return std::nullopt;
  const int64_t result =
      ScaleAndRoundShift(value, scale.multiplier, scale.shift);
  if (result < kInt32Min || result > kInt32Max) return std::nullopt;
  return static_cast<int32_t>(result);
}

std::optional<Int8Range> ActivationRange(Activation activation, double scale,
                                         int64_t zero_point) {
  if (!std::isfinite(scale) || scale <= 0 || !IsInt8(zero_point)) {
    return std::nullopt;
  }
  Int8Range range{-128, 127};
  switch (activation) {
    case Activation::kNone:
      break;
    case Activation::kRelu:
      range.minimum = zero_point;
      break;
    case Activation::kRelu6:
      range.minimum = zero_point;
      // Clamp in floating point before conversion to avoid overflow for tiny
      // positive scales. LiteRT quantization uses round-away-from-zero here.
      range.maximum = static_cast<int32_t>(
          std::min(127.0, std::round(6.0 / scale) + zero_point));
      break;
    default:
      return std::nullopt;
  }
  return range;
}

std::optional<Int8Range> ActivationRangeInt16(Activation activation,
                                              double scale,
                                              int64_t zero_point) {
  if (!std::isfinite(scale) || scale <= 0 || zero_point != 0) {
    return std::nullopt;
  }
  Int8Range range{-32768, 32767};
  switch (activation) {
    case Activation::kNone:
      break;
    case Activation::kRelu:
      range.minimum = 0;
      break;
    case Activation::kRelu6:
      range.minimum = 0;
      range.maximum =
          static_cast<int32_t>(std::min(32767.0, std::round(6.0 / scale)));
      break;
    default:
      return std::nullopt;
  }
  return range;
}

bool CanEmitNativeRescale(Scale32 s, int64_t minimum, int64_t maximum,
                          bool is_int48) {
  if (!is_int48 && !IsScale32RangeValid(s, minimum, maximum)) return false;
  double scale = std::ldexp(static_cast<double>(s.multiplier), -s.shift);
  if (!(scale > 0) || !std::isfinite(scale) || minimum > maximum) {
    return false;
  }
  int left = 0;
  while (std::llround(scale * 4.0) > INT16_MAX) {
    scale *= .5;
    if (++left > 31) return false;
  }
  while (left > 0) {
    const int step = std::min(12, left);
    const int64_t factor = int64_t{1} << step;
    if (minimum < INT32_MIN / factor || maximum > INT32_MAX / factor) {
      return false;
    }
    minimum *= factor;
    maximum *= factor;
    left -= step;
  }
  int exponent = 0;
  std::frexp(scale, &exponent);
  int shift = std::min(62, 15 - exponent);
  if (shift < 2) return false;
  int64_t multiplier = std::llround(std::ldexp(scale, shift));
  if (multiplier > INT16_MAX) {
    --shift;
    multiplier = std::llround(std::ldexp(scale, shift));
  }
  return shift >= 2 && multiplier > 0 && multiplier <= INT16_MAX;
}

std::optional<AccumulatorRange> DotProductRange(const int8_t* weights,
                                                int64_t count,
                                                int32_t input_zero_point,
                                                int32_t weight_zero_point,
                                                int32_t bias) {
  if (count < 0 || (count != 0 && weights == nullptr) ||
      !IsInt8(input_zero_point) || !IsInt8(weight_zero_point)) {
    return std::nullopt;
  }
  int64_t minimum = 0;
  int64_t maximum = 0;
  for (int64_t i = 0; i < count; ++i) {
    const int64_t weight = int32_t{weights[i]} - weight_zero_point;
    const int64_t min_input_product = (-128 - input_zero_point) * weight;
    const int64_t max_input_product = (127 - input_zero_point) * weight;
    minimum += std::min(min_input_product, max_input_product);
    maximum += std::max(min_input_product, max_input_product);
    // Bounds widen monotonically, so returning early is safe and keeps the
    // analysis itself from overflowing even for invalid model dimensions.
    if (minimum < kInt32Min || maximum > kInt32Max) return std::nullopt;
  }
  // TOSA accumulates the products before adding bias. Prove both stages;
  // cancellation by a large bias cannot make overflowing products legal.
  minimum += bias;
  maximum += bias;
  if (minimum < kInt32Min || maximum > kInt32Max) return std::nullopt;
  return AccumulatorRange{static_cast<int32_t>(minimum),
                          static_cast<int32_t>(maximum)};
}
std::optional<std::vector<int8_t>> RequantizeTable(double input_scale,
                                                   int32_t input_zero_point,
                                                   double output_scale,
                                                   int32_t output_zero_point) {
  if (!std::isfinite(input_scale) || !std::isfinite(output_scale) ||
      input_scale <= 0 || output_scale <= 0 || !IsInt8(input_zero_point) ||
      !IsInt8(output_zero_point)) {
    return std::nullopt;
  }
  const std::optional<Scale32> scale = MakeScale32(input_scale / output_scale);
  if (!scale || !IsScale32RangeValid(*scale, -128 - input_zero_point,
                                     127 - input_zero_point)) {
    return std::nullopt;
  }
  std::vector<int8_t> table;
  table.reserve(256);
  for (int code = -128; code <= 127; ++code) {
    const std::optional<int32_t> scaled =
        ApplyScale32(code - input_zero_point, *scale);
    if (!scaled) return std::nullopt;
    table.push_back(static_cast<int8_t>(std::clamp(
        int64_t{*scaled} + output_zero_point, int64_t{-128}, int64_t{127})));
  }
  return table;
}

std::optional<std::vector<int8_t>> HardSwishTable(float input_scale,
                                                  int32_t input_zero_point,
                                                  float output_scale,
                                                  int32_t output_zero_point) {
  if (!std::isfinite(input_scale) || !std::isfinite(output_scale) ||
      input_scale <= 0 || output_scale <= 0 || !IsInt8(input_zero_point) ||
      !IsInt8(output_zero_point)) {
    return std::nullopt;
  }
  const float hires = input_scale / 128.0f;
  std::optional<Scale32> output =
      MakeScale32(static_cast<float>(hires / output_scale));
  std::optional<Scale32> relu =
      MakeScale32(static_cast<float>(hires / (3.0f / 32768.0f)));
  if (!output || !relu || output->shift < 31) return std::nullopt;
  auto multiplier16 = [](int32_t m) {
    return static_cast<int32_t>(
        std::min(int64_t{32767}, (int64_t{m} + 32768) / 65536));
  };
  auto high = [](int32_t a, int32_t b) {
    const int64_t product = int64_t{a} * b;
    const int64_t nudge = product >= 0 ? 16384 : 1 - 16384;
    return static_cast<int32_t>(
        std::min(int64_t{32767}, (product + nudge) / 32768));
  };
  auto left = [](int32_t value, int shift) {
    if (!value) return 0;
    if (shift >= 16) return value > 0 ? 32767 : -32768;
    return static_cast<int32_t>(
        std::clamp(int64_t{value} * (int64_t{1} << shift), int64_t{-32768},
                   int64_t{32767}));
  };
  auto divide = [](int32_t value, int shift) {
    if (!shift) return value;
    const int64_t magnitude =
        (std::abs(int64_t{value}) + (int64_t{1} << (shift - 1))) /
        (int64_t{1} << shift);
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
    const int32_t result =
        divide(product, output->shift - 31) + output_zero_point;
    table.push_back(std::clamp(result, -128, 127));
  }
  return table;
}

namespace {

// `:quantization` intentionally has zero external dependencies in
// `compiler/BUILD` so `:compiler_plugin` (`-Os`, `ungrte = True`) stays
// minimal. The fixed-point helpers below mirror
// `tflite::GetInvSqrtQuantizedMultiplierExp` and `tflite::QuantizeMultiplier`
// without pulling TFLite headers into the compiler plugin.

int32_t SaturatingRoundingDoublingHighMul32(int32_t a, int32_t b) {
  if (a == kInt32Min && b == kInt32Min) return static_cast<int32_t>(kInt32Max);
  const int64_t ab = int64_t{a} * b;
  const int64_t nudge = ab >= 0 ? (int64_t{1} << 30) : (1 - (int64_t{1} << 30));
  int64_t quotient = (ab + nudge) / (int64_t{1} << 31);
  if (ab + nudge < 0 && (ab + nudge) % (int64_t{1} << 31) != 0) {
    --quotient;
  }
  return static_cast<int32_t>(quotient);
}

int32_t MultiplyByQuantizedMultiplier32(int32_t x, int32_t multiplier,
                                        int shift) {
  const int64_t total_shift = 31 - shift;
  if (total_shift >= 63) return 0;
  if (total_shift <= 0) {
    const int64_t prod = int64_t{x} * multiplier;
    if (prod == 0) return 0;
    const int64_t left_shift = -total_shift;
    if (left_shift >= 31) {
      return static_cast<int32_t>(prod > 0 ? kInt32Max : kInt32Min);
    }
    const int64_t shifted = prod * (int64_t{1} << left_shift);
    return static_cast<int32_t>(std::clamp(shifted, kInt32Min, kInt32Max));
  }
  const int64_t result =
      ScaleAndRoundShift(x, multiplier, static_cast<int>(total_shift));
  return static_cast<int32_t>(std::clamp(result, kInt32Min, kInt32Max));
}

bool DecomposeQuantizedMultiplier(double scale, int32_t* out_mult,
                                  int* out_shift) {
  if (!std::isfinite(scale) || scale <= 0) return false;
  if (auto s = MakeScale32(scale)) {
    const int shift = 31 - s->shift;
    if (shift >= -31) {
      *out_mult = s->multiplier;
      *out_shift = shift;
      return true;
    }
  }
  int shift = 0;
  const double frac = std::frexp(scale, &shift);
  int64_t mult64 = std::llround(frac * (int64_t{1} << 31));
  if (mult64 == (int64_t{1} << 31)) {
    mult64 /= 2;
    ++shift;
  }
  if (shift < -31) {
    *out_mult = 0;
    *out_shift = 0;
    return true;
  }
  if (shift > 30) return false;
  *out_mult = static_cast<int32_t>(mult64);
  *out_shift = shift;
  return true;
}

void GetInvSqrtMultiplier(int32_t input, int32_t* out_mult, int* out_shift) {
  if (input <= 1) {
    *out_mult = static_cast<int32_t>(kInt32Max);
    *out_shift = 0;
    return;
  }
  int shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++shift;
  }
  int lz = 0;
  uint32_t v = static_cast<uint32_t>(input);
  while ((v & 0x80000000u) == 0u) {
    v <<= 1;
    ++lz;
  }
  const int left_shift_pairs = ((lz - 1) / 2) - 1;
  shift -= left_shift_pairs;
  input <<= 2 * left_shift_pairs;
  const int32_t fixed_input = input / 2;
  const int32_t half_input = (fixed_input + 1) / 2;
  const int32_t half_three = (1 << 28) + (1 << 27);
  int32_t x = 1 << 28;
  auto rescale_f3 = [](int32_t a, int32_t b) {
    // Multiply two F3 fixed-point values and rescale back to F3
    // (`Rescale<3>(a * b)` in gemmlowp), safe from overflow for x in
    // [0.5, 1.5].
    const int64_t prod = SaturatingRoundingDoublingHighMul32(a, b);
    return static_cast<int32_t>(
        std::clamp<int64_t>(prod * 8, kInt32Min, kInt32Max));
  };
  for (int i = 0; i < 5; ++i) {
    const int32_t x2 = rescale_f3(x, x);
    const int32_t x3 = rescale_f3(x2, x);
    const int32_t hx3 = rescale_f3(half_input, x3);
    const int32_t t1 = rescale_f3(half_three, x);
    x = t1 - hx3;
  }
  // 1518500250 == round((sqrt(2) / 2) * 2^31) in F0.
  x = SaturatingRoundingDoublingHighMul32(x, 1518500250);
  if (shift < 0) {
    x <<= -shift;
    shift = 0;
  }
  // Convert right shift to left shift (kReverseShift = -1).
  *out_mult = x;
  *out_shift = -shift;
}

}  // namespace

std::optional<std::vector<int8_t>> RsqrtTable(float input_scale,
                                              int32_t input_zero_point,
                                              float output_scale,
                                              int32_t output_zero_point) {
  if (!std::isfinite(input_scale) || !std::isfinite(output_scale) ||
      input_scale <= 0 || output_scale <= 0 || !IsInt8(input_zero_point) ||
      !IsInt8(output_zero_point)) {
    return std::nullopt;
  }
  const double scale =
      1.0 / (std::sqrt(static_cast<double>(input_scale)) * output_scale);
  int32_t out_mult = 0;
  int out_shift = 0;
  if (!DecomposeQuantizedMultiplier(scale, &out_mult, &out_shift)) {
    return std::nullopt;
  }

  std::vector<int8_t> table;
  table.reserve(256);
  constexpr int32_t kShift = 20;
  for (int code = -128; code <= 127; ++code) {
    const int32_t value = code - input_zero_point;
    if (value <= 0) {
      table.push_back(127);
      continue;
    }
    int32_t inv_sqrt_mult = 0;
    int inv_sqrt_shift = 0;
    GetInvSqrtMultiplier(value, &inv_sqrt_mult, &inv_sqrt_shift);
    const int32_t data = MultiplyByQuantizedMultiplier32(
        1, inv_sqrt_mult, inv_sqrt_shift + kShift);
    const int32_t result =
        MultiplyByQuantizedMultiplier32(data, out_mult, out_shift - kShift) +
        output_zero_point;
    table.push_back(static_cast<int8_t>(std::clamp(result, -128, 127)));
  }
  return table;
}

}  // namespace litert::arm_vulkan_ml
