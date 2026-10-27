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

// Worst-case dot-product bounds, including actual bias and weights, for an
// arbitrary int8 activation tensor. This avoids assuming observed calibration
// ranges are runtime guarantees. A null return means int32 may overflow.
struct AccumulatorRange {
  int32_t minimum;
  int32_t maximum;
};
std::optional<AccumulatorRange> DotProductRange(
    const int8_t* weights, int64_t count, int32_t input_zero_point,
    int32_t weight_zero_point, int32_t bias);

// Compile-time int8 requantization table matching LiteRT double rounding.
// Rejects scales whose intermediate left shift cannot be proven safe.
std::optional<std::vector<int8_t>> RequantizeTable(float input_scale,
    int32_t input_zero_point, float output_scale, int32_t output_zero_point);

// Exhaustive int8 lookup using LiteRT's documented 16-bit fixed-point
// hard-swish arithmetic, computed at compile time without a runtime kernel.
std::optional<std::vector<int8_t>> HardSwishTable(float input_scale,
    int32_t input_zero_point, float output_scale, int32_t output_zero_point);

}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_QUANTIZATION_H_
