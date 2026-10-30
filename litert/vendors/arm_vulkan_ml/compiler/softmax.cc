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

#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::ResId;
using Tensor = tfsc::tosa::Tensor;
}  // namespace

const char* SoftmaxLutTypeName(SoftmaxLutType type) {
  switch (type) {
    case kLrtArmVulkanMLSoftmaxLutAuto:
      return "auto";
    case kLrtArmVulkanMLSoftmaxLutInt8:
      return "int8";
    case kLrtArmVulkanMLSoftmaxLutInt16:
      return "int16";
  }
  return "invalid";
}

std::optional<SoftmaxLutType> ParseSoftmaxLutType(const std::string& name) {
  if (name == "auto") return kLrtArmVulkanMLSoftmaxLutAuto;
  if (name == "int8") return kLrtArmVulkanMLSoftmaxLutInt8;
  if (name == "int16") return kLrtArmVulkanMLSoftmaxLutInt16;
  return std::nullopt;
}

std::optional<QuantizedSoftmaxPlan> AnalyzeQuantizedSoftmax(
    uint32_t channels, double input_scale, double beta, SoftmaxLutType type,
    std::string& reason) {
  auto reject = [&](const char* message) -> std::optional<QuantizedSoftmaxPlan> {
    reason = message;
    return std::nullopt;
  };
  if (!channels || channels > 4095) {
    return reject("INT8 SOFTMAX requires last-axis length 1..4095");
  }
  const double scale = input_scale * beta;
  if (!std::isfinite(input_scale) || input_scale <= 0 || !std::isfinite(beta) ||
      beta <= 0 || !std::isfinite(scale) || scale <= 0) {
    return reject("INT8 SOFTMAX requires finite positive scale and beta");
  }
  if (!ParseSoftmaxLutType(SoftmaxLutTypeName(type))) {
    return reject("Invalid SOFTMAX lookup-table type");
  }
  QuantizedSoftmaxPlan plan;
  plan.channels = channels;
  plan.beta = beta;
  // Compilation has no device-specific extension information. Auto chooses
  // the PRO-INT route. Forced INT16 requires EXT-INT16 at dispatch acceptance.
  plan.lut_type = type == kLrtArmVulkanMLSoftmaxLutAuto
                      ? kLrtArmVulkanMLSoftmaxLutInt8 : type;
  for (int i = 0; i < 256; ++i) {
    const double value =
        std::exp(static_cast<double>(i - 255) * scale) * 2147483648.0;
    plan.exponential[i] = static_cast<int32_t>(std::min(
        std::floor(value + .5), static_cast<double>(INT32_MAX)));
  }
  // Adapt the normalization precision to the signed INT32 domain. Prove both
  // sum(r) and (r*256 + floor(sum(r)/2)) fit for every input row, including
  // all-equal maxima. XNNPACK's unsigned normalization cannot be copied with
  // its UINT32 bounds into TOSA's signed INTDIV. At least one r is rmax, so
  // the denominator is strictly positive. No observation/calibration is used.
  for (int shift = 9; shift <= 30; ++shift) {
    const int64_t rmax =
        (int64_t{INT32_MAX} + (int64_t{1} << (shift - 1))) >> shift;
    const int64_t sum = rmax * channels;
    if (sum <= INT32_MAX && rmax * 256 + sum / 2 <= INT32_MAX) {
      plan.normalization_shift = static_cast<int8_t>(shift);
      reason.clear();
      return plan;
    }
  }
  return reject("INT8 SOFTMAX accumulation range not proven");
}

ResId EmitQuantizedSoftmax(tfsc::tosa::Graph& g, ResId input,
                          const std::vector<uint32_t>& shape,
                          const QuantizedSoftmaxPlan& p) {
  auto reduced = shape;
  reduced.back() = 1;
  Tensor wide(DataType::int32_t, shape), row(DataType::int32_t, reduced);
  Tensor bytes(DataType::int8_t, shape);
  const std::vector<uint32_t> scalar_shape(shape.size(), 1);
  auto constant = [&](int32_t value) {
    return g.AddReshapeOperator(g.AddTensorConstant(Attribute(value)),
                               g.AddTensorConstant(Attribute(scalar_shape)),
                               Tensor(DataType::int32_t, scalar_shape));
  };
  auto shift = [&](ResId value, int32_t bits, const Tensor& result, bool round) {
    return g.AddArithmeticRightShiftOperator(value, constant(bits),
                                            Attribute(round), result);
  };
  auto x = g.AddCastOperator(input, wide);
  const Attribute axis(static_cast<int32_t>(shape.size() - 1));
  auto maximum = g.AddReduceMaxOperator(x, axis, Attribute(uint32_t{1}), row);
  auto difference = g.AddSubOperator(x, maximum, wide);
  ResId exponential = nullptr;
  if (p.lut_type == kLrtArmVulkanMLSoftmaxLutInt8) {
    // d=-255..0 maps bijectively onto the INT8 TABLE index domain.
    auto index = g.AddCastOperator(
        g.AddAddOperator(difference, constant(127), wide), bytes);
    for (int lane = 0; lane < 4; ++lane) {
      std::vector<int8_t> table(256);
      const int bits = 24 - lane * 8;
      for (size_t i = 0; i < table.size(); ++i) {
        const uint32_t byte =
            (static_cast<uint32_t>(p.exponential[i]) >> bits) & 255;
        table[i] = static_cast<int8_t>(static_cast<int32_t>(byte) - 128);
      }
      auto lut = g.AddTensorConstant(Attribute(table));
      auto value = g.AddAddOperator(
          g.AddCastOperator(g.AddTableOperator(index, lut, bytes), wide),
          constant(128), wide);
      if (bits) {
        value = g.AddLogicalLeftShiftOperator(value, constant(bits), wide);
      }
      exponential = exponential ? g.AddAddOperator(exponential, value, wide) : value;
    }
  } else {
    // INT16 TABLE uses a Q9.7 address. The low seven bits are zero, so it
    // returns entry*128 exactly, without interpolation. Chunks of <=11 bits
    // also satisfy the signed INT16 adjacent-entry slope restriction.
    auto address = g.AddMulOperator(
        difference, constant(128), g.AddTensorConstant(Attribute(int8_t{0})),
        wide);
    auto index = g.AddCastOperator(address, Tensor(DataType::int16_t, shape));
    const int offsets[] = {21, 10, 0};
    const uint32_t masks[] = {0x7ff, 0x7ff, 0x3ff};
    for (int lane = 0; lane < 3; ++lane) {
      std::vector<int16_t> table(513);
      for (int i = 0; i < 513; ++i) {
        const int32_t value = p.exponential[std::clamp(i - 1, 0, 255)];
        table[i] = static_cast<int16_t>(
            (static_cast<uint32_t>(value) >> offsets[lane]) & masks[lane]);
      }
      auto value = g.AddTableOperator(
          index, g.AddTensorConstant(Attribute(table)), wide);
      if (lane < 2) {
        value = g.AddLogicalLeftShiftOperator(
            value, constant(offsets[lane] - 7), wide);
      } else {
        value = shift(value, 7, wide, false);
      }
      exponential = exponential ? g.AddAddOperator(exponential, value, wide) : value;
    }
  }
  auto numerator = shift(exponential, p.normalization_shift, wide, true);
  auto sum = g.AddReduceSumOperator(numerator, axis, row);
  auto scaled = g.AddLogicalLeftShiftOperator(numerator, constant(8), wide);
  auto rounded = g.AddAddOperator(scaled, shift(sum, 1, row, false), wide);
  auto probability = g.AddIntDivOperator(rounded, sum, wide);
  // Numerator and denominator are positive. Only the single-winner 256 case
  // needs saturation; no lower clamp is needed.
  probability = g.AddMinimumOperator(probability, constant(255),
                                     Attribute(uint32_t{1}), wide);
  return g.AddCastOperator(
      g.AddSubOperator(probability, constant(128), wide), bytes);
}

}  // namespace litert::arm_vulkan_ml
