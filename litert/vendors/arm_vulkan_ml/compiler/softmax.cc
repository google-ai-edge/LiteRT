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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/emission.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"

namespace litert::arm_vulkan_ml {
namespace {
using ::tfsc::tosa::Attribute;
using ::tfsc::tosa::DataType;
using ::tfsc::tosa::ResId;
using ::tfsc::tosa::Tensor;
}  // namespace

const char* SoftmaxLutTypeName(SoftmaxLutType type) {
  switch (type) {
    case SoftmaxLutType::kAuto:
      return "auto";
    case SoftmaxLutType::kInt8:
      return "int8";
    case SoftmaxLutType::kInt16:
      return "int16";
    case SoftmaxLutType::kProFp:
      return "pro_fp";
  }
  return "invalid";
}

std::optional<SoftmaxLutType> ParseSoftmaxLutType(const std::string& name) {
  if (name == "auto") return SoftmaxLutType::kAuto;
  if (name == "int8") return SoftmaxLutType::kInt8;
  if (name == "int16") return SoftmaxLutType::kInt16;
  if (name == "pro_fp") return SoftmaxLutType::kProFp;
  return std::nullopt;
}

std::optional<ProFpSoftmaxPlan> AnalyzeProFpSoftmax(
    uint32_t channels, double input_scale, double output_scale,
    int32_t output_zero_point, double beta, DataType output_type,
    std::string& reason) {
  auto reject = [&](const char* message) -> std::optional<ProFpSoftmaxPlan> {
    reason = message;
    return std::nullopt;
  };
  if (!channels || channels > 8192) {
    return reject("Quantized SOFTMAX requires last-axis length 1..8192");
  }
  const double scaled_beta = input_scale * beta;
  if (!std::isfinite(input_scale) || input_scale <= 0 || !std::isfinite(beta) ||
      beta <= 0 || !std::isfinite(scaled_beta) || scaled_beta <= 0) {
    return reject("Quantized SOFTMAX requires finite positive scale and beta");
  }
  if (output_type == DataType::int16_t) {
    if (!std::isfinite(output_scale) ||
        std::abs(output_scale - (1.0 / 32768.0)) > 1e-6 ||
        output_zero_point != 0) {
      return reject(
          "INT16 SOFTMAX requires symmetric output scale=1/32768 and "
          "zero_point=0");
    }
  } else if (output_scale != 1.0 / 256 || output_zero_point != -128) {
    return reject(
        "INT8 SOFTMAX requires matching shapes, scale=1/256 and "
        "zero_point=-128 output");
  }
  reason.clear();
  return ProFpSoftmaxPlan{channels, input_scale, output_scale,
                          output_zero_point, beta};
}

ResId EmitProFpSoftmax(tfsc::tosa::Graph& graph, ResId input,
                       const std::vector<uint32_t>& shape, DataType output_type,
                       const ProFpSoftmaxPlan& plan,
                       GraphConstantPool& constants) {
  std::vector<uint32_t> reduced = shape;
  reduced.back() = 1;
  const Tensor fp(DataType::float32_t, shape);
  const Tensor row(DataType::float32_t, reduced);
  const Tensor out(output_type, shape);
  const ResId zero_shift = graph.AddTensorConstant(Attribute(int8_t{0}));
  auto scalar = [&](float value) {
    return BroadcastScalar(graph, value, shape.size(), constants);
  };
  auto mul = [&](ResId a, ResId b, const Tensor& type) {
    return graph.AddMulOperator(a, b, zero_shift, type);
  };
  // Subtracting max in FP32 of exact integer inputs cancels input zero_point
  // and keeps exponents <= 0 before multiplying by (input_scale * beta).
  ResId q_f = graph.AddCastOperator(input, fp);
  const Attribute axis(static_cast<int32_t>(shape.size() - 1));
  ResId max_q =
      graph.AddReduceMaxOperator(q_f, axis, Attribute(uint32_t{0}), row);
  ResId diff = graph.AddSubOperator(q_f, max_q, fp);
  const float combined = static_cast<float>(plan.input_scale * plan.beta);
  if (combined > 0.0f && std::isfinite(combined)) {
    if (combined != 1.0f) {
      diff = mul(diff, scalar(combined), fp);
    }
  } else {
    diff = mul(mul(diff, scalar(static_cast<float>(plan.input_scale)), fp),
               scalar(static_cast<float>(plan.beta)), fp);
  }
  ResId exp_val = graph.AddExpOperator(diff, fp);
  ResId sum = graph.AddReduceSumOperator(exp_val, axis, row);
  ResId recip = graph.AddReciprocalOperator(sum, row);
  ResId prob = mul(exp_val, recip, fp);
  ResId scaled =
      mul(prob, scalar(static_cast<float>(1.0 / plan.output_scale)), fp);
  // Probabilities are nonnegative and output_scale > 0, so floor(x + 0.5)
  // matches round-to-nearest-away-from-zero.
  ResId rounded = graph.AddFloorOperator(
      graph.AddAddOperator(scaled, scalar(0.5f), fp), fp);
  if (plan.output_zero_point != 0) {
    rounded = graph.AddAddOperator(
        rounded, scalar(static_cast<float>(plan.output_zero_point)), fp);
  }
  const float lo = output_type == DataType::int16_t ? -32768.0f : -128.0f;
  const float hi = output_type == DataType::int16_t ? 32767.0f : 127.0f;
  rounded = graph.AddClampOperator(rounded, Attribute(lo), Attribute(hi),
                                   Attribute(int32_t{1}), fp);
  return graph.AddCastOperator(rounded, out);
}

std::optional<QuantizedSoftmaxPlan> AnalyzeQuantizedSoftmax(
    uint32_t channels, double input_scale, double beta, SoftmaxLutType type,
    std::string& reason) {
  auto reject =
      [&](const char* message) -> std::optional<QuantizedSoftmaxPlan> {
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
  if (type != SoftmaxLutType::kAuto && type != SoftmaxLutType::kInt8 &&
      type != SoftmaxLutType::kInt16) {
    return reject("Invalid SOFTMAX lookup-table type");
  }
  QuantizedSoftmaxPlan plan;
  plan.channels = channels;
  plan.beta = beta;
  // Compilation has no device-specific extension information. Auto chooses
  // the PRO-INT route. Forced INT16 requires EXT-INT16 at dispatch acceptance.
  plan.lut_type = type == SoftmaxLutType::kAuto ? SoftmaxLutType::kInt8 : type;
  for (int i = 0; i < 256; ++i) {
    const double value =
        std::exp(static_cast<double>(i - 255) * scale) * 2147483648.0;
    plan.exponential[i] = static_cast<int32_t>(
        std::min(std::floor(value + .5), static_cast<double>(INT32_MAX)));
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

ResId EmitQuantizedSoftmax(tfsc::tosa::Graph& graph, ResId input,
                           const std::vector<uint32_t>& shape,
                           const QuantizedSoftmaxPlan& plan) {
  std::vector<uint32_t> reduced = shape;
  reduced.back() = 1;
  Tensor wide(DataType::int32_t, shape), row(DataType::int32_t, reduced);
  Tensor bytes(DataType::int8_t, shape);
  const std::vector<uint32_t> scalar_shape(shape.size(), 1);
  auto constant = [&](int32_t value) {
    return ReshapeTensor(graph, graph.AddTensorConstant(Attribute(value)),
                         graph.AddTensorConstant(Attribute(scalar_shape)),
                         Tensor(DataType::int32_t, scalar_shape));
  };
  auto shift = [&](ResId value, int32_t bits, const Tensor& result,
                   bool round) {
    return graph.AddArithmeticRightShiftOperator(value, constant(bits),
                                                 Attribute(round), result);
  };
  ResId input_int32 = graph.AddCastOperator(input, wide);
  const Attribute axis(static_cast<int32_t>(shape.size() - 1));
  ResId maximum = graph.AddReduceMaxOperator(input_int32, axis,
                                             Attribute(uint32_t{1}), row);
  ResId difference = graph.AddSubOperator(input_int32, maximum, wide);
  ResId exponential = nullptr;
  if (plan.lut_type == SoftmaxLutType::kInt8) {
    // d=-255..0 maps bijectively onto the INT8 TABLE index domain.
    ResId index = graph.AddCastOperator(
        graph.AddAddOperator(difference, constant(127), wide), bytes);
    for (int lane = 0; lane < 4; ++lane) {
      std::vector<int8_t> table(256);
      const int bits = 24 - lane * 8;
      for (size_t i = 0; i < table.size(); ++i) {
        const uint32_t byte =
            (static_cast<uint32_t>(plan.exponential[i]) >> bits) & 255;
        table[i] = static_cast<int8_t>(static_cast<int32_t>(byte) - 128);
      }
      ResId lut = graph.AddTensorConstant(Attribute(table));
      ResId value = graph.AddAddOperator(
          graph.AddCastOperator(graph.AddTableOperator(index, lut, bytes),
                                wide),
          constant(128), wide);
      if (bits) {
        value = graph.AddLogicalLeftShiftOperator(value, constant(bits), wide);
      }
      exponential =
          exponential ? graph.AddAddOperator(exponential, value, wide) : value;
    }
  } else {
    // INT16 TABLE uses a Q9.7 address. The low seven bits are zero, so it
    // returns entry*128 exactly, without interpolation. Chunks of <=11 bits
    // also satisfy the signed INT16 adjacent-entry slope restriction.
    ResId address = graph.AddMulOperator(
        difference, constant(128),
        graph.AddTensorConstant(Attribute(int8_t{0})), wide);
    ResId index =
        graph.AddCastOperator(address, Tensor(DataType::int16_t, shape));
    constexpr std::array<int, 3> kOffsets = {21, 10, 0};
    constexpr std::array<uint32_t, 3> kMasks = {0x7ff, 0x7ff, 0x3ff};
    for (int lane = 0; lane < 3; ++lane) {
      std::vector<int16_t> table(513);
      for (int i = 0; i < 513; ++i) {
        const int32_t value = plan.exponential[std::clamp(i - 1, 0, 255)];
        table[i] = static_cast<int16_t>(
            (static_cast<uint32_t>(value) >> kOffsets[lane]) & kMasks[lane]);
      }
      ResId value = graph.AddTableOperator(
          index, graph.AddTensorConstant(Attribute(table)), wide);
      if (lane < 2) {
        value = graph.AddLogicalLeftShiftOperator(
            value, constant(kOffsets[lane] - 7), wide);
      } else {
        value = shift(value, 7, wide, false);
      }
      exponential =
          exponential ? graph.AddAddOperator(exponential, value, wide) : value;
    }
  }
  ResId numerator = shift(exponential, plan.normalization_shift, wide, true);
  ResId sum = graph.AddReduceSumOperator(numerator, axis, row);
  ResId scaled =
      graph.AddLogicalLeftShiftOperator(numerator, constant(8), wide);
  ResId rounded = graph.AddAddOperator(scaled, shift(sum, 1, row, false), wide);
  ResId probability = graph.AddIntDivOperator(rounded, sum, wide);
  // Numerator and denominator are positive. Only the single-winner 256 case
  // needs saturation; no lower clamp is needed.
  probability = graph.AddMinimumOperator(probability, constant(255),
                                         Attribute(uint32_t{1}), wide);
  return graph.AddCastOperator(
      graph.AddSubOperator(probability, constant(128), wide), bytes);
}

std::optional<Int16SoftmaxPlan> AnalyzeInt16Softmax(uint32_t channels,
                                                    double input_scale,
                                                    double output_scale,
                                                    double beta,
                                                    std::string& reason) {
  auto reject = [&](const char* message) -> std::optional<Int16SoftmaxPlan> {
    reason = message;
    return std::nullopt;
  };
  if (!channels || channels > 8192) {
    return reject("INT16 SOFTMAX requires last-axis length 1..8192");
  }
  const double scaled_beta = input_scale * beta;
  if (!std::isfinite(input_scale) || input_scale <= 0 || !std::isfinite(beta) ||
      beta <= 0 || !std::isfinite(scaled_beta) || scaled_beta <= 0) {
    return reject("INT16 SOFTMAX requires finite positive scale and beta");
  }
  if (!std::isfinite(output_scale) ||
      std::abs(output_scale - (1.0 / 32768.0)) > 1e-6) {
    return reject(
        "INT16 SOFTMAX requires symmetric output scale=1/32768 and "
        "zero_point=0");
  }
  const double eff_diff_scale =
      std::min(scaled_beta * (65535.0 / 10.0), 65536.0);
  std::optional<Scale32> diff_scale = MakeScale32(eff_diff_scale);
  std::optional<Scale32> out_scale =
      MakeScale32(1.0 / (output_scale * 32768.0));
  if (!diff_scale || !out_scale) {
    return reject("INT16 SOFTMAX scale is not representable");
  }
  Int16SoftmaxPlan plan;
  plan.channels = channels;
  plan.beta = beta;
  plan.diff_scale = *diff_scale;
  plan.output_scale = *out_scale;
  if (eff_diff_scale > 1.0) {
    plan.clamp_diff = true;
    plan.diff_min = std::max<int32_t>(
        -65535,
        static_cast<int32_t>(std::floor(-65536.0 / eff_diff_scale)) - 1);
  }
  if (!IsScale32RangeValid(plan.diff_scale, plan.diff_min, 0) ||
      !IsScale32RangeValid(plan.output_scale, 0, 65535)) {
    return reject("INT16 SOFTMAX rescale range is not proven");
  }
  std::optional<std::vector<int16_t>> exp_lut =
      PopulateInt16Table<double>(10.0 / 65535.0, 32767, 2.0 / 65535.0, 0,
                                 [](double value) { return std::exp(value); });
  std::optional<std::vector<int16_t>> recip_lut = PopulateInt16Table<double>(
      1.0 / 65535.0, -32768, 2.0 / 65535.0, 0,
      [](double value) { return 1.0 / (1.0 + value); });
  if (!exp_lut || !recip_lut) {
    return reject("INT16 SOFTMAX lookup table construction failed");
  }
  absl::c_copy(*exp_lut, plan.exp_lut.begin());
  absl::c_copy(*recip_lut, plan.one_over_one_plus_x_lut.begin());
  reason.clear();
  return plan;
}

ResId EmitInt16Softmax(tfsc::tosa::Graph& graph, ResId input,
                       const std::vector<uint32_t>& shape,
                       const Int16SoftmaxPlan& plan,
                       GraphConstantPool& constants) {
  std::vector<uint32_t> reduced = shape;
  reduced.back() = 1;
  Tensor wide(DataType::int32_t, shape), row(DataType::int32_t, reduced);
  const std::vector<uint32_t> scalar_shape(shape.size(), 1);
  auto constant = [&](int32_t value) {
    return ReshapeTensor(graph, graph.AddTensorConstant(Attribute(value)),
                         graph.AddTensorConstant(Attribute(scalar_shape)),
                         Tensor(DataType::int32_t, scalar_shape));
  };
  auto shift = [&](ResId value, int32_t bits, const Tensor& result,
                   bool round) {
    return graph.AddArithmeticRightShiftOperator(value, constant(bits),
                                                 Attribute(round), result);
  };
  auto lut_constant = [&](const std::array<int16_t, 513>& lut) {
    const auto* raw = reinterpret_cast<const uint8_t*>(lut.data());
    std::vector<uint8_t> bytes(raw, raw + lut.size() * sizeof(int16_t));
    return AddGraphConstant(graph, Tensor(DataType::int16_t, {513}),
                            std::move(bytes), constants);
  };
  ResId input_int32 = graph.AddCastOperator(input, wide);
  const Attribute axis(static_cast<int32_t>(shape.size() - 1));
  ResId maximum = graph.AddReduceMaxOperator(input_int32, axis,
                                             Attribute(uint32_t{1}), row);
  ResId diff = graph.AddSubOperator(input_int32, maximum, wide);
  if (plan.clamp_diff) {
    diff = graph.AddMaximumOperator(diff, constant(plan.diff_min),
                                    Attribute(uint32_t{1}), wide);
  }
  ResId sdiff = EmitNativeRescale(
      graph, diff, DataType::int32_t, DataType::int32_t, shape, 0, 0,
      {plan.diff_scale}, constants, {plan.diff_min}, {0});
  ResId sym_diff = graph.AddAddOperator(sdiff, constant(32767), wide);
  // Unit rescale to INT16 saturates directly to the int16 output range.
  ResId idx0 = EmitNativeRescale(
      graph, sym_diff, DataType::int32_t, DataType::int16_t, shape, 0, 0,
      {*MakeScale32(1.0)}, constants, {-196608}, {32767});
  ResId exp_q15 =
      shift(graph.AddTableOperator(idx0, lut_constant(plan.exp_lut), wide), 7,
            wide, true);
  ResId sum = graph.AddReduceSumOperator(exp_q15, axis, row);
  ResId clz = graph.AddClzOperator(sum, row);
  ResId headroom = graph.AddSubOperator(clz, constant(1), row);
  ResId lls = graph.AddLogicalLeftShiftOperator(sum, headroom, row);
  ResId shifted_sum = shift(lls, 14, row, true);
  ResId sym_sum = graph.AddSubOperator(shifted_sum, constant(98304), row);
  ResId idx1 = EmitNativeRescale(
      graph, sym_sum, DataType::int32_t, DataType::int16_t, reduced, 0, 0,
      {*MakeScale32(1.0)}, constants, {-65536}, {65536});
  ResId recip_q15 =
      shift(graph.AddTableOperator(
                idx1, lut_constant(plan.one_over_one_plus_x_lut), row),
            7, row, true);
  ResId prod = graph.AddMulOperator(
      exp_q15, recip_q15, graph.AddTensorConstant(Attribute(int8_t{0})), wide);
  ResId rsh = graph.AddSubOperator(constant(31), clz, row);
  ResId prob_q15 =
      graph.AddArithmeticRightShiftOperator(prod, rsh, Attribute(true), wide);
  return EmitNativeRescale(graph, prob_q15, DataType::int32_t,
                           DataType::int16_t, shape, 0, 0, {plan.output_scale},
                           constants, {0}, {65535});
}

}  // namespace litert::arm_vulkan_ml
