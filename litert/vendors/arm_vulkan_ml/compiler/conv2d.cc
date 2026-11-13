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

#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/OperatorEnum.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"

namespace litert::arm_vulkan_ml {
namespace {
using ::tfsc::tosa::Attribute;
using ::tfsc::tosa::DataType;
using ::tfsc::tosa::Tensor;

bool Elements(const std::array<uint32_t, 4>& shape, uint64_t& count) {
  count = 1;
  for (uint32_t dim : shape) {
    if (!dim || count > uint64_t{INT32_MAX} / dim) return false;
    count *= dim;
  }
  return true;
}

template <typename T>
tfsc::tosa::ResId Constant(tfsc::tosa::Graph& graph, const Tensor& tensor,
                           const std::vector<T>& data,
                           GraphConstantPool& constants) {
  GraphConstant constant{tensor, std::vector<uint8_t>(data.size() * sizeof(T))};
  std::memcpy(constant.bytes.data(), data.data(), constant.bytes.size());
  return AddGraphConstant(graph, tensor, std::move(constant.bytes), constants);
}
}  // namespace

tfsc::tosa::ResId AddGraphConstant(tfsc::tosa::Graph& graph,
                                   const tfsc::tosa::Tensor& tensor,
                                   std::vector<uint8_t> bytes,
                                   GraphConstantPool& constants) {
  if (constants.deduplicate) {
    for (const GraphConstant& constant : constants.values) {
      if (constant.id &&
          constant.tensor.GetDataType() == tensor.GetDataType() &&
          constant.tensor.GetTensorShape() == tensor.GetTensorShape() &&
          constant.bytes == bytes) {
        ++constants.reused;
        return constant.id;
      }
    }
  }
  tfsc::tosa::ResId id = graph.AddGraphConstant(tensor);
  constants.values.push_back({tensor, std::move(bytes), id});
  return id;
}

std::optional<Conv2dPlan> AnalyzeConv2d(Conv2dSpec spec, std::string& reason) {
  reason.clear();
  auto reject = [&](const char* message) -> std::optional<Conv2dPlan> {
    reason = message;
    return std::nullopt;
  };
  uint64_t input_count = 0;
  uint64_t weight_count = 0;
  uint64_t output_count = 0;
  if (!Elements(spec.input_shape, input_count) ||
      !Elements(spec.weight_shape, weight_count) ||
      !Elements(spec.output_shape, output_count)) {
    return reject(
        "CONV2D requires static nonempty shapes with int32 element counts");
  }
  const uint32_t channels = spec.weight_shape[0];
  if (spec.input_shape[0] != spec.output_shape[0] ||
      (spec.depthwise
           ? (channels % spec.input_shape[3] != 0 || spec.weight_shape[3] != 1)
           : (spec.input_shape[3] % spec.weight_shape[3] != 0 ||
              channels % (spec.input_shape[3] / spec.weight_shape[3]) != 0)) ||
      spec.output_shape[3] != channels) {
    return reject("CONV2D batch/channel dimensions disagree");
  }
  if (spec.weights.size() != weight_count || spec.bias.size() != channels ||
      spec.weight_scales.size() != channels ||
      spec.bias_scales.size() != channels) {
    return reject("CONV2D constant or per-channel quantization size mismatch");
  }
  if (!std::isfinite(spec.input_scale) || spec.input_scale <= 0 ||
      !std::isfinite(spec.output_scale) || spec.output_scale <= 0 ||
      spec.input_zero_point < -128 || spec.input_zero_point > 127) {
    return reject("CONV2D invalid activation quantization");
  }
  std::optional<Int8Range> activation = ActivationRange(
      spec.activation, spec.output_scale, spec.output_zero_point);
  if (!activation) {
    return reject("CONV2D invalid activation or output quantization");
  }
  Conv2dPlan plan;
  plan.convolution_input_shape = spec.input_shape;
  plan.convolution_padding = spec.padding;
  for (int axis = 0; axis < 2; ++axis) {
    const int64_t stride = spec.stride[axis];
    const int64_t dilation = spec.dilation[axis];
    const int64_t before = spec.padding[2 * axis];
    const int64_t after = spec.padding[2 * axis + 1];
    const int64_t kernel =
        (int64_t{spec.weight_shape[axis + 1]} - 1) * dilation + 1;
    const int64_t padded = int64_t{spec.input_shape[axis + 1]} + before + after;
    if (stride <= 0 || dilation <= 0 || before < 0 || after < 0 ||
        before >= kernel || after >= kernel || padded < kernel ||
        (padded - kernel) / stride + 1 != spec.output_shape[axis + 1]) {
      return reject(
          "CONV2D invalid padding, stride, dilation, or output shape");
    }
    // TOSA shape equations require exact divisibility; trailing elements that
    // LiteRT ignores must be removed from the declared padding or input.
    const int64_t trailing = (padded - kernel) % stride;
    const int64_t removed_padding = std::min(trailing, after);
    plan.convolution_padding[2 * axis + 1] -= removed_padding;
    plan.convolution_input_shape[axis + 1] -= trailing - removed_padding;
  }
  plan.activation_range = *activation;
  const uint64_t channel_size = weight_count / channels;
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const double expected_bias_scale =
        spec.input_scale * spec.weight_scales[channel];
    if (!std::isfinite(spec.weight_scales[channel]) ||
        spec.weight_scales[channel] <= 0 ||
        !std::isfinite(spec.bias_scales[channel]) ||
        spec.bias_scales[channel] <= 0 ||
        std::abs(spec.bias_scales[channel] - expected_bias_scale) >
            expected_bias_scale * 1e-5) {
      return reject(
          "CONV2D bias scale must equal input scale times weight scale");
    }
    std::optional<Scale32> scale =
        MakeScale32(expected_bias_scale / spec.output_scale);
    if (!scale) {
      return reject("CONV2D scale is not representable as TOSA scale32");
    }
    std::optional<AccumulatorRange> range = DotProductRange(
        spec.weights.data() + channel * channel_size, channel_size,
        spec.input_zero_point, 0, spec.bias[channel]);
    if (!range ||
        !IsScale32RangeValid(*scale, range->minimum, range->maximum)) {
      return reject("CONV2D cannot prove accumulator/RESCALE range safety");
    }
    plan.scales.push_back(*scale);
    plan.accumulator_ranges.push_back(*range);
  }
  plan.spec = std::move(spec);
  return plan;
}

tfsc::tosa::ResId EmitNativeRescale(
    tfsc::tosa::Graph& graph, tfsc::tosa::ResId input, DataType input_type,
    DataType output_type, const std::vector<uint32_t>& logical_shape,
    int32_t input_zero_point, int32_t output_zero_point,
    const std::vector<Scale32>& scales, GraphConstantPool& constants,
    const std::vector<int64_t>& minimum, const std::vector<int64_t>& maximum) {
  const std::vector<uint32_t> shape =
      logical_shape.empty() ? std::vector<uint32_t>{1} : logical_shape;
  if (scales.empty() || shape.empty() || minimum.empty() != maximum.empty() ||
      (scales.size() != 1 && scales.size() != shape.back()) ||
      (!minimum.empty() &&
       (minimum.size() != scales.size() || maximum.size() != scales.size()))) {
    return nullptr;
  }
  const bool wide = input_type == DataType::int48_t;
  const bool use16 = constants.rescale_multiplier_bits == 16 || wide;
  auto emit = [&](tfsc::tosa::ResId value, DataType source_type,
                  DataType dest_type, int32_t input_zp, int32_t output_zp,
                  const std::vector<int32_t>& mult,
                  const std::vector<int8_t>& shift) {
    tfsc::tosa::ResId multiplier_id;
    if (use16) {
      multiplier_id = Constant(
          graph,
          Tensor(DataType::int16_t, {static_cast<uint32_t>(mult.size())}),
          std::vector<int16_t>(mult.begin(), mult.end()), constants);
    } else {
      multiplier_id = Constant(
          graph,
          Tensor(DataType::int32_t, {static_cast<uint32_t>(mult.size())}), mult,
          constants);
    }
    tfsc::tosa::ResId shift_id = Constant(
        graph, Tensor(DataType::int8_t, {static_cast<uint32_t>(shift.size())}),
        shift, constants);
    tfsc::tosa::ResId zero =
        source_type == DataType::int48_t
            ? graph.AddTensorConstant(Attribute(int64_t{input_zp}))
            : graph.AddTensorConstant(Attribute(input_zp, source_type));
    return graph.AddRescaleOperator(
        value, multiplier_id, shift_id, zero,
        graph.AddTensorConstant(Attribute(output_zp, dest_type)),
        Attribute(!use16), Attribute(int32_t{1}), Attribute(mult.size() > 1),
        Attribute(false), Attribute(false), Tensor(dest_type, shape));
  };
  std::vector<int32_t> multipliers;
  std::vector<int8_t> shifts;
  if (!use16) {
    for (Scale32 scale : scales) {
      multipliers.push_back(scale.multiplier);
      shifts.push_back(scale.shift);
    }
    return emit(input, input_type, output_type, input_zero_point,
                output_zero_point, multipliers, shifts);
  }
  std::vector<double> residual;
  std::vector<int> amplification;
  std::vector<int64_t> min_bounds;
  std::vector<int64_t> max_bounds;
  for (size_t i = 0; i < scales.size(); ++i) {
    double scale =
        std::ldexp(static_cast<double>(scales[i].multiplier), -scales[i].shift);
    if (!(scale > 0) || !std::isfinite(scale)) return nullptr;
    int left = 0;
    while (std::llround(scale * 4.0) > INT16_MAX) {
      scale *= .5;
      if (++left > 31) return nullptr;
    }
    residual.push_back(scale);
    amplification.push_back(left);
    const int64_t lower = input_type == DataType::int8_t    ? -128
                          : input_type == DataType::int16_t ? -32768
                          : wide ? -(int64_t{1} << 47)
                                 : INT32_MIN;
    const int64_t upper = input_type == DataType::int8_t    ? 127
                          : input_type == DataType::int16_t ? 32767
                          : wide ? (int64_t{1} << 47) - 1
                                 : INT32_MAX;
    min_bounds.push_back(minimum.empty() ? lower - input_zero_point
                                         : minimum[i]);
    max_bounds.push_back(maximum.empty() ? upper - input_zero_point
                                         : maximum[i]);
  }
  auto pair16 = [](double scale, int32_t& multiplier, int8_t& shift) {
    int exponent = 0;
    std::frexp(scale, &exponent);
    int candidate_shift = std::min(62, 15 - exponent);
    if (candidate_shift < 2) return false;
    int64_t candidate_multiplier =
        std::llround(std::ldexp(scale, candidate_shift));
    if (candidate_multiplier > INT16_MAX) {
      --candidate_shift;
      candidate_multiplier = std::llround(std::ldexp(scale, candidate_shift));
    }
    if (candidate_shift < 2 || candidate_multiplier <= 0 ||
        candidate_multiplier > INT16_MAX) {
      return false;
    }
    multiplier = static_cast<int32_t>(candidate_multiplier);
    shift = static_cast<int8_t>(candidate_shift);
    return true;
  };
  while (std::any_of(amplification.begin(), amplification.end(),
                     [](int n) { return n != 0; })) {
    multipliers.clear();
    shifts.clear();
    for (size_t i = 0; i < amplification.size(); ++i) {
      int step = std::min(12, amplification[i]);
      int64_t factor = int64_t{1} << step;
      if (min_bounds[i] < INT32_MIN / factor ||
          max_bounds[i] > INT32_MAX / factor) {
        return nullptr;
      }
      min_bounds[i] *= factor;
      max_bounds[i] *= factor;
      int32_t multiplier = 0;
      int8_t shift = 0;
      if (!pair16(static_cast<double>(factor), multiplier, shift)) {
        return nullptr;
      }
      multipliers.push_back(multiplier);
      shifts.push_back(shift);
      amplification[i] -= step;
    }
    input = emit(input, input_type, DataType::int32_t, input_zero_point, 0,
                 multipliers, shifts);
    input_type = DataType::int32_t;
    input_zero_point = 0;
  }
  multipliers.clear();
  shifts.clear();
  for (double scale : residual) {
    int32_t multiplier = 0;
    int8_t shift = 0;
    if (!pair16(scale, multiplier, shift)) return nullptr;
    multipliers.push_back(multiplier);
    shifts.push_back(shift);
  }
  return emit(input, input_type, output_type, input_zero_point,
              output_zero_point, multipliers, shifts);
}

tfsc::tosa::ResId EmitRescale(tfsc::tosa::Graph& graph, tfsc::tosa::ResId input,
                              const std::vector<uint32_t>& shape,
                              int32_t output_zero_point,
                              const std::vector<Scale32>& scales,
                              const std::vector<AccumulatorRange>& ranges,
                              GraphConstantPool& constants) {
  if (shape.empty() || scales.empty() || scales.size() != ranges.size() ||
      (scales.size() != 1 && scales.size() != shape.back()) ||
      output_zero_point < -128 || output_zero_point > 127) {
    return nullptr;
  }
  for (size_t i = 0; i < scales.size(); ++i) {
    if (!IsScale32RangeValid(scales[i], ranges[i].minimum, ranges[i].maximum)) {
      return nullptr;
    }
  }
  std::vector<int64_t> min_bounds;
  std::vector<int64_t> max_bounds;
  for (AccumulatorRange range : ranges) {
    min_bounds.push_back(range.minimum);
    max_bounds.push_back(range.maximum);
  }
  return EmitNativeRescale(graph, input, DataType::int32_t, DataType::int8_t,
                           shape, 0, output_zero_point, scales, constants,
                           min_bounds, max_bounds);
}

tfsc::tosa::ResId EmitConv2d(tfsc::tosa::Graph& graph, tfsc::tosa::ResId input,
                             const Conv2dPlan& plan,
                             GraphConstantPool& constants,
                             tfsc::tosa::ResId* accumulator) {
  const Conv2dSpec& spec = plan.spec;
  const uint32_t groups =
      spec.depthwise ? 1 : spec.input_shape[3] / spec.weight_shape[3];
  if (groups > 1) {
    const uint32_t input_channels = spec.weight_shape[3];
    const uint32_t output_channels = spec.output_shape[3] / groups;
    const size_t group_weights = spec.weights.size() / groups;
    std::vector<tfsc::tosa::ResId> outputs;
    std::vector<tfsc::tosa::ResId> accumulators;
    for (uint32_t group = 0; group < groups; ++group) {
      Conv2dSpec part{};
      part.input_shape = spec.input_shape;
      part.input_shape[3] = input_channels;
      part.weight_shape = spec.weight_shape;
      part.weight_shape[0] = output_channels;
      part.output_shape = spec.output_shape;
      part.output_shape[3] = output_channels;
      part.padding = spec.padding;
      part.stride = spec.stride;
      part.dilation = spec.dilation;
      part.input_scale = spec.input_scale;
      part.output_scale = spec.output_scale;
      part.input_zero_point = spec.input_zero_point;
      part.output_zero_point = spec.output_zero_point;
      part.activation = spec.activation;
      part.weights.assign(spec.weights.begin() + group * group_weights,
                          spec.weights.begin() + (group + 1) * group_weights);
      part.bias.assign(spec.bias.begin() + group * output_channels,
                       spec.bias.begin() + (group + 1) * output_channels);
      part.weight_scales.assign(
          spec.weight_scales.begin() + group * output_channels,
          spec.weight_scales.begin() + (group + 1) * output_channels);
      part.bias_scales.assign(
          spec.bias_scales.begin() + group * output_channels,
          spec.bias_scales.begin() + (group + 1) * output_channels);
      std::string reason;
      std::optional<Conv2dPlan> subplan =
          AnalyzeConv2d(std::move(part), reason);
      if (!subplan) return nullptr;
      std::vector<uint32_t> shape(spec.input_shape.begin(),
                                  spec.input_shape.end());
      shape[3] = input_channels;
      tfsc::tosa::ResId slice = graph.AddSliceOperator(
          input,
          graph.AddTensorConstant(Attribute(
              std::vector<uint32_t>{0, 0, 0, group * input_channels})),
          graph.AddTensorConstant(Attribute(shape)),
          Tensor(DataType::int8_t, shape));
      tfsc::tosa::ResId acc;
      outputs.push_back(EmitConv2d(graph, slice, *subplan, constants,
                                   accumulator ? &acc : nullptr));
      if (accumulator) accumulators.push_back(acc);
    }
    const std::vector<uint32_t> shape(spec.output_shape.begin(),
                                      spec.output_shape.end());
    if (accumulator) {
      *accumulator =
          graph.AddConcatOperator(accumulators, Attribute(int32_t{3}),
                                  Tensor(DataType::int32_t, shape));
    }
    return graph.AddConcatOperator(outputs, Attribute(int32_t{3}),
                                   Tensor(DataType::int8_t, shape));
  }
  Tensor weight_type(DataType::int8_t,
                     {spec.weight_shape.begin(), spec.weight_shape.end()});
  Tensor output_type(DataType::int8_t,
                     {spec.output_shape.begin(), spec.output_shape.end()});
  Tensor acc_type(DataType::int32_t, output_type.GetTensorShape());
  if (plan.convolution_input_shape != spec.input_shape) {
    tfsc::tosa::ResId start =
        graph.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0)));
    tfsc::tosa::ResId size = graph.AddTensorConstant(
        Attribute(std::vector<uint32_t>(plan.convolution_input_shape.begin(),
                                        plan.convolution_input_shape.end())));
    input = graph.AddSliceOperator(
        input, start, size,
        Tensor(DataType::int8_t, {plan.convolution_input_shape.begin(),
                                  plan.convolution_input_shape.end()}));
    if (tfsc::IsError(input)) return input;
  }
  std::vector<int8_t> weights = spec.weights;
  if (spec.depthwise) {
    const uint32_t channels = spec.weight_shape[0];
    const size_t kernel = size_t{spec.weight_shape[1]} * spec.weight_shape[2];
    for (size_t kernel_idx = 0; kernel_idx < kernel; ++kernel_idx) {
      for (size_t channel = 0; channel < channels; ++channel) {
        weights[kernel_idx * channels + channel] =
            spec.weights[channel * kernel + kernel_idx];
      }
    }
    weight_type = Tensor(DataType::int8_t,
                         {spec.weight_shape[1], spec.weight_shape[2],
                          spec.input_shape[3], channels / spec.input_shape[3]});
  }
  tfsc::tosa::ResId weight = Constant(graph, weight_type, weights, constants);
  tfsc::tosa::ResId bias =
      Constant(graph, Tensor(DataType::int32_t, {spec.weight_shape[0]}),
               spec.bias, constants);
  tfsc::tosa::ResId input_zp = graph.AddTensorConstant(
      Attribute(static_cast<int8_t>(spec.input_zero_point)));
  tfsc::tosa::ResId weight_zp = graph.AddTensorConstant(Attribute(int8_t{0}));
  // TOSA.001000.1 enum encodings (not serialization-library DType values):
  // acc_type_t::INT32 = 1, rounding_mode_t::SINGLE_ROUND = 1.
  tfsc::tosa::ResId conv = graph.AddOperator(
      spec.depthwise ? tfsc::tosa::OperatorEnum::DepthwiseConv2d
                     : tfsc::tosa::OperatorEnum::Conv2d,
      {input, weight, bias, input_zp, weight_zp}, {acc_type},
      {Attribute(std::vector<int32_t>(plan.convolution_padding.begin(),
                                      plan.convolution_padding.end())),
       Attribute(std::vector<int32_t>(spec.stride.begin(), spec.stride.end())),
       Attribute(
           std::vector<int32_t>(spec.dilation.begin(), spec.dilation.end())),
       Attribute(int32_t{1}), Attribute(false)})[0];
  if (tfsc::IsError(conv)) return conv;
  if (accumulator) *accumulator = conv;
  tfsc::tosa::ResId rescale = EmitRescale(
      graph, conv, output_type.GetTensorShape(), spec.output_zero_point,
      plan.scales, plan.accumulator_ranges, constants);
  if (tfsc::IsError(rescale)) return rescale;
  if (plan.activation_range.minimum == -128 &&
      plan.activation_range.maximum == 127) {
    return rescale;
  }
  return graph.AddClampOperator(
      rescale, Attribute(static_cast<int8_t>(plan.activation_range.minimum)),
      Attribute(static_cast<int8_t>(plan.activation_range.maximum)),
      Attribute(int32_t{1}), output_type);
}
}  // namespace litert::arm_vulkan_ml
