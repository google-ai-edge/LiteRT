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
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include "TosaForSpirvCodegen.hpp"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::Tensor;

bool Elements(const std::array<uint32_t, 4>& shape, uint64_t& count) {
  count = 1;
  for (auto dim : shape) {
    if (!dim || count > uint64_t{INT32_MAX} / dim) return false;
    count *= dim;
  }
  return true;
}

template <typename T>
tfsc::tosa::ResId Constant(tfsc::tosa::Graph& graph, const Tensor& tensor,
                           const std::vector<T>& data,
                           std::vector<GraphConstant>& constants) {
  GraphConstant constant{tensor, std::vector<uint8_t>(data.size() * sizeof(T))};
  std::memcpy(constant.bytes.data(), data.data(), constant.bytes.size());
  constants.push_back(std::move(constant));
  return graph.AddGraphConstant(tensor);
}
}  // namespace

std::optional<Conv2dPlan> AnalyzeConv2d(Conv2dSpec s, std::string& reason) {
  reason.clear();
  auto reject = [&](const char* message) -> std::optional<Conv2dPlan> {
    reason = message;
    return std::nullopt;
  };
  uint64_t input_count, weight_count, output_count;
  if (!Elements(s.input_shape, input_count) ||
      !Elements(s.weight_shape, weight_count) ||
      !Elements(s.output_shape, output_count))
    return reject("CONV2D requires static nonempty shapes with int32 element counts");
  const uint32_t channels = s.weight_shape[0];
  if (s.input_shape[0] != s.output_shape[0] ||
      (s.depthwise ? (s.input_shape[3] != channels || s.weight_shape[3] != 1) :
       (s.input_shape[3] % s.weight_shape[3] != 0 || channels % (s.input_shape[3] / s.weight_shape[3]) != 0)) || s.output_shape[3] != channels)
    return reject("CONV2D batch/channel dimensions disagree");
  if (s.weights.size() != weight_count || s.bias.size() != channels ||
      s.weight_scales.size() != channels || s.bias_scales.size() != channels)
    return reject("CONV2D constant or per-channel quantization size mismatch");
  if (!std::isfinite(s.input_scale) || s.input_scale <= 0 ||
      !std::isfinite(s.output_scale) || s.output_scale <= 0 ||
      s.input_zero_point < -128 || s.input_zero_point > 127)
    return reject("CONV2D invalid activation quantization");
  auto activation = ActivationRange(s.activation, s.output_scale, s.output_zero_point);
  if (!activation) return reject("CONV2D invalid activation or output quantization");
  Conv2dPlan plan;
  plan.convolution_input_shape = s.input_shape;
  plan.convolution_padding = s.padding;
  for (int axis = 0; axis < 2; ++axis) {
    const int64_t stride = s.stride[axis], dilation = s.dilation[axis];
    const int64_t before = s.padding[2 * axis], after = s.padding[2 * axis + 1];
    const int64_t kernel = (int64_t{s.weight_shape[axis + 1]} - 1) * dilation + 1;
    const int64_t padded = int64_t{s.input_shape[axis + 1]} + before + after;
    if (stride <= 0 || dilation <= 0 || before < 0 || after < 0 ||
        before >= kernel || after >= kernel || padded < kernel ||
        (padded - kernel) / stride + 1 != s.output_shape[axis + 1])
      return reject("CONV2D invalid padding, stride, dilation, or output shape");
    // TOSA shape equations require exact divisibility; trailing elements that
    // LiteRT ignores must be removed from the declared padding or input.
    const int64_t trailing = (padded - kernel) % stride;
    const int64_t removed_padding = std::min(trailing, after);
    plan.convolution_padding[2 * axis + 1] -= removed_padding;
    plan.convolution_input_shape[axis + 1] -= trailing - removed_padding;
  }
  plan.activation_range = *activation;
  const uint64_t channel_size = weight_count / channels;
  for (uint32_t c = 0; c < channels; ++c) {
    const double expected_bias_scale = s.input_scale * s.weight_scales[c];
    if (!std::isfinite(s.weight_scales[c]) || s.weight_scales[c] <= 0 ||
        !std::isfinite(s.bias_scales[c]) || s.bias_scales[c] <= 0 ||
        std::abs(s.bias_scales[c] - expected_bias_scale) > expected_bias_scale * 1e-5)
      return reject("CONV2D bias scale must equal input scale times weight scale");
    auto scale = MakeScale32(expected_bias_scale / s.output_scale);
    if (!scale) return reject("CONV2D scale is not representable as TOSA scale32");
    auto range = DotProductRange(s.weights.data() + c * channel_size,
                                 channel_size, s.input_zero_point, 0, s.bias[c]);
    if (!range || !IsScale32RangeValid(*scale, range->minimum, range->maximum))
      return reject("CONV2D cannot prove accumulator/RESCALE range safety");
    plan.scales.push_back(*scale);
    plan.accumulator_ranges.push_back(*range);
  }
  plan.spec = std::move(s);
  return plan;
}

bool UsesStagedRounding(const Conv2dPlan& plan) {
  for (size_t c = 0; c < plan.scales.size(); ++c) {
    auto first = plan.scales[c];
    first.shift = std::min<int8_t>(first.shift, 31);
    const auto range = plan.accumulator_ranges[c];
    if (!IsScale32RangeValid(first, range.minimum, range.maximum)) return false;
    auto minimum = ApplyScale32(range.minimum, plan.scales[c]);
    auto maximum = ApplyScale32(range.maximum, plan.scales[c]);
    // The composed result differs from one rounding by at most one code.
    // Prove that the final identity rescale also meets TOSA's input bound.
    if (!minimum || !maximum ||
        !IsScale32RangeValid(Scale32{1 << 30, 30}, int64_t{*minimum} - 1,
                             int64_t{*maximum} + 1)) return false;
  }
  return true;
}

tfsc::tosa::ResId EmitRescale(tfsc::tosa::Graph& graph, tfsc::tosa::ResId input,
    const std::vector<uint32_t>& shape, int32_t output_zero_point,
    const std::vector<Scale32>& scales, const std::vector<AccumulatorRange>& ranges,
    std::vector<GraphConstant>& constants) {
  if(shape.empty() || scales.empty() || scales.size()!=ranges.size() ||
     (scales.size()!=1 && scales.size()!=shape.back()))return nullptr;
  Tensor output_type(DataType::int8_t,shape),acc_type(DataType::int32_t,shape);
  std::vector<bool> legal;
  for(size_t i=0;i<scales.size();++i) {
    Conv2dPlan check{};check.scales={scales[i]};check.accumulator_ranges={ranges[i]};
    legal.push_back(UsesStagedRounding(check));
  }
  // A bias-underflow guard can make one channel large. Keep the exact rounding
  // path for the remaining channels instead of downgrading the entire tensor.
  if(scales.size()>1 && !std::all_of(legal.begin(),legal.end(),[&](bool v){return v==legal[0];})) {
    std::vector<tfsc::tosa::ResId> pieces;
    for(size_t first=0;first<scales.size();) {
      size_t last=first+1;while(last<scales.size() && legal[last]==legal[first])++last;
      auto subshape=shape;subshape.back()=last-first;
      std::vector<uint32_t> start(shape.size(),0);start.back()=first;
      auto slice=graph.AddSliceOperator(input,graph.AddTensorConstant(Attribute(start)),
          graph.AddTensorConstant(Attribute(subshape)),Tensor(DataType::int32_t,subshape));
      pieces.push_back(EmitRescale(graph,slice,subshape,output_zero_point,
          {scales.begin()+first,scales.begin()+last},{ranges.begin()+first,ranges.begin()+last},constants));
      first=last;
    }
    return graph.AddConcatOperator(pieces,Attribute(static_cast<int32_t>(shape.size()-1)),output_type);
  }
  // Compose LiteRT's two rounding stages from base-profile SINGLE_ROUND
  // operations when the proven accumulator range permits a shift-31 stage.
  // Larger accumulators retain the original, range-checked single rescale.
  const bool staged = std::all_of(legal.begin(),legal.end(),[](bool v){return v;});
  std::vector<int32_t> multipliers;
  std::vector<int8_t> shifts;
  std::vector<int32_t> tail_shifts, correction;
  for (auto scale : scales) {
    multipliers.push_back(scale.multiplier);
    const int first = staged ? std::min<int>(scale.shift, 31) : scale.shift;
    shifts.push_back(first);
    tail_shifts.push_back(scale.shift - first);
    correction.push_back(scale.shift > first ? 1 : 0);
  }
  auto multiplier = Constant(graph, Tensor(DataType::int32_t, {static_cast<uint32_t>(scales.size())}),
                              multipliers, constants);
  auto shift = Constant(graph, Tensor(DataType::int8_t, {static_cast<uint32_t>(scales.size())}),
                         shifts, constants);
  auto zero = graph.AddTensorConstant(Attribute(int32_t{0}));
  auto output_zp = graph.AddTensorConstant(Attribute(static_cast<int8_t>(output_zero_point)));
  tfsc::tosa::ResId rescale;
  if (staged) {
    auto value = graph.AddRescaleOperator(input, multiplier, shift, zero, zero,
        Attribute(true), Attribute(int32_t{1}), Attribute(scales.size()>1), Attribute(false),
        Attribute(false), acc_type);
    std::vector<uint32_t> channel_shape(shape.size(), 1);
    channel_shape.back() = static_cast<uint32_t>(scales.size());
    auto adjustment = Constant(graph, Tensor(DataType::int32_t, channel_shape),
                               correction, constants);
    auto right_shift = Constant(graph, Tensor(DataType::int32_t, channel_shape),
                                tail_shifts, constants);
    auto broadcast_zero = Constant(graph, Tensor(DataType::int32_t, std::vector<uint32_t>(shape.size(),1)),
                                   std::vector<int32_t>{0}, constants);
    auto negative = graph.AddGreaterOperator(broadcast_zero, value,
        Tensor(DataType::bool_t, shape));
    auto adjusted = graph.AddSubOperator(value, adjustment, acc_type);
    value = graph.AddSelectOperator(negative, adjusted, value, acc_type);
    value = graph.AddArithmeticRightShiftOperator(value, right_shift,
                                                   Attribute(true), acc_type);
    auto identity_multiplier = Constant(graph, Tensor(DataType::int32_t, {1}),
        std::vector<int32_t>{int32_t{1} << 30}, constants);
    auto identity_shift = Constant(graph, Tensor(DataType::int8_t, {1}),
        std::vector<int8_t>{30}, constants);
    rescale = graph.AddRescaleOperator(value, identity_multiplier, identity_shift,
        zero, output_zp, Attribute(true), Attribute(int32_t{1}), Attribute(false),
        Attribute(false), Attribute(false), output_type);
  } else {
    rescale = graph.AddRescaleOperator(input, multiplier, shift, zero, output_zp,
        Attribute(true), Attribute(int32_t{1}), Attribute(scales.size()>1), Attribute(false),
        Attribute(false), output_type);
  }
  return rescale;
}

tfsc::tosa::ResId EmitConv2d(tfsc::tosa::Graph& graph,
                            tfsc::tosa::ResId input, const Conv2dPlan& plan,
                            std::vector<GraphConstant>& constants,
                            tfsc::tosa::ResId* accumulator) {
  const auto& s = plan.spec;
  const uint32_t groups=s.depthwise?1:s.input_shape[3]/s.weight_shape[3];
  if(groups>1) {
    const uint32_t input_channels=s.weight_shape[3],output_channels=s.output_shape[3]/groups;
    const size_t group_weights=s.weights.size()/groups;
    std::vector<tfsc::tosa::ResId> outputs,accumulators;
    for(uint32_t group=0;group<groups;++group) {
      Conv2dSpec part{};
      part.input_shape=s.input_shape;part.input_shape[3]=input_channels;
      part.weight_shape=s.weight_shape;part.weight_shape[0]=output_channels;
      part.output_shape=s.output_shape;part.output_shape[3]=output_channels;
      part.padding=s.padding;part.stride=s.stride;part.dilation=s.dilation;
      part.input_scale=s.input_scale;part.output_scale=s.output_scale;
      part.input_zero_point=s.input_zero_point;part.output_zero_point=s.output_zero_point;
      part.activation=s.activation;
      part.weights.assign(s.weights.begin()+group*group_weights,s.weights.begin()+(group+1)*group_weights);
      part.bias.assign(s.bias.begin()+group*output_channels,s.bias.begin()+(group+1)*output_channels);
      part.weight_scales.assign(s.weight_scales.begin()+group*output_channels,s.weight_scales.begin()+(group+1)*output_channels);
      part.bias_scales.assign(s.bias_scales.begin()+group*output_channels,s.bias_scales.begin()+(group+1)*output_channels);
      std::string reason;auto subplan=AnalyzeConv2d(std::move(part),reason);
      if(!subplan)return nullptr;
      auto shape=std::vector<uint32_t>(s.input_shape.begin(),s.input_shape.end());shape[3]=input_channels;
      auto slice=graph.AddSliceOperator(input,graph.AddTensorConstant(Attribute(std::vector<uint32_t>{0,0,0,group*input_channels})),
          graph.AddTensorConstant(Attribute(shape)),Tensor(DataType::int8_t,shape));
      tfsc::tosa::ResId acc;
      outputs.push_back(EmitConv2d(graph,slice,*subplan,constants,accumulator?&acc:nullptr));
      if(accumulator)accumulators.push_back(acc);
    }
    const std::vector<uint32_t> shape(s.output_shape.begin(),s.output_shape.end());
    if(accumulator)*accumulator=graph.AddConcatOperator(accumulators,Attribute(int32_t{3}),Tensor(DataType::int32_t,shape));
    return graph.AddConcatOperator(outputs,Attribute(int32_t{3}),Tensor(DataType::int8_t,shape));
  }
  Tensor weight_type(DataType::int8_t, {s.weight_shape.begin(), s.weight_shape.end()});
  Tensor output_type(DataType::int8_t, {s.output_shape.begin(), s.output_shape.end()});
  Tensor acc_type(DataType::int32_t, output_type.GetTensorShape());
  if (plan.convolution_input_shape != s.input_shape) {
    auto start = graph.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0)));
    auto size = graph.AddTensorConstant(Attribute(std::vector<uint32_t>(
        plan.convolution_input_shape.begin(), plan.convolution_input_shape.end())));
    input = graph.AddSliceOperator(input, start, size, Tensor(DataType::int8_t,
        {plan.convolution_input_shape.begin(), plan.convolution_input_shape.end()}));
    if (tfsc::IsError(input)) return input;
  }
  auto weights = s.weights;
  if (s.depthwise) {
    const uint32_t channels = s.weight_shape[0];
    const size_t kernel = size_t{s.weight_shape[1]} * s.weight_shape[2];
    for (size_t k = 0; k < kernel; ++k)
      for (size_t c = 0; c < channels; ++c)
        weights[k * channels + c] = s.weights[c * kernel + k];
    weight_type = Tensor(DataType::int8_t, {s.weight_shape[1], s.weight_shape[2], channels, 1});
  }
  auto weight = Constant(graph, weight_type, weights, constants);
  auto bias = Constant(graph, Tensor(DataType::int32_t, {s.weight_shape[0]}),
                        s.bias, constants);
  auto input_zp = graph.AddTensorConstant(Attribute(static_cast<int8_t>(s.input_zero_point)));
  auto weight_zp = graph.AddTensorConstant(Attribute(int8_t{0}));
  // TOSA.001000.1 enum encodings (not serialization-library DType values):
  // acc_type_t::INT32 = 1, rounding_mode_t::SINGLE_ROUND = 1.
  auto conv = graph.AddOperator(s.depthwise ? tfsc::tosa::OperatorEnum::DepthwiseConv2d
                                            : tfsc::tosa::OperatorEnum::Conv2d,
      {input, weight, bias, input_zp, weight_zp}, {acc_type}, {
      Attribute(std::vector<int32_t>(plan.convolution_padding.begin(),
                                     plan.convolution_padding.end())),
      Attribute(std::vector<int32_t>(s.stride.begin(), s.stride.end())),
      Attribute(std::vector<int32_t>(s.dilation.begin(), s.dilation.end())),
      Attribute(int32_t{1}), Attribute(false)})[0];
  if (tfsc::IsError(conv)) return conv;
  if (accumulator) *accumulator = conv;
  auto rescale = EmitRescale(graph,conv,output_type.GetTensorShape(),
      s.output_zero_point,plan.scales,plan.accumulator_ranges,constants);
  if (tfsc::IsError(rescale)) return rescale;
  return graph.AddClampOperator(rescale,
      Attribute(static_cast<int8_t>(plan.activation_range.minimum)),
      Attribute(static_cast<int8_t>(plan.activation_range.maximum)), Attribute(int32_t{1}),
      output_type);
}
}  // namespace litert::arm_vulkan_ml
