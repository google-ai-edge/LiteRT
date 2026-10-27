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

#include "litert/vendors/arm_vulkan_ml/compiler/conv2d_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include "litert/c/litert_op_options.h"

namespace litert::arm_vulkan_ml {
namespace {
bool Shape4(const Tensor& tensor, ElementType type, std::array<uint32_t, 4>& shape) {
  auto ranked = tensor.RankedTensorType();
  if (!ranked || ranked->ElementType() != type || ranked->Layout().HasStrides()) return false;
  auto dims = ranked->Layout().Dimensions();
  if (dims.size() != 4) return false;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) return false;
    shape[i] = dims[i];
  }
  return true;
}

bool Scales(const Tensor& tensor, uint32_t channels, int axis, std::vector<double>& scales) {
  if (tensor.QTypeId() == QuantizationTypeId::PerTensor) {
    auto q = tensor.PerTensorQuantization();
    if (q.zero_point != 0) return false;
    scales.assign(channels, q.scale);
  } else if (tensor.QTypeId() == QuantizationTypeId::PerChannel) {
    auto q = tensor.PerChannelQuantization();
    if (q.quantized_dimension != axis || q.num_channels != channels) return false;
    for (uint32_t i = 0; i < channels; ++i) {
      if (q.zero_points[i] != 0) return false;
      scales.push_back(q.scales[i]);
    }
  } else return false;
  return true;
}
}  // namespace

std::optional<Conv2dPlan> AnalyzeConv2d(const Op& op, std::string& reason) {
  auto reject = [&](const char* message) -> std::optional<Conv2dPlan> {
    reason = message;
    return std::nullopt;
  };
  const bool depthwise = op.Code() == kLiteRtOpCodeTflDepthwiseConv2d;
  if (op.Code() != kLiteRtOpCodeTflConv2d && !depthwise) return reject("not convolution");
  auto inputs = op.Inputs();
  auto outputs = op.Outputs();
  if (inputs.size() != 3 || outputs.size() != 1)
    return reject("CONV_2D needs input, constant weights, and constant bias");
  Conv2dSpec s{};
  if (!Shape4(inputs[0], ElementType::Int8, s.input_shape) ||
      !Shape4(inputs[1], ElementType::Int8, s.weight_shape) ||
      !Shape4(outputs[0], ElementType::Int8, s.output_shape))
    return reject("CONV_2D requires dense static rank-4 int8 tensors");
  s.depthwise = depthwise;
  if (depthwise) {
    if (s.weight_shape[0] != 1 || s.weight_shape[3] != s.input_shape[3])
      return reject("Depthwise convolution requires depth_multiplier=1");
    s.weight_shape = {s.weight_shape[3], s.weight_shape[1], s.weight_shape[2], 1};
  }
  const auto& bias = inputs[2];
  auto bias_type = bias.RankedTensorType();
  if (!bias_type || bias_type->ElementType() != ElementType::Int32 ||
      bias_type->Layout().HasStrides() ||
      bias_type->Layout().Dimensions().size() != 1 ||
      bias_type->Layout().Dimensions()[0] != s.weight_shape[0])
    return reject("CONV_2D requires a dense int32 bias vector");
  if (inputs[0].QTypeId() != QuantizationTypeId::PerTensor ||
      outputs[0].QTypeId() != QuantizationTypeId::PerTensor)
    return reject("CONV_2D activation quantization must be per-tensor");
  auto iq = inputs[0].PerTensorQuantization();
  auto oq = outputs[0].PerTensorQuantization();
  if (iq.zero_point < -128 || iq.zero_point > 127 ||
      oq.zero_point < -128 || oq.zero_point > 127)
    return reject("CONV_2D activation zero point is outside int8");
  s.input_scale = iq.scale;
  s.output_scale = oq.scale;
  s.input_zero_point = iq.zero_point;
  s.output_zero_point = oq.zero_point;
  if (!Scales(inputs[1], s.weight_shape[0], depthwise ? 3 : 0, s.weight_scales) ||
      !Scales(bias, s.weight_shape[0], 0, s.bias_scales))
    return reject("CONV_2D requires symmetric per-tensor or output-channel weights/bias");
  auto weights = inputs[1].WeightsData<int8_t>();
  auto biases = bias.WeightsData<int32_t>();
  if (!weights || !biases) return reject("CONV_2D weights and bias must be valid constants");
  s.weights.assign(weights->begin(), weights->end());
  s.bias.assign(biases->begin(), biases->end());
  if (depthwise) {
    const size_t channels = s.weight_shape[0];
    const size_t kernel = size_t{s.weight_shape[1]} * s.weight_shape[2];
    for (size_t c = 0; c < channels; ++c)
      for (size_t k = 0; k < kernel; ++k)
        s.weights[c * kernel + k] = (*weights)[k * channels + c];
  }
  uint32_t padding, activation;
  auto padding_fn = depthwise ? LiteRtGetDepthwiseConv2dPaddingOption : LiteRtGetConv2dPaddingOption;
  auto activation_fn = depthwise ? LiteRtGetDepthwiseConv2dFusedActivationOption : LiteRtGetConv2dFusedActivationOption;
  auto stride_h = depthwise ? LiteRtGetDepthwiseConv2dStrideHOption : LiteRtGetConv2dStrideHOption;
  auto stride_w = depthwise ? LiteRtGetDepthwiseConv2dStrideWOption : LiteRtGetConv2dStrideWOption;
  auto dilation_h = depthwise ? LiteRtGetDepthwiseConv2dDilationHOption : LiteRtGetConv2dDilationHOption;
  auto dilation_w = depthwise ? LiteRtGetDepthwiseConv2dDilationWOption : LiteRtGetConv2dDilationWOption;
  if (padding_fn(op.Get(), &padding) != kLiteRtStatusOk ||
      activation_fn(op.Get(), &activation) != kLiteRtStatusOk ||
      stride_h(op.Get(), &s.stride[0]) != kLiteRtStatusOk ||
      stride_w(op.Get(), &s.stride[1]) != kLiteRtStatusOk ||
      dilation_h(op.Get(), &s.dilation[0]) != kLiteRtStatusOk ||
      dilation_w(op.Get(), &s.dilation[1]) != kLiteRtStatusOk)
    return reject("Convolution option lookup failed");
  if (depthwise) {
    int32_t multiplier;
    if (LiteRtGetDepthwiseConv2dDepthMultiplierOption(op.Get(), &multiplier) != kLiteRtStatusOk || multiplier != 1)
      return reject("Depthwise convolution requires depth_multiplier=1");
  }
  switch (activation) {
    case 0: s.activation = Activation::kNone; break;
    case 1: s.activation = Activation::kRelu; break;
    case 3: s.activation = Activation::kRelu6; break;
    default: return reject("CONV_2D unsupported fused activation");
  }
  if (padding > 1) return reject("CONV_2D unknown padding");
  for (int axis = 0; axis < 2; ++axis) {
    if (s.stride[axis] <= 0 || s.dilation[axis] <= 0)
      return reject("CONV_2D invalid stride/dilation");
    if (padding == 0) {
      const int64_t expected = (int64_t{s.input_shape[axis + 1]} +
                                s.stride[axis] - 1) / s.stride[axis];
      if (s.output_shape[axis + 1] != expected)
        return reject("CONV_2D SAME output shape is not ceil(input/stride)");
      int64_t kernel = (int64_t{s.weight_shape[axis + 1]} - 1) * s.dilation[axis] + 1;
      int64_t total = std::max(int64_t{0},
          (int64_t{s.output_shape[axis + 1]} - 1) * s.stride[axis] + kernel - s.input_shape[axis + 1]);
      if (total > INT32_MAX) return reject("CONV_2D padding overflow");
      s.padding[2 * axis] = total / 2;
      s.padding[2 * axis + 1] = total - total / 2;
    }
  }
  return AnalyzeConv2d(std::move(s), reason);
}
}  // namespace litert::arm_vulkan_ml
