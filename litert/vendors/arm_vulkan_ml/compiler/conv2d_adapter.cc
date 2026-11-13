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
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"

namespace litert::arm_vulkan_ml {
namespace {
bool Shape4(const Tensor& tensor, ElementType type,
            std::array<uint32_t, 4>& shape) {
  Expected<RankedTensorType> ranked = tensor.RankedTensorType();
  if (!ranked || ranked->ElementType() != type ||
      ranked->Layout().HasStrides()) {
    return false;
  }
  absl::Span<const int32_t> dims = ranked->Layout().Dimensions();
  if (dims.size() != 4) return false;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) return false;
    shape[i] = dims[i];
  }
  return true;
}

bool Scales(const Tensor& tensor, uint32_t channels, int axis,
            std::vector<double>& scales) {
  if (tensor.QTypeId() == QuantizationTypeId::PerTensor) {
    LiteRtQuantizationPerTensor per_tensor = tensor.PerTensorQuantization();
    if (per_tensor.zero_point != 0) return false;
    scales.assign(channels, per_tensor.scale);
  } else if (tensor.QTypeId() == QuantizationTypeId::PerChannel) {
    LiteRtQuantizationPerChannel per_channel = tensor.PerChannelQuantization();
    if (per_channel.quantized_dimension != axis ||
        per_channel.num_channels != channels) {
      return false;
    }
    for (uint32_t i = 0; i < channels; ++i) {
      if (per_channel.zero_points[i] != 0) return false;
      scales.push_back(per_channel.scales[i]);
    }
  } else {
    return false;
  }
  return true;
}
}  // namespace

std::optional<Conv2dPlan> AnalyzeConv2d(const Op& op, std::string& reason) {
  auto reject = [&](const char* message) -> std::optional<Conv2dPlan> {
    reason = message;
    return std::nullopt;
  };
  const bool depthwise = op.Code() == kLiteRtOpCodeTflDepthwiseConv2d;
  if (op.Code() != kLiteRtOpCodeTflConv2d && !depthwise) {
    return reject("not convolution");
  }
  OpInputs inputs = op.Inputs();
  OpOutputs outputs = op.Outputs();
  if (inputs.size() != 3 || outputs.size() != 1) {
    return reject("CONV_2D needs input, constant weights, and constant bias");
  }
  Conv2dSpec spec{};
  if (!Shape4(inputs[0], ElementType::Int8, spec.input_shape) ||
      !Shape4(inputs[1], ElementType::Int8, spec.weight_shape) ||
      !Shape4(outputs[0], ElementType::Int8, spec.output_shape)) {
    return reject("CONV_2D requires dense static rank-4 int8 tensors");
  }
  spec.depthwise = depthwise;
  if (depthwise) {
    if (spec.weight_shape[0] != 1 ||
        spec.weight_shape[3] % spec.input_shape[3] != 0) {
      return reject(
          "Depthwise output channels must be a multiple of input channels");
    }
    spec.weight_shape = {spec.weight_shape[3], spec.weight_shape[1],
                         spec.weight_shape[2], 1};
  }
  const Tensor& bias = inputs[2];
  Expected<RankedTensorType> bias_type = bias.RankedTensorType();
  if (!bias_type || bias_type->ElementType() != ElementType::Int32 ||
      bias_type->Layout().HasStrides() ||
      bias_type->Layout().Dimensions().size() != 1 ||
      bias_type->Layout().Dimensions()[0] != spec.weight_shape[0]) {
    return reject("CONV_2D requires a dense int32 bias vector");
  }
  if (inputs[0].QTypeId() != QuantizationTypeId::PerTensor ||
      outputs[0].QTypeId() != QuantizationTypeId::PerTensor) {
    return reject("CONV_2D activation quantization must be per-tensor");
  }
  LiteRtQuantizationPerTensor input_quantization =
      inputs[0].PerTensorQuantization();
  LiteRtQuantizationPerTensor output_quantization =
      outputs[0].PerTensorQuantization();
  if (input_quantization.zero_point < -128 ||
      input_quantization.zero_point > 127 ||
      output_quantization.zero_point < -128 ||
      output_quantization.zero_point > 127) {
    return reject("CONV_2D activation zero point is outside int8");
  }
  spec.input_scale = input_quantization.scale;
  spec.output_scale = output_quantization.scale;
  spec.input_zero_point = input_quantization.zero_point;
  spec.output_zero_point = output_quantization.zero_point;
  if (!Scales(inputs[1], spec.weight_shape[0], depthwise ? 3 : 0,
              spec.weight_scales) ||
      !Scales(bias, spec.weight_shape[0], 0, spec.bias_scales)) {
    return reject(
        "CONV_2D requires symmetric per-tensor or output-channel weights/bias");
  }
  Expected<absl::Span<const int8_t>> weights = inputs[1].WeightsData<int8_t>();
  Expected<absl::Span<const int32_t>> biases = bias.WeightsData<int32_t>();
  if (!weights || !biases) {
    return reject("CONV_2D weights and bias must be valid constants");
  }
  spec.weights.assign(weights->begin(), weights->end());
  spec.bias.assign(biases->begin(), biases->end());
  if (depthwise) {
    const size_t channels = spec.weight_shape[0];
    const size_t kernel = size_t{spec.weight_shape[1]} * spec.weight_shape[2];
    for (size_t channel = 0; channel < channels; ++channel) {
      for (size_t kernel_idx = 0; kernel_idx < kernel; ++kernel_idx) {
        spec.weights[channel * kernel + kernel_idx] =
            (*weights)[kernel_idx * channels + channel];
      }
    }
  }
  uint32_t padding = 0;
  uint32_t activation = 0;
  auto padding_fn = depthwise ? LiteRtGetDepthwiseConv2dPaddingOption
                              : LiteRtGetConv2dPaddingOption;
  auto activation_fn = depthwise ? LiteRtGetDepthwiseConv2dFusedActivationOption
                                 : LiteRtGetConv2dFusedActivationOption;
  auto stride_h = depthwise ? LiteRtGetDepthwiseConv2dStrideHOption
                            : LiteRtGetConv2dStrideHOption;
  auto stride_w = depthwise ? LiteRtGetDepthwiseConv2dStrideWOption
                            : LiteRtGetConv2dStrideWOption;
  auto dilation_h = depthwise ? LiteRtGetDepthwiseConv2dDilationHOption
                              : LiteRtGetConv2dDilationHOption;
  auto dilation_w = depthwise ? LiteRtGetDepthwiseConv2dDilationWOption
                              : LiteRtGetConv2dDilationWOption;
  if (padding_fn(op.Get(), &padding) != kLiteRtStatusOk ||
      activation_fn(op.Get(), &activation) != kLiteRtStatusOk ||
      stride_h(op.Get(), &spec.stride[0]) != kLiteRtStatusOk ||
      stride_w(op.Get(), &spec.stride[1]) != kLiteRtStatusOk ||
      dilation_h(op.Get(), &spec.dilation[0]) != kLiteRtStatusOk ||
      dilation_w(op.Get(), &spec.dilation[1]) != kLiteRtStatusOk) {
    return reject("Convolution option lookup failed");
  }
  if (depthwise) {
    int32_t multiplier = 0;
    if (LiteRtGetDepthwiseConv2dDepthMultiplierOption(op.Get(), &multiplier) !=
            kLiteRtStatusOk ||
        multiplier <= 0 ||
        int64_t{multiplier} * spec.input_shape[3] != spec.output_shape[3]) {
      return reject("Depthwise multiplier disagrees with channel dimensions");
    }
  }
  switch (activation) {
    case 0:
      spec.activation = Activation::kNone;
      break;
    case 1:
      spec.activation = Activation::kRelu;
      break;
    case 3:
      spec.activation = Activation::kRelu6;
      break;
    default:
      return reject("CONV_2D unsupported fused activation");
  }
  if (padding > 1) return reject("CONV_2D unknown padding");
  for (int axis = 0; axis < 2; ++axis) {
    if (spec.stride[axis] <= 0 || spec.dilation[axis] <= 0) {
      return reject("CONV_2D invalid stride/dilation");
    }
    if (padding == 0) {
      const int64_t expected =
          (int64_t{spec.input_shape[axis + 1]} + spec.stride[axis] - 1) /
          spec.stride[axis];
      if (spec.output_shape[axis + 1] != expected) {
        return reject("CONV_2D SAME output shape is not ceil(input/stride)");
      }
      int64_t kernel =
          (int64_t{spec.weight_shape[axis + 1]} - 1) * spec.dilation[axis] + 1;
      int64_t total =
          std::max(int64_t{0}, (int64_t{spec.output_shape[axis + 1]} - 1) *
                                       spec.stride[axis] +
                                   kernel - spec.input_shape[axis + 1]);
      if (total > INT32_MAX) return reject("CONV_2D padding overflow");
      spec.padding[2 * axis] = total / 2;
      spec.padding[2 * axis + 1] = total - total / 2;
    }
  }
  return AnalyzeConv2d(std::move(spec), reason);
}
}  // namespace litert::arm_vulkan_ml
