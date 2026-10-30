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

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace litert::arm_vulkan_ml::tool {
Expected<std::vector<int8_t>> QuantizedReference(
    const tflite::ModelT& model, const std::vector<int8_t>& input) {
  // Independent mathematical references for types absent from local CPU
  // kernels. No production analysis, LUT or fixed-point helpers are shared.
  auto invalid = [] { return Error(kLiteRtStatusErrorInvalidArgument, "Invalid FP64 SQUARE/CONV_3D reference model"); };
  if (model.subgraphs.size() != 1) return invalid();
  const auto& graph = *model.subgraphs[0];
  if (graph.operators.size() != 1 || graph.inputs.size() != 1 || graph.outputs.size() != 1) return invalid();
  const auto& op = *graph.operators[0];
  if (op.opcode_index >= model.operator_codes.size() || op.inputs.empty() || op.outputs.size() != 1 ||
      graph.inputs[0] != op.inputs[0] || graph.outputs[0] != op.outputs[0]) return invalid();
  for (auto id : op.inputs) if (id < 0 || id >= graph.tensors.size()) return invalid();
  if (op.outputs[0] < 0 || op.outputs[0] >= graph.tensors.size()) return invalid();
  const auto& source = *graph.tensors[op.inputs[0]];
  const auto& output = *graph.tensors[op.outputs[0]];
  auto count = [](const tflite::TensorT& tensor) {
    size_t n = 1;
    for (auto d : tensor.shape) { if (d <= 0 || n > (64u * 1024u * 1024u) / d) return size_t{0}; n *= d; }
    return n;
  };
  auto affine = [](const tflite::TensorT& tensor) {
    return tensor.type == tflite::TensorType_INT8 && tensor.quantization &&
        tensor.quantization->scale.size() == 1 && tensor.quantization->zero_point.size() == 1 &&
        std::isfinite(tensor.quantization->scale[0]) && tensor.quantization->scale[0] > 0 &&
        tensor.quantization->zero_point[0] >= -128 && tensor.quantization->zero_point[0] <= 127;
  };
  if (!affine(source) || !affine(output) || !count(source) || count(source) != input.size() || !count(output)) return invalid();
  const double iscale = source.quantization->scale[0], oscale = output.quantization->scale[0];
  const int64_t izp = source.quantization->zero_point[0], ozp = output.quantization->zero_point[0];
  auto quantize = [&](double value) {
    return static_cast<int8_t>(std::clamp(std::round(value / oscale) + ozp, -128.0, 127.0));
  };
  std::vector<int8_t> result(count(output));
  const auto code = model.operator_codes[op.opcode_index]->builtin_code;
  if (code == tflite::BuiltinOperator_SQUARE) {
    if (op.inputs.size() != 1 || source.shape != output.shape) return invalid();
    for (size_t i = 0; i < input.size(); ++i) {
      const double value = (input[i] - izp) * iscale;
      result[i] = quantize(value * value);
    }
    return result;
  }
  if (code != tflite::BuiltinOperator_CONV_3D || op.inputs.size() != 3 ||
      source.shape.size() != 5 || output.shape.size() != 5) return invalid();
  const auto& weight = *graph.tensors[op.inputs[1]];
  const auto& bias = *graph.tensors[op.inputs[2]];
  const auto* options = op.builtin_options.AsConv3DOptions();
  if (!options || weight.type != tflite::TensorType_INT8 || weight.shape.size() != 5 ||
      bias.type != tflite::TensorType_INT32 || bias.shape != std::vector<int32_t>{output.shape[4]} ||
      weight.buffer >= model.buffers.size() || bias.buffer >= model.buffers.size() || !weight.quantization ||
      source.shape[0] != output.shape[0] || weight.shape[3] != source.shape[4] || weight.shape[4] != output.shape[4]) return invalid();
  const auto& wbytes = model.buffers[weight.buffer]->data;
  const auto& bbytes = model.buffers[bias.buffer]->data;
  if (!count(weight) || wbytes.size() != count(weight) || bbytes.size() != count(bias) * sizeof(int32_t)) return invalid();
  const auto& scales = weight.quantization->scale;
  const auto& zero_points = weight.quantization->zero_point;
  const int channels = output.shape[4];
  if ((scales.size() != 1 && (scales.size() != channels || weight.quantization->quantized_dimension != 4)) ||
      zero_points.size() != scales.size()) return invalid();
  for (size_t i = 0; i < scales.size(); ++i)
    if (!std::isfinite(scales[i]) || scales[i] <= 0 || zero_points[i] != 0) return invalid();
  const int32_t stride[] = {options->stride_d, options->stride_h, options->stride_w};
  const int32_t dilation[] = {options->dilation_d_factor, options->dilation_h_factor, options->dilation_w_factor};
  int64_t pad[3];
  if (options->padding != tflite::Padding_SAME && options->padding != tflite::Padding_VALID) return invalid();
  for (int a = 0; a < 3; ++a) {
    if (stride[a] <= 0 || dilation[a] <= 0) return invalid();
    const int64_t kernel = (int64_t{weight.shape[a]} - 1) * dilation[a] + 1;
    pad[a] = options->padding == tflite::Padding_SAME ?
        std::max(int64_t{0}, (int64_t{output.shape[a + 1]} - 1) * stride[a] + kernel - source.shape[a + 1]) / 2 : 0;
  }
  if (options->fused_activation_function != tflite::ActivationFunctionType_NONE &&
      options->fused_activation_function != tflite::ActivationFunctionType_RELU &&
      options->fused_activation_function != tflite::ActivationFunctionType_RELU6) return invalid();
  size_t index = 0;
  for (int n = 0; n < output.shape[0]; ++n)
    for (int d = 0; d < output.shape[1]; ++d)
      for (int h = 0; h < output.shape[2]; ++h)
        for (int w = 0; w < output.shape[3]; ++w)
          for (int c = 0; c < channels; ++c) {
            int32_t bias_value;
            std::memcpy(&bias_value, bbytes.data() + c * sizeof(int32_t), sizeof(int32_t));
            int64_t accumulator = bias_value;
            for (int kd = 0; kd < weight.shape[0]; ++kd)
              for (int kh = 0; kh < weight.shape[1]; ++kh)
                for (int kw = 0; kw < weight.shape[2]; ++kw) {
                  const int64_t id = int64_t{d} * stride[0] + int64_t{kd} * dilation[0] - pad[0];
                  const int64_t ih = int64_t{h} * stride[1] + int64_t{kh} * dilation[1] - pad[1];
                  const int64_t iw = int64_t{w} * stride[2] + int64_t{kw} * dilation[2] - pad[2];
                  if (id < 0 || id >= source.shape[1] || ih < 0 || ih >= source.shape[2] || iw < 0 || iw >= source.shape[3]) continue;
                  for (int ic = 0; ic < source.shape[4]; ++ic) {
                    const size_t si = ((((n * source.shape[1] + id) * source.shape[2] + ih) * source.shape[3] + iw) * source.shape[4] + ic);
                    const size_t wi = ((((kd * weight.shape[1] + kh) * weight.shape[2] + kw) * weight.shape[3] + ic) * channels + c);
                    int8_t coefficient;
                    std::memcpy(&coefficient, wbytes.data() + wi, sizeof(int8_t));
                    accumulator += (input[si] - izp) * coefficient;
                  }
                }
            double value = accumulator * iscale * scales[scales.size() == 1 ? 0 : c];
            if (options->fused_activation_function != tflite::ActivationFunctionType_NONE) value = std::max(0.0, value);
            if (options->fused_activation_function == tflite::ActivationFunctionType_RELU6) value = std::min(6.0, value);
            result[index++] = quantize(value);
          }
  return result;
}
}  // namespace litert::arm_vulkan_ml::tool
