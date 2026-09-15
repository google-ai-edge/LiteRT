/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/gemma4/helpers/float_activation_fully_connected.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "xnnpack.h"

namespace litert::tensor::examples::gemma4 {

absl::Status ValidateFloatActivationFullyConnected(
    const TensorHandle& input, const TensorHandle& weights) {
  LRT_TENSOR_RETURN_IF_ERROR(input.GetStatus());
  LRT_TENSOR_RETURN_IF_ERROR(weights.GetStatus());
  if (input.GetType() != Type::kFP32 || input.GetQuantization() != nullptr) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires unquantized FP32 input");
  }
  const auto& input_shape = input.GetShape();
  const auto& weight_shape = weights.GetShape();
  if (input_shape.empty() || weight_shape.size() != 2) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires input rank >= 1 and matrix "
        "weights");
  }
  if (weight_shape[0] <= 1 || weight_shape[1] <= 0 ||
      input_shape.back() != weight_shape[1]) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires matching input channels and "
        "at least two output channels");
  }
  if ((weights.GetType() != Type::kI4 && weights.GetType() != Type::kI8) ||
      weights.GetQuantization() == nullptr ||
      weights.GetBufferPtr() == nullptr) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires constant quantized INT4 or "
        "INT8 weights");
  }
  if (weights.GetType() == Type::kI4 && weight_shape[1] % 2 != 0) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires even INT4 input channels "
        "for portable XNNPACK kernels");
  }
  const size_t rows = weight_shape[0];
  const size_t columns = weight_shape[1];
  if (rows > std::numeric_limits<size_t>::max() / columns) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected weight dimensions overflow size_t");
  }
  const size_t required_bytes = BufferSize(weights.GetType(), rows * columns);
  const auto buffer = weights.GetBufferPtr();
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t buffer_bytes, buffer->ByteSize());
  if (buffer_bytes < required_bytes) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected weight buffer is too small");
  }
  auto source = buffer->Lock();
  if (source.data() == nullptr) {
    return absl::FailedPreconditionError(
        "FloatActivationFullyConnected could not lock weight data");
  }
  if (source.size() < required_bytes) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected locked weight span is too small");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& quantization,
      weights.GetQuantization()->As<PerChannelAffineQuantization>());
  if (quantization.quantized_dimension != 0 ||
      quantization.scales.size() != static_cast<size_t>(weight_shape[0])) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected requires one scale per output channel "
        "on axis 0");
  }
  for (float scale : quantization.scales) {
    if (!std::isfinite(scale) || scale <= 0.0f) {
      return absl::InvalidArgumentError(
          "FloatActivationFullyConnected requires finite positive scales");
    }
  }
  for (int64_t zero_point : quantization.zero_points) {
    if (zero_point != 0) {
      return absl::InvalidArgumentError(
          "FloatActivationFullyConnected requires zero points equal to 0");
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4

namespace litert::tensor::graph {

absl::Status
OpMixin<FloatActivationFullyConnectedOperation, XnnpackMixinTag>::ToXnnpack(
    const Operation& op, XnnpackBuildContext& ctx) const {
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected expects two inputs");
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      examples::gemma4::ValidateFloatActivationFullyConnected(
          TensorHandle(op.inputs[0]), TensorHandle(op.inputs[1])));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, GetOutputs(op));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected expects one output");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& output_info, GetInfo(outputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& input_info, GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& weight_info, GetInfo(op.inputs[1]));
  auto output_shape = input_info.shape;
  output_shape.back() = weight_info.shape[0];
  if (output_info.type != Type::kFP32 || output_info.quantization != nullptr ||
      output_info.shape != output_shape) {
    return absl::InvalidArgumentError(
        "FloatActivationFullyConnected output must preserve leading input "
        "dimensions and use unquantized FP32");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  uint32_t weights_id;
  if (weight_info.type == Type::kI4) {
    // FP32 qc4w kernels subtract the supplied zero point from unsigned
    // nibbles. Tensor INT4 storage uses signed two's-complement nibbles, as
    // consumed by the INT8-activation kernels. Convert only the packed bytes
    // for this operation, preserving the original tensor for other consumers.
    const size_t rows = weight_info.shape[0];
    const size_t columns = weight_info.shape[1];
    const size_t bytes_per_row = columns / 2;
    auto source = weight_info.buffer->Lock();
    if (source.data() == nullptr ||
        source.size() < BufferSize(Type::kI4, rows * columns)) {
      return absl::InvalidArgumentError(
          "FloatActivationFullyConnected INT4 buffer is unavailable or too "
          "small");
    }
    const auto* signed_bytes = reinterpret_cast<const uint8_t*>(source.data());
    auto& offset_bytes = ctx.graph().constant_buffers().emplace_back(
        rows * bytes_per_row, static_cast<char>(0x88));
    for (size_t i = 0; i < offset_bytes.size(); ++i) {
      offset_bytes[i] = static_cast<char>(signed_bytes[i] ^ 0x88);
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const auto& quantization,
        weight_info.quantization->As<PerChannelAffineQuantization>());
    // Both arrays must survive the subgraph and its runtimes. The graph owns
    // these packed bytes and the small scale array; no dense FP32 weights are
    // materialized.
    auto& scales =
        ctx.graph().dequantized_buffers().emplace_back(quantization.scales);
    const size_t dims[] = {rows, columns};
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_channelwise_quantized_tensor_value_v3(
        ctx.subgraph(), xnn_datatype_qcint4, /*zero_point=*/8, scales.data(),
        /*num_dims=*/2, /*channel_dim=*/0, dims, offset_bytes.data(),
        XNN_INVALID_VALUE_ID, /*flags=*/0, &weights_id,
        /*channelwise_zero_point=*/nullptr))
        << "FloatActivationFullyConnected INT4 weights";
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(weights_id, ctx.DefineValue(op.inputs[1]));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(outputs[0]));
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_fully_connected(
      ctx.subgraph(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, weights_id,
      XNN_INVALID_VALUE_ID, output_id, /*flags=*/0))
      << "FloatActivationFullyConnected";
  return absl::OkStatus();
}

}  // namespace litert::tensor::graph
