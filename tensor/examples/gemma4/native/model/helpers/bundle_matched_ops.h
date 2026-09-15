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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_BUNDLE_MATCHED_OPS_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_BUNDLE_MATCHED_OPS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xnnpack.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/examples/gemma4/native/model/helpers/mobile_fully_connected.h"

namespace litert::tensor::graph {
struct BundleDynamicInt2FullyConnectedOperation : Operation {
  absl::string_view GetName() const override {
    return "BundleDynamicInt2FullyConnected";
  }
  LRT_TENSOR_DEFINE_OPERATION_TYPE_IDENTIFICATION
};
template <>
class OpMixin<BundleDynamicInt2FullyConnectedOperation, XnnpackMixinTag>
    : public XnnpackOperation {
public:
  absl::Status ToXnnpack(const Operation &op,
                         XnnpackBuildContext &ctx) const override;
};
} // namespace litert::tensor::graph

namespace litert::tensor::examples::gemma4::native {

inline absl::Status
ValidateBundleDynamicInt2FullyConnected(const TensorHandle &input,
                                        const TensorHandle &weights) {
  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateFloatActivationFullyConnected(input, weights));
  if (weights.GetType() != Type::kI4 || weights.GetShape()[1] % 4 != 0) {
    return absl::InvalidArgumentError(
        "Bundle INT2 head requires signed INT4 container and input channels "
        "divisible by four");
  }
  return absl::OkStatus();
}

// The bundle exporter widens signed two-bit weights into signed INT4 tensor
// storage because the tensor frontend has no INT2 Type. Restore the exact
// original two-bit payload; do not use an INT4 kernel or FP32 dequantization.
// Codes outside [-2,1] are rejected during lowering, before XNNPACK reads them.
// The input conversion and FC definitions match the CPU delegate's hybrid FC:
// FP32 -> qdint8 (one quantization group per last-dimension row) -> qcint2 FC
// -> FP32. XNNPACK may optimize this subgraph just as it optimizes the
// delegate.
template <class... Mixins>
Tensor<Mixins...> BundleDynamicInt2FullyConnected(
    Tensor<Mixins...> input, Tensor<Mixins...> weights,
    source_location loc = source_location::current()) {
  LRT_TENSOR_RETURN_IF_ERROR(
      ValidateBundleDynamicInt2FullyConnected(input, weights));
  auto op = std::make_shared<graph::BundleDynamicInt2FullyConnectedOperation>();
  RegisterMixins<Mixins...>(op);
  AddInputs(op, input, weights);
  Tensor<Mixins...> output = AddOutput(op, loc);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto &info, graph::GetInfo(output.GetRaw()));
  info.shape = input.GetShape();
  info.shape.back() = weights.GetShape()[0];
  info.type = Type::kFP32;
  graph::OpDebugger::DebugOp(*op);
  return output;
}

// Input is already multiplied by sqrt(embed_dim), as required by the bundle.
// The one global INT8 FC produces [..., num_layers * per_layer_dim]. Split only
// after its static output requantization, preserving that full-matrix boundary.
template <class... Mixins>
std::vector<Tensor<Mixins...>> BundlePerLayerProjectionParts(
    Tensor<Mixins...> scaled_embedding, Tensor<Mixins...> full_weight,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>> &weights,
    int num_layers, int per_layer_dim) {
  auto fail = [&](const std::string &message) {
    return std::vector<Tensor<Mixins...>>(
        std::max(1, num_layers), Tensor<Mixins...>(graph::ErrorTensor(
                                     absl::InvalidArgumentError(message))));
  };
  if (num_layers <= 0 || per_layer_dim <= 0 ||
      num_layers > std::numeric_limits<int>::max() / per_layer_dim ||
      scaled_embedding.GetShape().empty() ||
      full_weight.GetShape().size() != 2 ||
      full_weight.GetType() != Type::kI8 ||
      full_weight.GetShape()[0] != num_layers * per_layer_dim ||
      full_weight.GetShape()[1] != scaled_embedding.GetShape().back() ||
      full_weight.GetName() != "model.per_layer_model_projection.weight") {
    return fail("Bundle global PLE requires the named full INT8 matrix with "
                "shape [num_layers * per_layer_dim, embed_dim]");
  }
  if (!weights.contains("model.per_layer_model_projection.input_scale") ||
      !weights.contains("model.per_layer_model_projection.output_scale")) {
    return fail("Bundle global PLE requires both original activation scales");
  }
  Tensor<Mixins...> projected =
      MobileFullyConnected(scaled_embedding, full_weight, &weights);
  Shape expanded_shape = scaled_embedding.GetShape();
  expanded_shape.back() = num_layers;
  expanded_shape.push_back(per_layer_dim);
  Tensor<Mixins...> expanded = Reshape(projected, expanded_shape);
  Shape part_shape = scaled_embedding.GetShape();
  part_shape.back() = per_layer_dim;
  Shape offsets(expanded_shape.size(), 0);
  Shape sizes = expanded_shape;
  sizes[sizes.size() - 2] = 1;
  std::vector<Tensor<Mixins...>> parts;
  parts.reserve(num_layers);
  for (int layer = 0; layer < num_layers; ++layer) {
    offsets[offsets.size() - 2] = layer;
    parts.push_back(Reshape(Slice(expanded, offsets, sizes), part_shape));
  }
  return parts;
}
} // namespace litert::tensor::examples::gemma4::native

namespace litert::tensor::graph {
inline absl::Status
OpMixin<BundleDynamicInt2FullyConnectedOperation, XnnpackMixinTag>::ToXnnpack(
    const Operation &op, XnnpackBuildContext &ctx) const {
  if (op.inputs.size() != 2) {
    return absl::InvalidArgumentError("Bundle INT2 FC expects two inputs");
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      examples::gemma4::native::ValidateBundleDynamicInt2FullyConnected(
          TensorHandle(op.inputs[0]), TensorHandle(op.inputs[1])));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto outputs, GetOutputs(op));
  if (outputs.size() != 1) {
    return absl::InvalidArgumentError("Bundle INT2 FC expects one output");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto &in, GetInfo(op.inputs[0]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto &weight, GetInfo(op.inputs[1]));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto &out, GetInfo(outputs[0]));
  auto expected_shape = in.shape;
  expected_shape.back() = weight.shape[0];
  if (out.type != Type::kFP32 || out.quantization ||
      out.shape != expected_shape) {
    return absl::InvalidArgumentError("Bundle INT2 FC output must be FP32 "
                                      "and preserve leading dimensions");
  }
  const size_t elements =
      static_cast<size_t>(weight.shape[0]) * weight.shape[1];
  auto source = weight.buffer->Lock();
  if (source.data() == nullptr || source.size() < elements / 2) {
    return absl::InvalidArgumentError(
        "Bundle INT2 FC weight buffer unavailable");
  }
  const auto *bytes = reinterpret_cast<const uint8_t *>(source.data());
  std::vector<char> int2(elements / 4, 0);
  for (size_t i = 0; i < elements; ++i) {
    const int code = (bytes[i / 2] >> (4 * (i % 2))) & 15;
    const int signed_value = code >= 8 ? code - 16 : code;
    if (signed_value < -2 || signed_value > 1) {
      return absl::InvalidArgumentError(
          "Bundle INT2 FC received a widened weight outside [-2,1]");
    }
    int2[i / 4] = static_cast<char>(
        static_cast<uint8_t>(int2[i / 4]) |
        ((static_cast<uint8_t>(signed_value) & 3) << (2 * (i % 4))));
  }
  auto &packed = ctx.graph().constant_buffers().emplace_back(std::move(int2));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto &q, weight.quantization->As<PerChannelAffineQuantization>());
  auto &scales = ctx.graph().dequantized_buffers().emplace_back(q.scales);
  uint32_t weight_id, dynamic_id;
  const size_t weight_dims[] = {static_cast<size_t>(weight.shape[0]),
                                static_cast<size_t>(weight.shape[1])};
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_channelwise_quantized_tensor_value_v3(
      ctx.subgraph(), xnn_datatype_qcint2, 0, scales.data(), 2, 0, weight_dims,
      packed.data(), XNN_INVALID_VALUE_ID, 0, &weight_id, nullptr));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t input_id, ctx.DefineValue(op.inputs[0]));
  std::vector<size_t> dims(in.shape.begin(), in.shape.end());
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_dynamically_quantized_tensor_value(
      ctx.subgraph(), xnn_datatype_qdint8, dims.size(), 1, dims.data(),
      XNN_INVALID_VALUE_ID, 0, &dynamic_id));
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_unary(
      ctx.subgraph(), xnn_unary_convert, nullptr, input_id, dynamic_id, 0));
  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t output_id, ctx.DefineValue(outputs[0]));
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_fully_connected(
      ctx.subgraph(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), dynamic_id, weight_id,
      XNN_INVALID_VALUE_ID, output_id, 0));
  return absl::OkStatus();
}
} // namespace litert::tensor::graph
#endif // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_BUNDLE_MATCHED_OPS_H_
