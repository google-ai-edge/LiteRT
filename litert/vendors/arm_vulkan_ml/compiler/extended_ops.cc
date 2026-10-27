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

#include "litert/vendors/arm_vulkan_ml/compiler/extended_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include "litert/c/litert_op_options.h"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::ResId;
using TosaTensor = tfsc::tosa::Tensor;
uint64_t Count(const std::vector<uint32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
}
bool Ints(const Tensor& tensor, std::vector<int32_t>& values, int rank = -1) {
  auto type = tensor.RankedTensorType();
  if (!type || type->ElementType() != ElementType::Int32 || type->Layout().HasStrides()) return false;
  if (rank >= 0 && type->Layout().Dimensions().size() != static_cast<size_t>(rank)) return false;
  auto data = tensor.WeightsData<int32_t>();
  if (!data) return false;
  uint64_t count = 1;
  for (auto dim : type->Layout().Dimensions()) {
    if (dim <= 0 || count > uint64_t{INT32_MAX} / dim) return false;
    count *= dim;
  }
  if (count != data->size()) return false;
  values.assign(data->begin(), data->end());
  return true;
}
template <typename T>
ResId Constant(tfsc::tosa::Graph& graph, const std::vector<T>& data,
               const std::vector<uint32_t>& shape, std::vector<GraphConstant>& constants) {
  TosaTensor type(tfsc::tosa::GetDataType<T>(), shape);
  GraphConstant constant{type, std::vector<uint8_t>(data.size() * sizeof(T))};
  std::memcpy(constant.bytes.data(), data.data(), constant.bytes.size());
  constants.push_back(std::move(constant));
  return graph.AddGraphConstant(type);
}
}  // namespace

bool IsExtendedOperation(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflGather || code == kLiteRtOpCodeTflReduceMax ||
      code == kLiteRtOpCodeTflReduceMin || code == kLiteRtOpCodeTflResizeBilinear ||
      code == kLiteRtOpCodeTflResizeNearestNeighbor || code == kLiteRtOpCodeTflTransposeConv;
}

bool AnalyzeExtendedOperation(const Op& op, OperationPlan& p, std::string& reason) {
  auto reject = [&](const char* message) { reason = message; return false; };
  const auto in = op.Inputs();
  const auto& input = p.inputs[0];
  const auto& output = p.output;
  if (input.floating != output.floating) return reject("Extended op requires matching element types");
  if (!input.floating && (input.scale != output.scale || input.zero_point != output.zero_point))
    return reject("Extended op requires preserved quantization");
  if (p.code == kLiteRtOpCodeTflGather) {
    int32_t batch_dims;
    if (in.size() != 2 || !Ints(in[1], p.parameters) ||
        LiteRtGetGatherAxisOption(op.Get(), &p.axis) != kLiteRtStatusOk ||
        LiteRtGetGatherBatchDimsOption(op.Get(), &batch_dims) != kLiteRtStatusOk || batch_dims != 0)
      return reject("GATHER requires constant INT32 indices and batch_dims=0");
    if (p.axis < 0) p.axis += static_cast<int32_t>(input.shape.size());
    if (p.axis < 0 || p.axis >= input.shape.size()) return reject("Invalid GATHER axis");
    for (int32_t index : p.parameters)
      if (index < 0 || index >= input.shape[p.axis]) return reject("GATHER index out of range");
    auto indices = in[1].RankedTensorType();
    std::vector<uint32_t> shape(input.shape.begin(), input.shape.begin() + p.axis);
    for (auto dim : indices->Layout().Dimensions()) shape.push_back(static_cast<uint32_t>(dim));
    shape.insert(shape.end(), input.shape.begin() + p.axis + 1, input.shape.end());
    if (shape != output.shape) return reject("GATHER output shape mismatch");
  } else if (p.code == kLiteRtOpCodeTflReduceMax || p.code == kLiteRtOpCodeTflReduceMin) {
    bool keep;
    auto get_keep = p.code == kLiteRtOpCodeTflReduceMax ? LiteRtGetReduceMaxKeepDimsOption : LiteRtGetReduceMinKeepDimsOption;
    if (in.size() != 2 || !Ints(in[1], p.parameters) || p.parameters.empty() ||
        get_keep(op.Get(), &keep) != kLiteRtStatusOk) return reject("Reduction requires constant nonempty INT32 axes");
    std::set<int32_t> axes;
    for (auto axis : p.parameters) {
      if (axis < 0) axis += static_cast<int32_t>(input.shape.size());
      if (axis < 0 || axis >= input.shape.size()) return reject("Reduction axis out of range");
      axes.insert(axis);
    }
    p.parameters.assign(axes.begin(), axes.end());
    std::vector<uint32_t> shape;
    for (size_t axis = 0; axis < input.shape.size(); ++axis) {
      if (!axes.count(static_cast<int32_t>(axis))) shape.push_back(input.shape[axis]);
      else if (keep) shape.push_back(1);
    }
    if (shape != output.shape) return reject("Reduction output shape mismatch");
  } else if (p.code == kLiteRtOpCodeTflResizeNearestNeighbor || p.code == kLiteRtOpCodeTflResizeBilinear) {
    const bool nearest = p.code == kLiteRtOpCodeTflResizeNearestNeighbor;
    bool align, half;
    auto get_align = nearest ? LiteRtGetResizeNearestNeighborAlignCornersOption : LiteRtGetResizeBilinearAlignCornersOption;
    auto get_half = nearest ? LiteRtGetResizeNearestNeighborHalfPixelCenterOption : LiteRtGetResizeBilinearHalfPixelCenterOption;
    if (in.size() != 2 || input.shape.size() != 4 || output.shape.size() != 4 ||
        !Ints(in[1], p.parameters, 1) || p.parameters.size() != 2 ||
        get_align(op.Get(), &align) != kLiteRtStatusOk || get_half(op.Get(), &half) != kLiteRtStatusOk ||
        (align && half) || (!nearest && !input.floating))
      return reject("RESIZE requires static NHWC sizes; bilinear currently requires FP32");
    if (input.shape[0] != output.shape[0] || input.shape[3] != output.shape[3] ||
        p.parameters[0] != output.shape[1] || p.parameters[1] != output.shape[2])
      return reject("RESIZE shape mismatch");
    p.parameters.clear(); p.padding.clear(); p.kernel.clear();
    for (int axis : {1, 2}) {
      const int32_t source = static_cast<int32_t>(input.shape[axis]);
      const int32_t target = static_cast<int32_t>(output.shape[axis]);
      if (align && (source == 1 || target == 1)) return reject("Aligned resize singleton dimensions not supported");
      int32_t n = align ? target - 1 : target;
      int32_t d = align ? source - 1 : source;
      const int32_t divisor = std::gcd(n, d);
      n = 2 * (n / divisor); d = 2 * (d / divisor);
      int32_t offset = half ? (d - n) / 2 : 0;
      if (nearest && !align) offset = (half ? d / 2 : 0) - n / 2;
      const int64_t border = int64_t{target - 1} * d - int64_t{source - 1} * n + offset;
      if (n > 2048 || n > 256 * d || d >= 16 * n || offset < -n || offset >= 16 * n ||
          border < -16 * n || border >= n) return reject("RESIZE scale/offset/border outside TOSA limits");
      p.parameters.push_back(n); p.parameters.push_back(d);
      p.padding.push_back(offset); p.kernel.push_back(static_cast<int32_t>(border));
    }
  } else {
    if (!input.floating || in.size() < 3 || in.size() > 4 || input.shape.size() != 4 || output.shape.size() != 4)
      return reject("TRANSPOSE_CONV currently requires FP32 NHWC input and constant weights");
    if (!Ints(in[0], p.parameters, 1) || p.parameters.size() != 4 ||
        !std::equal(p.parameters.begin(), p.parameters.end(), output.shape.begin()))
      return reject("TRANSPOSE_CONV requires constant matching output shape");
    auto weights = in[1].RankedTensorType();
    auto data = in[1].WeightsData<float>();
    if (!weights || weights->ElementType() != ElementType::Float32 || weights->Layout().HasStrides() ||
        weights->Layout().Dimensions().size() != 4 || !data) return reject("Invalid TRANSPOSE_CONV weights");
    for (auto dim : weights->Layout().Dimensions()) {
      if (dim <= 0 || dim > 8192) return reject("Invalid TRANSPOSE_CONV weight dimensions");
      p.weight_shape.push_back(static_cast<uint32_t>(dim));
    }
    if (Count(p.weight_shape) != data->size() || p.weight_shape[0] != output.shape[3] ||
        p.weight_shape[3] != input.shape[3] || input.shape[0] != output.shape[0])
      return reject("TRANSPOSE_CONV channels/batch mismatch");
    p.float_weights.assign(data->begin(), data->end());
    p.float_bias.assign(output.shape[3], 0.0f);
    if (in.size() == 4) {
      auto bias_type = in[3].RankedTensorType();
      auto bias = in[3].WeightsData<float>();
      if (!bias_type || bias_type->ElementType() != ElementType::Float32 || bias_type->Layout().HasStrides() ||
          bias_type->Layout().Dimensions().size() != 1 || bias_type->Layout().Dimensions()[0] != output.shape[3] ||
          !bias || bias->size() != output.shape[3])
        return reject("Invalid TRANSPOSE_CONV bias");
      p.float_bias.assign(bias->begin(), bias->end());
    }
    for (float value : p.float_weights) if (!std::isfinite(value)) return reject("Nonfinite TRANSPOSE_CONV weights");
    for (float value : p.float_bias) if (!std::isfinite(value)) return reject("Nonfinite TRANSPOSE_CONV bias");
    uint32_t padding;
    p.stride.resize(2);
    if (LiteRtGetTransposeConvPaddingOption(op.Get(), &padding) != kLiteRtStatusOk || padding > 1 ||
        LiteRtGetTransposeConvStrideHOption(op.Get(), &p.stride[0]) != kLiteRtStatusOk ||
        LiteRtGetTransposeConvStrideWOption(op.Get(), &p.stride[1]) != kLiteRtStatusOk ||
        LiteRtGetTransposeConvFusedActivationOption(op.Get(), &p.float_activation) != kLiteRtStatusOk ||
        (p.float_activation != 0 && p.float_activation != 1 && p.float_activation != 3))
      return reject("Unsupported TRANSPOSE_CONV options");
    for (int axis = 0; axis < 2; ++axis) {
      const int64_t k = p.weight_shape[axis + 1], stride = p.stride[axis];
      if (stride <= 0 || stride > k) return reject("TRANSPOSE_CONV requires stride <= kernel");
      const int64_t full = (input.shape[axis + 1] - 1) * stride + k;
      const int64_t expected = padding == 0 ? input.shape[axis + 1] * stride : full;
      if (output.shape[axis + 1] != expected) return reject("TRANSPOSE_CONV output geometry mismatch");
      const int32_t crop = static_cast<int32_t>(full - expected);
      p.padding.push_back(-crop / 2); p.padding.push_back(-(crop - crop / 2));
    }
  }
  reason.clear();
  return true;
}

ResId EmitExtendedOperation(tfsc::tosa::Graph& g, ResId input, const OperationPlan& p,
                            std::vector<GraphConstant>& constants) {
  const auto type = p.inputs[0].floating ? DataType::float32_t : DataType::int8_t;
  TosaTensor output(type, p.output.shape);
  auto reshape = [&](ResId value, const std::vector<uint32_t>& shape) {
    return g.AddReshapeOperator(value, g.AddTensorConstant(Attribute(shape)), TosaTensor(type, shape));
  };
  if (p.code == kLiteRtOpCodeTflGather) {
    uint32_t batches = 1, channels = 1;
    const auto& shape = p.inputs[0].shape;
    for (int i = 0; i < p.axis; ++i) batches *= shape[i];
    for (size_t i = p.axis + 1; i < shape.size(); ++i) channels *= shape[i];
    const auto width = static_cast<uint32_t>(p.parameters.size());
    std::vector<int32_t> indices;
    for (uint32_t batch = 0; batch < batches; ++batch)
      indices.insert(indices.end(), p.parameters.begin(), p.parameters.end());
    input = reshape(input, {batches, shape[p.axis], channels});
    auto result = g.AddGatherOperator(input, Constant(g, indices, {batches, width}, constants),
                                      TosaTensor(type, {batches, width, channels}));
    return reshape(result, p.output.shape);
  }
  if (p.code == kLiteRtOpCodeTflReduceMax || p.code == kLiteRtOpCodeTflReduceMin) {
    auto shape = p.inputs[0].shape;
    for (auto axis : p.parameters) {
      shape[axis] = 1;
      input = p.code == kLiteRtOpCodeTflReduceMax
          ? g.AddReduceMaxOperator(input, Attribute(axis), Attribute(int32_t{1}), TosaTensor(type, shape))
          : g.AddReduceMinOperator(input, Attribute(axis), Attribute(int32_t{1}), TosaTensor(type, shape));
    }
    return reshape(input, p.output.shape);
  }
  if (p.code == kLiteRtOpCodeTflResizeNearestNeighbor || p.code == kLiteRtOpCodeTflResizeBilinear) {
    return g.AddResizeOperator(input, g.AddTensorConstant(Attribute(p.parameters)),
        g.AddTensorConstant(Attribute(p.padding)), g.AddTensorConstant(Attribute(p.kernel)),
        Attribute(int32_t{p.code == kLiteRtOpCodeTflResizeNearestNeighbor ? 1 : 2}), output);
  }
  auto zero = g.AddTensorConstant(Attribute(0.0f));
  auto result = g.AddTransposeConv2dOperator(input,
      Constant(g, p.float_weights, p.weight_shape, constants),
      Constant(g, p.float_bias, {p.output.shape[3]}, constants), zero, zero,
      Attribute(p.padding), Attribute(p.stride), Attribute(int32_t{3}), Attribute(false), output);
  if (p.float_activation) result = g.AddClampOperator(result, Attribute(0.0f),
      Attribute(p.float_activation == 3 ? 6.0f : std::numeric_limits<float>::infinity()),
      Attribute(int32_t{1}), output);
  return result;
}
}  // namespace litert::arm_vulkan_ml
