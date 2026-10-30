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

#include "litert/vendors/arm_vulkan_ml/compiler/decomposition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>

#include "TosaForSpirvCodegen.hpp"
#include "litert/c/litert_op_options.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::ResId;
using TT = tfsc::tosa::Tensor;
using Shape = std::vector<uint32_t>;

uint64_t Count(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), uint64_t{1},
                         std::multiplies<uint64_t>());
}
bool Read(const Tensor& tensor, DecompositionTensor& info, bool raw = false) {
  const auto type = tensor.RankedTensorType();
  if (!type || type->Layout().HasStrides()) return false;
  switch (tensor.ElementType()) {
    case ElementType::Int8: info.type = DataType::int8_t; break;
    case ElementType::Float32: info.type = DataType::float32_t; break;
    case ElementType::Complex64: info.type = DataType::float32_t; info.complex = true; break;
    case ElementType::Int32: info.type = DataType::int32_t; break;
    case ElementType::Bool: info.type = DataType::bool_t; break;
    default: return false;
  }
  uint64_t count = 1;
  for (int32_t dim : type->Layout().Dimensions()) {
    if (dim <= 0 || dim > 8192 || count > (64u * 1024u * 1024u) / dim)
      return false;
    count *= dim;
    info.shape.push_back(static_cast<uint32_t>(dim));
  }
  if (info.shape.size() > 6) return false;
  info.scalar = info.shape.empty();
  if (info.complex) {
    if (info.shape.size() >= 6 || count > 32u * 1024u * 1024u) return false;
    info.shape.push_back(2);
  } else if (info.scalar) info.shape = {1};
  if (info.type == DataType::int8_t &&
      (!raw || tensor.QTypeId() == QuantizationTypeId::PerTensor)) {
    if (tensor.QTypeId() != QuantizationTypeId::PerTensor) return false;
    const auto q = tensor.PerTensorQuantization();
    if (!std::isfinite(q.scale) || q.scale <= 0 ||
        q.zero_point < -128 || q.zero_point > 127) return false;
    info.scale = q.scale;
    info.zero_point = q.zero_point;
  }
  return true;
}
bool Ints(const Tensor& tensor, std::vector<int32_t>& values) {
  auto type = tensor.RankedTensorType();
  auto data = tensor.WeightsData<int32_t>();
  if (!type || tensor.ElementType() != ElementType::Int32 ||
      type->Layout().HasStrides() || !data || data->empty()) return false;
  uint64_t size = 1;
  for (int32_t dim : type->Layout().Dimensions()) {
    if (dim <= 0 || size > 8192u / static_cast<uint32_t>(dim)) return false;
    size *= dim;
  }
  if (size != data->size()) return false;
  values.assign(data->begin(), data->end());
  return true;
}
bool SameQuant(const DecompositionTensor& a, const DecompositionTensor& b) {
  return a.type == b.type && a.complex == b.complex && a.scale == b.scale && a.zero_point == b.zero_point;
}
bool SymmetricScales(const Tensor& tensor, uint32_t count, int axis,
                     std::vector<double>& result) {
  if (tensor.QTypeId() == QuantizationTypeId::PerTensor) {
    auto q = tensor.PerTensorQuantization();
    if (q.zero_point != 0 || !std::isfinite(q.scale) || q.scale <= 0) return false;
    result.assign(count, q.scale);
    return true;
  }
  if (tensor.QTypeId() != QuantizationTypeId::PerChannel) return false;
  auto q = tensor.PerChannelQuantization();
  if (q.quantized_dimension != axis || q.num_channels != count) return false;
  for (uint32_t i = 0; i < count; ++i) {
    if (q.zero_points[i] != 0 || !std::isfinite(q.scales[i]) || q.scales[i] <= 0) return false;
    result.push_back(q.scales[i]);
  }
  return true;
}
template <typename T>
bool ConstantScalar(const Tensor& tensor, T& value) {
  auto data = tensor.WeightsData<T>();
  auto type = tensor.RankedTensorType();
  if (!data || data->size() != 1 || !type || type->Layout().HasStrides()) return false;
  for (auto dim : type->Layout().Dimensions()) if (dim != 1) return false;
  value = (*data)[0];
  return true;
}
bool Broadcast(const std::vector<Shape>& inputs, Shape& result) {
  size_t rank = 1;
  for (const auto& shape : inputs) rank = std::max(rank, shape.size());
  if (rank > 6) return false;
  result.assign(rank, 1);
  for (auto shape : inputs) {
    shape.insert(shape.begin(), rank - shape.size(), 1);
    for (size_t i = 0; i < rank; ++i) {
      if (shape[i] != 1 && result[i] != 1 && shape[i] != result[i]) return false;
      result[i] = std::max(result[i], shape[i]);
    }
  }
  return Count(result) <= 64u * 1024u * 1024u;
}
ResId Reshape(tfsc::tosa::Graph& g, ResId value, DataType type,
              const Shape& shape) {
  return g.AddReshapeOperator(value, g.AddTensorConstant(Attribute(shape)),
                               TT(type, shape));
}
template <typename T>
ResId Scalar(tfsc::tosa::Graph& g, T value, size_t rank) {
  return Reshape(g, g.AddTensorConstant(Attribute(value)),
                 tfsc::tosa::GetDataType<T>(), Shape(rank, 1));
}
template <typename T>
ResId Vector(tfsc::tosa::Graph& g, const std::vector<T>& values,
             const Shape& shape) {
  return Reshape(g, g.AddTensorConstant(Attribute(values)),
                 tfsc::tosa::GetDataType<T>(), shape);
}
template <typename T>
ResId ExternalConstant(tfsc::tosa::Graph& g, const std::vector<T>& values,
                       const Shape& shape, GraphConstantPool& constants) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return AddGraphConstant(g, TT(tfsc::tosa::GetDataType<T>(), shape), std::move(bytes), constants);
}
}  // namespace

std::shared_ptr<DecompositionPlan> AnalyzeDecomposition(
    const Op& op, std::string& reason) {
  auto reject = [&](const char* message) -> std::shared_ptr<DecompositionPlan> {
    reason = message;
    return nullptr;
  };
  auto p = std::make_shared<DecompositionPlan>();
  p->code = op.Code();
  const auto in = op.Inputs();
  const auto out = op.Outputs();
  const auto* registration = FindOperationRegistration(p->code);
  const bool raw = p->code == kLiteRtOpCodeTflCast;
  if (!registration || out.size() != 1 || registration->data_input >= in.size())
    return reject("Missing registered decomposition operands");
  const auto* variant = FindOperationVariant(*registration,
      static_cast<LiteRtElementType>(in[registration->data_input].ElementType()),
      static_cast<LiteRtElementType>(out[0].ElementType()));
  if (!variant || variant->route != LoweringRoute::kDecomposed ||
      !Read(in[registration->data_input], p->data, raw) ||
      !Read(out[0], p->output, raw)) return reject("Invalid decomposition tensor types/shapes");
  auto add = [&](int index) {
    DecompositionTensor info;
    if (index < 0 || index >= in.size() || !Read(in[index], info, raw)) return false;
    p->runtime_inputs.push_back(index);
    p->inputs.push_back(std::move(info));
    return true;
  };
  auto same = [&] { return p->data.shape == p->output.shape && p->data.scalar == p->output.scalar; };
  auto unary = [&] { return in.size() == 1 && same() && add(0); };
  const auto code = p->code;
  const auto type = p->data.type;
  const auto& shape = p->data.shape;
  const auto& output = p->output.shape;
  if (code == kLiteRtOpCodeTflSelect || code == kLiteRtOpCodeTflSelectV2) {
    if (in.size() != 3 || !add(0) || !add(1) || !add(2) ||
        p->inputs[0].type != DataType::bool_t ||
        !SameQuant(p->inputs[1], p->output) || !SameQuant(p->inputs[2], p->output))
      return reject("SELECT requires BOOL condition and matching data types/quantization");
    // TFLite explicitly preserves the declared output shape when all four
    // tensors contain one element, including mixed scalar/vector ranks.
    if (Count(p->inputs[0].shape) == 1 && Count(p->inputs[1].shape) == 1 &&
        Count(p->inputs[2].shape) == 1 && Count(output) == 1) {
      for (auto& tensor : p->inputs) tensor.shape.assign(output.size(), 1);
      reason.clear();
      return p;
    }
    if (code == kLiteRtOpCodeTflSelect &&
        (p->inputs[1].shape != p->inputs[2].shape ||
         p->inputs[1].scalar != p->inputs[2].scalar))
      return reject("Legacy SELECT requires identical data shapes");
    if (code == kLiteRtOpCodeTflSelect && p->inputs[0].shape.size() == 1 &&
        p->inputs[0].shape[0] == shape[0] && !p->inputs[0].scalar)
      p->inputs[0].shape.resize(shape.size(), 1);
    if (code == kLiteRtOpCodeTflSelect && !p->inputs[0].scalar && p->inputs[0].shape != shape &&
        !(p->inputs[0].shape[0] == shape[0] &&
          std::all_of(p->inputs[0].shape.begin() + 1, p->inputs[0].shape.end(),
                      [](uint32_t dim) { return dim == 1; })))
      return reject("Legacy SELECT condition must match data or its first dimension");
    Shape expected;
    if (!Broadcast({p->inputs[0].shape, p->inputs[1].shape, p->inputs[2].shape}, expected) ||
        expected != output || p->output.scalar !=
            (p->inputs[0].scalar && p->inputs[1].scalar && p->inputs[2].scalar))
      return reject("SELECT broadcast/output shape mismatch");
  } else if (code == kLiteRtOpCodeTflGreater || code == kLiteRtOpCodeTflGreaterEqual ||
             code == kLiteRtOpCodeTflLess || code == kLiteRtOpCodeTflLessEqual) {
    Shape expected;
    if (in.size() != 2 || !add(0) || !add(1) ||
        !SameQuant(p->inputs[0], p->inputs[1]) ||
        p->output.type != DataType::bool_t ||
        !Broadcast({shape, p->inputs[1].shape}, expected) || expected != output ||
        p->output.scalar != (p->data.scalar && p->inputs[1].scalar))
      return reject("Comparison requires matching operand types/quantization and static broadcasting");
  } else if (code == kLiteRtOpCodeTflPrelu) {
    Shape expected;
    if (in.size() != 2 || !add(0) || !add(1) || !same() ||
        p->inputs[1].type != type || p->inputs[1].complex != p->data.complex || p->output.type != type ||
        (p->data.scalar && !p->inputs[1].scalar) ||
        !Broadcast({shape, p->inputs[1].shape}, expected) || expected != shape)
      return reject("PRELU requires compatible alpha broadcasting and matching types");
    if (type == DataType::int8_t) {
      const double scales[] = {p->data.scale / p->output.scale,
          p->data.scale * p->inputs[1].scale / p->output.scale};
      const AccumulatorRange ranges[] = {{-128 - p->data.zero_point, 127 - p->data.zero_point},
                                         {-65025, 65025}};
      for (int i = 0; i < 2; ++i) {
        auto scale = MakeScale32(scales[i]);
        if (!scale || !IsScale32RangeValid(*scale, ranges[i].minimum, ranges[i].maximum))
          return reject("PRELU rescale range is not proven");
        p->scales.push_back(*scale);
        p->ranges.push_back(ranges[i]);
      }
    }
  } else if (code == kLiteRtOpCodeTflLeakyRelu || code == kLiteRtOpCodeTflReluN1To1 ||
             code == kLiteRtOpCodeTflRelu0To1 || code == kLiteRtOpCodeTflExp ||
             code == kLiteRtOpCodeTflLog || code == kLiteRtOpCodeTflRsqrt ||
             code == kLiteRtOpCodeTflGelu || code == kLiteRtOpCodeTflSquare) {
    if (!unary() || p->output.type != type) return reject("Activation requires matching shapes/types");
    float alpha = 0;
    if (code == kLiteRtOpCodeTflLeakyRelu &&
        (LiteRtGetLeakyReluAlphaOption(op.Get(), &alpha) != kLiteRtStatusOk || !std::isfinite(alpha)))
      return reject("Invalid LEAKY_RELU alpha");
    p->coefficients = {alpha};
    bool approximate = false;
    if (code == kLiteRtOpCodeTflGelu &&
        LiteRtGetGeluApproximateOption(op.Get(), &approximate) != kLiteRtStatusOk)
      return reject("Missing GELU options");
    p->flag = approximate;
    if (type == DataType::int8_t) {
      if ((code == kLiteRtOpCodeTflLog || code == kLiteRtOpCodeTflRsqrt) &&
          p->data.zero_point != -128)
        return reject("INT8 LOG/RSQRT currently require a nonnegative representable input domain");
      for (int value = -128; value <= 127; ++value) {
        const double x = (value - p->data.zero_point) * p->data.scale;
        double y = 0;
        if (code == kLiteRtOpCodeTflLeakyRelu) y = x >= 0 ? x : alpha * x;
        else if (code == kLiteRtOpCodeTflReluN1To1) y = std::clamp(x, -1.0, 1.0);
        else if (code == kLiteRtOpCodeTflRelu0To1) y = std::clamp(x, 0.0, 1.0);
        else if (code == kLiteRtOpCodeTflExp) y = std::exp(x);
        else if (code == kLiteRtOpCodeTflLog) y = std::log(x);
        else if (code == kLiteRtOpCodeTflRsqrt) y = 1.0 / std::sqrt(x);
        else if (code == kLiteRtOpCodeTflSquare) y = x * x;
        else y = approximate ? .5 * x * (1 + std::tanh(.7978845608028654 * (x + .044715 * x * x * x)))
                             : .5 * x * std::erfc(-x * .7071067811865475);
        if (std::isnan(y)) return reject("Activation domain produces NaN");
        const double q = std::round(y / p->output.scale) + p->output.zero_point;
        p->table.push_back(static_cast<int8_t>(std::clamp(q, -128.0, 127.0)));
      }
    }
  } else if (code == kLiteRtOpCodeTflSpaceToDepth || code == kLiteRtOpCodeTflDepthToSpace) {
    auto option = code == kLiteRtOpCodeTflSpaceToDepth
                      ? LiteRtGetSpaceToDepthBlockSizeOption : LiteRtGetDepthToSpaceBlockSizeOption;
    if (in.size() != 1 || !add(0) || shape.size() != 4 || output.size() != 4 || !SameQuant(p->data, p->output) ||
        option(op.Get(), &p->block) != kLiteRtStatusOk || p->block <= 0 || p->block > 90)
      return reject("Space/depth rearrangement requires NHWC and a bounded positive block size");
    Shape expected = shape;
    const uint32_t b = p->block;
    if (code == kLiteRtOpCodeTflSpaceToDepth) {
      if (shape[1] % b || shape[2] % b) return reject("Spatial dimensions are not divisible by block size");
      expected = {shape[0], shape[1] / b, shape[2] / b, shape[3] * b * b};
    } else {
      if (shape[3] % (b * b)) return reject("Channels are not divisible by block area");
      expected = {shape[0], shape[1] * b, shape[2] * b, shape[3] / (b * b)};
    }
    if (expected != output) return reject("Space/depth output shape mismatch");
  } else if (code == kLiteRtOpCodeTflSpaceToBatchNd || code == kLiteRtOpCodeTflBatchToSpaceNd) {
    if (in.size() != 3 || !add(0) || shape.size() != 4 || output.size() != 4 ||
        !SameQuant(p->data, p->output) || !Ints(in[1], p->parameters) ||
        p->parameters.size() != 2 || !Ints(in[2], p->padding) || p->padding.size() != 4)
      return reject("Space/batch rearrangement requires NHWC and constant 2D block/padding controls");
    int64_t area = 1;
    for (auto b : p->parameters) { if (b <= 0 || b > 8192) return reject("Invalid block size"); area *= b; }
    for (auto pad : p->padding) if (pad < 0 || pad > 8192) return reject("Invalid padding/crop");
    Shape expected = shape;
    if (code == kLiteRtOpCodeTflSpaceToBatchNd) {
      if (area * shape[0] > 8192) return reject("Space-to-batch batch exceeds limit");
      expected[0] *= area;
      p->intermediate_shape = shape;
      for (int a = 0; a < 2; ++a) {
        int64_t n = int64_t{shape[a + 1]} + p->padding[2 * a] + p->padding[2 * a + 1];
        if (n > 8192 || n % p->parameters[a]) return reject("Padded dimensions must divide by block size");
        p->intermediate_shape[a + 1] = n;
        expected[a + 1] = n / p->parameters[a];
      }
    } else {
      if (shape[0] % area) return reject("Batch must divide by block area");
      expected[0] /= area;
      p->intermediate_shape = expected;
      for (int a = 0; a < 2; ++a) {
        int64_t n = int64_t{shape[a + 1]} * p->parameters[a];
        int64_t cropped = n - p->padding[2 * a] - p->padding[2 * a + 1];
        if (n > 8192 || cropped <= 0) return reject("Invalid batch-to-space crop geometry");
        p->intermediate_shape[a + 1] = n;
        expected[a + 1] = cropped;
      }
    }
    if (expected != output || Count(p->intermediate_shape) > 64u * 1024u * 1024u)
      return reject("Space/batch output or materialization limit mismatch");
  } else if (code == kLiteRtOpCodeTflMirrorPad) {
    uint32_t mode;
    if (in.size() != 2 || !add(0) || p->data.scalar || p->output.scalar ||
        !SameQuant(p->data, p->output) ||
        !Ints(in[1], p->padding) || p->padding.size() != shape.size() * 2 ||
        LiteRtGetMirrorPadModeOption(op.Get(), &mode) != kLiteRtStatusOk || mode > 1)
      return reject("MIRROR_PAD requires constant REFLECT/SYMMETRIC padding");
    p->flag = mode == 1;
    Shape expected = shape;
    for (size_t a = 0; a < shape.size(); ++a) {
      const int64_t limit = int64_t{shape[a]} - (p->flag ? 0 : 1);
      const int32_t left = p->padding[2 * a], right = p->padding[2 * a + 1];
      if (left < 0 || right < 0 || left > limit || right > limit)
        return reject("Mirror padding exceeds source dimension");
      expected[a] += left + right;
    }
    if (expected != output) return reject("MIRROR_PAD output shape mismatch");
  } else if (code == kLiteRtOpCodeTflBroadcastTo) {
    Shape expected;
    if (in.size() != 2 || !add(0) || p->output.scalar || !SameQuant(p->data, p->output) ||
        !Ints(in[1], p->parameters) || p->parameters.size() != output.size() ||
        !std::equal(p->parameters.begin(), p->parameters.end(), output.begin()) ||
        !Broadcast({shape, output}, expected) || expected != output)
      return reject("BROADCAST_TO requires constant compatible shape and preserved quantization");
  } else if (code == kLiteRtOpCodeTflGatherNd) {
    if (in.size() != 2 || !add(0) || p->data.scalar || !SameQuant(p->data, p->output) ||
        !Ints(in[1], p->parameters))
      return reject("GATHER_ND requires constant in-range INT32 indices and preserved quantization");
    auto index_type = in[1].RankedTensorType();
    auto dims = index_type->Layout().Dimensions();
    if (dims.empty() || dims.back() < 1 || dims.back() > shape.size())
      return reject("GATHER_ND index depth exceeds data rank");
    const uint32_t depth = dims.back();
    const uint32_t width = p->parameters.size() / depth;
    uint64_t rows = 1, channels = 1;
    for (size_t a = 0; a < shape.size(); ++a)
      (a < depth ? rows : channels) *= shape[a];
    if (rows > 8192 || channels > 8192 || width > 8192)
      return reject("GATHER_ND materialization exceeds level limits");
    for (uint32_t row = 0; row < width; ++row) {
      int32_t flattened = 0;
      for (uint32_t a = 0; a < depth; ++a) {
        const int32_t index = p->parameters[row * depth + a];
        if (index < 0 || index >= shape[a]) return reject("GATHER_ND index out of range");
        flattened = flattened * shape[a] + index;
      }
      p->indices.push_back(flattened);
    }
    Shape expected(dims.begin(), dims.end() - 1);
    expected.insert(expected.end(), shape.begin() + depth, shape.end());
    const bool scalar = expected.empty();
    if (scalar) expected = {1};
    if (expected != output || scalar != p->output.scalar)
      return reject("GATHER_ND output shape mismatch");
    p->intermediate_shape = {1, static_cast<uint32_t>(rows), static_cast<uint32_t>(channels)};
  } else if (code == kLiteRtOpCodeTflOneHot) {
    if (in.size() != 4 || !add(0) || type != DataType::int32_t ||
        !Ints(in[1], p->parameters) || p->parameters.size() != 1 ||
        p->parameters[0] <= 0 || p->parameters[0] > 8192 ||
        LiteRtGetOneHotAxisOption(op.Get(), &p->axis) != kLiteRtStatusOk)
      return reject("ONE_HOT requires INT32 indices and constant positive bounded depth");
    Shape expected = p->data.scalar ? Shape{} : shape;
    if (p->axis == -1) p->axis = expected.size();
    if (p->axis < 0 || p->axis > expected.size()) return reject("ONE_HOT axis out of range");
    expected.insert(expected.begin() + p->axis, p->parameters[0]);
    if (expected != output) return reject("ONE_HOT output shape mismatch");
    for (int i : {2, 3}) {
      DecompositionTensor info;
      if (!Read(in[i], info) || !SameQuant(info, p->output))
        return reject("ONE_HOT on/off values must match output type and quantization");
      if (p->output.type == DataType::float32_t) {
        float value;
        if (!ConstantScalar(in[i], value) || !std::isfinite(value))
          return reject("ONE_HOT requires finite constant on/off scalars");
        p->coefficients.push_back(value);
      } else {
        int32_t value;
        if (p->output.type == DataType::int8_t) {
          int8_t small;
          if (!ConstantScalar(in[i], small)) return reject("Invalid ONE_HOT scalar");
          value = small;
        } else if (p->output.type == DataType::bool_t) {
          auto bytes = in[i].Weights().Bytes();
          if (bytes.size() != 1 || Count(info.shape) != 1) return reject("Invalid ONE_HOT BOOL scalar");
          value = bytes[0] != 0;
        } else if (!ConstantScalar(in[i], value)) return reject("Invalid ONE_HOT scalar");
        p->parameters.push_back(value);
      }
    }
  } else if (code == kLiteRtOpCodeTflSparseToDense) {
    if (in.size() != 4 || !add(2) || p->output.scalar || p->output.type != type ||
        !Ints(in[0], p->indices) || !Ints(in[1], p->parameters) ||
        p->parameters.size() != output.size() ||
        !std::equal(p->parameters.begin(), p->parameters.end(), output.begin()))
      return reject("SPARSE_TO_DENSE requires constant indices/output shape and matching data types");
    const auto* options = internal::GetTflOptions(*op.Get()).AsSparseToDenseOptions();
    auto index_type = in[0].RankedTensorType();
    auto index_dims = index_type->Layout().Dimensions();
    if (!options || index_dims.size() > 2 ||
        (index_dims.size() == 2 ? index_dims[1] != output.size() : output.size() != 1))
      return reject("SPARSE_TO_DENSE index rank/depth mismatch");
    const uint32_t width = p->indices.size() / output.size();
    if ((!p->data.scalar && shape != Shape{width}) || Count(output) > 8192)
      return reject("SPARSE_TO_DENSE requires scalar or per-index values and bounded output");
    std::set<int32_t> used;
    for (uint32_t row = 0; row < width; ++row) {
      int32_t flat = 0;
      for (size_t a = 0; a < output.size(); ++a) {
        const int32_t index = p->indices[row * output.size() + a];
        if (index < 0 || index >= output[a]) return reject("SPARSE_TO_DENSE index out of range");
        flat = flat * output[a] + index;
      }
      if (!used.insert(flat).second ||
          (options->validate_indices && row && flat <= p->indices[row - 1]))
        return reject("SPARSE_TO_DENSE requires unique indices in the requested order");
      p->indices[row] = flat;
    }
    p->indices.resize(width);
    if (type == DataType::float32_t) {
      float value;
      if (in[3].ElementType() != ElementType::Float32 || !ConstantScalar(in[3], value) || !std::isfinite(value))
        return reject("SPARSE_TO_DENSE requires finite constant default value");
      p->coefficients = {value};
    } else {
      int32_t value;
      if (in[3].ElementType() != ElementType::Int32 || !ConstantScalar(in[3], value))
        return reject("SPARSE_TO_DENSE requires constant INT32 default value");
      p->parameters = {value};
    }
  } else if (code == kLiteRtOpCodeTflBucketize) {
    const auto* options = internal::GetTflOptions(*op.Get()).AsBucketizeOptions();
    if (!unary() || !options || options->boundaries.size() > 256 ||
        p->output.type != DataType::int32_t)
      return reject("BUCKETIZE requires matching shapes and at most 256 boundaries");
    p->coefficients.assign(options->boundaries.begin(), options->boundaries.end());
    if (!std::is_sorted(p->coefficients.begin(), p->coefficients.end()) ||
        !std::all_of(p->coefficients.begin(), p->coefficients.end(), [](float x) { return std::isfinite(x); }))
      return reject("BUCKETIZE boundaries must be sorted and finite");
  } else if (code == kLiteRtOpCodeTflFakeQuant) {
    const auto* options = internal::GetTflOptions(*op.Get()).AsFakeQuantOptions();
    if (!unary() || !options || options->num_bits < 2 || options->num_bits > 16 ||
        !std::isfinite(options->min) || !std::isfinite(options->max) || options->min >= options->max)
      return reject("FAKE_QUANT requires a finite ordered range and 2..16 bits");
    const int32_t qmin = options->narrow_range ? 1 : 0;
    const int32_t qmax = (int32_t{1} << options->num_bits) - 1;
    const float scale = (options->max - options->min) / (qmax - qmin);
    if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(1.0f / scale))
      return reject("FAKE_QUANT scale is not representable");
    const float zero = std::clamp(std::round(qmin - options->min / scale),
                                  static_cast<float>(qmin), static_cast<float>(qmax));
    p->coefficients = {(qmin - zero) * scale, (qmax - zero) * scale, scale};
    if (!std::isfinite(p->coefficients[0]) || !std::isfinite(p->coefficients[1]))
      return reject("FAKE_QUANT nudged limits are not representable");
  } else if (code == kLiteRtOpCodeTflCast) {
    if (!unary()) return reject("CAST requires matching shapes");
  } else if (code == kLiteRtOpCodeTflAddN) {
    if (in.size() < 2 || !same()) return reject("ADD_N requires at least two same-shaped inputs");
    for (size_t i = 0; i < in.size(); ++i)
      if (!add(i) || p->inputs.back().shape != shape || p->inputs.back().type != type ||
          p->inputs.back().complex != p->data.complex ||
          p->inputs.back().scalar != p->data.scalar)
        return reject("ADD_N input shapes/types differ");
  } else if (code == kLiteRtOpCodeTflRfft2d) {
    if (in.size() != 2 || !add(0) || p->data.complex || !p->output.complex ||
        shape.size() < 2 || !Ints(in[1], p->parameters) || p->parameters.size() != 2)
      return reject("RFFT2D requires FP32 input and constant two-dimensional FFT lengths");
    for (auto length : p->parameters)
      if (length <= 0 || length > 8192 || (length & (length - 1)))
        return reject("RFFT2D lengths must be positive bounded powers of two");
    Shape expected = shape;
    expected[expected.size() - 2] = p->parameters[0];
    expected.back() = p->parameters[1] / 2 + 1;
    expected.push_back(2);
    if (expected != output) return reject("RFFT2D complex output shape mismatch");
    p->intermediate_shape = shape;
    p->intermediate_shape[shape.size() - 2] = p->parameters[0];
    p->intermediate_shape.back() = p->parameters[1];
    const uint64_t batches = Count(Shape(shape.begin(), shape.end() - 2));
    if (batches > 8192 || Count(p->intermediate_shape) > 64u * 1024u * 1024u)
      return reject("RFFT2D batch/materialization exceeds limits");
    p->block = batches;
    p->extensions = {"EXT-FFT"};
  } else if (code == kLiteRtOpCodeTflReal || code == kLiteRtOpCodeTflImag) {
    if (in.size() != 1 || !add(0) || !p->data.complex || p->output.complex)
      return reject("REAL/IMAG requires COMPLEX64 input and FP32 output");
    auto expected = shape; expected.pop_back();
    if (expected.empty()) expected = {1};
    if (expected != output || p->data.scalar != p->output.scalar)
      return reject("REAL/IMAG output shape mismatch");
  } else if (code == kLiteRtOpCodeTflConv3d) {
    DecompositionTensor weight, bias;
    if (in.size() != 3 || !add(0) || shape.size() != 5 || output.size() != 5 ||
        !Read(in[1], weight, true) || weight.type != DataType::int8_t || weight.shape.size() != 5 ||
        !Read(in[2], bias) || bias.type != DataType::int32_t || bias.scalar || bias.shape.size() != 1 ||
        p->output.type != type || weight.shape[3] != shape[4] || bias.shape[0] != weight.shape[4])
      return reject("INT8 CONV3D requires NDHWC activations, constant DHWIO weights and INT32 bias");
    auto weights = in[1].WeightsData<int8_t>();
    auto biases = in[2].WeightsData<int32_t>();
    const uint32_t channels = weight.shape[4];
    std::vector<double> weight_scales, bias_scales;
    if (!weights || !biases || weights->size() != Count(weight.shape) || biases->size() != channels ||
        !SymmetricScales(in[1], channels, 4, weight_scales) || !SymmetricScales(in[2], channels, 0, bias_scales))
      return reject("CONV3D requires symmetric per-tensor/per-output-channel weight and bias quantization");
    const size_t kernel_size = weights->size() / channels;
    p->weights.resize(weights->size());
    for (uint32_t c = 0; c < channels; ++c)
      for (size_t k = 0; k < kernel_size; ++k)
        p->weights[c * kernel_size + k] = (*weights)[k * channels + c];
    p->bias.assign(biases->begin(), biases->end());
    p->weight_shape = {channels, weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};
    for (uint32_t c = 0; c < channels; ++c) {
      const double scale = p->data.scale * weight_scales[c];
      if (!std::isfinite(scale) || scale <= 0 || std::abs(bias_scales[c] - scale) > scale * 1e-5)
        return reject("CONV3D bias scale must match input times weight scale");
      auto rescale = MakeScale32(scale / p->output.scale);
      auto range = DotProductRange(p->weights.data() + c * kernel_size, kernel_size,
          p->data.zero_point, 0, p->bias[c]);
      if (!rescale || !range || !IsScale32RangeValid(*rescale, range->minimum, range->maximum))
        return reject("CONV3D accumulator/rescale range is not proven");
      p->scales.push_back(*rescale); p->ranges.push_back(*range);
    }
    uint32_t padding, fused;
    p->strides.resize(3); p->dilations.resize(3); p->padding.resize(6);
    if (LiteRtGetConv3dPaddingOption(op.Get(), &padding) != kLiteRtStatusOk || padding > 1 ||
        LiteRtGetConv3dStrideDOption(op.Get(), &p->strides[0]) != kLiteRtStatusOk ||
        LiteRtGetConv3dStrideHOption(op.Get(), &p->strides[1]) != kLiteRtStatusOk ||
        LiteRtGetConv3dStrideWOption(op.Get(), &p->strides[2]) != kLiteRtStatusOk ||
        LiteRtGetConv3dDilationDOption(op.Get(), &p->dilations[0]) != kLiteRtStatusOk ||
        LiteRtGetConv3dDilationHOption(op.Get(), &p->dilations[1]) != kLiteRtStatusOk ||
        LiteRtGetConv3dDilationWOption(op.Get(), &p->dilations[2]) != kLiteRtStatusOk ||
        LiteRtGetConv3dFusedActivationOption(op.Get(), &fused) != kLiteRtStatusOk ||
        (fused != 0 && fused != 1 && fused != 3)) return reject("Invalid CONV3D options");
    auto activation = ActivationRange(fused == 0 ? Activation::kNone : fused == 1 ? Activation::kRelu : Activation::kRelu6,
        p->output.scale, p->output.zero_point);
    if (!activation) return reject("Invalid CONV3D activation range");
    p->activation = *activation;
    auto expected = shape; expected[4] = channels;
    p->intermediate_shape = shape;
    for (size_t a = 0; a < 3; ++a) {
      const int64_t stride = p->strides[a], dilation = p->dilations[a];
      const int64_t kernel = (int64_t{weight.shape[a]} - 1) * dilation + 1, n = shape[a + 1];
      if (stride <= 0 || stride > 8192 || dilation <= 0 || dilation > 8192 ||
          kernel > 8192 || (padding == 1 && n < kernel)) return reject("Invalid CONV3D geometry");
      const int64_t count = padding == 0 ? (n + stride - 1) / stride : (n - kernel) / stride + 1;
      const int64_t total = padding == 0 ? std::max(int64_t{0}, (count - 1) * stride + kernel - n) : 0;
      p->padding[2 * a] = total / 2; p->padding[2 * a + 1] = total - total / 2;
      const int64_t trailing = (n + total - kernel) % stride;
      const int64_t removed = std::min(trailing, int64_t{p->padding[2 * a + 1]});
      p->padding[2 * a + 1] -= removed;
      p->intermediate_shape[a + 1] -= trailing - removed;
      expected[a + 1] = count;
    }
    if (expected != output) return reject("CONV3D output shape mismatch");
  } else if (code == kLiteRtOpCodeTflMean) {
    if (in.size() != 2 || !add(0) || p->data.scalar || p->output.type != type ||
        !Ints(in[1], p->parameters) ||
        LiteRtGetMeanKeepDimsOption(op.Get(), &p->flag) != kLiteRtStatusOk)
      return reject("INT8 MEAN requires constant nonempty reduction axes");
    std::set<int32_t> axes;
    uint64_t count = 1;
    auto reduced = shape;
    for (auto axis : p->parameters) {
      if (axis < 0) axis += shape.size();
      if (axis < 0 || axis >= shape.size()) return reject("MEAN axis out of range");
      if (axes.insert(axis).second) { count *= shape[axis]; reduced[axis] = 1; }
    }
    auto expected = reduced;
    if (!p->flag) for (auto a = axes.rbegin(); a != axes.rend(); ++a) expected.erase(expected.begin() + *a);
    const bool scalar = expected.empty();
    if (scalar) expected = {1};
    if (expected != output || scalar != p->output.scalar)
      return reject("MEAN output shape mismatch");
    const int64_t lo = int64_t{-128 - p->data.zero_point} * static_cast<int64_t>(count);
    const int64_t hi = int64_t{127 - p->data.zero_point} * static_cast<int64_t>(count);
    auto scale = MakeScale32(p->data.scale / (count * p->output.scale));
    if (!scale || lo < INT32_MIN || hi > INT32_MAX || !IsScale32RangeValid(*scale, lo, hi))
      return reject("MEAN INT32 accumulation/rescale range is not proven");
    p->scales = {*scale};
    p->ranges = {{static_cast<int32_t>(lo), static_cast<int32_t>(hi)}};
    p->parameters.assign(axes.begin(), axes.end());
    p->intermediate_shape = reduced;
  } else if (code == kLiteRtOpCodeTflBatchMatmul) {
    bool adj_x, adj_y;
    if (in.size() != 2 || !add(0) || !add(1) || shape.size() < 2 ||
        p->inputs[1].shape.size() < 2 || p->inputs[1].type != type || p->output.type != type ||
        LiteRtGetBatchMatmulAdjXOption(op.Get(), &adj_x) != kLiteRtStatusOk ||
        LiteRtGetBatchMatmulAdjYOption(op.Get(), &adj_y) != kLiteRtStatusOk)
      return reject("INT8 BATCH_MATMUL requires compatible static quantized operands");
    p->parameters = {adj_x ? 1 : 0, adj_y ? 1 : 0};
    auto lhs = shape, rhs = p->inputs[1].shape;
    if (adj_x) std::swap(lhs[lhs.size() - 2], lhs.back());
    if (adj_y) std::swap(rhs[rhs.size() - 2], rhs.back());
    const size_t rank = std::max(lhs.size(), rhs.size());
    lhs.insert(lhs.begin(), rank - lhs.size(), 1); rhs.insert(rhs.begin(), rank - rhs.size(), 1);
    if (lhs.back() != rhs[rank - 2]) return reject("BATCH_MATMUL contraction dimension mismatch");
    Shape expected(rank, 1);
    for (size_t i = 0; i + 2 < rank; ++i) {
      if (lhs[i] != rhs[i] && lhs[i] != 1 && rhs[i] != 1) return reject("BATCH_MATMUL batch broadcast mismatch");
      expected[i] = std::max(lhs[i], rhs[i]);
    }
    const uint64_t batches = Count(expected);
    expected[rank - 2] = lhs[rank - 2]; expected.back() = rhs.back();
    if (expected != output || batches > 8192 ||
        batches * lhs[rank - 2] * lhs.back() > 64u * 1024u * 1024u ||
        batches * rhs[rank - 2] * rhs.back() > 64u * 1024u * 1024u)
      return reject("BATCH_MATMUL output/materialization limit mismatch");
    const int64_t a[] = {-128 - p->data.zero_point, 127 - p->data.zero_point};
    const int64_t b[] = {-128 - p->inputs[1].zero_point, 127 - p->inputs[1].zero_point};
    int64_t lo = 0, hi = 0;
    for (auto x : a) for (auto y : b) { lo = std::min(lo, x * y); hi = std::max(hi, x * y); }
    lo *= lhs.back(); hi *= lhs.back();
    auto scale = MakeScale32(p->data.scale * p->inputs[1].scale / p->output.scale);
    if (!scale || lo < INT32_MIN || hi > INT32_MAX || !IsScale32RangeValid(*scale, lo, hi))
      return reject("BATCH_MATMUL accumulator/rescale range is not proven");
    p->scales = {*scale}; p->ranges = {{static_cast<int32_t>(lo), static_cast<int32_t>(hi)}};
    p->intermediate_shape = {static_cast<uint32_t>(batches), lhs[rank - 2], rhs.back()};
  } else if (code == kLiteRtOpCodeTflSquaredDifference || code == kLiteRtOpCodeTflAtan2 ||
             code == kLiteRtOpCodeTflFloorDiv || code == kLiteRtOpCodeTflFloorMod) {
    Shape expected;
    if (in.size() != 2 || !add(0) || !add(1) || p->inputs[1].type != type ||
        p->inputs[1].complex != p->data.complex ||
        !Broadcast({shape, p->inputs[1].shape}, expected) || expected != output ||
        p->output.scalar != (p->data.scalar && p->inputs[1].scalar))
      return reject("Binary decomposition requires compatible static broadcasting");
    if (code == kLiteRtOpCodeTflAtan2 &&
        (shape != p->inputs[1].shape || p->data.scalar != p->inputs[1].scalar))
      return reject("ATAN2 requires identical operand shapes");
    if (type == DataType::int32_t) {
      if (!Ints(in[1], p->parameters)) return reject("Integer division requires constant divisors");
      for (int32_t d : p->parameters) if (d <= 0) return reject("Integer divisors must be positive");
    } else if (code == kLiteRtOpCodeTflFloorDiv || code == kLiteRtOpCodeTflFloorMod) {
      auto divisors = in[1].WeightsData<float>();
      if (!divisors || divisors->empty() || divisors->size() > 8192 ||
          divisors->size() != Count(p->inputs[1].shape))
        return reject("FP32 floor division/modulo requires constant power-of-two divisors");
      for (float divisor : *divisors) {
        int exponent;
        if (!std::isnormal(divisor) || std::abs(std::frexp(divisor, &exponent)) != .5f ||
            !std::isnormal(1.0f / divisor))
          return reject("FP32 floor division/modulo requires normal power-of-two divisors and reciprocals");
        const float reciprocal = 1.0f / divisor;
        p->coefficients.push_back(code == kLiteRtOpCodeTflFloorMod ? std::abs(reciprocal) : reciprocal);
      }
    }
  } else if (code == kLiteRtOpCodeTflArgMin) {
    if (in.size() != 2 || !add(0) || p->data.scalar || !Ints(in[1], p->parameters) ||
        p->parameters.size() != 1 || p->output.type != DataType::int32_t)
      return reject("ARG_MIN requires one constant axis and INT32 output");
    p->axis = p->parameters[0];
    if (p->axis < 0) p->axis += shape.size();
    if (p->axis < 0 || p->axis >= shape.size()) return reject("ARG_MIN axis out of range");
    auto expected = shape;
    expected.erase(expected.begin() + p->axis);
    if (expected.empty() != p->output.scalar) return reject("ARG_MIN output rank mismatch");
    if (expected.empty()) expected = {1};
    if (expected != output) return reject("ARG_MIN output shape mismatch");
  } else if (code == kLiteRtOpCodeTflShape || code == kLiteRtOpCodeTflRank) {
    if (in.size() != 1 || !add(0) || p->output.type != DataType::int32_t)
      return reject("Shape metadata requires INT32 output");
    const int32_t rank = p->data.scalar ? 0 : shape.size();
    p->parameters = code == kLiteRtOpCodeTflRank ? std::vector<int32_t>{rank}
                                               : std::vector<int32_t>(shape.begin(), shape.end());
    if ((code == kLiteRtOpCodeTflRank && !p->output.scalar) ||
        (code == kLiteRtOpCodeTflShape && (rank == 0 || p->output.scalar || output != Shape{static_cast<uint32_t>(rank)})))
      return reject("Shape metadata output rank/type mismatch");
  } else if (code == kLiteRtOpCodeTflZerosLike) {
    if (!unary()) return reject("ZEROS_LIKE requires matching shapes");
  } else if (code == kLiteRtOpCodeTflFill) {
    if (in.size() != 2 || !add(1) || !p->data.scalar || p->output.scalar ||
        !Ints(in[0], p->parameters) || p->parameters.size() != output.size() ||
        !std::equal(p->parameters.begin(), p->parameters.end(), output.begin()))
      return reject("FILL requires scalar value and constant output dimensions");
  } else if (code == kLiteRtOpCodeTflSign || code == kLiteRtOpCodeTflRound ||
             code == kLiteRtOpCodeTflElu || code == kLiteRtOpCodeTflLogSoftmax ||
             code == kLiteRtOpCodeTflL2Normalization) {
    if (!unary()) return reject("Unary decomposition requires matching shapes");
    if (p->data.scalar && (code == kLiteRtOpCodeTflLogSoftmax || code == kLiteRtOpCodeTflL2Normalization))
      return reject("Normalization requires a non-scalar last axis");
    if (code == kLiteRtOpCodeTflL2Normalization) {
      const auto* options = internal::GetTflOptions(*op.Get()).AsL2NormOptions();
      if (!options || options->fused_activation_function != tflite::ActivationFunctionType_NONE)
        return reject("L2_NORMALIZATION requires no fused activation");
    }
  } else return reject("Unimplemented decomposition");
  reason.clear();
  return p;
}

ResId EmitDecomposition(tfsc::tosa::Graph& g, const std::vector<ResId>& inputs,
                       const DecompositionPlan& p, GraphConstantPool& constants) {
  const auto& shape = p.output.shape;
  const auto type = p.output.type;
  const size_t rank = shape.size();
  const TT output(type, shape), wide(DataType::int32_t, shape);
  const TT boolean(DataType::bool_t, shape), fp(DataType::float32_t, shape);
  auto scalar = [&](int32_t value) { return Scalar(g, value, rank); };
  auto floating = [&](float value) { return Scalar(g, value, rank); };
  auto multiply = [&](ResId x, ResId y, const TT& result) {
    return g.AddMulOperator(x, y, g.AddTensorConstant(Attribute(int8_t{0})), result);
  };
  auto reshape = [&](ResId value, DataType dtype, const Shape& target) {
    return Reshape(g, value, dtype, target);
  };
  std::vector<ResId> v;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto s = p.inputs[i].shape;
    s.insert(s.begin(), rank > s.size() ? rank - s.size() : 0, 1);
    v.push_back(reshape(inputs[i], p.inputs[i].type, s));
  }
  if (!p.table.empty())
    return g.AddTableOperator(v[0], g.AddTensorConstant(Attribute(p.table)), output);
  const auto code = p.code;
  if (code == kLiteRtOpCodeTflSelect || code == kLiteRtOpCodeTflSelectV2)
    return g.AddSelectOperator(v[0], v[1], v[2], output);
  if (code == kLiteRtOpCodeTflGreater || code == kLiteRtOpCodeTflGreaterEqual ||
      code == kLiteRtOpCodeTflLess || code == kLiteRtOpCodeTflLessEqual) {
    // Casts preserve operand broadcasting; explicitly use each padded shape.
    if (p.data.type == DataType::int8_t) {
      for (size_t i = 0; i < v.size(); ++i) {
        auto s = p.inputs[i].shape; s.insert(s.begin(), rank - s.size(), 1);
        v[i] = g.AddCastOperator(reshape(inputs[i], DataType::int8_t, s), TT(DataType::int32_t, s));
      }
    }
    if (code == kLiteRtOpCodeTflLess || code == kLiteRtOpCodeTflLessEqual) std::swap(v[0], v[1]);
    return code == kLiteRtOpCodeTflGreater || code == kLiteRtOpCodeTflLess
        ? g.AddGreaterOperator(v[0], v[1], boolean)
        : g.AddGreaterEqualOperator(v[0], v[1], boolean);
  }
  if (code == kLiteRtOpCodeTflPrelu) {
    if (type == DataType::float32_t)
      return g.AddSelectOperator(g.AddGreaterEqualOperator(v[0], floating(0), boolean),
          v[0], multiply(v[0], v[1], output), output);
    auto centered = g.AddSubOperator(g.AddCastOperator(v[0], wide), scalar(p.data.zero_point), wide);
    auto a = p.inputs[1].shape; a.insert(a.begin(), rank - a.size(), 1);
    auto alpha = g.AddSubOperator(g.AddCastOperator(v[1], TT(DataType::int32_t, a)),
                                  scalar(p.inputs[1].zero_point), TT(DataType::int32_t, a));
    auto product = multiply(centered, alpha, wide);
    auto positive = EmitRescale(g, centered, shape, p.output.zero_point,
        {p.scales[0]}, {p.ranges[0]}, constants);
    auto negative = EmitRescale(g, product, shape, p.output.zero_point,
        {p.scales[1]}, {p.ranges[1]}, constants);
    return g.AddSelectOperator(g.AddGreaterEqualOperator(centered, scalar(0), boolean), positive, negative, output);
  }
  if (code == kLiteRtOpCodeTflLeakyRelu)
    return g.AddSelectOperator(g.AddGreaterEqualOperator(v[0], floating(0), boolean),
        v[0], multiply(v[0], floating(p.coefficients[0]), output), output);
  if (code == kLiteRtOpCodeTflReluN1To1 || code == kLiteRtOpCodeTflRelu0To1)
    return g.AddClampOperator(v[0], Attribute(code == kLiteRtOpCodeTflReluN1To1 ? -1.0f : 0.0f),
                              Attribute(1.0f), Attribute(uint32_t{0}), output);
  if (code == kLiteRtOpCodeTflSpaceToDepth || code == kLiteRtOpCodeTflDepthToSpace) {
    const auto& s = p.data.shape;
    const uint32_t b = p.block;
    Shape a = code == kLiteRtOpCodeTflSpaceToDepth
        ? Shape{s[0], s[1] / b, b, s[2] / b, b, s[3]}
        : Shape{s[0], s[1], s[2], b, b, s[3] / (b * b)};
    const std::vector<uint32_t> perm{0, 1, 3, 2, 4, 5};
    Shape transposed; for (auto index : perm) transposed.push_back(a[index]);
    return reshape(g.AddTransposeOperator(reshape(inputs[0], type, a), Attribute(perm),
        TT(type, transposed)), type, shape);
  }
  if (code == kLiteRtOpCodeTflSpaceToBatchNd || code == kLiteRtOpCodeTflBatchToSpaceNd) {
    const auto& s = p.data.shape;
    const uint32_t b0 = p.parameters[0], b1 = p.parameters[1];
    if (code == kLiteRtOpCodeTflSpaceToBatchNd) {
      std::vector<int32_t> padding{0, 0, p.padding[0], p.padding[1], p.padding[2], p.padding[3], 0, 0};
      auto fill = type == DataType::float32_t ? g.AddTensorConstant(Attribute(0.0f)) :
          type == DataType::int8_t ? g.AddTensorConstant(Attribute(static_cast<int8_t>(p.data.zero_point))) :
                                     g.AddTensorConstant(Attribute(int32_t{0}));
      auto padded = g.AddPadOperator(inputs[0], g.AddTensorConstant(Attribute(padding)), fill,
                                     TT(type, p.intermediate_shape));
      Shape a{s[0], shape[1], b0, shape[2], b1, s[3]};
      Shape b{b0, b1, s[0], shape[1], shape[2], s[3]};
      auto transposed = g.AddTransposeOperator(reshape(padded, type, a),
          Attribute(std::vector<uint32_t>{2, 4, 0, 1, 3, 5}), TT(type, b));
      return reshape(transposed, type, shape);
    }
    Shape a{b0, b1, shape[0], s[1], s[2], s[3]};
    Shape b{shape[0], s[1], b0, s[2], b1, s[3]};
    auto expanded = reshape(g.AddTransposeOperator(reshape(inputs[0], type, a),
        Attribute(std::vector<uint32_t>{2, 3, 0, 4, 1, 5}), TT(type, b)), type, p.intermediate_shape);
    return g.AddSliceOperator(expanded,
        g.AddTensorConstant(Attribute(std::vector<int32_t>{0, p.padding[0], p.padding[2], 0})),
        g.AddTensorConstant(Attribute(shape)), output);
  }
  if (code == kLiteRtOpCodeTflMirrorPad) {
    auto current = p.data.shape;
    auto value = inputs[0];
    for (size_t a = 0; a < current.size(); ++a) {
      std::vector<ResId> pieces;
      for (int side = 0; side < 3; ++side) {
        if (side == 1) { pieces.push_back(value); continue; }
        const int32_t width = p.padding[2 * a + (side == 2)];
        if (!width) continue;
        std::vector<int32_t> begin(current.size(), 0);
        auto part = current; part[a] = width;
        begin[a] = side == 0 ? (p.flag ? 0 : 1) : current[a] - width - (p.flag ? 0 : 1);
        auto slice = g.AddSliceOperator(value, g.AddTensorConstant(Attribute(begin)),
            g.AddTensorConstant(Attribute(part)), TT(type, part));
        pieces.push_back(g.AddReverseOperator(slice, Attribute(static_cast<int32_t>(a)), TT(type, part)));
      }
      current[a] += p.padding[2 * a] + p.padding[2 * a + 1];
      if (pieces.size() > 1) value = g.AddConcatOperator(pieces, Attribute(static_cast<int32_t>(a)), TT(type, current));
    }
    return value;
  }
  if (code == kLiteRtOpCodeTflBroadcastTo || code == kLiteRtOpCodeTflFill || code == kLiteRtOpCodeTflZerosLike) {
    auto value = v[0];
    auto source = p.inputs[0].shape;
    if (code == kLiteRtOpCodeTflZerosLike) {
      value = type == DataType::float32_t ? floating(0) : scalar(0);
      source.assign(rank, 1);
    }
    source.insert(source.begin(), rank - source.size(), 1);
    Shape multiples(rank);
    for (size_t i = 0; i < rank; ++i) multiples[i] = shape[i] / source[i];
    return g.AddTileOperator(reshape(value, type, source),
        g.AddTensorConstant(Attribute(multiples)), output);
  }
  if (code == kLiteRtOpCodeTflShape || code == kLiteRtOpCodeTflRank)
    return Vector(g, p.parameters, shape);
  if (code == kLiteRtOpCodeTflReal || code == kLiteRtOpCodeTflImag) {
    auto component = p.data.shape; component.back() = 1;
    std::vector<int32_t> begin(component.size(), 0);
    begin.back() = code == kLiteRtOpCodeTflImag ? 1 : 0;
    auto value = g.AddSliceOperator(inputs[0], g.AddTensorConstant(Attribute(begin)),
        g.AddTensorConstant(Attribute(component)), TT(DataType::float32_t, component));
    return reshape(value, DataType::float32_t, shape);
  }
  if (code == kLiteRtOpCodeTflRfft2d) {
    auto source_shape = p.data.shape, cropped = source_shape;
    const size_t r = source_shape.size();
    auto source = inputs[0];
    cropped[r - 2] = std::min(source_shape[r - 2], p.intermediate_shape[r - 2]);
    cropped.back() = std::min(source_shape.back(), p.intermediate_shape.back());
    if (cropped != source_shape) source = g.AddSliceOperator(source,
        g.AddTensorConstant(Attribute(std::vector<int32_t>(r, 0))),
        g.AddTensorConstant(Attribute(cropped)), TT(DataType::float32_t, cropped));
    if (cropped != p.intermediate_shape) {
      std::vector<int32_t> padding(2 * r, 0);
      padding[2 * (r - 2) + 1] = p.intermediate_shape[r - 2] - cropped[r - 2];
      padding.back() = p.intermediate_shape.back() - cropped.back();
      source = g.AddPadOperator(source, g.AddTensorConstant(Attribute(padding)),
          g.AddTensorConstant(Attribute(0.0f)), TT(DataType::float32_t, p.intermediate_shape));
    }
    Shape fft{static_cast<uint32_t>(p.block), static_cast<uint32_t>(p.parameters[0]),
               static_cast<uint32_t>(p.parameters[1])};
    auto result_shape = fft; result_shape.back() = fft.back() / 2 + 1;
    auto result = g.AddRfft2dOperator(reshape(source, DataType::float32_t, fft),
        Attribute(false), TT(DataType::float32_t, result_shape), TT(DataType::float32_t, result_shape));
    // Interleaved real/imaginary FP32 storage is the COMPLEX64 byte layout.
    auto component = shape; component.back() = 1;
    return g.AddConcatOperator({reshape(result[0], DataType::float32_t, component),
        reshape(result[1], DataType::float32_t, component)}, Attribute(static_cast<int32_t>(rank - 1)), output);
  }
  if (code == kLiteRtOpCodeTflGatherNd) {
    auto source = reshape(inputs[0], type, p.intermediate_shape);
    const Shape selected{1, static_cast<uint32_t>(p.indices.size()), p.intermediate_shape[2]};
    auto indices = Vector(g, p.indices, {1, selected[1]});
    return reshape(g.AddGatherOperator(source, indices, TT(type, selected)), type, shape);
  }
  if (code == kLiteRtOpCodeTflOneHot) {
    Shape broadcast = shape;
    broadcast[p.axis] = 1;
    Shape index_shape(rank, 1);
    index_shape[p.axis] = shape[p.axis];
    std::vector<int32_t> indices(shape[p.axis]);
    std::iota(indices.begin(), indices.end(), 0);
    auto condition = g.AddEqualOperator(reshape(inputs[0], DataType::int32_t, broadcast),
        Vector(g, indices, index_shape), boolean);
    ResId on, off;
    if (type == DataType::float32_t) {
      on = floating(p.coefficients[0]); off = floating(p.coefficients[1]);
    } else if (type == DataType::int8_t) {
      on = Scalar(g, static_cast<int8_t>(p.parameters[1]), rank);
      off = Scalar(g, static_cast<int8_t>(p.parameters[2]), rank);
    } else if (type == DataType::bool_t) {
      on = g.AddEqualOperator(scalar(p.parameters[1]), scalar(1), TT(DataType::bool_t, Shape(rank, 1)));
      off = g.AddEqualOperator(scalar(p.parameters[2]), scalar(1), TT(DataType::bool_t, Shape(rank, 1)));
    } else {
      on = scalar(p.parameters[1]); off = scalar(p.parameters[2]);
    }
    return g.AddSelectOperator(condition, on, off, output);
  }
  if (code == kLiteRtOpCodeTflSparseToDense) {
    const uint32_t width = p.indices.size(), size = Count(shape);
    auto fill = type == DataType::float32_t ? Scalar(g, p.coefficients[0], 3)
                                          : Scalar(g, p.parameters[0], 3);
    const Shape base{1, size, 1}, updates{1, width, 1};
    auto initial = g.AddTileOperator(fill, g.AddTensorConstant(Attribute(base)), TT(type, base));
    auto values = reshape(inputs[0], type, p.data.scalar ? Shape{1, 1, 1} : updates);
    if (p.data.scalar) values = g.AddTileOperator(values,
        g.AddTensorConstant(Attribute(updates)), TT(type, updates));
    return reshape(g.AddScatterOperator(initial, Vector(g, p.indices, {1, width}), values,
                                         TT(type, base)), type, shape);
  }
  if (code == kLiteRtOpCodeTflBucketize) {
    auto source = v[0];
    if (p.data.type != DataType::float32_t) source = g.AddCastOperator(source, fp);
    auto count = scalar(0);
    for (float boundary : p.coefficients) {
      auto match = g.AddGreaterEqualOperator(source, floating(boundary), boolean);
      count = g.AddAddOperator(count, g.AddSelectOperator(match, scalar(1), scalar(0), wide), wide);
    }
    // upper_bound places NaN after all boundaries because every less-than
    // comparison against NaN is false.
    return g.AddSelectOperator(g.AddEqualOperator(source, source, boolean), count,
                                scalar(p.coefficients.size()), wide);
  }
  if (code == kLiteRtOpCodeTflFakeQuant) {
    auto bounded = g.AddClampOperator(v[0], Attribute(p.coefficients[0]),
        Attribute(p.coefficients[1]), Attribute(uint32_t{0}), fp);
    auto shifted = g.AddSubOperator(bounded, floating(p.coefficients[0]), fp);
    auto rounded = g.AddFloorOperator(g.AddAddOperator(
        multiply(shifted, floating(1.0f / p.coefficients[2]), fp), floating(.5f), fp), fp);
    return g.AddAddOperator(multiply(rounded, floating(p.coefficients[2]), fp),
                             floating(p.coefficients[0]), fp);
  }
  if (code == kLiteRtOpCodeTflAddN) {
    auto result = v[0];
    for (size_t i = 1; i < v.size(); ++i) result = g.AddAddOperator(result, v[i], output);
    return result;
  }
  if (code == kLiteRtOpCodeTflMean) {
    auto current = p.data.shape;
    const size_t input_rank = current.size();
    auto value = g.AddSubOperator(g.AddCastOperator(inputs[0], TT(DataType::int32_t, current)),
        Scalar(g, p.data.zero_point, input_rank), TT(DataType::int32_t, current));
    for (auto axis : p.parameters) {
      current[axis] = 1;
      value = g.AddReduceSumOperator(value, Attribute(axis), TT(DataType::int32_t, current));
    }
    value = reshape(value, DataType::int32_t, shape);
    return EmitRescale(g, value, shape, p.output.zero_point, p.scales, p.ranges, constants);
  }
  if (code == kLiteRtOpCodeTflConv3d) {
    auto source = inputs[0];
    if (p.intermediate_shape != p.data.shape)
      source = g.AddSliceOperator(source, g.AddTensorConstant(Attribute(std::vector<int32_t>(5, 0))),
          g.AddTensorConstant(Attribute(p.intermediate_shape)), TT(DataType::int8_t, p.intermediate_shape));
    auto weight = ExternalConstant(g, p.weights, p.weight_shape, constants);
    auto bias = ExternalConstant(g, p.bias, {p.weight_shape[0]}, constants);
    auto acc = g.AddConv3dOperator(source, weight, bias,
        g.AddTensorConstant(Attribute(static_cast<int8_t>(p.data.zero_point))),
        g.AddTensorConstant(Attribute(int8_t{0})), Attribute(p.padding), Attribute(p.strides),
        Attribute(p.dilations), Attribute(int32_t{1}), Attribute(false), wide);
    auto value = EmitRescale(g, acc, shape, p.output.zero_point, p.scales, p.ranges, constants);
    return g.AddClampOperator(value, Attribute(static_cast<int8_t>(p.activation.minimum)),
        Attribute(static_cast<int8_t>(p.activation.maximum)), Attribute(uint32_t{1}), output);
  }
  if (code == kLiteRtOpCodeTflBatchMatmul) {
    std::vector<ResId> values = inputs;
    for (size_t i = 0; i < 2; ++i) {
      auto source = p.inputs[i].shape;
      if (p.parameters[i]) {
        std::vector<uint32_t> permutation(source.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::swap(permutation[permutation.size() - 2], permutation.back());
        std::swap(source[source.size() - 2], source.back());
        values[i] = g.AddTransposeOperator(values[i], Attribute(permutation), TT(DataType::int8_t, source));
      }
      source.insert(source.begin(), rank - source.size(), 1);
      values[i] = reshape(values[i], DataType::int8_t, source);
      Shape multiples(rank, 1), expanded = source;
      for (size_t a = 0; a + 2 < rank; ++a) { multiples[a] = shape[a] / source[a]; expanded[a] = shape[a]; }
      if (expanded != source) values[i] = g.AddTileOperator(values[i],
          g.AddTensorConstant(Attribute(multiples)), TT(DataType::int8_t, expanded));
      values[i] = reshape(values[i], DataType::int8_t,
          {p.intermediate_shape[0], expanded[rank - 2], expanded.back()});
    }
    auto acc = g.AddMatmulOperator(values[0], values[1],
        g.AddTensorConstant(Attribute(static_cast<int8_t>(p.data.zero_point))),
        g.AddTensorConstant(Attribute(static_cast<int8_t>(p.inputs[1].zero_point))),
        TT(DataType::int32_t, p.intermediate_shape));
    return EmitRescale(g, reshape(acc, DataType::int32_t, shape), shape,
        p.output.zero_point, p.scales, p.ranges, constants);
  }
  if (code == kLiteRtOpCodeTflCast) {
    if (p.data.type == type) return inputs[0];
    if (type == DataType::bool_t) {
      auto zero = p.data.type == DataType::float32_t ? floating(0) : scalar(0);
      auto source = v[0];
      if (p.data.type == DataType::int8_t) source = g.AddCastOperator(source, wide);
      if (p.data.type == DataType::bool_t) return source;
      return g.AddLogicalNotOperator(g.AddEqualOperator(source, zero, boolean), boolean);
    }
    auto source = v[0];
    if (p.data.type == DataType::bool_t) {
      // SELECT also avoids requiring a dedicated BOOL-to-floating cast shader.
      if (type == DataType::float32_t)
        return g.AddSelectOperator(source, floating(1), floating(0), output);
      source = g.AddSelectOperator(source, scalar(1), scalar(0), wide);
    }
    if (p.data.type == DataType::float32_t && type != DataType::float32_t) {
      auto negative = g.AddGreaterOperator(floating(0), source, boolean);
      auto absolute = g.AddAbsOperator(source, fp);
      auto integer = g.AddFloorOperator(absolute, fp);
      source = g.AddSelectOperator(negative,
          g.AddNegateOperator(integer, g.AddTensorConstant(Attribute(0.0f)),
              g.AddTensorConstant(Attribute(0.0f)), fp), integer, fp);
    }
    if (type == DataType::int8_t) {
      source = g.AddCastOperator(source, wide);
      auto low = g.AddBitwiseAndOperator(source, scalar(255), wide);
      auto sign = g.AddGreaterOperator(low, scalar(127), boolean);
      source = g.AddSelectOperator(sign, g.AddSubOperator(low, scalar(256), wide), low, wide);
    }
    return g.AddCastOperator(source, output);
  }
  if (code == kLiteRtOpCodeTflArgMin) {
    ResId source = inputs[0];
    auto input_shape = p.data.shape;
    int32_t axis = p.axis;
    if (input_shape.size() == 1) {
      input_shape.insert(input_shape.begin(), 1);
      source = reshape(source, p.data.type, input_shape);
      axis = 1;
    }
    if (p.data.type == DataType::float32_t) {
      source = g.AddNegateOperator(source, g.AddTensorConstant(Attribute(0.0f)),
          g.AddTensorConstant(Attribute(0.0f)), TT(DataType::float32_t, input_shape));
    } else {
      // Integer ARGMAX is absent from some implementations. Reducing values
      // first, then the matching index, preserves all INT32 bits and first
      // ties without a lossy conversion to FP32 (including INT32_MIN).
      const TT data(DataType::int32_t, input_shape);
      if (p.data.type != DataType::int32_t) source = g.AddCastOperator(source, data);
      auto reduced = input_shape; reduced[axis] = 1;
      auto minimum = g.AddReduceMinOperator(source, Attribute(axis), Attribute(uint32_t{1}), TT(DataType::int32_t, reduced));
      auto match = g.AddEqualOperator(source, minimum, TT(DataType::bool_t, input_shape));
      Shape index_shape(input_shape.size(), 1); index_shape[axis] = input_shape[axis];
      std::vector<int32_t> indices(input_shape[axis]); std::iota(indices.begin(), indices.end(), 0);
      auto selected = g.AddSelectOperator(match, Vector(g, indices, index_shape),
          Scalar(g, int32_t{INT32_MAX}, input_shape.size()), data);
      auto result = g.AddReduceMinOperator(selected, Attribute(axis), Attribute(uint32_t{1}), TT(DataType::int32_t, reduced));
      return reshape(result, DataType::int32_t, shape);
    }
    auto reduced = input_shape; reduced.erase(reduced.begin() + axis);
    auto value = g.AddArgMaxOperator(source, Attribute(axis), Attribute(uint32_t{1}),
                                     TT(DataType::int32_t, reduced));
    return reshape(value, DataType::int32_t, shape);
  }
  if (code == kLiteRtOpCodeTflSign) {
    auto zero = type == DataType::float32_t ? floating(0) : scalar(0);
    auto one = type == DataType::float32_t ? floating(1) : scalar(1);
    auto minus = type == DataType::float32_t ? floating(-1) : scalar(-1);
    auto negative = g.AddSelectOperator(g.AddGreaterOperator(zero, v[0], boolean), minus, zero, output);
    return g.AddSelectOperator(g.AddGreaterOperator(v[0], zero, boolean), one, negative, output);
  }
  if (code == kLiteRtOpCodeTflRound) {
    auto floor = g.AddFloorOperator(v[0], fp);
    auto fraction = g.AddSubOperator(v[0], floor, fp);
    auto half_floor = g.AddFloorOperator(multiply(floor, floating(.5f), fp), fp);
    auto odd = g.AddGreaterOperator(g.AddSubOperator(floor, multiply(half_floor, floating(2), fp), fp), floating(0), boolean);
    auto tie = g.AddLogicalAndOperator(g.AddEqualOperator(fraction, floating(.5f), boolean), odd, boolean);
    auto up = g.AddLogicalOrOperator(g.AddGreaterOperator(fraction, floating(.5f), boolean), tie, boolean);
    return g.AddSelectOperator(up, g.AddAddOperator(floor, floating(1), fp), floor, fp);
  }
  if (code == kLiteRtOpCodeTflElu) {
    auto negative = g.AddSubOperator(g.AddExpOperator(v[0], fp), floating(1), fp);
    return g.AddSelectOperator(g.AddGreaterEqualOperator(v[0], floating(0), boolean), v[0], negative, fp);
  }
  if (code == kLiteRtOpCodeTflSquaredDifference) {
    auto difference = g.AddSubOperator(v[0], v[1], fp);
    return multiply(difference, difference, fp);
  }
  if (code == kLiteRtOpCodeTflAtan2) {
    // Reduce to atan(z), |z| <= sqrt(2)-1, then use the convergent odd
    // Taylor series. This needs only PRO-FP operations and no INT16 TABLE.
    for (size_t i = 0; i < 2; ++i) {
      auto source = p.inputs[i].shape;
      source.insert(source.begin(), rank - source.size(), 1);
      Shape multiples(rank);
      for (size_t a = 0; a < rank; ++a) multiples[a] = shape[a] / source[a];
      if (source != shape) v[i] = g.AddTileOperator(v[i],
          g.AddTensorConstant(Attribute(multiples)), fp);
    }
    auto ay = g.AddAbsOperator(v[0], fp);
    auto ax = g.AddAbsOperator(v[1], fp);
    auto small = g.AddMinimumOperator(ay, ax, Attribute(uint32_t{0}), fp);
    auto large = g.AddMaximumOperator(ay, ax, Attribute(uint32_t{0}), fp);
    auto factor = g.AddSelectOperator(g.AddGreaterOperator(floating(1e-10f), large, boolean),
        floating(1e10f), floating(1), fp);
    factor = g.AddSelectOperator(g.AddGreaterOperator(large, floating(1e10f), boolean),
        floating(1e-10f), factor, fp);
    auto denominator = multiply(large, factor, fp);
    denominator = g.AddSelectOperator(g.AddEqualOperator(large, floating(0), boolean), floating(1), denominator, fp);
    auto t = multiply(multiply(small, factor, fp), g.AddReciprocalOperator(denominator, fp), fp);
    auto both_infinite = g.AddEqualOperator(small, floating(std::numeric_limits<float>::infinity()), boolean);
    t = g.AddSelectOperator(both_infinite, floating(1), t, fp);
    auto reduced = g.AddGreaterOperator(t, floating(.414213562373095f), boolean);
    auto z = g.AddSelectOperator(reduced,
        multiply(g.AddSubOperator(t, floating(1), fp),
            g.AddReciprocalOperator(g.AddAddOperator(t, floating(1), fp), fp), fp), t, fp);
    auto square = multiply(z, z, fp);
    auto series = floating(-1.0f / 19);
    for (int n = 8; n >= 0; --n)
      series = g.AddAddOperator(floating((n % 2 ? -1.0f : 1.0f) / (2 * n + 1)),
                                  multiply(square, series, fp), fp);
    auto angle = g.AddAddOperator(multiply(z, series, fp),
        g.AddSelectOperator(reduced, floating(.785398163397448f), floating(0), fp), fp);
    angle = g.AddSelectOperator(g.AddGreaterOperator(ay, ax, boolean),
        g.AddSubOperator(floating(1.570796326794897f), angle, fp), angle, fp);
    auto negative = [&](ResId value) {
      return g.AddLogicalOrOperator(g.AddGreaterOperator(floating(0), value, boolean),
          g.AddGreaterOperator(floating(0), g.AddReciprocalOperator(value, fp), boolean), boolean);
    };
    angle = g.AddSelectOperator(negative(v[1]),
        g.AddSubOperator(floating(3.141592653589793f), angle, fp), angle, fp);
    angle = g.AddSelectOperator(negative(v[0]),
        g.AddNegateOperator(angle, g.AddTensorConstant(Attribute(0.0f)),
            g.AddTensorConstant(Attribute(0.0f)), fp), angle, fp);
    auto finite_or_infinite = g.AddLogicalAndOperator(g.AddEqualOperator(v[0], v[0], boolean),
        g.AddEqualOperator(v[1], v[1], boolean), boolean);
    return g.AddSelectOperator(finite_or_infinite, angle, g.AddAddOperator(v[0], v[1], fp), fp);
  }
  if (code == kLiteRtOpCodeTflFloorDiv || code == kLiteRtOpCodeTflFloorMod) {
    if (type == DataType::int32_t) {
      auto quotient = g.AddIntDivOperator(v[0], v[1], wide);
      auto remainder = g.AddSubOperator(v[0], multiply(quotient, v[1], wide), wide);
      return g.AddSelectOperator(g.AddGreaterOperator(scalar(0), remainder, boolean),
          g.AddAddOperator(remainder, v[1], wide), remainder, wide);
    }
    auto lhs = p.inputs[0].shape, rhs = p.inputs[1].shape;
    lhs.insert(lhs.begin(), rank - lhs.size(), 1);
    rhs.insert(rhs.begin(), rank - rhs.size(), 1);
    // These constant reciprocals are exact powers of two. A general native
    // RECIPROCAL can turn floor(41/41) into zero, so unsupported divisors
    // must be rejected rather than relying on approximate division here.
    auto reciprocal = Vector(g, p.coefficients, rhs);
    if (code == kLiteRtOpCodeTflFloorDiv)
      return g.AddFloorOperator(multiply(v[0], reciprocal, fp), fp);
    auto absolute_a = g.AddAbsOperator(v[0], TT(DataType::float32_t, lhs));
    auto absolute_b = g.AddAbsOperator(v[1], TT(DataType::float32_t, rhs));
    auto threshold = multiply(absolute_b, floating(16777216.0f), TT(DataType::float32_t, rhs));
    auto small = g.AddGreaterOperator(absolute_b, absolute_a, boolean);
    auto large = g.AddGreaterEqualOperator(absolute_a, threshold, boolean);
    auto special = g.AddLogicalOrOperator(small, large, boolean);
    auto source = g.AddSelectOperator(special, floating(0), v[0], fp);
    auto negative_a = g.AddGreaterOperator(floating(0), v[0], TT(DataType::bool_t, lhs));
    auto negative_b = g.AddGreaterOperator(floating(0), v[1], TT(DataType::bool_t, rhs));
    auto quotient = g.AddFloorOperator(multiply(g.AddAbsOperator(source, fp), reciprocal, fp), fp);
    quotient = g.AddSelectOperator(negative_a,
        g.AddNegateOperator(quotient, g.AddTensorConstant(Attribute(0.0f)),
            g.AddTensorConstant(Attribute(0.0f)), fp), quotient, fp);
    // Truncating before multiplication keeps q*|b| finite. For |a/b| >=
    // 2^24, FP32 a is already an exact multiple of the dyadic divisor;
    // forming the quotient could overflow, but its remainder is signed zero.
    auto remainder = g.AddSubOperator(source, multiply(quotient, absolute_b, fp), fp);
    remainder = g.AddSelectOperator(small, v[0], remainder, fp);
    remainder = g.AddSelectOperator(large, multiply(v[0], floating(0), fp), remainder, fp);
    auto nonzero = g.AddLogicalNotOperator(g.AddEqualOperator(remainder, floating(0), boolean), boolean);
    auto opposite = g.AddLogicalXorOperator(negative_a, negative_b, boolean);
    auto adjust = g.AddLogicalAndOperator(nonzero, opposite, boolean);
    return g.AddSelectOperator(adjust, g.AddAddOperator(remainder, v[1], fp), remainder, fp);
  }
  if (code == kLiteRtOpCodeTflLogSoftmax || code == kLiteRtOpCodeTflL2Normalization) {
    auto reduced = shape; reduced.back() = 1;
    const TT row(DataType::float32_t, reduced);
    const Attribute axis(static_cast<int32_t>(rank - 1));
    if (code == kLiteRtOpCodeTflLogSoftmax) {
      auto maximum = g.AddReduceMaxOperator(v[0], axis, Attribute(uint32_t{0}), row);
      auto shifted = g.AddSubOperator(v[0], maximum, fp);
      auto sum = g.AddReduceSumOperator(g.AddExpOperator(shifted, fp), axis, row);
      return g.AddSubOperator(shifted, g.AddLogOperator(sum, row), fp);
    }
    auto sum = g.AddReduceSumOperator(multiply(v[0], v[0], fp), axis, row);
    // The float reference clamps the L2 norm to 1e-6.
    auto bounded = g.AddMaximumOperator(sum, Scalar(g, 1e-12f, rank), Attribute(uint32_t{0}), row);
    return multiply(v[0], g.AddRsqrtOperator(bounded, row), fp);
  }
  return nullptr;
}
}  // namespace litert::arm_vulkan_ml
