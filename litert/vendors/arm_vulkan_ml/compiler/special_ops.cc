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

#include "litert/vendors/arm_vulkan_ml/compiler/special_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "flatbuffers/flexbuffers.h"
#include "litert/c/litert_op_options.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/emission.h"

namespace litert::arm_vulkan_ml {
namespace {
using D = tfsc::tosa::DataType;
using T = tfsc::tosa::Tensor;
using A = tfsc::tosa::Attribute;
using G = tfsc::tosa::Graph;
using V = tfsc::tosa::ResId;
using Shape = std::vector<uint32_t>;
uint64_t Count(const Shape& s) {
  return std::accumulate(s.begin(), s.end(), uint64_t{1},
                         std::multiplies<uint64_t>());
}
std::optional<QuantizedTensor> Read(const Tensor& tensor) {
  auto type = tensor.RankedTensorType();
  if (!type || type->Layout().HasStrides()) return std::nullopt;
  QuantizedTensor q{{}, 1, 0, type->ElementType() == ElementType::Float32};
  if (!q.floating) {
    if ((type->ElementType() != ElementType::Int8 &&
         type->ElementType() != ElementType::Int16) ||
        tensor.QTypeId() != QuantizationTypeId::PerTensor)
      return std::nullopt;
    auto params = tensor.PerTensorQuantization();
    const int lo = type->ElementType() == ElementType::Int8 ? -128 : -32768;
    const int hi = type->ElementType() == ElementType::Int8 ? 127 : 32767;
    if (!std::isfinite(params.scale) || params.scale <= 0 ||
        params.zero_point < lo || params.zero_point > hi)
      return std::nullopt;
    q.scale = params.scale;
    q.zero_point = params.zero_point;
    if (type->ElementType() == ElementType::Int16 && q.zero_point != 0)
      return std::nullopt;
  }
  uint64_t count = 1;
  for (auto dim : type->Layout().Dimensions()) {
    if (dim <= 0 || dim > 8192 ||
        count > (64u * 1024u * 1024u) / static_cast<uint32_t>(dim))
      return std::nullopt;
    count *= dim;
    q.shape.push_back(static_cast<uint32_t>(dim));
  }
  if (q.shape.size() > 6 || Count(q.shape) > 64u * 1024u * 1024u)
    return std::nullopt;
  if (q.shape.empty()) q.shape = {1};
  return q;
}
bool SameDimensions(const Tensor& a, const Tensor& b) {
  auto x = a.RankedTensorType(), y = b.RankedTensorType();
  if (!x || !y) return false;
  const auto xd = x->Layout().Dimensions(), yd = y->Layout().Dimensions();
  return xd.size() == yd.size() && std::equal(xd.begin(), xd.end(), yd.begin());
}
D Type(ElementType type) {
  return type == ElementType::Float32 ? D::float32_t
         : type == ElementType::Int16 ? D::int16_t
                                      : D::int8_t;
}
bool Scales(const Tensor& tensor, uint32_t n, std::vector<double>& scales) {
  if (tensor.QTypeId() == QuantizationTypeId::PerTensor) {
    auto q = tensor.PerTensorQuantization();
    if (q.zero_point) return false;
    scales.assign(n, q.scale);
  } else if (tensor.QTypeId() == QuantizationTypeId::PerChannel) {
    auto q = tensor.PerChannelQuantization();
    if (q.quantized_dimension != 0 || q.num_channels != n) return false;
    for (uint32_t i = 0; i < n; ++i) {
      if (q.zero_points[i]) return false;
      scales.push_back(q.scales[i]);
    }
  } else
    return false;
  return std::all_of(scales.begin(), scales.end(),
                     [](double s) { return std::isfinite(s) && s > 0; });
}
template <typename X>
V Constant(G& g, const std::vector<X>& data, const Shape& shape,
           GraphConstantPool& pool) {
  std::vector<uint8_t> bytes(data.size() * sizeof(X));
  std::memcpy(bytes.data(), data.data(), bytes.size());
  return AddGraphConstant(g, T(tfsc::tosa::GetDataType<X>(), shape),
                          std::move(bytes), pool);
}
V Reshape(G& g, V x, D type, const Shape& shape) {
  return ReshapeTensor(g, x, g.AddTensorConstant(A(shape)), T(type, shape));
}
}  // namespace

bool IsSpecialOperation(const Op& op) {
  if (op.Code() == kLiteRtOpCodeShloComposite) return true;
  if (op.Inputs().empty() || op.Outputs().empty()) return false;
  if (op.Code() == kLiteRtOpCodeTflFullyConnected)
    return op.Inputs()[0].ElementType() == ElementType::Int8 ||
           op.Inputs()[0].ElementType() == ElementType::Int16;
  const bool wide = op.Inputs()[0].ElementType() == ElementType::Int16 ||
                    op.Outputs()[0].ElementType() == ElementType::Int16;
  return wide && (op.Code() == kLiteRtOpCodeTflQuantize ||
                  op.Code() == kLiteRtOpCodeTflDequantize ||
                  op.Code() == kLiteRtOpCodeTflMinimum ||
                  op.Code() == kLiteRtOpCodeTflMaximum ||
                  op.Code() == kLiteRtOpCodeTflSub);
}
std::optional<OperationPlan> AnalyzeSpecialOperation(const Op& op,
                                                     std::string& reason) {
  auto fail = [&](const char* text) -> std::optional<OperationPlan> {
    reason = text;
    return std::nullopt;
  };
  const auto in = op.Inputs();
  const auto out = op.Outputs();
  if (in.empty() || out.size() != 1)
    return fail("Expected data operands and one result");
  auto output = Read(out[0]);
  if (!output)
    return fail("Invalid static special-operation output or quantization");
  OperationPlan plan{};
  plan.code = op.Code();
  auto p = std::make_shared<SpecialPlan>();
  p->output = *output;
  p->input_type = Type(in[0].ElementType());
  p->output_type = Type(out[0].ElementType());
  auto add = [&](size_t i) {
    if (i >= in.size()) return false;
    auto info = Read(in[i]);
    if (!info) return false;
    p->inputs.push_back(*info);
    plan.runtime_inputs.push_back(static_cast<int>(i));
    return true;
  };
  if (!add(0))
    return fail("Invalid static special-operation input or quantization");
  const auto shape = p->inputs[0].shape;
  auto original_type = in[0].RankedTensorType();
  if (op.Code() == kLiteRtOpCodeShloComposite) {
    const auto* opts =
        internal::GetTflOptions2(*op.Get()).AsStableHLOCompositeOptions();
    const auto* registration =
        opts ? FindNormalizationRegistration(opts->name) : nullptr;
    if (!opts || !registration || opts->version != registration->version ||
        opts->composite_attributes.empty() ||
        !flexbuffers::VerifyBuffer(opts->composite_attributes.data(),
                                   opts->composite_attributes.size()))
      return fail(
          "Unregistered normalization composite version or invalid attributes");
    auto root = flexbuffers::GetRoot(opts->composite_attributes);
    if (!root.IsMap()) return fail("Normalization attributes must be a map");
    auto attrs = root.AsMap();
    if (!attrs["epsilon"].IsNumeric())
      return fail("Normalization requires epsilon");
    p->epsilon = attrs["epsilon"].AsDouble();
    if (!std::isfinite(p->epsilon) ||
        !std::isfinite(static_cast<float>(p->epsilon)) ||
        static_cast<float>(p->epsilon) <= 0 || !p->inputs[0].floating ||
        !p->output.floating || shape != output->shape ||
        !SameDimensions(in[0], out[0]) ||
        original_type->Layout().Dimensions().empty())
      return fail(
          "Normalization requires matching static FP32 data and positive "
          "finite epsilon");
    if (!attrs["axis"].IsNull() &&
        (!attrs["axis"].IsInt() ||
         (attrs["axis"].AsInt64() != -1 &&
          attrs["axis"].AsInt64() != static_cast<int64_t>(shape.size()) - 1)))
      return fail("Normalization currently requires the last axis");
    if (!attrs["axes"].IsNull())
      return fail("Normalization axes lists are not supported");
    if (!attrs["channel_axis"].IsNull()) {
      if (!attrs["channel_axis"].IsInt())
        return fail("Normalization channel_axis must be an integer");
      int64_t axis = attrs["channel_axis"].AsInt64();
      if (axis < 0) axis += shape.size();
      if (axis != static_cast<int64_t>(shape.size()) - 1)
        return fail("Normalization currently requires the last axis");
    }
    if (opts->name == "odml.group_norm") {
      if (!attrs["num_groups"].IsInt() || attrs["num_groups"].AsInt64() != 1 ||
          !attrs["sub_type"].IsInt() || attrs["sub_type"].AsInt64() != 1 ||
          in.size() < 2 || in.size() > 3)
        return fail("Only the one-group LayerNorm composite form is supported");
      p->kind = SpecialPlan::Kind::kLayerNorm;
    } else if (opts->name == "odml.rms_norm") {
      if (in.size() != 2) return fail("RMSNorm requires data and scale");
      p->kind = SpecialPlan::Kind::kRmsNorm;
    } else {
      if (in.size() != 1) return fail("L2Norm requires one data input");
      p->kind = SpecialPlan::Kind::kL2Norm;
    }
    for (size_t i = 1; i < in.size(); ++i)
      if (!add(i) || !p->inputs.back().floating ||
          p->inputs.back().shape != Shape{shape.back()})
        return fail("Normalization scale/bias must be FP32 channel vectors");
  } else if (op.Code() == kLiteRtOpCodeTflFullyConnected) {
    p->kind = SpecialPlan::Kind::kFullyConnected;
    const bool wide = p->input_type == D::int16_t;
    if (in.size() != 3 || p->output_type != p->input_type)
      return fail("FC requires matching INT8 or INT16 data types");
    auto wt = in[1].RankedTensorType(), bt = in[2].RankedTensorType();
    if (!wt || !bt || wt->Layout().HasStrides() || bt->Layout().HasStrides() ||
        wt->ElementType() != ElementType::Int8 ||
        bt->ElementType() != (wide ? ElementType::Int64 : ElementType::Int32) ||
        wt->Layout().Dimensions().size() != 2 ||
        bt->Layout().Dimensions().size() != 1)
      return fail(
          "FC requires constant INT8 matrix weights and matching INT32/INT64 "
          "bias");
    auto dims = wt->Layout().Dimensions();
    if (dims[0] <= 0 || dims[0] > 8192 || dims[1] <= 0 || dims[1] > 8192 ||
        bt->Layout().Dimensions()[0] != dims[0])
      return fail("FC weight/bias dimensions disagree");
    uint32_t format, fused;
    bool keep;
    if (LiteRtGetFullyConnectedWeightsFormatOption(op.Get(), &format) !=
            kLiteRtStatusOk ||
        format ||
        LiteRtGetFullyConnectedKeepNumDimsOption(op.Get(), &keep) !=
            kLiteRtStatusOk ||
        LiteRtGetFullyConnectedFusedActivationOption(op.Get(), &fused) !=
            kLiteRtStatusOk ||
        (fused != 0 && fused != 1 && fused != 3))
      return fail("Unsupported FC options");
    p->depth = static_cast<uint32_t>(dims[1]);
    p->channels = static_cast<uint32_t>(dims[0]);
    if (Count(shape) % p->depth || Count(shape) / p->depth > 8192)
      return fail("FC flattened batch exceeds limits");
    p->batch = static_cast<uint32_t>(Count(shape) / p->depth);
    Shape expected{p->batch, p->channels};
    if (keep) {
      if (original_type->Layout().Dimensions().empty())
        return fail("FC keep_num_dims requires non-scalar input");
      if (shape.back() != p->depth)
        return fail("FC keep_num_dims contraction mismatch");
      expected = shape;
      expected.back() = p->channels;
    }
    if (expected != output->shape) return fail("FC output dimensions disagree");
    auto weights = in[1].WeightsData<int8_t>();
    if (!weights || weights->size() != uint64_t{p->channels} * p->depth)
      return fail("FC weights are missing or truncated");
    p->weights.assign(weights->begin(), weights->end());
    if (wide) {
      auto bias = in[2].WeightsData<int64_t>();
      if (!bias) return fail("FC INT64 bias is missing");
      p->bias.assign(bias->begin(), bias->end());
    } else {
      auto bias = in[2].WeightsData<int32_t>();
      if (!bias) return fail("FC INT32 bias is missing");
      p->bias.assign(bias->begin(), bias->end());
    }
    std::vector<double> weights_scale, bias_scale;
    if (p->bias.size() != p->channels ||
        !Scales(in[1], p->channels, weights_scale) ||
        !Scales(in[2], p->channels, bias_scale))
      return fail("FC constant quantization is invalid");
    const int64_t low = wide ? -32768 : -128, high = wide ? 32767 : 127;
    const int64_t limit = wide ? (int64_t{1} << 47) : (int64_t{1} << 31);
    for (size_t c = 0; c < p->channels; ++c) {
      if (p->bias[c] < -limit || p->bias[c] >= limit)
        return fail("FC bias exceeds accumulator range");
      int64_t lo = p->bias[c], hi = lo;
      for (size_t k = 0; k < p->depth; ++k) {
        const int64_t w = p->weights[c * p->depth + k];
        const auto a = (low - p->inputs[0].zero_point) * w,
                   b = (high - p->inputs[0].zero_point) * w;
        lo += std::min(a, b);
        hi += std::max(a, b);
      }
      if (lo < -limit || hi >= limit)
        return fail("FC cannot prove accumulator range");
      const double scale = p->inputs[0].scale * weights_scale[c];
      if (std::abs(bias_scale[c] - scale) > scale * 1e-5)
        return fail("FC bias scales disagree");
      auto s = MakeScale32(scale / output->scale);
      if (!s || (!wide && !IsScale32RangeValid(*s, lo, hi)))
        return fail("FC rescale range is invalid");
      p->scales.push_back(*s);
      p->minimum.push_back(lo);
      p->maximum.push_back(hi);
    }
    p->activation = {static_cast<int32_t>(low), static_cast<int32_t>(high)};
    if (fused)
      p->activation.minimum =
          std::max(p->activation.minimum, output->zero_point);
    if (fused == 3)
      p->activation.maximum = static_cast<int32_t>(
          std::clamp(std::round(6.0 / output->scale) + output->zero_point,
                     static_cast<double>(low), static_cast<double>(high)));
    if (wide) plan.required_extensions = {"EXT-INT16"};
  } else if (op.Code() == kLiteRtOpCodeTflQuantize ||
             op.Code() == kLiteRtOpCodeTflDequantize) {
    p->kind = op.Code() == kLiteRtOpCodeTflQuantize
                  ? SpecialPlan::Kind::kQuantize
                  : SpecialPlan::Kind::kDequantize;
    if (in.size() != 1 || shape != output->shape ||
        !SameDimensions(in[0], out[0]) ||
        (p->kind == SpecialPlan::Kind::kQuantize
             ? !p->inputs[0].floating || p->output_type != D::int16_t
             : p->input_type != D::int16_t || !p->output.floating))
      return fail("INT16 conversion requires FP32/INT16 with matching shapes");
    if (p->kind == SpecialPlan::Kind::kQuantize &&
        !std::isfinite(static_cast<float>(1.0 / output->scale)))
      return fail("INT16 quantization reciprocal scale exceeds FP32");
    plan.required_extensions = {"EXT-INT16"};
  } else if (op.Code() == kLiteRtOpCodeTflSub) {
    if (in.size() != 2 || !add(1) ||
        in[1].ElementType() != ElementType::Int16 ||
        p->input_type != D::int16_t || p->output_type != D::int16_t)
      return fail("INT16 SUB requires two INT16 operands");
    Shape a = p->inputs[0].shape, b = p->inputs[1].shape;
    const size_t rank = std::max(a.size(), b.size());
    a.insert(a.begin(), rank - a.size(), 1);
    b.insert(b.begin(), rank - b.size(), 1);
    Shape expected;
    for (size_t i = 0; i < rank; ++i) {
      if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
        return fail("INT16 SUB broadcast mismatch");
      expected.push_back(std::max(a[i], b[i]));
    }
    for (const auto& q : p->inputs)
      if (q.floating || q.scale != output->scale ||
          q.zero_point != output->zero_point)
        return fail(
            "INT16 SUB currently requires identical symmetric quantization");
    uint32_t fused;
    if (expected != output->shape ||
        LiteRtGetSubFusedActivationOption(op.Get(), &fused) !=
            kLiteRtStatusOk ||
        fused)
      return fail(
          "INT16 SUB requires matching output shape and no fused activation");
    p->kind = SpecialPlan::Kind::kSubtract;
    plan.required_extensions = {"EXT-INT16"};
  } else {
    if (in.size() != 2 || !add(1) || p->input_type != D::int16_t ||
        p->output_type != D::int16_t)
      return fail("INT16 bound requires matching integer operands");
    for (const auto& q : p->inputs)
      if (q.floating || q.scale != output->scale ||
          q.zero_point != output->zero_point)
        return fail("INT16 bound quantization must match");
    if (p->inputs[1].shape != Shape{1} || !in[1].HasWeights() ||
        shape != output->shape)
      return fail("INT16 bound requires a scalar constant");
    auto bound = in[1].WeightsData<int16_t>();
    if (!bound || bound->size() != 1)
      return fail("Invalid INT16 bound constant");
    p->activation = {-32768, 32767};
    if (op.Code() == kLiteRtOpCodeTflMinimum)
      p->activation.maximum = (*bound)[0];
    else
      p->activation.minimum = (*bound)[0];
    p->kind = op.Code() == kLiteRtOpCodeTflMinimum
                  ? SpecialPlan::Kind::kMinimum
                  : SpecialPlan::Kind::kMaximum;
    p->inputs.resize(1);
    plan.runtime_inputs.resize(1);
    plan.required_extensions = {"EXT-INT16"};
  }
  plan.special = std::move(p);
  plan.inputs = plan.special->inputs;
  plan.output = *output;
  reason.clear();
  return plan;
}

V EmitSpecialOperation(G& g, const std::vector<V>& v, const SpecialPlan& p,
                       GraphConstantPool& pool) {
  const auto& shape = p.output.shape;
  const T out(p.output_type, shape), fp(D::float32_t, shape);
  const auto scalar = [&](float x) {
    return BroadcastScalar(g, x, shape.size(), pool);
  };
  const auto zero_shift = g.AddTensorConstant(A(int8_t{0}));
  auto multiply = [&](V a, V b, const T& type) {
    return g.AddMulOperator(a, b, zero_shift, type);
  };
  if (p.kind == SpecialPlan::Kind::kFullyConnected) {
    const bool wide = p.input_type == D::int16_t;
    Shape expanded{p.batch, 1, 1, p.channels};
    auto x = Reshape(g, v[0], p.input_type, {p.batch, 1, 1, p.depth});
    auto w = Constant(g, p.weights, {p.channels, 1, 1, p.depth}, pool);
    V bias;
    if (wide)
      bias = Constant(g, p.bias, {p.channels}, pool);
    else
      bias = Constant(g, std::vector<int32_t>(p.bias.begin(), p.bias.end()),
                      {p.channels}, pool);
    const auto acc = wide ? D::int48_t : D::int32_t;
    auto value = g.AddConv2dOperator(
        x, w, bias,
        g.AddTensorConstant(A(p.inputs[0].zero_point, p.input_type)),
        g.AddTensorConstant(A(int8_t{0})), A(std::vector<int32_t>{0, 0, 0, 0}),
        A(std::vector<int32_t>{1, 1}), A(std::vector<int32_t>{1, 1}),
        A(int32_t{wide ? 4 : 1}), A(false), T(acc, expanded));
    value = EmitNativeRescale(g, value, acc, p.output_type, expanded, 0,
                              p.output.zero_point, p.scales, pool, p.minimum,
                              p.maximum);
    const int low = wide ? -32768 : -128, high = wide ? 32767 : 127;
    if (p.activation.minimum != low || p.activation.maximum != high)
      value = g.AddClampOperator(value, A(p.activation.minimum, p.output_type),
                                 A(p.activation.maximum, p.output_type),
                                 A(int32_t{1}), T(p.output_type, expanded));
    return Reshape(g, value, p.output_type, shape);
  }
  if (p.kind == SpecialPlan::Kind::kSubtract) {
    std::vector<V> expanded;
    for (size_t i = 0; i < v.size(); ++i) {
      auto operand_shape = p.inputs[i].shape;
      auto value = g.AddCastOperator(v[i], T(D::int32_t, operand_shape));
      operand_shape.insert(operand_shape.begin(),
                           shape.size() - operand_shape.size(), 1);
      expanded.push_back(Reshape(g, value, D::int32_t, operand_shape));
    }
    auto value =
        g.AddSubOperator(expanded[0], expanded[1], T(D::int32_t, shape));
    // The difference fits INT32. A unit rescale saturates to INT16 exactly;
    // narrowing the subtraction before saturation would wrap at the endpoints.
    return EmitNativeRescale(g, value, D::int32_t, D::int16_t, shape, 0, 0,
                             {*MakeScale32(1)}, pool, {-65535}, {65535});
  }
  if (p.kind == SpecialPlan::Kind::kMinimum ||
      p.kind == SpecialPlan::Kind::kMaximum)
    return g.AddClampOperator(
        v[0], A(static_cast<int16_t>(p.activation.minimum)),
        A(static_cast<int16_t>(p.activation.maximum)), A(int32_t{1}), out);
  if (p.kind == SpecialPlan::Kind::kDequantize) {
    auto value = g.AddCastOperator(v[0], fp);
    return multiply(value, scalar(static_cast<float>(p.inputs[0].scale)), fp);
  }
  if (p.kind == SpecialPlan::Kind::kQuantize) {
    auto value =
        multiply(v[0], scalar(static_cast<float>(1.0 / p.output.scale)), fp);
    auto positive =
        g.AddGreaterEqualOperator(value, scalar(0), T(D::bool_t, shape));
    auto up = g.AddFloorOperator(g.AddAddOperator(value, scalar(.5f), fp), fp);
    auto negative = g.AddSubOperator(scalar(0), value, fp);
    auto down = g.AddSubOperator(
        scalar(0),
        g.AddFloorOperator(g.AddAddOperator(negative, scalar(.5f), fp), fp),
        fp);
    value = g.AddSelectOperator(positive, up, down, fp);
    value =
        g.AddClampOperator(value, A(-32768.0f), A(32767.0f), A(int32_t{0}), fp);
    return g.AddCastOperator(value, out);
  }
  auto row_shape = shape;
  row_shape.back() = 1;
  const T row(D::float32_t, row_shape);
  const A axis(static_cast<int32_t>(shape.size() - 1));
  auto mean = [&](V value) {
    return multiply(g.AddReduceSumOperator(value, axis, row),
                    scalar(1.0f / shape.back()), row);
  };
  auto centered = v[0];
  if (p.kind == SpecialPlan::Kind::kLayerNorm)
    centered = g.AddSubOperator(v[0], mean(v[0]), fp);
  auto square = multiply(centered, centered, fp);
  auto variance = p.kind == SpecialPlan::Kind::kL2Norm
                      ? g.AddReduceSumOperator(square, axis, row)
                      : mean(square);
  auto bounded =
      g.AddAddOperator(variance, scalar(static_cast<float>(p.epsilon)), row);
  auto value = multiply(centered, g.AddRsqrtOperator(bounded, row), fp);
  if (v.size() > 1)
    value = multiply(value,
                     Reshape(g, v[1], D::float32_t,
                             [&] {
                               auto s = Shape(shape.size(), 1);
                               s.back() = shape.back();
                               return s;
                             }()),
                     fp);
  if (v.size() > 2)
    value = g.AddAddOperator(value,
                             Reshape(g, v[2], D::float32_t,
                                     [&] {
                                       auto s = Shape(shape.size(), 1);
                                       s.back() = shape.back();
                                       return s;
                                     }()),
                             fp);
  return value;
}
}  // namespace litert::arm_vulkan_ml
