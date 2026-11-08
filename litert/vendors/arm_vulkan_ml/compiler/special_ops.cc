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
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <string>
#include <optional>
#include <vector>

#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/emission.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

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
  const auto dt = type->ElementType() == ElementType::Float32 ? D::float32_t
                  : type->ElementType() == ElementType::Int32 ? D::int32_t
                  : type->ElementType() == ElementType::Int16 ? D::int16_t
                                                              : D::int8_t;
  QuantizedTensor q{
      {}, 1, 0, type->ElementType() == ElementType::Float32, dt};
  if (!q.floating) {
    if ((type->ElementType() != ElementType::Int8 &&
         type->ElementType() != ElementType::Int16 &&
         type->ElementType() != ElementType::Int32) ||
        tensor.QTypeId() != QuantizationTypeId::PerTensor)
      return std::nullopt;
    auto params = tensor.PerTensorQuantization();
    const int64_t lo = type->ElementType() == ElementType::Int8    ? -128
                       : type->ElementType() == ElementType::Int16 ? -32768
                                                                   : INT32_MIN;
    const int64_t hi = type->ElementType() == ElementType::Int8    ? 127
                       : type->ElementType() == ElementType::Int16 ? 32767
                                                                   : INT32_MAX;
    if (!std::isfinite(params.scale) || params.scale <= 0 ||
        params.zero_point < lo || params.zero_point > hi)
      return std::nullopt;
    q.scale = params.scale;
    q.zero_point = static_cast<int32_t>(params.zero_point);
    if ((type->ElementType() == ElementType::Int16 ||
         type->ElementType() == ElementType::Int32) &&
        q.zero_point != 0)
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
         : type == ElementType::Int32 ? D::int32_t
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
  const auto in = op.Inputs();
  const auto out = op.Outputs();
  if (in.empty() || out.empty()) return false;
  if (op.Code() == kLiteRtOpCodeTflFullyConnected)
    return in[0].ElementType() == ElementType::Int8 ||
           in[0].ElementType() == ElementType::Int16;
  if (op.Code() == kLiteRtOpCodeTflQuantize) {
    const auto a = in[0].ElementType(), b = out[0].ElementType();
    return (a == ElementType::Float32 && b == ElementType::Int16) ||
           (a == ElementType::Int16 && b == ElementType::Int8) ||
           (a == ElementType::Int8 && b == ElementType::Int16);
  }
  if (op.Code() == kLiteRtOpCodeTflDequantize)
    return in[0].ElementType() == ElementType::Int16 &&
           out[0].ElementType() == ElementType::Float32;
  if (op.Code() == kLiteRtOpCodeTflMinimum ||
      op.Code() == kLiteRtOpCodeTflMaximum) {
    if (in.size() != 2 || in[0].ElementType() != ElementType::Int16 ||
        in[1].ElementType() != ElementType::Int16 ||
        out[0].ElementType() != ElementType::Int16 || !in[1].HasWeights() ||
        !SameDimensions(in[0], out[0])) {
      return false;
    }
    auto bound = Read(in[1]);
    return bound && bound->shape == Shape{1};
  }
  return false;
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
    p->input_types.push_back(Type(in[i].ElementType()));
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
    const bool fp_norm = p->inputs[0].floating && p->output.floating;
    const bool i16_norm =
        p->input_type == D::int16_t&& p->output_type ==
        D::int16_t&& std::isfinite(static_cast<float>(1.0 / output->scale));
    if (!std::isfinite(p->epsilon) ||
        !std::isfinite(static_cast<float>(p->epsilon)) ||
        static_cast<float>(p->epsilon) <= 0 || (!fp_norm && !i16_norm) ||
        shape != output->shape || !SameDimensions(in[0], out[0]) ||
        original_type->Layout().Dimensions().empty())
      return fail(
          "Normalization requires matching static FP32/INT16 data and positive "
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
    for (size_t i = 1; i < in.size(); ++i) {
      if (!add(i) || p->inputs.back().shape != Shape{shape.back()})
        return fail("Normalization scale/bias must be channel vectors");
      if (fp_norm) {
        if (!p->inputs.back().floating)
          return fail("FP32 normalization scale/bias must be FP32");
      } else {
        const auto param_type = p->input_types.back();
        const bool ok = p->inputs.back().floating || param_type == D::int16_t ||
                        (i == 2 && param_type == D::int32_t);
        if (!ok)
          return fail(
              "INT16 normalization scale/bias must be FP32 or symmetric "
              "INT16/INT32");
      }
    }
    if (i16_norm) plan.required_extensions = {"EXT-INT16"};
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
        !SameDimensions(in[0], out[0]))
      return fail("INT16 conversion requires matching shapes");
    if (p->kind == SpecialPlan::Kind::kDequantize) {
      if (p->input_type != D::int16_t || !p->output.floating)
        return fail(
            "INT16 conversion requires FP32/INT16 with matching shapes");
    } else if (p->inputs[0].floating) {
      if (p->output_type != D::int16_t)
        return fail(
            "INT16 conversion requires FP32/INT16 with matching shapes");
      if (!std::isfinite(static_cast<float>(1.0 / output->scale)))
        return fail("INT16 quantization reciprocal scale exceeds FP32");
    } else if ((p->input_type == D::int16_t&& p->output_type == D::int8_t) ||
               (p->input_type == D::int8_t&& p->output_type == D::int16_t)) {
      const int64_t lo = (p->input_type == D::int8_t ? -128 : -32768) -
                         p->inputs[0].zero_point;
      const int64_t hi =
          (p->input_type == D::int8_t ? 127 : 32767) - p->inputs[0].zero_point;
      auto s = MakeScale32(p->inputs[0].scale / output->scale);
      if (!s || !IsScale32RangeValid(*s, lo, hi))
        return fail("INT8/INT16 QUANTIZE rescale range is invalid");
      p->scales = {*s};
      p->minimum = {lo};
      p->maximum = {hi};
    } else {
      return fail("INT16 conversion requires FP32/INT16 with matching shapes");
    }
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
    p->input_types.resize(1);
    plan.runtime_inputs.resize(1);
    plan.required_extensions = {"EXT-INT16"};
  }
  plan.output_type = p->output_type;
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
  auto quantize_fp_to_i16 = [&](V val) {
    auto scaled =
        multiply(val, scalar(static_cast<float>(1.0 / p.output.scale)), fp);
    auto positive =
        g.AddGreaterEqualOperator(scaled, scalar(0), T(D::bool_t, shape));
    auto up = g.AddFloorOperator(g.AddAddOperator(scaled, scalar(.5f), fp), fp);
    auto negative = g.AddSubOperator(scalar(0), scaled, fp);
    auto down = g.AddSubOperator(
        scalar(0),
        g.AddFloorOperator(g.AddAddOperator(negative, scalar(.5f), fp), fp),
        fp);
    scaled = g.AddSelectOperator(positive, up, down, fp);
    scaled = g.AddClampOperator(scaled, A(-32768.0f), A(32767.0f),
                                A(int32_t{0}), fp);
    return g.AddCastOperator(scaled, out);
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
    if (!p.inputs[0].floating) {
      return EmitNativeRescale(g, v[0], p.input_type, p.output_type, shape,
                               p.inputs[0].zero_point, p.output.zero_point,
                               p.scales, pool, p.minimum, p.maximum);
    }
    return quantize_fp_to_i16(v[0]);
  }
  auto row_shape = shape;
  row_shape.back() = 1;
  const T row(D::float32_t, row_shape);
  const A axis(static_cast<int32_t>(shape.size() - 1));
  auto mean = [&](V value) {
    return multiply(g.AddReduceSumOperator(value, axis, row),
                    scalar(1.0f / shape.back()), row);
  };
  auto data = v[0];
  if (!p.inputs[0].floating)
    data = multiply(g.AddCastOperator(data, fp),
                    scalar(static_cast<float>(p.inputs[0].scale)), fp);
  auto centered = data;
  if (p.kind == SpecialPlan::Kind::kLayerNorm)
    centered = g.AddSubOperator(data, mean(data), fp);
  auto square = multiply(centered, centered, fp);
  auto variance = p.kind == SpecialPlan::Kind::kL2Norm
                      ? g.AddReduceSumOperator(square, axis, row)
                      : mean(square);
  auto bounded =
      g.AddAddOperator(variance, scalar(static_cast<float>(p.epsilon)), row);
  auto value = multiply(centered, g.AddRsqrtOperator(bounded, row), fp);
  Shape channel_shape(shape.size(), 1);
  channel_shape.back() = shape.back();
  const T fp_channel(D::float32_t, channel_shape);
  auto channel_param = [&](size_t i) {
    auto param = Reshape(g, v[i], p.input_types[i], channel_shape);
    if (!p.inputs[i].floating) {
      param = g.AddCastOperator(param, fp_channel);
      param = multiply(param, scalar(static_cast<float>(p.inputs[i].scale)),
                       fp_channel);
    }
    return param;
  };
  if (v.size() > 1) value = multiply(value, channel_param(1), fp);
  if (v.size() > 2) value = g.AddAddOperator(value, channel_param(2), fp);
  return p.output.floating ? value : quantize_fp_to_i16(value);
}
}  // namespace litert::arm_vulkan_ml
