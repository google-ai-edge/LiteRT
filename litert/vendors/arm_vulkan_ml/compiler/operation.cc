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

#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_op_code.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_common.h"
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/OperatorEnum.hpp"  // from @tosa_for_spirv_codegen
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d_adapter.h"
#include "litert/vendors/arm_vulkan_ml/compiler/decomposition.h"
#include "litert/vendors/arm_vulkan_ml/compiler/emission.h"
#include "litert/vendors/arm_vulkan_ml/compiler/extended_ops.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"
#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"
#include "litert/vendors/arm_vulkan_ml/compiler/special_ops.h"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::Graph;
using tfsc::tosa::ResId;
using TosaTensor = tfsc::tosa::Tensor;
bool CanEmitNativeRescale(Scale32 s, int64_t lo, int64_t hi) {
  if (!IsScale32RangeValid(s, lo, hi)) return false;
  double scale = std::ldexp(static_cast<double>(s.multiplier), -s.shift);
  if (!(scale > 0) || !std::isfinite(scale)) return false;
  int left = 0;
  while (std::llround(scale * 4.0) > INT16_MAX) {
    scale *= 0.5;
    if (++left > 31) return false;
  }
  auto pair16 = [](double sc) {
    int exponent = 0;
    std::frexp(sc, &exponent);
    int sh = std::min(62, 15 - exponent);
    if (sh < 2) return false;
    int64_t m = std::llround(std::ldexp(sc, sh));
    if (m > INT16_MAX) {
      --sh;
      m = std::llround(std::ldexp(sc, sh));
    }
    return sh >= 2 && m > 0 && m <= INT16_MAX;
  };
  while (left != 0) {
    int step = std::min(12, left);
    int64_t factor = int64_t{1} << step;
    if (lo < INT32_MIN / factor || hi > INT32_MAX / factor) return false;
    if (!pair16(static_cast<double>(factor))) return false;
    lo *= factor;
    hi *= factor;
    left -= step;
  }
  return pair16(scale);
}
std::optional<QuantizedTensor> ReadTensor(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type || type->Layout().HasStrides()) return std::nullopt;
  QuantizedTensor result{{}, 1.0, 0};
  if (type->ElementType() == ElementType::Float32) {
    result.floating = true;
    result.type = DataType::float32_t;
  } else if (type->ElementType() == ElementType::Int8) {
    if (t.QTypeId() != QuantizationTypeId::PerTensor) return std::nullopt;
    auto q = t.PerTensorQuantization();
    if (!std::isfinite(q.scale) || q.scale <= 0 || q.zero_point < -128 ||
        q.zero_point > 127)
      return std::nullopt;
    result.scale = q.scale;
    result.zero_point = q.zero_point;
    result.type = DataType::int8_t;
  } else if (type->ElementType() == ElementType::Int16) {
    if (t.QTypeId() != QuantizationTypeId::PerTensor) return std::nullopt;
    auto q = t.PerTensorQuantization();
    if (!std::isfinite(q.scale) || q.scale <= 0 || q.zero_point != 0)
      return std::nullopt;
    result.scale = q.scale;
    result.zero_point = 0;
    result.type = DataType::int16_t;
  } else {
    return std::nullopt;
  }
  uint64_t count = 1;
  auto dims = type->Layout().Dimensions();
  if (dims.size() > 6) return std::nullopt;
  for (auto d : dims) {
    if (d <= 0 || d > 8192 || count > uint64_t{INT32_MAX} / d)
      return std::nullopt;
    count *= d;
    result.shape.push_back(d);
  }
  return result;
}
uint64_t Count(const std::vector<uint32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), uint64_t{1},
                         std::multiplies<uint64_t>());
}
bool SameQuant(const QuantizedTensor& a, const QuantizedTensor& b) {
  return a.floating == b.floating && a.type == b.type && a.scale == b.scale &&
         a.zero_point == b.zero_point;
}
std::optional<Activation> ActivationCode(uint32_t code) {
  switch (code) {
    case 0:
      return Activation::kNone;
    case 1:
      return Activation::kRelu;
    case 3:
      return Activation::kRelu6;
    default:
      return std::nullopt;
  }
}
template <typename T>
ResId Constant(Graph& graph, const std::vector<T>& data,
               const std::vector<uint32_t>& shape,
               GraphConstantPool& constants) {
  const auto type = tfsc::tosa::GetDataType<T>();
  TosaTensor tensor(type, shape);
  GraphConstant constant{tensor, std::vector<uint8_t>(data.size() * sizeof(T))};
  std::memcpy(constant.bytes.data(), data.data(), constant.bytes.size());
  return AddGraphConstant(graph, tensor, std::move(constant.bytes), constants);
}
ResId Rescale(Graph& g, ResId input, const std::vector<uint32_t>& shape,
              DataType input_type, DataType output_type, int32_t izp,
              int32_t ozp, const std::vector<Scale32>& scales,
              GraphConstantPool& constants, bool per_channel = false) {
  return EmitNativeRescale(g, input, input_type, output_type, shape, izp, ozp,
                           scales, constants);
}
ResId Clamp(Graph& g, ResId input, const OperationPlan& p) {
  const int32_t lo = p.output_type == DataType::int16_t ? -32768 : -128;
  const int32_t hi = p.output_type == DataType::int16_t ? 32767 : 127;
  if (p.activation.minimum == lo && p.activation.maximum == hi) return input;
  return g.AddClampOperator(
      input, Attribute(p.activation.minimum, p.output_type),
      Attribute(p.activation.maximum, p.output_type), Attribute(int32_t{1}),
      TosaTensor(p.output_type, p.output.shape));
}
ResId Reshape(Graph& g, ResId input, DataType type,
              const std::vector<uint32_t>& shape) {
  return ReshapeTensor(g, input, g.AddTensorConstant(Attribute(shape)),
                       TosaTensor(type, shape));
}

std::optional<OperationPlan> AnalyzeTypedOperation(const Op& op,
                                                   OperationPlan p,
                                                   std::string& reason) {
  auto reject = [&](const char* s) -> std::optional<OperationPlan> {
    reason = s;
    return std::nullopt;
  };
  auto in = op.Inputs();
  auto out = op.Outputs();
  if (out.size() != 1 || in.empty())
    return reject("Typed operation requires one output");
  auto read = [&](const Tensor& t, QuantizedTensor& q, DataType& type) {
    auto rt = t.RankedTensorType();
    if (!rt || rt->Layout().HasStrides()) return false;
    switch (rt->ElementType()) {
      case ElementType::Bool:
        type = DataType::bool_t;
        break;
      case ElementType::UInt32:
        if (p.code != kLiteRtOpCodeTflRightShift) return false;
        [[fallthrough]];
      case ElementType::Int32:
        type = DataType::int32_t;
        break;
      case ElementType::Float32:
        type = DataType::float32_t;
        break;
      default:
        return false;
    }
    q.floating = type == DataType::float32_t;
    uint64_t count = 1;
    for (auto d : rt->Layout().Dimensions()) {
      if (d <= 0 || d > 8192 || count > uint64_t{INT32_MAX} / d) return false;
      count *= d;
      q.shape.push_back(static_cast<uint32_t>(d));
    }
    return !q.shape.empty() && q.shape.size() <= 6;
  };
  QuantizedTensor lhs;
  DataType lt;
  if (!read(in[0], lhs, lt) || !read(out[0], p.output, p.output_type))
    return reject("Unsupported typed tensor layout/type");
  p.typed = true;
  p.inputs = {lhs};
  p.runtime_inputs = {0};
  p.input_types = {lt};
  auto expected = lhs.shape;
  const bool reduce = p.code == kLiteRtOpCodeTflReduceAll ||
                      p.code == kLiteRtOpCodeTflReduceAny;
  const bool argmax = p.code == kLiteRtOpCodeTflArgMax;
  if (reduce || argmax) {
    if (in.size() != 2) return reject("Expected constant axes");
    auto rt = in[1].RankedTensorType();
    auto axes = in[1].WeightsData<int32_t>();
    if (!rt || rt->ElementType() != ElementType::Int32 ||
        rt->Layout().HasStrides() || !axes || axes->empty())
      return reject("Expected constant INT32 axes");
    std::set<int32_t> unique;
    for (auto a : *axes) {
      if (a < 0) a += static_cast<int32_t>(lhs.shape.size());
      if (a < 0 || a >= lhs.shape.size() || !unique.insert(a).second)
        return reject("Invalid reduction axis");
    }
    p.parameters.assign(unique.begin(), unique.end());
    if (argmax) {
      if (lt != DataType::float32_t || p.output_type != DataType::int32_t ||
          unique.size() != 1)
        return reject("ARGMAX requires FP32 to INT32 and one axis");
      expected.erase(expected.begin() + p.parameters[0]);
    } else {
      if (lt != DataType::bool_t || p.output_type != DataType::bool_t)
        return reject("Logical reduction requires BOOL");
      auto* opts = internal::GetTflOptions(*op.Get()).AsReducerOptions();
      if (!opts) return reject("Missing reduction options");
      expected.clear();
      for (size_t i = 0; i < lhs.shape.size(); ++i) {
        if (!unique.count(static_cast<int32_t>(i)))
          expected.push_back(lhs.shape[i]);
        else if (opts->keep_dims)
          expected.push_back(1);
      }
    }
  } else if (p.code == kLiteRtOpCodeTflLogicalNot) {
    if (in.size() != 1 || lt != DataType::bool_t || p.output_type != lt)
      return reject("LOGICAL_NOT requires BOOL");
  } else {
    QuantizedTensor rhs;
    DataType rt;
    if (in.size() != 2 || !read(in[1], rhs, rt) || lt != rt)
      return reject("Binary types differ");
    if (p.code == kLiteRtOpCodeTflRightShift) {
      auto shifts = in[1].WeightsData<uint32_t>();
      if (in[0].ElementType() != ElementType::UInt32 ||
          in[1].ElementType() != ElementType::UInt32 ||
          out[0].ElementType() != ElementType::UInt32 || !shifts ||
          shifts->empty())
        return reject("RIGHT_SHIFT requires UINT32 data and constant shifts");
      for (auto shift : *shifts)
        if (shift > 31) return reject("RIGHT_SHIFT constant out of range");
    } else if (p.code == kLiteRtOpCodeTflFloorDiv) {
      auto divisors = in[1].WeightsData<int32_t>();
      if (lt != DataType::int32_t || p.output_type != lt || !divisors ||
          divisors->empty())
        return reject("INT32 FLOOR_DIV requires positive constant divisors");
      for (auto d : *divisors)
        if (d <= 0) return reject("FLOOR_DIV divisor must be positive");
    } else if (p.code == kLiteRtOpCodeTflBitwiseXor) {
      if (lt != DataType::int32_t || p.output_type != lt)
        return reject("BITWISE_XOR requires INT32");
    } else if (p.output_type != DataType::bool_t ||
               ((p.code == kLiteRtOpCodeTflLogicalAnd ||
                 p.code == kLiteRtOpCodeTflLogicalOr) &&
                lt != DataType::bool_t))
      return reject("Expected BOOL output");
    size_t rank = std::max(lhs.shape.size(), rhs.shape.size());
    auto a = lhs.shape, b = rhs.shape;
    a.insert(a.begin(), rank - a.size(), 1);
    b.insert(b.begin(), rank - b.size(), 1);
    expected.resize(rank);
    for (size_t i = 0; i < rank; ++i) {
      if (a[i] != b[i] && a[i] != 1 && b[i] != 1)
        return reject("Typed broadcast mismatch");
      expected[i] = std::max(a[i], b[i]);
    }
    if (p.code == kLiteRtOpCodeTflBitwiseXor && a == expected) {
      auto rhs_weights = in[1].WeightsData<int32_t>();
      if (rhs_weights && !rhs_weights->empty() &&
          std::all_of(rhs_weights->begin(), rhs_weights->end(),
                      [](int32_t x) { return x == -1; }))
        p.parameters = {-1};
    }
    p.inputs[0].shape = a;
    rhs.shape = b;
    p.inputs.push_back(rhs);
    p.input_types.push_back(rt);
    p.runtime_inputs.push_back(1);
  }
  if (expected != p.output.shape)
    return reject("Typed output shape mismatch");
  return p;
}
}  // namespace

bool AnalyzeFloatConvolution(const Op& op, OperationPlan& p,
                             std::string& reason) {
  auto in = op.Inputs();
  if (in.size() != 3 || p.inputs[0].shape.size() != 4 ||
      p.output.shape.size() != 4)
    return false;
  auto wt = in[1].RankedTensorType(), bt = in[2].RankedTensorType();
  if (!wt || !bt || wt->ElementType() != ElementType::Float32 ||
      bt->ElementType() != ElementType::Float32 || wt->Layout().HasStrides() ||
      bt->Layout().HasStrides() || wt->Layout().Dimensions().size() != 4 ||
      bt->Layout().Dimensions().size() != 1)
    return false;
  for (auto d : wt->Layout().Dimensions()) {
    if (d <= 0 || d > 8192) return false;
    p.weight_shape.push_back(d);
  }
  auto weights = in[1].WeightsData<float>(), bias = in[2].WeightsData<float>();
  if (!weights || !bias || weights->size() != Count(p.weight_shape) ||
      bias->size() != p.output.shape[3] ||
      bt->Layout().Dimensions()[0] != p.output.shape[3])
    return false;
  p.float_weights.assign(weights->begin(), weights->end());
  p.float_bias.assign(bias->begin(), bias->end());
  for (float v : p.float_weights)
    if (!std::isfinite(v)) return false;
  for (float v : p.float_bias)
    if (!std::isfinite(v)) return false;
  const bool dw = p.code == kLiteRtOpCodeTflDepthwiseConv2d;
  if (p.inputs[0].shape[0] != p.output.shape[0]) return false;
  if (dw) {
    int32_t multiplier;
    if (LiteRtGetDepthwiseConv2dDepthMultiplierOption(op.Get(), &multiplier) !=
            kLiteRtStatusOk ||
        multiplier <= 0 || p.weight_shape[0] != 1 ||
        p.weight_shape[3] != p.output.shape[3] ||
        int64_t{multiplier} * p.inputs[0].shape[3] != p.output.shape[3])
      return false;
  } else if (p.weight_shape[0] != p.output.shape[3] ||
             p.weight_shape[3] != p.inputs[0].shape[3])
    return false;
  auto padding_fn =
      dw ? LiteRtGetDepthwiseConv2dPaddingOption : LiteRtGetConv2dPaddingOption;
  auto activation_fn = dw ? LiteRtGetDepthwiseConv2dFusedActivationOption
                          : LiteRtGetConv2dFusedActivationOption;
  auto sh =
      dw ? LiteRtGetDepthwiseConv2dStrideHOption : LiteRtGetConv2dStrideHOption;
  auto sw =
      dw ? LiteRtGetDepthwiseConv2dStrideWOption : LiteRtGetConv2dStrideWOption;
  auto dh = dw ? LiteRtGetDepthwiseConv2dDilationHOption
               : LiteRtGetConv2dDilationHOption;
  auto dw_fn = dw ? LiteRtGetDepthwiseConv2dDilationWOption
                  : LiteRtGetConv2dDilationWOption;
  uint32_t padding;
  p.stride.resize(2);
  p.dilation.resize(2);
  p.padding.assign(4, 0);
  if (padding_fn(op.Get(), &padding) != kLiteRtStatusOk || padding > 1 ||
      activation_fn(op.Get(), &p.float_activation) != kLiteRtStatusOk ||
      !ActivationCode(p.float_activation) ||
      sh(op.Get(), &p.stride[0]) != kLiteRtStatusOk ||
      sw(op.Get(), &p.stride[1]) != kLiteRtStatusOk ||
      dh(op.Get(), &p.dilation[0]) != kLiteRtStatusOk ||
      dw_fn(op.Get(), &p.dilation[1]) != kLiteRtStatusOk)
    return false;
  p.cropped_shape = p.inputs[0].shape;
  for (int axis = 0; axis < 2; ++axis) {
    if (p.stride[axis] <= 0 || p.dilation[axis] <= 0) return false;
    int64_t n = p.inputs[0].shape[axis + 1],
            k = (static_cast<int64_t>(p.weight_shape[axis + 1]) - 1) *
                    p.dilation[axis] +
                1;
    int64_t expected = padding == 0 ? (n + p.stride[axis] - 1) / p.stride[axis]
                                    : (n - k) / p.stride[axis] + 1;
    if ((padding == 1 && n < k) || expected != p.output.shape[axis + 1])
      return false;
    int64_t total =
        padding == 0
            ? std::max(int64_t{0}, (expected - 1) * p.stride[axis] + k - n)
            : 0;
    if (total > INT32_MAX) return false;
    p.padding[2 * axis] = total / 2;
    p.padding[2 * axis + 1] = total - total / 2;
    p.cropped_shape[axis + 1] -= (n + total - k) % p.stride[axis];
  }
  return true;
}

std::optional<OperationPlan> AnalyzeOperation(const Op& op, std::string& reason,
                                              SoftmaxLutType softmax_lut) {
  auto reject = [&](const char* s) -> std::optional<OperationPlan> {
    reason = s;
    return std::nullopt;
  };
  if (op.Code() == kLiteRtOpCodeShloComposite)
    return AnalyzeSpecialOperation(op, reason);
  OperationPlan p{};
  p.code = op.Code();
  auto in = op.Inputs();
  auto out = op.Outputs();
  const auto* registration = FindOperationRegistration(p.code);
  if (!registration)
    return reject("TFLite operator has no registered lowering");
  if (out.empty() || registration->data_input >= in.size())
    return reject("Missing registered data input or output");
  const auto* variant = FindOperationVariant(
      *registration,
      static_cast<LiteRtElementType>(
          in[registration->data_input].ElementType()),
      static_cast<LiteRtElementType>(out[0].ElementType()));
  if (!variant)
    return reject(
        "TFLite data input/output type pair has no registered lowering");
  if (IsSpecialOperation(op)) return AnalyzeSpecialOperation(op, reason);
  p.route = variant->route;
  if (p.route == LoweringRoute::kDecomposed) {
    p.decomposition = AnalyzeDecomposition(op, reason);
    if (!p.decomposition) return std::nullopt;
    p.runtime_inputs = p.decomposition->runtime_inputs;
    const auto& data = p.decomposition->data;
    const auto& output = p.decomposition->output;
    p.output_type = output.type;
    p.inputs.push_back({data.shape, data.scale, data.zero_point,
                        data.type == DataType::float32_t, data.type});
    p.output = {output.shape, output.scale, output.zero_point,
                output.type == DataType::float32_t, output.type};
    p.table = p.decomposition->table;
    p.table16 = p.decomposition->table16;
    p.required_extensions = p.decomposition->extensions;
    return p;
  }
  if (p.route == LoweringRoute::kTyped)
    return AnalyzeTypedOperation(op, std::move(p), reason);
  if (p.route == LoweringRoute::kPortable) {
    p.portable = AnalyzePortableOperation(op, reason);
    if (!p.portable) return std::nullopt;
    p.runtime_inputs = p.portable->inputs;
    p.output_type = p.portable->data_type;
    p.inputs.push_back({p.portable->shapes[0], 1, 0, !p.portable->quantized,
                        p.portable->data_type});
    if (p.portable->data_type == DataType::int16_t)
      p.required_extensions = {"EXT-INT16"};
    return p;
  }
  if (in.empty() || out.size() != 1)
    return reject("Expected data inputs and exactly one output");
  const int data_index = registration->data_input;
  if (data_index >= in.size()) return reject("Missing data input");
  auto input = ReadTensor(in[data_index]);
  auto output = ReadTensor(out[0]);
  if (!input || !output || input->shape.empty() || output->shape.empty())
    return reject(
        "Requires static dense FP32 or per-tensor affine INT8/INT16 data "
        "tensors");
  p.inputs.push_back(*input);
  p.output = *output;
  p.output_type = output->type;
  p.runtime_inputs = {data_index};
  if (input->type == DataType::int16_t || output->type == DataType::int16_t)
    p.required_extensions = {"EXT-INT16"};
  if (p.code == kLiteRtOpCodeTflSoftmax) {
    float beta;
    if (in.size() != 1 || input->floating || output->floating ||
        input->type != output->type || input->shape != output->shape ||
        LiteRtGetSoftmaxBetaOption(op.Get(), &beta) != kLiteRtStatusOk)
      return reject("Quantized SOFTMAX requires matching shapes and types");
    if (input->type == DataType::int16_t) {
      p.softmax16 = AnalyzeInt16Softmax(input->shape.back(), input->scale,
                                        output->scale, beta, reason);
      if (!p.softmax16) return std::nullopt;
      return p;
    }
    if (output->scale != 1.0 / 256 || output->zero_point != -128)
      return reject(
          "INT8 SOFTMAX requires matching shapes, scale=1/256 and "
          "zero_point=-128 output");
    p.softmax = AnalyzeQuantizedSoftmax(input->shape.back(), input->scale, beta,
                                        softmax_lut, reason);
    if (!p.softmax) return std::nullopt;
    return p;
  }
  if (p.route == LoweringRoute::kExtended) {
    if (!AnalyzeExtendedOperation(op, p, reason)) return std::nullopt;
    return p;
  }
  if (input->floating && output->floating &&
      (p.code == kLiteRtOpCodeTflAdd || p.code == kLiteRtOpCodeTflMul)) {
    if (in.size() != 2) return reject("FP32 arithmetic requires two inputs");
    auto rhs = ReadTensor(in[1]);
    if (!rhs || !rhs->floating)
      return reject("FP32 arithmetic requires matching types");
    const size_t rank = std::max(input->shape.size(), rhs->shape.size());
    auto lhs_shape = input->shape, rhs_shape = rhs->shape;
    lhs_shape.insert(lhs_shape.begin(), rank - lhs_shape.size(), 1);
    rhs_shape.insert(rhs_shape.begin(), rank - rhs_shape.size(), 1);
    std::vector<uint32_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      if (lhs_shape[i] != rhs_shape[i] && lhs_shape[i] != 1 &&
          rhs_shape[i] != 1)
        return reject("FP32 arithmetic broadcast mismatch");
      shape[i] = std::max(lhs_shape[i], rhs_shape[i]);
    }
    if (shape != output->shape)
      return reject("FP32 arithmetic output shape mismatch");
    auto activation_fn = p.code == kLiteRtOpCodeTflAdd
                             ? LiteRtGetAddFusedActivationOption
                             : LiteRtGetMulFusedActivationOption;
    if (activation_fn(op.Get(), &p.float_activation) != kLiteRtStatusOk ||
        !ActivationCode(p.float_activation))
      return reject("Unsupported FP32 activation");
    p.inputs.push_back(*rhs);
    p.runtime_inputs.push_back(1);
    reason.clear();
    return p;
  }
  if (p.code == kLiteRtOpCodeTflConv2d ||
      p.code == kLiteRtOpCodeTflDepthwiseConv2d) {
    if (input->floating) {
      if (!AnalyzeFloatConvolution(op, p, reason))
        return reject("Unsupported floating-point convolution geometry");
      return p;
    }
    p.convolution = AnalyzeConv2d(op, reason);
    if (!p.convolution) return std::nullopt;
    return p;
  }
  auto constant_ints = [&](int index) -> bool {
    if (index >= in.size()) return false;
    auto type = in[index].RankedTensorType();
    if (!type || type->ElementType() != ElementType::Int32 ||
        type->Layout().HasStrides())
      return false;
    auto values = in[index].WeightsData<int32_t>();
    if (!values) return false;
    p.parameters.assign(values->begin(), values->end());
    return true;
  };
  switch (p.code) {
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6: {
      if (in.size() != 1 || input->type != output->type ||
          input->shape != output->shape)
        return reject("RELU requires matching shapes");
      const auto act_kind = p.code == kLiteRtOpCodeTflRelu ? Activation::kRelu
                                                           : Activation::kRelu6;
      if (input->type == DataType::int16_t) {
        auto scale = MakeScale32(input->scale / output->scale);
        auto activation =
            ActivationRangeInt16(act_kind, output->scale, output->zero_point);
        if (!scale || !activation ||
            (!SameQuant(*input, *output) &&
             !CanEmitNativeRescale(*scale, -32768, 32767)))
          return reject("Invalid INT16 RELU quantization");
        p.activation = *activation;
        if (!SameQuant(*input, *output)) {
          p.scales.push_back(*scale);
          p.min64 = {-32768};
          p.max64 = {32767};
        }
        break;
      }
      auto scale = MakeScale32(input->scale / output->scale);
      auto activation =
          ActivationRange(act_kind, output->scale, output->zero_point);
      if (!scale || !activation) return reject("Invalid RELU quantization");
      p.activation = *activation;
      if (SameQuant(*input, *output)) break;
      for (int code = -128; code <= 127; ++code) {
        auto first = *scale;
        first.shift = std::min<int8_t>(first.shift, 31);
        auto value = ApplyScale32(code - input->zero_point, first);
        if (!value) return reject("RELU scale range invalid");
        int shift = scale->shift - first.shift;
        int64_t rounded = *value;
        if (shift) {
          int64_t magnitude =
              (std::abs(rounded) + (int64_t{1} << (shift - 1))) /
              (int64_t{1} << shift);
          rounded = rounded < 0 ? -magnitude : magnitude;
        }
        p.table.push_back(std::clamp(rounded + output->zero_point,
                                     int64_t{activation->minimum},
                                     int64_t{activation->maximum}));
      }
      break;
    }
    case kLiteRtOpCodeTflReshape: {
      if (in.size() != 2 || !SameQuant(*input, *output) ||
          Count(input->shape) != Count(output->shape) || !constant_ints(1) ||
          p.parameters.size() != output->shape.size())
        return reject(
            "RESHAPE requires a constant compatible shape and unchanged "
            "quantization");
      int inferred = 0;
      for (size_t i = 0; i < p.parameters.size(); ++i) {
        if (p.parameters[i] == -1)
          ++inferred;
        else if (p.parameters[i] <= 0 ||
                 static_cast<uint32_t>(p.parameters[i]) != output->shape[i])
          return reject("RESHAPE shape disagrees");
      }
      if (inferred > 1)
        return reject("RESHAPE has multiple inferred dimensions");
      break;
    }
    case kLiteRtOpCodeTflSlice: {
      if (in.size() != 3 || !SameQuant(*input, *output) || !constant_ints(1) ||
          p.parameters.size() != input->shape.size() ||
          output->shape.size() != input->shape.size())
        return reject(
            "SLICE requires constant starts/sizes and matching quantization");
      p.kernel = p.parameters;
      if (!constant_ints(2) || p.parameters.size() != input->shape.size())
        return reject("Invalid SLICE size");
      for (size_t i = 0; i < p.parameters.size(); ++i) {
        int64_t start = p.kernel[i], size = p.parameters[i];
        if (size == -1) size = static_cast<int64_t>(input->shape[i]) - start;
        if (start < 0 || size <= 0 || start + size > input->shape[i] ||
            size != output->shape[i])
          return reject("SLICE bounds mismatch");
        p.parameters[i] = size;
      }
      break;
    }
    case kLiteRtOpCodeTflConcatenation: {
      uint32_t fused;
      if (LiteRtGetConcatenationAxisOption(op.Get(), &p.axis) !=
              kLiteRtStatusOk ||
          LiteRtGetConcatenationFusedActivationOption(op.Get(), &fused) !=
              kLiteRtStatusOk ||
          fused)
        return reject("CONCAT requires no fused activation");
      if (p.axis < 0) p.axis += output->shape.size();
      if (p.axis < 0 || p.axis >= output->shape.size())
        return reject("Invalid CONCAT axis");
      p.inputs.clear();
      p.runtime_inputs.clear();
      uint64_t total = 0;
      for (size_t n = 0; n < in.size(); ++n) {
        auto tensor = ReadTensor(in[n]);
        if (!tensor || !SameQuant(*tensor, *output) ||
            tensor->shape.size() != output->shape.size())
          return reject("CONCAT requires matching types/quantization and rank");
        for (size_t i = 0; i < output->shape.size(); ++i)
          if (i != p.axis && tensor->shape[i] != output->shape[i])
            return reject("CONCAT shape mismatch");
        total += tensor->shape[p.axis];
        p.inputs.push_back(*tensor);
        p.runtime_inputs.push_back(n);
      }
      if (total != output->shape[p.axis])
        return reject("CONCAT output extent mismatch");
      break;
    }
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflQuantize: {
      if (p.code == kLiteRtOpCodeTflQuantize && !input->floating &&
          !output->floating) {
        if (in.size() != 1 || input->type != output->type ||
            input->shape != output->shape)
          return reject("Quantized QUANTIZE requires identical shapes");
        if (SameQuant(*input, *output)) break;
        if (input->type == DataType::int16_t) {
          auto scale = MakeScale32(input->scale / output->scale);
          if (!scale || !CanEmitNativeRescale(*scale, -32768, 32767))
            return reject("INT16 QUANTIZE scale range is not supported");
          p.scales.push_back(*scale);
          p.min64 = {-32768};
          p.max64 = {32767};
          break;
        }
        auto table = RequantizeTable(input->scale, input->zero_point,
                                     output->scale, output->zero_point);
        if (!table) return reject("INT8 QUANTIZE scale range is not supported");
        p.table = std::move(*table);
        break;
      }
      const bool dequant = p.code == kLiteRtOpCodeTflDequantize;
      if (in.size() != 1 || input->shape != output->shape ||
          input->floating != !dequant || output->floating != dequant)
        return reject("Requires int8/float32 conversion with identical shapes");
      break;
    }
    case kLiteRtOpCodeTflAbs: {
      if (in.size() != 1 || input->floating || output->floating ||
          input->type != DataType::int8_t || output->type != DataType::int8_t ||
          input->shape != output->shape)
        return reject("INT8 ABS requires matching shapes");
      auto table = RequantizeTable(static_cast<float>(input->scale), -128,
                                   static_cast<float>(output->scale),
                                   output->zero_point);
      if (!table) return reject("INT8 ABS rescale range not proven");
      for (int code = -128; code <= 127; ++code)
        p.table.push_back((*table)[std::abs(code - input->zero_point)]);
      break;
    }
    case kLiteRtOpCodeTflTanh: {
      if (in.size() != 1 || input->floating || output->floating ||
          input->type != output->type || input->shape != output->shape)
        return reject("TANH requires matching shapes and types");
      if (input->type == DataType::int16_t) {
        if (output->scale != 1.0 / 32768.0 || output->zero_point != 0)
          return reject("INT16 TANH requires output scale 1/32768, zp 0");
        auto lut = PopulateInt16Table<float>(
            static_cast<float>(input->scale), 0,
            static_cast<float>(output->scale), 0,
            [](float x) { return std::tanh(x); });
        if (!lut) return reject("INT16 TANH lookup table construction failed");
        p.table16 = std::move(*lut);
        break;
      }
      if (output->scale != 1.0 / 128 || output->zero_point != 0)
        return reject("INT8 TANH requires output scale 1/128, zero point 0");
      for (int code = -128; code <= 127; ++code) {
        float x = static_cast<float>(input->scale) * (code - input->zero_point);
        float y = std::tanh(x);
        p.table.push_back(static_cast<int8_t>(
            std::clamp(std::round(y / static_cast<float>(output->scale)) +
                           static_cast<float>(output->zero_point),
                       -128.0f, 127.0f)));
      }
      break;
    }
    case kLiteRtOpCodeTflLogistic: {
      if (in.size() != 1 || input->shape != output->shape ||
          input->type != output->type)
        return reject("LOGISTIC requires matching shapes and types");
      if (input->type == DataType::int16_t) {
        if (output->scale != 1.0 / 32768.0 || output->zero_point != 0)
          return reject("INT16 LOGISTIC requires output scale 1/32768, zp 0");
        auto lut = PopulateInt16Table<float>(
            static_cast<float>(input->scale), 0,
            static_cast<float>(output->scale), 0,
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        if (!lut)
          return reject("INT16 LOGISTIC lookup table construction failed");
        p.table16 = std::move(*lut);
        break;
      }
      if (!input->floating) {
        if (output->scale != 1.0 / 256 || output->zero_point != -128)
          return reject(
              "INT8 LOGISTIC requires scale 1/256 and zero point -128");
        // Match LiteRT's reference LUT using float arithmetic, then store in
        // TOSA TABLE's signed input order (-128 through 127).
        for (int code = -128; code <= 127; ++code) {
          float x =
              static_cast<float>(input->scale) * (code - input->zero_point);
          float y = 1.0f / (1.0f + std::exp(-x));
          int value = static_cast<int>(std::round(y * 256.0f)) - 128;
          p.table.push_back(std::clamp(value, -128, 127));
        }
      }
      break;
    }
    case kLiteRtOpCodeTflHardSwish: {
      if (in.size() != 1 || input->shape != output->shape ||
          input->floating != output->floating)
        return reject("HARD_SWISH requires matching shapes and types");
      if (!input->floating) {
        auto table = HardSwishTable(input->scale, input->zero_point,
                                    output->scale, output->zero_point);
        if (!table)
          return reject("HARD_SWISH fixed-point scales are not representable");
        p.table = std::move(*table);
      }
      break;
    }
    case kLiteRtOpCodeTflMul: {
      if (in.size() != 2) return reject("MUL requires two inputs");
      auto rhs = ReadTensor(in[1]);
      if (!rhs || rhs->floating || rhs->type != input->type ||
          output->type != input->type)
        return reject("MUL requires matching quantized inputs");
      const size_t rank = std::max(input->shape.size(), rhs->shape.size());
      auto lhs_shape = input->shape, rhs_shape = rhs->shape;
      lhs_shape.insert(lhs_shape.begin(), rank - lhs_shape.size(), 1);
      rhs_shape.insert(rhs_shape.begin(), rank - rhs_shape.size(), 1);
      std::vector<uint32_t> shape(rank);
      for (size_t i = 0; i < rank; ++i) {
        if (lhs_shape[i] != rhs_shape[i] && lhs_shape[i] != 1 &&
            rhs_shape[i] != 1)
          return reject("MUL broadcast mismatch");
        shape[i] = std::max(lhs_shape[i], rhs_shape[i]);
      }
      if (shape != output->shape) return reject("MUL output shape mismatch");
      uint32_t fused;
      if (LiteRtGetMulFusedActivationOption(op.Get(), &fused) !=
          kLiteRtStatusOk)
        return reject("Missing MUL activation");
      auto activation = ActivationCode(fused);
      if (!activation) return reject("Unsupported MUL activation");
      auto scale = MakeScale32(input->scale * rhs->scale / output->scale);
      if (input->type == DataType::int16_t) {
        auto act_range = ActivationRangeInt16(*activation, output->scale,
                                              output->zero_point);
        const int64_t lo = -int64_t{32768} * 32767;
        const int64_t hi = int64_t{32768} * 32768;
        if (!act_range || !scale || !CanEmitNativeRescale(*scale, lo, hi))
          return reject("INT16 MUL rescale range invalid");
        p.activation = *act_range;
        p.inputs.push_back(*rhs);
        p.runtime_inputs.push_back(1);
        p.scales.push_back(*scale);
        p.min64 = {lo};
        p.max64 = {hi};
        break;
      }
      p.activation =
          *ActivationRange(*activation, output->scale, output->zero_point);
      int64_t lo = 0, hi = 0;
      for (int64_t a : {-128 - input->zero_point, 127 - input->zero_point})
        for (int64_t b : {-128 - rhs->zero_point, 127 - rhs->zero_point}) {
          lo = std::min(lo, a * b);
          hi = std::max(hi, a * b);
        }
      if (!scale || !IsScale32RangeValid(*scale, lo, hi))
        return reject("MUL rescale range invalid");
      p.inputs.push_back(*rhs);
      p.runtime_inputs.push_back(1);
      p.scales.push_back(*scale);
      p.ranges.push_back({static_cast<int32_t>(lo), static_cast<int32_t>(hi)});
      break;
    }
    case kLiteRtOpCodeTflTranspose: {
      if (in.size() != 2 || !constant_ints(1) || !SameQuant(*input, *output) ||
          p.parameters.size() != input->shape.size() ||
          output->shape.size() != input->shape.size())
        return reject(
            "TRANSPOSE requires a constant permutation and matching "
            "quantization");
      std::set<int32_t> axes;
      for (size_t i = 0; i < p.parameters.size(); ++i) {
        int axis = p.parameters[i];
        if (axis < 0 || axis >= input->shape.size() ||
            !axes.insert(axis).second || output->shape[i] != input->shape[axis])
          return reject("Invalid TRANSPOSE permutation/output shape");
      }
      break;
    }
    case kLiteRtOpCodeTflPad:
    case kLiteRtOpCodeTflPadv2: {
      if (in.size() != (p.code == kLiteRtOpCodeTflPadv2 ? 3 : 2) ||
          !constant_ints(1) || !SameQuant(*input, *output) ||
          p.parameters.size() != input->shape.size() * 2 ||
          output->shape.size() != input->shape.size())
        return reject(
            "PAD requires constant nonnegative padding and matching "
            "quantization");
      p.pad_value = input->zero_point;
      if (p.code == kLiteRtOpCodeTflPadv2) {
        auto fill = ReadTensor(in[2]);
        if (!fill || !SameQuant(*fill, *input))
          return reject(
              "PADV2 requires a constant scalar with matching quantization");
        if (input->type == DataType::int16_t) {
          auto value = in[2].WeightsData<int16_t>();
          if (!value || value->size() != 1)
            return reject(
                "PADV2 requires a constant scalar with matching quantization");
          p.pad_value = (*value)[0];
        } else {
          auto value = in[2].WeightsData<int8_t>();
          if (!value || value->size() != 1)
            return reject(
                "PADV2 requires a constant scalar with matching quantization");
          p.pad_value = (*value)[0];
        }
      }
      for (size_t i = 0; i < input->shape.size(); ++i) {
        if (p.parameters[2 * i] < 0 || p.parameters[2 * i + 1] < 0 ||
            uint64_t{input->shape[i]} + p.parameters[2 * i] +
                    p.parameters[2 * i + 1] !=
                output->shape[i])
          return reject("Invalid PAD shape");
      }
      break;
    }
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflAdd: {
      if (in.size() != 2) return reject("ADD requires two inputs");
      auto rhs = ReadTensor(in[1]);
      if (!rhs || rhs->floating || rhs->type != input->type ||
          output->type != input->type)
        return reject("ADD requires matching quantized inputs");
      const size_t rank = std::max(input->shape.size(), rhs->shape.size());
      auto lhs_shape = input->shape, rhs_shape = rhs->shape;
      lhs_shape.insert(lhs_shape.begin(), rank - lhs_shape.size(), 1);
      rhs_shape.insert(rhs_shape.begin(), rank - rhs_shape.size(), 1);
      std::vector<uint32_t> shape(rank);
      for (size_t i = 0; i < rank; ++i) {
        if (lhs_shape[i] != rhs_shape[i] && lhs_shape[i] != 1 &&
            rhs_shape[i] != 1)
          return reject("ADD broadcast mismatch");
        shape[i] = std::max(lhs_shape[i], rhs_shape[i]);
      }
      if (shape != output->shape) return reject("ADD output shape mismatch");
      uint32_t fused;
      if ((p.code == kLiteRtOpCodeTflSub ? LiteRtGetSubFusedActivationOption
                                         : LiteRtGetAddFusedActivationOption)(
              op.Get(), &fused) != kLiteRtStatusOk)
        return reject("Cannot read ADD activation");
      auto activation = ActivationCode(fused);
      if (!activation) return reject("Unsupported ADD activation");
      if (input->type == DataType::int16_t) {
        auto act_range = ActivationRangeInt16(*activation, output->scale,
                                              output->zero_point);
        if (!act_range) return reject("Invalid INT16 ADD/SUB activation");
        p.activation = *act_range;
        p.inputs.push_back(*rhs);
        p.runtime_inputs.push_back(1);
        if (input->scale == rhs->scale && input->scale == output->scale) {
          p.equal_scale_fast_path = true;
          p.scales.push_back(*MakeScale32(1.0));
          p.min64 = {-65536};
          p.max64 = {65536};
          break;
        }
        const double twice_max = 2.0 * std::max(input->scale, rhs->scale);
        bool ok = false;
        // Start at 8192 so (s_i / twice_max) * factor <= 4096 fits in a single
        // 16-bit multiplier RESCALE without two-stage amplification splitting,
        // while keeping 27 bits of intermediate i32 precision.
        for (double factor = 8192.0; factor >= 1.0; factor *= 0.5) {
          auto s0 = MakeScale32(input->scale / twice_max * factor);
          auto s1 = MakeScale32(rhs->scale / twice_max * factor);
          auto s2 = MakeScale32(twice_max / (factor * output->scale));
          if (!s0 || !s1 || !s2 ||
              !CanEmitNativeRescale(*s0, -32768, 32767) ||
              !CanEmitNativeRescale(*s1, -32768, 32767))
            continue;
          const int64_t l0 = *ApplyScale32(-32768, *s0);
          const int64_t h0 = *ApplyScale32(32767, *s0);
          const int64_t l1 = *ApplyScale32(-32768, *s1);
          const int64_t h1 = *ApplyScale32(32767, *s1);
          const int64_t sum_lo =
              p.code == kLiteRtOpCodeTflSub ? (l0 - h1) : (l0 + l1);
          const int64_t sum_hi =
              p.code == kLiteRtOpCodeTflSub ? (h0 - l1) : (h0 + h1);
          if (sum_lo < INT32_MIN || sum_hi > INT32_MAX ||
              !CanEmitNativeRescale(*s2, sum_lo, sum_hi))
            continue;
          p.scales = {*s0, *s1, *s2};
          p.min64 = {-32768, -32768, sum_lo};
          p.max64 = {32767, 32767, sum_hi};
          ok = true;
          break;
        }
        if (!ok) return reject("INT16 ADD/SUB rescale range invalid");
        break;
      }
      p.activation =
          *ActivationRange(*activation, output->scale, output->zero_point);
      p.inputs.push_back(*rhs);
      p.runtime_inputs.push_back(1);
      const double twice_max = 2 * std::max(input->scale, rhs->scale);
      for (double scale : {input->scale / twice_max * 1048576,
                           rhs->scale / twice_max * 1048576,
                           twice_max / (1048576 * output->scale)}) {
        auto encoded = MakeScale32(scale);
        if (!encoded) return reject("ADD scale not representable");
        p.scales.push_back(*encoded);
      }
      int64_t minimum = 0, maximum = 0;
      for (int i = 0; i < 2; ++i) {
        int32_t lo = -128 - p.inputs[i].zero_point,
                hi = 127 - p.inputs[i].zero_point;
        if (!IsScale32RangeValid(p.scales[i], lo, hi))
          return reject("ADD input rescale range invalid");
        if (p.code == kLiteRtOpCodeTflSub && i == 1) {
          minimum -= *ApplyScale32(hi, p.scales[i]);
          maximum -= *ApplyScale32(lo, p.scales[i]);
        } else {
          minimum += *ApplyScale32(lo, p.scales[i]);
          maximum += *ApplyScale32(hi, p.scales[i]);
        }
      }
      if (!IsScale32RangeValid(p.scales[2], minimum, maximum))
        return reject("ADD accumulator range invalid");
      p.ranges.push_back(
          {static_cast<int32_t>(minimum), static_cast<int32_t>(maximum)});
      break;
    }
    case kLiteRtOpCodeTflMaxPool2d:
    case kLiteRtOpCodeTflAveragePool2d: {
      if (in.size() != 1 || input->shape.size() != 4 ||
          output->shape.size() != 4 || !SameQuant(*input, *output) ||
          input->shape[0] != output->shape[0] ||
          input->shape[3] != output->shape[3])
        return reject(
            "Pool requires NHWC, preserved batch/channels and quantization");
      bool avg = p.code == kLiteRtOpCodeTflAveragePool2d;
      auto get_padding = avg ? LiteRtGetAveragePool2dPaddingOption
                             : LiteRtGetMaxPool2dPaddingOption;
      auto get_activation = avg ? LiteRtGetAveragePool2dFusedActivationOption
                                : LiteRtGetMaxPool2dFusedActivationOption;
      auto get_h = avg ? LiteRtGetAveragePool2dFilterHeightOption
                       : LiteRtGetMaxPool2dFilterHeightOption;
      auto get_w = avg ? LiteRtGetAveragePool2dFilterWidthOption
                       : LiteRtGetMaxPool2dFilterWidthOption;
      auto get_sh = avg ? LiteRtGetAveragePool2dStrideHOption
                        : LiteRtGetMaxPool2dStrideHOption;
      auto get_sw = avg ? LiteRtGetAveragePool2dStrideWOption
                        : LiteRtGetMaxPool2dStrideWOption;
      uint32_t padding, fused;
      p.kernel.resize(2);
      p.stride.resize(2);
      if (get_padding(op.Get(), &padding) != kLiteRtStatusOk || padding > 1 ||
          get_activation(op.Get(), &fused) != kLiteRtStatusOk ||
          get_h(op.Get(), &p.kernel[0]) != kLiteRtStatusOk ||
          get_w(op.Get(), &p.kernel[1]) != kLiteRtStatusOk ||
          get_sh(op.Get(), &p.stride[0]) != kLiteRtStatusOk ||
          get_sw(op.Get(), &p.stride[1]) != kLiteRtStatusOk)
        return reject("Invalid pool options");
      auto activation = ActivationCode(fused);
      if (!activation) return reject("Unsupported pool activation");
      p.float_activation = fused;
      if (!output->floating) {
        auto act_range =
            output->type == DataType::int16_t
                ? ActivationRangeInt16(*activation, output->scale,
                                       output->zero_point)
                : ActivationRange(*activation, output->scale,
                                  output->zero_point);
        if (!act_range) return reject("Invalid pool activation");
        p.activation = *act_range;
      }
      p.cropped_shape = input->shape;
      p.padding.assign(4, 0);
      for (int axis = 0; axis < 2; ++axis) {
        int64_t d = input->shape[axis + 1], k = p.kernel[axis],
                stride = p.stride[axis];
        if (k <= 0 || k > 8192 || stride <= 0 || stride > 8192 ||
            (padding == 1 && d < k))
          return reject("Invalid pool geometry");
        const int64_t count =
            padding == 0 ? (d + stride - 1) / stride : (d - k) / stride + 1;
        if (count != output->shape[axis + 1])
          return reject("Pool output shape mismatch");
        const int64_t total =
            padding == 0 ? std::max(int64_t{0}, (count - 1) * stride + k - d)
                         : 0;
        p.padding[2 * axis] = total / 2;
        p.padding[2 * axis + 1] = total - total / 2;
        const int64_t trailing = (d + total - k) % stride;
        const int64_t removed =
            std::min(trailing, int64_t{p.padding[2 * axis + 1]});
        p.padding[2 * axis + 1] -= removed;
        p.cropped_shape[axis + 1] -= trailing - removed;
      }
      // TOSA AVG_POOL2D excludes padded elements from its divisor. Prove that
      // the centered INT8 sum fits INT32 regardless of stride or output
      // geometry.
      const uint64_t max_mag =
          input->type == DataType::int16_t ? 32768 : 255;
      if (avg && !input->floating &&
          uint64_t{static_cast<uint32_t>(p.kernel[0])} * p.kernel[1] * max_mag >
              INT32_MAX)
        return reject("AVG_POOL requires a bounded integer window");
      break;
    }
    case kLiteRtOpCodeTflSum: {
      bool keep;
      if (in.size() != 2 || input->floating || output->floating ||
          input->type != output->type || !constant_ints(1) ||
          p.parameters.empty() ||
          LiteRtGetSumKeepDimsOption(op.Get(), &keep) != kLiteRtStatusOk ||
          !keep)
        return reject("SUM requires constant axes and keep_dims=true");
      std::set<int32_t> axes;
      auto shape = input->shape;
      uint64_t n = 1;
      for (auto& axis : p.parameters) {
        if (axis < 0) axis += shape.size();
        if (axis < 0 || axis >= shape.size() || !axes.insert(axis).second)
          return reject("Invalid SUM axes");
        n *= shape[axis];
        shape[axis] = 1;
      }
      const uint64_t max_mag = input->type == DataType::int16_t ? 32768 : 255;
      if (shape != output->shape || n > uint64_t{INT32_MAX} / max_mag)
        return reject("SUM shape or accumulation range invalid");
      const int64_t lo =
          ((input->type == DataType::int16_t ? -32768 : -128) -
           int64_t{input->zero_point}) *
          static_cast<int64_t>(n);
      const int64_t hi =
          ((input->type == DataType::int16_t ? 32767 : 127) -
           int64_t{input->zero_point}) *
          static_cast<int64_t>(n);
      auto scale = MakeScale32(input->scale / output->scale);
      if (!scale ||
          (input->type == DataType::int16_t
               ? !CanEmitNativeRescale(*scale, lo, hi)
               : !IsScale32RangeValid(*scale, lo, hi)))
        return reject("SUM rescale range invalid");
      p.scales.push_back(*scale);
      p.min64 = {lo};
      p.max64 = {hi};
      break;
    }
    default:
      return reject("Operator lowering not implemented");
  }
  reason.clear();
  return p;
}

ResId EmitFloatOperation(Graph& g, const std::vector<ResId>& inputs,
                         const OperationPlan& p, GraphConstantPool& constants) {
  const auto& q = p.inputs[0];
  auto value = inputs[0];
  TosaTensor ftype(DataType::float32_t, p.output.shape);
  auto scalar = [&](float x) {
    return BroadcastScalar(g, x, q.shape.size(), constants);
  };
  auto mul = [&](ResId a, ResId b) {
    return g.AddMulOperator(a, b, g.AddTensorConstant(Attribute(int8_t{0})),
                            ftype);
  };
  switch (p.code) {
    case kLiteRtOpCodeTflDequantize:
      value = g.AddCastOperator(value, ftype);
      if (q.zero_point != 0)
        value = g.AddSubOperator(
            value, scalar(static_cast<float>(q.zero_point)), ftype);
      return mul(value, scalar(static_cast<float>(q.scale)));
    case kLiteRtOpCodeTflQuantize: {
      // Match affine quantization: round the scaled value before adding zp.
      value = mul(value, scalar(static_cast<float>(1.0 / p.output.scale)));
      auto positive = g.AddGreaterEqualOperator(
          value, scalar(0), TosaTensor(DataType::bool_t, p.output.shape));
      auto up = g.AddFloorOperator(g.AddAddOperator(value, scalar(.5f), ftype),
                                   ftype);
      auto negative = g.AddSubOperator(scalar(0), value, ftype);
      auto down = g.AddSubOperator(
          scalar(0),
          g.AddFloorOperator(g.AddAddOperator(negative, scalar(.5f), ftype),
                             ftype),
          ftype);
      value = g.AddSelectOperator(positive, up, down, ftype);
      value = g.AddAddOperator(
          value, scalar(static_cast<float>(p.output.zero_point)), ftype);
      value = g.AddClampOperator(value, Attribute(-128.0f), Attribute(127.0f),
                                 Attribute(int32_t{1}), ftype);
      return g.AddCastOperator(value,
                               TosaTensor(DataType::int8_t, p.output.shape));
    }
    case kLiteRtOpCodeTflReshape:
      return Reshape(g, value, DataType::float32_t, p.output.shape);
    case kLiteRtOpCodeTflTranspose:
      return g.AddTransposeOperator(value, Attribute(p.parameters), ftype);
    case kLiteRtOpCodeTflPad:
      return g.AddPadOperator(value,
                              g.AddTensorConstant(Attribute(p.parameters)),
                              g.AddTensorConstant(Attribute(0.0f)), ftype);
    case kLiteRtOpCodeTflLogistic:
      return g.AddSigmoidOperator(value, ftype);
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul: {
      auto lhs_shape = p.inputs[0].shape, rhs_shape = p.inputs[1].shape;
      lhs_shape.insert(lhs_shape.begin(),
                       p.output.shape.size() - lhs_shape.size(), 1);
      rhs_shape.insert(rhs_shape.begin(),
                       p.output.shape.size() - rhs_shape.size(), 1);
      auto lhs = Reshape(g, value, DataType::float32_t, lhs_shape);
      auto rhs = Reshape(g, inputs[1], DataType::float32_t, rhs_shape);
      value = p.code == kLiteRtOpCodeTflAdd ? g.AddAddOperator(lhs, rhs, ftype)
                                            : mul(lhs, rhs);
      if (p.float_activation)
        value = g.AddClampOperator(
            value, Attribute(0.0f),
            Attribute(p.float_activation == 3
                          ? 6.0f
                          : std::numeric_limits<float>::infinity()),
            Attribute(int32_t{1}), ftype);
      return value;
    }
    case kLiteRtOpCodeTflAveragePool2d: {
      auto zero = g.AddTensorConstant(Attribute(0.0f));
      value = g.AddAvgPool2dOperator(
          value, zero, zero, Attribute(p.kernel), Attribute(p.stride),
          Attribute(std::vector<int32_t>(4, 0)), Attribute(int32_t{3}), ftype);
      if (p.float_activation)
        value = g.AddClampOperator(
            value, Attribute(0.0f),
            Attribute(p.float_activation == 3
                          ? 6.0f
                          : std::numeric_limits<float>::infinity()),
            Attribute(int32_t{1}), ftype);
      return value;
    }
    case kLiteRtOpCodeTflHardSwish: {
      auto gate = g.AddAddOperator(value, scalar(3), ftype);
      gate = g.AddClampOperator(gate, Attribute(0.0f), Attribute(6.0f),
                                Attribute(int32_t{1}), ftype);
      return mul(value, mul(gate, scalar(1.0f / 6.0f)));
    }
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d: {
      if (p.cropped_shape != q.shape)
        value = g.AddSliceOperator(
            value, g.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0))),
            g.AddTensorConstant(Attribute(p.cropped_shape)),
            TosaTensor(DataType::float32_t, p.cropped_shape));
      bool dw = p.code == kLiteRtOpCodeTflDepthwiseConv2d;
      auto shape = p.weight_shape;
      if (dw) shape = {shape[1], shape[2], q.shape[3], shape[3] / q.shape[3]};
      auto weights = Constant(g, p.float_weights, shape, constants);
      auto bias = Constant(g, p.float_bias, {p.output.shape[3]}, constants);
      auto zero = g.AddTensorConstant(Attribute(0.0f));
      value = g.AddOperator(
          dw ? tfsc::tosa::OperatorEnum::DepthwiseConv2d
             : tfsc::tosa::OperatorEnum::Conv2d,
          {value, weights, bias, zero, zero}, {ftype},
          {Attribute(p.padding), Attribute(p.stride), Attribute(p.dilation),
           Attribute(int32_t{3}), Attribute(false)})[0];
      if (p.float_activation)
        value = g.AddClampOperator(
            value, Attribute(0.0f),
            Attribute(p.float_activation == 3
                          ? 6.0f
                          : std::numeric_limits<float>::infinity()),
            Attribute(int32_t{1}), ftype);
      return value;
    }
    default:
      return nullptr;
  }
}

ResId EmitOperation(Graph& g, const std::vector<ResId>& inputs,
                    const OperationPlan& p, GraphConstantPool& constants) {
  if (p.special) return EmitSpecialOperation(g, inputs, *p.special, constants);
  if (p.portable) {
    auto result = EmitPortableOperation(g, inputs, *p.portable, constants);
    return result.size() == 1 ? result[0] : nullptr;
  }
  if (p.decomposition)
    return EmitDecomposition(g, inputs, *p.decomposition, constants);
  if (p.typed) {
    auto type = [&](DataType dt, const std::vector<uint32_t>& shape) {
      return TosaTensor(dt, shape);
    };
    std::vector<ResId> v;
    v.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
      v.push_back(ReshapeTensor(
          g, inputs[i], g.AddTensorConstant(Attribute(p.inputs[i].shape)),
          type(p.input_types[i], p.inputs[i].shape)));
    auto output = type(p.output_type, p.output.shape);
    switch (p.code) {
      case kLiteRtOpCodeTflLogicalAnd:
        return g.AddLogicalAndOperator(v[0], v[1], output);
      case kLiteRtOpCodeTflLogicalOr:
        return g.AddLogicalOrOperator(v[0], v[1], output);
      case kLiteRtOpCodeTflLogicalNot:
        return g.AddLogicalNotOperator(v[0], output);
      case kLiteRtOpCodeTflBitwiseXor: {
        if (!p.parameters.empty()) return g.AddBitwiseNotOperator(v[0], output);
        // XOR with all ones is the standard bitwise complement idiom.
        return g.AddBitwiseXorOperator(v[0], v[1], output);
      }
      case kLiteRtOpCodeTflRightShift:
        return g.AddLogicalRightShiftOperator(v[0], v[1], output);
      case kLiteRtOpCodeTflFloorDiv: {
        auto constant = [&](int32_t x) {
          auto id = g.AddTensorConstant(Attribute(x));
          return ReshapeTensor(
              g, id,
              g.AddTensorConstant(
                  Attribute(std::vector<uint32_t>(p.output.shape.size(), 1))),
              type(DataType::int32_t,
                   std::vector<uint32_t>(p.output.shape.size(), 1)));
        };
        auto q = g.AddIntDivOperator(v[0], v[1], output);
        auto product = g.AddMulOperator(
            q, v[1], g.AddTensorConstant(Attribute(int8_t{0})), output);
        auto rem = g.AddSubOperator(v[0], product, output);
        auto bt = type(DataType::bool_t, p.output.shape);
        auto negative = g.AddGreaterOperator(
            constant(0), v[0], type(DataType::bool_t, p.inputs[0].shape));
        auto nonzero = g.AddLogicalNotOperator(
            g.AddEqualOperator(rem, constant(0), bt), bt);
        auto adjust =
            g.AddSelectOperator(g.AddLogicalAndOperator(negative, nonzero, bt),
                                constant(-1), constant(0), output);
        return g.AddAddOperator(q, adjust, output);
      }
      case kLiteRtOpCodeTflEqual:
        if (p.input_types[0] == DataType::bool_t)
          return g.AddLogicalNotOperator(
              g.AddLogicalXorOperator(v[0], v[1], output), output);
        return g.AddEqualOperator(v[0], v[1], output);
      case kLiteRtOpCodeTflNotEqual:
        if (p.input_types[0] == DataType::bool_t)
          return g.AddLogicalXorOperator(v[0], v[1], output);
        return g.AddLogicalNotOperator(g.AddEqualOperator(v[0], v[1], output),
                                       output);
      case kLiteRtOpCodeTflArgMax:
        return g.AddArgMaxOperator(v[0], Attribute(p.parameters[0]),
                                   Attribute(uint32_t{1}), output);
      default: {
        auto shape = p.inputs[0].shape;
        auto x = v[0];
        for (auto axis : p.parameters) {
          shape[axis] = 1;
          x = p.code == kLiteRtOpCodeTflReduceAll
                  ? g.AddReduceAllOperator(x, Attribute(axis),
                                           type(DataType::bool_t, shape))
                  : g.AddReduceAnyOperator(x, Attribute(axis),
                                           type(DataType::bool_t, shape));
        }
        return ReshapeTensor(
            g, x, g.AddTensorConstant(Attribute(p.output.shape)), output);
      }
    }
  }

  auto input = inputs[0];
  const auto& q = p.inputs[0];
  if (p.softmax16)
    return EmitInt16Softmax(g, input, p.output.shape, *p.softmax16, constants);
  if (p.softmax)
    return EmitQuantizedSoftmax(g, input, p.output.shape, *p.softmax);
  TosaTensor out(p.output.floating ? DataType::float32_t : p.output_type,
                 p.output.shape);
  if (p.route == LoweringRoute::kExtended)
    return EmitExtendedOperation(g, input, p, constants);
  if (q.floating || p.output.floating)
    return EmitFloatOperation(g, inputs, p, constants);
  switch (p.code) {
    case kLiteRtOpCodeTflReshape:
      return Reshape(g, input, p.output_type, p.output.shape);
    case kLiteRtOpCodeTflSlice:
      return g.AddSliceOperator(input, g.AddTensorConstant(Attribute(p.kernel)),
                                g.AddTensorConstant(Attribute(p.parameters)),
                                out);
    case kLiteRtOpCodeTflConcatenation:
      return g.AddConcatOperator(inputs, Attribute(p.axis), out);
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6:
      if (SameQuant(q, p.output)) return Clamp(g, input, p);
      if (p.output_type == DataType::int16_t) {
        auto wide = g.AddCastOperator(
            input, TosaTensor(DataType::int32_t, p.output.shape));
        auto rescaled = EmitNativeRescale(
            g, wide, DataType::int32_t, DataType::int16_t, p.output.shape, 0, 0,
            p.scales, constants, p.min64, p.max64);
        return Clamp(g, rescaled, p);
      }
      return g.AddTableOperator(input, Constant(g, p.table, {256}, constants),
                                out);
    case kLiteRtOpCodeTflQuantize:
      if (SameQuant(q, p.output)) return input;
      if (p.output_type == DataType::int16_t) {
        auto wide = g.AddCastOperator(
            input, TosaTensor(DataType::int32_t, p.output.shape));
        return EmitNativeRescale(g, wide, DataType::int32_t, DataType::int16_t,
                                 p.output.shape, 0, 0, p.scales, constants,
                                 p.min64, p.max64);
      }
      return g.AddTableOperator(input, Constant(g, p.table, {256}, constants),
                                out);
    case kLiteRtOpCodeTflAbs:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflLogistic:
    case kLiteRtOpCodeTflHardSwish:
      if (p.output_type == DataType::int16_t) {
        auto raw_val = g.AddTableOperator(
            input, Constant(g, p.table16, {513}, constants),
            TosaTensor(DataType::int32_t, p.output.shape));
        return EmitNativeRescale(g, raw_val, DataType::int32_t,
                                 DataType::int16_t, p.output.shape, 0, 0,
                                 {*MakeScale32(1.0 / 128.0)}, constants,
                                 {-4194304}, {4194303});
      }
      return g.AddTableOperator(input, Constant(g, p.table, {256}, constants),
                                out);
    case kLiteRtOpCodeTflMul: {
      if (p.output_type == DataType::int16_t) {
        const auto lhs_src =
            q.shape.empty() ? std::vector<uint32_t>{1} : q.shape;
        const auto rhs_src = p.inputs[1].shape.empty()
                                 ? std::vector<uint32_t>{1}
                                 : p.inputs[1].shape;
        auto lhs =
            g.AddCastOperator(input, TosaTensor(DataType::int32_t, lhs_src));
        auto rhs = g.AddCastOperator(inputs[1],
                                     TosaTensor(DataType::int32_t, rhs_src));
        auto shape = q.shape;
        shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
        lhs = Reshape(g, lhs, DataType::int32_t, shape);
        shape = p.inputs[1].shape;
        shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
        rhs = Reshape(g, rhs, DataType::int32_t, shape);
        auto value = g.AddMulOperator(
            lhs, rhs, g.AddTensorConstant(Attribute(int8_t{0})),
            TosaTensor(DataType::int32_t, p.output.shape));
        return Clamp(
            g,
            EmitNativeRescale(g, value, DataType::int32_t, DataType::int16_t,
                              p.output.shape, 0, 0, p.scales, constants,
                              p.min64, p.max64),
            p);
      }
      auto lhs = Rescale(g, input, q.shape, DataType::int8_t, DataType::int32_t,
                         q.zero_point, 0, {*MakeScale32(1)}, constants);
      auto rhs = Rescale(g, inputs[1], p.inputs[1].shape, DataType::int8_t,
                         DataType::int32_t, p.inputs[1].zero_point, 0,
                         {*MakeScale32(1)}, constants);
      auto shape = q.shape;
      shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
      lhs = Reshape(g, lhs, DataType::int32_t, shape);
      shape = p.inputs[1].shape;
      shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
      rhs = Reshape(g, rhs, DataType::int32_t, shape);
      auto value =
          g.AddMulOperator(lhs, rhs, g.AddTensorConstant(Attribute(int8_t{0})),
                           TosaTensor(DataType::int32_t, p.output.shape));
      return Clamp(g,
                   EmitRescale(g, value, p.output.shape, p.output.zero_point,
                               p.scales, p.ranges, constants),
                   p);
    }
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflDepthwiseConv2d:
      return EmitConv2d(g, input, *p.convolution, constants);
    case kLiteRtOpCodeTflTranspose:
      return g.AddTransposeOperator(input, Attribute(p.parameters), out);
    case kLiteRtOpCodeTflPadv2:
    case kLiteRtOpCodeTflPad:
      return g.AddPadOperator(
          input, g.AddTensorConstant(Attribute(p.parameters)),
          g.AddTensorConstant(Attribute(p.pad_value, p.output_type)), out);
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflAdd: {
      if (p.output_type == DataType::int16_t) {
        ResId lhs, rhs;
        if (p.equal_scale_fast_path) {
          const auto lhs_src =
              q.shape.empty() ? std::vector<uint32_t>{1} : q.shape;
          const auto rhs_src = p.inputs[1].shape.empty()
                                   ? std::vector<uint32_t>{1}
                                   : p.inputs[1].shape;
          lhs =
              g.AddCastOperator(input, TosaTensor(DataType::int32_t, lhs_src));
          rhs = g.AddCastOperator(inputs[1],
                                  TosaTensor(DataType::int32_t, rhs_src));
        } else {
          lhs = EmitNativeRescale(
              g, input, DataType::int16_t, DataType::int32_t, q.shape, 0, 0,
              {p.scales[0]}, constants, {p.min64[0]}, {p.max64[0]});
          rhs = EmitNativeRescale(
              g, inputs[1], DataType::int16_t, DataType::int32_t,
              p.inputs[1].shape, 0, 0, {p.scales[1]}, constants, {p.min64[1]},
              {p.max64[1]});
        }
        auto shape = q.shape;
        shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
        lhs = Reshape(g, lhs, DataType::int32_t, shape);
        shape = p.inputs[1].shape;
        shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
        rhs = Reshape(g, rhs, DataType::int32_t, shape);
        const auto sum_tensor = TosaTensor(DataType::int32_t, p.output.shape);
        auto sum = p.code == kLiteRtOpCodeTflSub
                       ? g.AddSubOperator(lhs, rhs, sum_tensor)
                       : g.AddAddOperator(lhs, rhs, sum_tensor);
        const auto& out_scale =
            p.equal_scale_fast_path ? p.scales[0] : p.scales[2];
        const int64_t out_lo =
            p.equal_scale_fast_path ? p.min64[0] : p.min64[2];
        const int64_t out_hi =
            p.equal_scale_fast_path ? p.max64[0] : p.max64[2];
        return Clamp(
            g,
            EmitNativeRescale(g, sum, DataType::int32_t, DataType::int16_t,
                              p.output.shape, 0, 0, {out_scale}, constants,
                              {out_lo}, {out_hi}),
            p);
      }
      auto lhs = Rescale(g, input, q.shape, DataType::int8_t, DataType::int32_t,
                         q.zero_point, 0, {p.scales[0]}, constants);
      auto rhs = Rescale(g, inputs[1], p.inputs[1].shape, DataType::int8_t,
                         DataType::int32_t, p.inputs[1].zero_point, 0,
                         {p.scales[1]}, constants);
      auto shape = q.shape;
      shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
      lhs = Reshape(g, lhs, DataType::int32_t, shape);
      shape = p.inputs[1].shape;
      shape.insert(shape.begin(), p.output.shape.size() - shape.size(), 1);
      rhs = Reshape(g, rhs, DataType::int32_t, shape);
      auto sum =
          p.code == kLiteRtOpCodeTflSub
              ? g.AddSubOperator(lhs, rhs,
                                 TosaTensor(DataType::int32_t, p.output.shape))
              : g.AddAddOperator(lhs, rhs,
                                 TosaTensor(DataType::int32_t, p.output.shape));
      return Clamp(g,
                   EmitRescale(g, sum, p.output.shape, p.output.zero_point,
                               {p.scales[2]}, p.ranges, constants),
                   p);
    }
    case kLiteRtOpCodeTflAveragePool2d: {
      if (p.cropped_shape != q.shape) {
        input = g.AddSliceOperator(
            input, g.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0))),
            g.AddTensorConstant(Attribute(p.cropped_shape)),
            TosaTensor(p.output_type, p.cropped_shape));
      }
      // Native TOSA rounding is intentionally allowed to differ from LiteRT
      // at quantization boundaries; avoid expanding pooling into integer ops.
      auto result = g.AddAvgPool2dOperator(
          input, g.AddTensorConstant(Attribute(q.zero_point, p.output_type)),
          g.AddTensorConstant(Attribute(p.output.zero_point, p.output_type)),
          Attribute(p.kernel), Attribute(p.stride), Attribute(p.padding),
          Attribute(int32_t{1}), out);
      return Clamp(g, result, p);
    }
    case kLiteRtOpCodeTflMaxPool2d: {
      if (p.cropped_shape != q.shape)
        input = g.AddSliceOperator(
            input, g.AddTensorConstant(Attribute(std::vector<uint32_t>(4, 0))),
            g.AddTensorConstant(Attribute(p.cropped_shape)),
            TosaTensor(p.output_type, p.cropped_shape));
      auto result = g.AddMaxPool2dOperator(
          input, Attribute(p.kernel), Attribute(p.stride), Attribute(p.padding),
          Attribute(int32_t{1}), out);
      return Clamp(g, result, p);
    }
    case kLiteRtOpCodeTflSum: {
      if (p.output_type == DataType::int16_t) {
        const auto src_shape =
            q.shape.empty() ? std::vector<uint32_t>{1} : q.shape;
        auto value =
            g.AddCastOperator(input, TosaTensor(DataType::int32_t, src_shape));
        auto shape = q.shape;
        for (auto axis : p.parameters) {
          shape[axis] = 1;
          value = g.AddReduceSumOperator(value, Attribute(axis),
                                         TosaTensor(DataType::int32_t, shape));
        }
        return EmitNativeRescale(g, value, DataType::int32_t, DataType::int16_t,
                                 p.output.shape, 0, 0, p.scales, constants,
                                 p.min64, p.max64);
      }
      auto value =
          Rescale(g, input, q.shape, DataType::int8_t, DataType::int32_t,
                  q.zero_point, 0, {*MakeScale32(1)}, constants);
      auto shape = q.shape;
      for (auto axis : p.parameters) {
        shape[axis] = 1;
        value = g.AddReduceSumOperator(value, Attribute(axis),
                                       TosaTensor(DataType::int32_t, shape));
      }
      return Rescale(g, value, p.output.shape, DataType::int32_t,
                     DataType::int8_t, 0, p.output.zero_point, p.scales,
                     constants);
    }
    default:
      return nullptr;
  }
}
}  // namespace litert::arm_vulkan_ml
