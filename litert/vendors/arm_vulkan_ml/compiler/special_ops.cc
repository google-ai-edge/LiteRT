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
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/emission.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/quantization.h"

namespace litert::arm_vulkan_ml {
namespace {
using ::tfsc::tosa::Attribute;
using ::tfsc::tosa::DataType;
using ::tfsc::tosa::Graph;
using ::tfsc::tosa::ResId;
using TosaTensor = ::tfsc::tosa::Tensor;
using Shape = std::vector<uint32_t>;

uint64_t Count(const Shape& s) {
  return std::accumulate(s.begin(), s.end(), uint64_t{1},
                         std::multiplies<uint64_t>());
}
std::optional<QuantizedTensor> Read(const Tensor& tensor) {
  auto type = tensor.RankedTensorType();
  if (!type || type->Layout().HasStrides()) return std::nullopt;
  const DataType dt =
      type->ElementType() == ElementType::Float32 ? DataType::float32_t
      : type->ElementType() == ElementType::Int32 ? DataType::int32_t
      : type->ElementType() == ElementType::Int16 ? DataType::int16_t
                                                  : DataType::int8_t;
  QuantizedTensor q{{}, 1, 0, type->ElementType() == ElementType::Float32, dt};
  if (!q.floating) {
    if ((type->ElementType() != ElementType::Int8 &&
         type->ElementType() != ElementType::Int16 &&
         type->ElementType() != ElementType::Int32) ||
        tensor.QTypeId() != QuantizationTypeId::PerTensor) {
      return std::nullopt;
    }
    auto params = tensor.PerTensorQuantization();
    const int64_t lo = type->ElementType() == ElementType::Int8    ? -128
                       : type->ElementType() == ElementType::Int16 ? -32768
                                                                   : INT32_MIN;
    const int64_t hi = type->ElementType() == ElementType::Int8    ? 127
                       : type->ElementType() == ElementType::Int16 ? 32767
                                                                   : INT32_MAX;
    if (!std::isfinite(params.scale) || params.scale <= 0 ||
        params.zero_point < lo || params.zero_point > hi) {
      return std::nullopt;
    }
    q.scale = params.scale;
    q.zero_point = static_cast<int32_t>(params.zero_point);
    if ((type->ElementType() == ElementType::Int16 ||
         type->ElementType() == ElementType::Int32) &&
        q.zero_point != 0) {
      return std::nullopt;
    }
  }
  uint64_t count = 1;
  for (auto dim : type->Layout().Dimensions()) {
    if (dim <= 0 || dim > 65536 ||
        count > (64u * 1024u * 1024u) / static_cast<uint32_t>(dim)) {
      return std::nullopt;
    }
    count *= dim;
    q.shape.push_back(static_cast<uint32_t>(dim));
  }
  if (q.shape.size() > 6 || Count(q.shape) > 64u * 1024u * 1024u) {
    return std::nullopt;
  }
  if (q.shape.empty()) q.shape = {1};
  return q;
}
bool SameDimensions(const Tensor& a, const Tensor& b) {
  auto x = a.RankedTensorType(), y = b.RankedTensorType();
  if (!x || !y) return false;
  const auto xd = x->Layout().Dimensions(), yd = y->Layout().Dimensions();
  return xd.size() == yd.size() && std::equal(xd.begin(), xd.end(), yd.begin());
}
DataType Type(ElementType type) {
  return type == ElementType::Float32 ? DataType::float32_t
         : type == ElementType::Int32 ? DataType::int32_t
         : type == ElementType::Int16 ? DataType::int16_t
                                      : DataType::int8_t;
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
  } else {
    return false;
  }
  return std::all_of(scales.begin(), scales.end(),
                     [](double s) { return std::isfinite(s) && s > 0; });
}
template <typename X>
ResId Constant(Graph& graph, const std::vector<X>& data, const Shape& shape,
               GraphConstantPool& pool) {
  std::vector<uint8_t> bytes(data.size() * sizeof(X));
  std::memcpy(bytes.data(), data.data(), bytes.size());
  return AddGraphConstant(graph,
                          TosaTensor(tfsc::tosa::GetDataType<X>(), shape),
                          std::move(bytes), pool);
}
ResId Reshape(Graph& graph, ResId x, DataType type, const Shape& shape) {
  return ReshapeTensor(graph, x, graph.AddTensorConstant(Attribute(shape)),
                       TosaTensor(type, shape));
}

// Fixed-point INT16 LayerNorm / RMSNorm Q-format pipeline:
// 1. Centering (`centered`): RMSNorm stays in Q0 (`centered_bits = 16`,
//    `centered_q_bits = 0`); LayerNorm rescales mean to Q8 (`centered_bits =
//    24`, `centered_q_bits = 8`) via 1-stage or 2-stage (`norm_mean_outer` x
//    `norm_mean_inner`) INT32 accumulation.
// 2. Per-row normalization: `abs_max` -> `clz` -> `left_shift_bits = clz - (32
//    - centered_bits)` aligns `centered` to `centered_bits` (`centered_norm`).
//    Shifting right by `sq_shift_bits = centered_bits - sq_term_bits`
//    (`sq_term_bits = (30 - ceil_log2(d)) / 2`) bounds `sum(sq_term^2)` within
//    signed INT32; shifting right by `q15_shift_bits = centered_bits - 15`
//    yields `centered_q15` in Q15.
// 3. Variance + epsilon & rsqrt LUT: `eps_rel_shift` aligns `eps_mant *
//    2^eps_shift` to the squared-sum exponent (downshifting `sum_sq` by even
//    `sum_sq_downshift` when needed). `var_sum_total >= 1` is normalized by
//    `even_headroom` into [2^29, 2^31 - 1], and its top 15 bits [8192, 32767]
//    index the 513-entry `norm_rsqrt_lut` (`TABLE`) producing `rsqrt_q15`.
// 4. Output rescaling & affine: `centered_q15 * rsqrt_q15` is rescaled by
//    `sqrt(d) / 2^(9 + (15 - sq_term_bits))` and aligned by `net_out_shift =
//    even_headroom / 2 - sum_sq_downshift / 2 - (20 - norm_qy)` to
//    `Q(norm_qy)` (`norm_qy = 14 - ceil(ceil_log2(d) / 2)`), then scaled by
//    optional gamma/beta (with Q4 bias rescaling when in range).
bool AnalyzeInt16IntegerNorm(const Op& op, SpecialPlan& p,
                             const QuantizedTensor& output, const Shape& shape,
                             Int16NormMode int16_norm_mode,
                             std::string& reason) {
  auto fail = [&](const char* text) {
    reason = text;
    return false;
  };
  const auto in = op.Inputs();
  const bool all_int =
      std::all_of(p.inputs.begin(), p.inputs.end(),
                  [](const QuantizedTensor& q) { return !q.floating; });
  const bool want_integer_norm = int16_norm_mode == Int16NormMode::kProInt ||
                                 (int16_norm_mode == Int16NormMode::kAuto &&
                                  p.kind == SpecialPlan::Kind::kLayerNorm);
  const uint64_t norm_dim64 = [&]() -> uint64_t {
    if (!p.norm_group_spatial) return shape.back();
    uint64_t spatial = 1;
    for (size_t i = 1; i + 1 < shape.size(); ++i) spatial *= shape[i];
    return spatial * (shape.back() / p.norm_groups);
  }();
  if (int16_norm_mode == Int16NormMode::kProInt &&
      norm_dim64 > (uint64_t{1} << 22)) {
    return fail("INT16 normalization group size exceeds integer range");
  }
  if (!want_integer_norm || !all_int || norm_dim64 > (uint64_t{1} << 22) ||
      (p.kind != SpecialPlan::Kind::kLayerNorm &&
       p.kind != SpecialPlan::Kind::kRmsNorm)) {
    return true;
  }

  const uint32_t norm_dim = static_cast<uint32_t>(norm_dim64);
  int norm_dim_ceil_log2 = 0;
  while ((uint32_t{1} << norm_dim_ceil_log2) < norm_dim) ++norm_dim_ceil_log2;
  const int sq_term_bits = (30 - norm_dim_ceil_log2) / 2;
  const int norm_out_q_bits = 14 - ((norm_dim_ceil_log2 + 1) / 2);
  const bool is_layer_norm = p.kind == SpecialPlan::Kind::kLayerNorm;
  const int centered_bits = is_layer_norm ? 24 : 16;
  const int centered_q_bits = is_layer_norm ? 8 : 0;
  const int norm_mul_extra_shift = 15 - sq_term_bits;
  const double eps_scaled = (static_cast<double>(norm_dim) * p.epsilon /
                             (p.inputs[0].scale * p.inputs[0].scale)) *
                            std::ldexp(1.0, 2 * centered_q_bits);
  int eps_exp = 0;
  const double eps_frac = std::frexp(eps_scaled, &eps_exp);
  int64_t eps_mantissa = std::llround(std::ldexp(eps_frac, 15));
  if (eps_mantissa >= 32768) {
    eps_mantissa >>= 1;
    ++eps_exp;
  }
  auto rsqrt_lut = PopulateInt16Table<double>(
      1.0 / 4096.0, 0, 1.0 / 32768.0, 0,
      [](double u) { return 1.0 / std::sqrt(std::max(u, 2.0)); });
  auto unit_scale = MakeScale32(1.0);
  auto norm_scale = MakeScale32(std::sqrt(static_cast<double>(norm_dim)) /
                                std::ldexp(1.0, 9 + norm_mul_extra_shift));
  auto mean_scale = unit_scale;
  int64_t mean_lo = 0;
  int64_t mean_hi = 0;
  if (is_layer_norm) {
    if (int64_t{32768} * norm_dim <= std::numeric_limits<int32_t>::max()) {
      p.norm_mean_outer = 1;
      p.norm_mean_inner = norm_dim;
      mean_scale = MakeScale32(256.0 / static_cast<double>(norm_dim));
      mean_lo = -int64_t{32768} * norm_dim;
      mean_hi = int64_t{32767} * norm_dim;
    } else {
      uint32_t best_inner = 0;
      for (uint32_t f = 2; static_cast<uint64_t>(f) * f <= norm_dim; ++f) {
        if (norm_dim % f == 0 && f <= 32768 && (norm_dim / f) <= 32768) {
          best_inner = f;
        }
      }
      if (best_inner > 0) {
        const uint32_t d_inner = best_inner;
        const uint32_t d_outer = norm_dim / best_inner;
        int32_t q_mid = 0;
        for (int32_t candidate : {8, 4, 2, 0}) {
          if ((int64_t{32768} << candidate) * d_outer <=
              std::numeric_limits<int32_t>::max()) {
            q_mid = candidate;
            break;
          }
        }
        auto stage1 =
            MakeScale32(std::ldexp(1.0, q_mid) / static_cast<double>(d_inner));
        if (!stage1 || !CanEmitNativeRescale(*stage1, -int64_t{32768} * d_inner,
                                             int64_t{32767} * d_inner)) {
          return fail("INT16 normalization stage-1 mean rescale failed");
        }
        p.norm_mean_outer = d_outer;
        p.norm_mean_inner = d_inner;
        p.norm_mean_q_mid = q_mid;
        p.norm_mean_stage1_scale = *stage1;
        mean_scale = MakeScale32(
            256.0 / (static_cast<double>(d_outer) * std::ldexp(1.0, q_mid)));
        mean_lo = -(int64_t{32768} << q_mid) * d_outer;
        mean_hi = (int64_t{32767} << q_mid) * d_outer;
      } else {
        int32_t pre_shift = 1;
        while (pre_shift < 15 && (int64_t{32768} >> pre_shift) * norm_dim >
                                     std::numeric_limits<int32_t>::max()) {
          ++pre_shift;
        }
        p.norm_mean_outer = 1;
        p.norm_mean_inner = norm_dim;
        p.norm_mean_pre_shift = pre_shift;
        mean_scale =
            MakeScale32(256.0 * static_cast<double>(int64_t{1} << pre_shift) /
                        static_cast<double>(norm_dim));
        mean_lo = -(int64_t{32768} >> pre_shift) * norm_dim;
        mean_hi = (int64_t{32767} >> pre_shift) * norm_dim;
      }
    }
  }
  if (!std::isfinite(eps_scaled) || eps_scaled <= 0 || eps_mantissa <= 0 ||
      !rsqrt_lut || !unit_scale ||
      !CanEmitNativeRescale(*unit_scale, 8192, 32767) || !norm_scale ||
      !CanEmitNativeRescale(*norm_scale, -(int64_t{1} << 30),
                            int64_t{1} << 30) ||
      !mean_scale ||
      (is_layer_norm && !CanEmitNativeRescale(*mean_scale, mean_lo, mean_hi))) {
    return fail("INT16 normalization parameters are not representable");
  }
  const double norm_out_scale = (in.size() > 1 ? p.inputs[1].scale : 1.0) /
                                std::ldexp(1.0, norm_out_q_bits);
  const int64_t scaled_bound =
      in.size() > 1 ? (int64_t{1} << 29) : (int64_t{1} << 15);
  p.integer_i16_norm = true;
  p.norm_p = sq_term_bits;
  p.norm_qy = norm_out_q_bits;
  p.norm_c_bits = centered_bits;
  p.norm_eps_mant = static_cast<int32_t>(eps_mantissa);
  p.norm_eps_shift_const = eps_exp - 15;
  p.norm_rsqrt_lut = std::move(*rsqrt_lut);
  p.scales = {*unit_scale, *norm_scale, *mean_scale};
  if (in.size() <= 2) {
    auto out_scale = MakeScale32(norm_out_scale / output.scale);
    if (!out_scale ||
        !CanEmitNativeRescale(*out_scale, -scaled_bound, scaled_bound)) {
      return fail("INT16 normalization output rescale out of range");
    }
    p.scales.push_back(*out_scale);
    return true;
  }
  int64_t bias_lo =
      p.input_types[2] == DataType::int16_t ? -32768 : -(int64_t{1} << 24);
  int64_t bias_hi =
      p.input_types[2] == DataType::int16_t ? 32767 : (int64_t{1} << 24);
  if (p.input_types[2] == DataType::int16_t) {
    if (auto w = in[2].WeightsData<int16_t>(); w && !w->empty()) {
      auto [mn, mx] = std::minmax_element(w->begin(), w->end());
      bias_lo = *mn;
      bias_hi = *mx;
    }
  } else if (auto w = in[2].WeightsData<int32_t>(); w && !w->empty()) {
    auto [mn, mx] = std::minmax_element(w->begin(), w->end());
    bias_lo = *mn;
    bias_hi = *mx;
  }
  bool configured = false;
  for (bool try_q4 : {true, false}) {
    const double q_factor = try_q4 ? 16.0 : 1.0;
    auto gamma_scale = MakeScale32(norm_out_scale * q_factor / output.scale);
    auto beta_scale = MakeScale32(p.inputs[2].scale * q_factor / output.scale);
    auto final_scale = try_q4 ? MakeScale32(1.0 / 16.0) : unit_scale;
    const int64_t final_bound = try_q4 ? (int64_t{1} << 29) : 32767;
    if (gamma_scale && beta_scale && final_scale &&
        CanEmitNativeRescale(*gamma_scale, -scaled_bound, scaled_bound) &&
        CanEmitNativeRescale(*beta_scale, bias_lo, bias_hi) &&
        CanEmitNativeRescale(*final_scale, -final_bound, final_bound)) {
      p.norm_q4_bias = try_q4;
      p.scales.push_back(*gamma_scale);
      p.scales.push_back(*beta_scale);
      p.scales.push_back(*final_scale);
      p.minimum = {bias_lo};
      p.maximum = {bias_hi};
      configured = true;
      break;
    }
  }
  if (!configured) {
    return fail("INT16 LayerNorm bias rescale out of range");
  }
  return true;
}

struct NormWorkLayout {
  const SpecialPlan& plan;
  Shape shape;
  Shape work_shape;
  Shape row_shape;
  Shape channel_shape;
  uint32_t gn_batch = 1;
  uint32_t gn_spatial = 1;
  uint32_t gn_groups = 1;
  uint32_t gn_group_ch = 1;

  ResId ToWork(Graph& graph, ResId x, DataType dt) const {
    if (!plan.norm_group_spatial) return x;
    if (gn_groups == 1 || gn_spatial == 1) {
      return Reshape(graph, x, dt, work_shape);
    }
    ResId r4 =
        Reshape(graph, x, dt, {gn_batch, gn_spatial, gn_groups, gn_group_ch});
    ResId tr = graph.AddTransposeOperator(
        r4, Attribute(std::vector<int32_t>{0, 2, 1, 3}),
        TosaTensor(dt, {gn_batch, gn_groups, gn_spatial, gn_group_ch}));
    return Reshape(graph, tr, dt, work_shape);
  }

  ResId FromWork(Graph& graph, ResId x, DataType dt) const {
    if (!plan.norm_group_spatial) return x;
    if (gn_groups == 1 || gn_spatial == 1) {
      return Reshape(graph, x, dt, shape);
    }
    ResId r4 =
        Reshape(graph, x, dt, {gn_batch, gn_groups, gn_spatial, gn_group_ch});
    ResId tr = graph.AddTransposeOperator(
        r4, Attribute(std::vector<int32_t>{0, 2, 1, 3}),
        TosaTensor(dt, {gn_batch, gn_spatial, gn_groups, gn_group_ch}));
    return Reshape(graph, tr, dt, shape);
  }
};

ResId EmitInt16IntegerNorm(Graph& graph, const std::vector<ResId>& inputs,
                           const SpecialPlan& plan,
                           const NormWorkLayout& layout, const Attribute& axis,
                           GraphConstantPool& pool) {
  const Shape& shape = layout.shape;
  const Shape& work_shape = layout.work_shape;
  const Shape& row_shape = layout.row_shape;
  const Shape& channel_shape = layout.channel_shape;
  const TosaTensor work_wide(DataType::int32_t, work_shape);
  const TosaTensor wide(DataType::int32_t, shape);
  const TosaTensor row(DataType::int32_t, row_shape);
  const TosaTensor row_bool(DataType::bool_t, row_shape);
  const auto work_i32 = [&](int32_t v) {
    return BroadcastScalar(graph, v, work_shape.size(), pool);
  };
  const auto mul_i32 = [&](ResId a, ResId b, const TosaTensor& tensor_type) {
    return graph.AddMulOperator(
        a, b, graph.AddTensorConstant(Attribute(int8_t{0})), tensor_type);
  };

  ResId centered = graph.AddCastOperator(
      layout.ToWork(graph, inputs[0], plan.input_type), work_wide);
  if (plan.kind == SpecialPlan::Kind::kLayerNorm) {
    ResId mean_q8;
    if (plan.norm_mean_stage1_scale) {
      Shape s4 = work_shape;
      s4.back() = plan.norm_mean_outer;
      s4.push_back(plan.norm_mean_inner);
      Shape mid_row = s4;
      mid_row.back() = 1;
      ResId c4 = Reshape(graph, centered, DataType::int32_t, s4);
      ResId sum_inner = graph.AddReduceSumOperator(
          c4, Attribute(static_cast<int32_t>(s4.size() - 1)),
          TosaTensor(DataType::int32_t, mid_row));
      ResId mean_inner = EmitNativeRescale(
          graph, sum_inner, DataType::int32_t, DataType::int32_t, mid_row, 0, 0,
          {*plan.norm_mean_stage1_scale}, pool,
          {-int64_t{32768} * static_cast<int64_t>(plan.norm_mean_inner)},
          {int64_t{32767} * static_cast<int64_t>(plan.norm_mean_inner)});
      Shape s3 = work_shape;
      s3.back() = plan.norm_mean_outer;
      ResId mean_inner3 = Reshape(graph, mean_inner, DataType::int32_t, s3);
      ResId sum1 = graph.AddReduceSumOperator(mean_inner3, axis, row);
      mean_q8 =
          EmitNativeRescale(graph, sum1, DataType::int32_t, DataType::int32_t,
                            row_shape, 0, 0, {plan.scales[2]}, pool,
                            {-(int64_t{32768} << plan.norm_mean_q_mid) *
                             static_cast<int64_t>(plan.norm_mean_outer)},
                            {(int64_t{32767} << plan.norm_mean_q_mid) *
                             static_cast<int64_t>(plan.norm_mean_outer)});
    } else {
      ResId sum_in = centered;
      if (plan.norm_mean_pre_shift > 0) {
        sum_in = graph.AddArithmeticRightShiftOperator(
            centered, work_i32(plan.norm_mean_pre_shift), Attribute(false),
            work_wide);
      }
      ResId sum1 = graph.AddReduceSumOperator(sum_in, axis, row);
      const int64_t lo = -(int64_t{32768} >> plan.norm_mean_pre_shift) *
                         static_cast<int64_t>(work_shape.back());
      const int64_t hi = (int64_t{32767} >> plan.norm_mean_pre_shift) *
                         static_cast<int64_t>(work_shape.back());
      mean_q8 = EmitNativeRescale(graph, sum1, DataType::int32_t,
                                  DataType::int32_t, row_shape, 0, 0,
                                  {plan.scales[2]}, pool, {lo}, {hi});
    }
    ResId centered_q8 =
        graph.AddLogicalLeftShiftOperator(centered, work_i32(8), work_wide);
    centered = graph.AddSubOperator(centered_q8, mean_q8, work_wide);
  }

  ResId abs_centered = graph.AddAbsOperator(centered, work_wide);
  ResId max_centered = graph.AddReduceMaxOperator(abs_centered, axis,
                                                  Attribute(uint32_t{1}), row);
  ResId clz_max = graph.AddClzOperator(max_centered, row);
  ResId left_shift_bits =
      graph.AddSubOperator(clz_max, work_i32(32 - plan.norm_c_bits), row);
  ResId centered_norm =
      graph.AddLogicalLeftShiftOperator(centered, left_shift_bits, work_wide);
  const int32_t sq_shift_bits = plan.norm_c_bits - plan.norm_p;
  const int32_t q15_shift_bits = plan.norm_c_bits - 15;
  ResId sq_term = graph.AddArithmeticRightShiftOperator(
      centered_norm, work_i32(sq_shift_bits), Attribute(false), work_wide);
  ResId centered_q15 = graph.AddArithmeticRightShiftOperator(
      centered_norm, work_i32(q15_shift_bits), Attribute(true), work_wide);
  ResId sq = mul_i32(sq_term, sq_term, work_wide);
  ResId sum_sq = graph.AddReduceSumOperator(sq, axis, row);

  ResId eps_rel_shift = graph.AddAddOperator(
      work_i32(plan.norm_eps_shift_const - 2 * sq_shift_bits),
      graph.AddLogicalLeftShiftOperator(left_shift_bits, work_i32(1), row),
      row);
  ResId shift_exceeds_14 =
      graph.AddGreaterOperator(eps_rel_shift, work_i32(14), row_bool);
  ResId downshift_even_raw = graph.AddBitwiseAndOperator(
      graph.AddSubOperator(eps_rel_shift, work_i32(13), row), work_i32(-2),
      row);
  ResId sum_sq_downshift = graph.AddSelectOperator(
      shift_exceeds_14,
      graph.AddMinimumOperator(downshift_even_raw, work_i32(30),
                               Attribute(uint32_t{1}), row),
      work_i32(0), row);
  ResId clamped_eps_shift =
      graph.AddSubOperator(eps_rel_shift, sum_sq_downshift, row);
  clamped_eps_shift = graph.AddMaximumOperator(clamped_eps_shift, work_i32(-31),
                                               Attribute(uint32_t{1}), row);
  clamped_eps_shift = graph.AddMinimumOperator(clamped_eps_shift, work_i32(14),
                                               Attribute(uint32_t{1}), row);
  ResId eps_left_shift = graph.AddMaximumOperator(
      clamped_eps_shift, work_i32(0), Attribute(uint32_t{1}), row);
  ResId eps_right_shift = graph.AddMaximumOperator(
      graph.AddSubOperator(work_i32(0), clamped_eps_shift, row), work_i32(0),
      Attribute(uint32_t{1}), row);
  ResId eps_adj = graph.AddArithmeticRightShiftOperator(
      graph.AddLogicalLeftShiftOperator(work_i32(plan.norm_eps_mant),
                                        eps_left_shift, row),
      eps_right_shift, Attribute(false), row);
  ResId sum_sq_adj = graph.AddArithmeticRightShiftOperator(
      sum_sq, sum_sq_downshift, Attribute(false), row);
  ResId var_sum_total =
      graph.AddMaximumOperator(graph.AddAddOperator(sum_sq_adj, eps_adj, row),
                               work_i32(1), Attribute(uint32_t{1}), row);

  ResId clz_var_sum = graph.AddClzOperator(var_sum_total, row);
  ResId even_headroom = graph.AddBitwiseAndOperator(
      graph.AddSubOperator(clz_var_sum, work_i32(1), row), work_i32(-2), row);
  ResId var_sum_norm =
      graph.AddLogicalLeftShiftOperator(var_sum_total, even_headroom, row);
  ResId lut_idx_i32 = graph.AddArithmeticRightShiftOperator(
      var_sum_norm, work_i32(16), Attribute(false), row);
  ResId lut_idx_i16 = EmitNativeRescale(
      graph, lut_idx_i32, DataType::int32_t, DataType::int16_t, row_shape, 0, 0,
      {plan.scales[0]}, pool, {8192}, {32767});
  ResId lut_id = Constant(graph, plan.norm_rsqrt_lut, {513}, pool);
  ResId rsqrt_raw = graph.AddTableOperator(lut_idx_i16, lut_id, row);
  ResId rsqrt_q15 = graph.AddArithmeticRightShiftOperator(
      rsqrt_raw, work_i32(7), Attribute(true), row);

  ResId prod = mul_i32(centered_q15, rsqrt_q15, work_wide);
  ResId norm_base = EmitNativeRescale(
      graph, prod, DataType::int32_t, DataType::int32_t, work_shape, 0, 0,
      {plan.scales[1]}, pool, {-(int64_t{1} << 30)}, {int64_t{1} << 30});
  ResId net_out_shift = graph.AddSubOperator(
      graph.AddSubOperator(
          graph.AddArithmeticRightShiftOperator(even_headroom, work_i32(1),
                                                Attribute(false), row),
          graph.AddArithmeticRightShiftOperator(sum_sq_downshift, work_i32(1),
                                                Attribute(false), row),
          row),
      work_i32(20 - plan.norm_qy), row);
  ResId out_left_shift = graph.AddMinimumOperator(
      graph.AddMaximumOperator(net_out_shift, work_i32(0),
                               Attribute(uint32_t{1}), row),
      work_i32(31), Attribute(uint32_t{1}), row);
  ResId out_right_shift = graph.AddMinimumOperator(
      graph.AddMaximumOperator(
          graph.AddSubOperator(work_i32(0), net_out_shift, row), work_i32(0),
          Attribute(uint32_t{1}), row),
      work_i32(31), Attribute(uint32_t{1}), row);
  ResId norm_out_work = graph.AddArithmeticRightShiftOperator(
      graph.AddLogicalLeftShiftOperator(norm_base, out_left_shift, work_wide),
      out_right_shift, Attribute(true), work_wide);
  ResId norm_out = layout.FromWork(graph, norm_out_work, DataType::int32_t);

  ResId scaled = norm_out;
  if (inputs.size() > 1) {
    ResId gamma = Reshape(graph, inputs[1], plan.input_types[1], channel_shape);
    if (plan.input_types[1] != DataType::int32_t) {
      gamma = graph.AddCastOperator(
          gamma, TosaTensor(DataType::int32_t, channel_shape));
    }
    scaled = mul_i32(norm_out, gamma, wide);
  }
  const int64_t scaled_bound =
      inputs.size() > 1 ? (int64_t{1} << 29) : (int64_t{1} << 15);
  if (inputs.size() <= 2) {
    return EmitNativeRescale(graph, scaled, DataType::int32_t,
                             DataType::int16_t, shape, 0, 0, {plan.scales[3]},
                             pool, {-scaled_bound}, {scaled_bound});
  }
  ResId y_scaled = EmitNativeRescale(
      graph, scaled, DataType::int32_t, DataType::int32_t, shape, 0, 0,
      {plan.scales[3]}, pool, {-scaled_bound}, {scaled_bound});
  ResId beta_reshaped =
      Reshape(graph, inputs[2], plan.input_types[2], channel_shape);
  ResId beta_i32 =
      plan.input_types[2] == DataType::int32_t
          ? beta_reshaped
          : graph.AddCastOperator(beta_reshaped,
                                  TosaTensor(DataType::int32_t, channel_shape));
  ResId beta_scaled = EmitNativeRescale(
      graph, beta_i32, DataType::int32_t, DataType::int32_t, channel_shape, 0,
      0, {plan.scales[4]}, pool, {plan.minimum[0]}, {plan.maximum[0]});
  ResId sum_out = graph.AddAddOperator(y_scaled, beta_scaled, wide);
  if (!plan.norm_q4_bias) {
    const auto orig_i32 = [&](int32_t v) {
      return BroadcastScalar(graph, v, shape.size(), pool);
    };
    sum_out = graph.AddMinimumOperator(
        graph.AddMaximumOperator(sum_out, orig_i32(-32768),
                                 Attribute(uint32_t{1}), wide),
        orig_i32(32767), Attribute(uint32_t{1}), wide);
    return EmitNativeRescale(graph, sum_out, DataType::int32_t,
                             DataType::int16_t, shape, 0, 0, {plan.scales[5]},
                             pool, {-32768}, {32767});
  }
  return EmitNativeRescale(graph, sum_out, DataType::int32_t, DataType::int16_t,
                           shape, 0, 0, {plan.scales[5]}, pool,
                           {-(int64_t{1} << 29)}, {int64_t{1} << 29});
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
std::optional<OperationPlan> AnalyzeSpecialOperation(
    const Op& op, std::string& reason, Int16NormMode int16_norm_mode) {
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
    const bool i16_norm = p->input_type == DataType::int16_t&& p->output_type ==
                          DataType::int16_t&& std::isfinite(
                              static_cast<float>(1.0 / output->scale));
    const bool i8_norm = p->input_type == DataType::int8_t&& p->output_type ==
                         DataType::int8_t&& std::isfinite(
                             static_cast<float>(1.0 / output->scale));
    if (!std::isfinite(p->epsilon) ||
        !std::isfinite(static_cast<float>(p->epsilon)) ||
        static_cast<float>(p->epsilon) <= 0 ||
        (!fp_norm && !i16_norm && !i8_norm) || shape != output->shape ||
        !SameDimensions(in[0], out[0]) ||
        original_type->Layout().Dimensions().empty())
      return fail(
          "Normalization requires matching static FP32/INT8/INT16 data and "
          "positive finite epsilon");
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
      if (!attrs["num_groups"].IsInt() || !attrs["sub_type"].IsInt() ||
          in.empty() || in.size() > 3)
        return fail("Invalid odml.group_norm composite attributes");
      const int64_t sub_type = attrs["sub_type"].AsInt64();
      const int64_t num_groups = attrs["num_groups"].AsInt64();
      if (sub_type == 1) {
        if (num_groups != 1)
          return fail(
              "Only the one-group LayerNorm composite form is supported");
      } else if (sub_type == 0) {
        if (shape.size() < 2 || num_groups < 1 ||
            shape.back() % static_cast<uint64_t>(num_groups) != 0)
          return fail(
              "GroupNorm requires rank >= 2 and num_groups dividing channels");
        p->norm_group_spatial = true;
        p->norm_groups = static_cast<uint32_t>(num_groups);
      } else {
        return fail("Unsupported odml.group_norm sub_type");
      }
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
        const DataType param_type = p->input_types.back();
        const bool ok = p->inputs.back().floating ||
                        param_type == DataType::int16_t ||
                        (i8_norm && param_type == DataType::int8_t) ||
                        (i == 2 && param_type == DataType::int32_t);
        if (!ok) {
          return fail(
              "Quantized normalization scale/bias must be FP32 or "
              "matching INT8/INT16/INT32");
        }
      }
    }
    if (i16_norm) {
      plan.required_extensions = {"EXT-INT16"};
      if (!AnalyzeInt16IntegerNorm(op, *p, *output, shape, int16_norm_mode,
                                   reason)) {
        return std::nullopt;
      }
    }
  } else if (op.Code() == kLiteRtOpCodeTflFullyConnected) {
    p->kind = SpecialPlan::Kind::kFullyConnected;
    const bool wide = p->input_type == DataType::int16_t;
    if ((in.size() != 2 && in.size() != 3) || p->output_type != p->input_type)
      return fail("FC requires matching INT8 or INT16 data types");
    const bool has_bias = in.size() == 3;
    auto wt = in[1].RankedTensorType();
    if (!wt || wt->Layout().HasStrides() ||
        wt->ElementType() != ElementType::Int8 ||
        wt->Layout().Dimensions().size() != 2)
      return fail(
          "FC requires constant INT8 matrix weights and matching INT32/INT64 "
          "bias");
    auto bt = has_bias ? in[2].RankedTensorType()
                       : litert::Unexpected(Status::kErrorInvalidArgument);
    if (has_bias && (!bt || bt->Layout().HasStrides() ||
                     bt->ElementType() !=
                         (wide ? ElementType::Int64 : ElementType::Int32) ||
                     bt->Layout().Dimensions().size() != 1))
      return fail(
          "FC requires constant INT8 matrix weights and matching INT32/INT64 "
          "bias");
    auto dims = wt->Layout().Dimensions();
    if (dims[0] <= 0 || dims[0] > 65536 || dims[1] <= 0 || dims[1] > 65536 ||
        (has_bias && bt->Layout().Dimensions()[0] != dims[0]))
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
    if (!has_bias) {
      p->bias.assign(p->channels, 0);
    } else if (wide) {
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
        (has_bias && !Scales(in[2], p->channels, bias_scale)))
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
      if (has_bias && std::abs(bias_scale[c] - scale) > scale * 1e-5)
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
      if (p->input_type != DataType::int16_t || !p->output.floating) {
        return fail(
            "INT16 conversion requires FP32/INT16 with matching shapes");
      }
    } else if (p->inputs[0].floating) {
      if (p->output_type != DataType::int16_t) {
        return fail(
            "INT16 conversion requires FP32/INT16 with matching shapes");
      }
      if (!std::isfinite(static_cast<float>(1.0 / output->scale))) {
        return fail("INT16 quantization reciprocal scale exceeds FP32");
      }
    } else if ((p->input_type == DataType::int16_t&& p->output_type ==
                DataType::int8_t) ||
               (p->input_type == DataType::int8_t&& p->output_type ==
                DataType::int16_t)) {
      const int64_t lo = (p->input_type == DataType::int8_t ? -128 : -32768) -
                         p->inputs[0].zero_point;
      const int64_t hi = (p->input_type == DataType::int8_t ? 127 : 32767) -
                         p->inputs[0].zero_point;
      std::optional<Scale32> s =
          MakeScale32(p->inputs[0].scale / output->scale);
      if (!s || !IsScale32RangeValid(*s, lo, hi)) {
        return fail("INT8/INT16 QUANTIZE rescale range is invalid");
      }
      p->scales = {*s};
      p->minimum = {lo};
      p->maximum = {hi};
    } else {
      return fail("INT16 conversion requires FP32/INT16 with matching shapes");
    }
    plan.required_extensions = {"EXT-INT16"};
  } else {
    if (in.size() != 2 || !add(1) || p->input_type != DataType::int16_t ||
        p->output_type != DataType::int16_t) {
      return fail("INT16 bound requires matching integer operands");
    }
    for (const QuantizedTensor& q : p->inputs) {
      if (q.floating || q.scale != output->scale ||
          q.zero_point != output->zero_point) {
        return fail("INT16 bound quantization must match");
      }
    }
    if (p->inputs[1].shape != Shape{1} || !in[1].HasWeights() ||
        shape != output->shape) {
      return fail("INT16 bound requires a scalar constant");
    }
    auto bound = in[1].WeightsData<int16_t>();
    if (!bound || bound->size() != 1) {
      return fail("Invalid INT16 bound constant");
    }
    p->activation = {-32768, 32767};
    if (op.Code() == kLiteRtOpCodeTflMinimum) {
      p->activation.maximum = (*bound)[0];
    } else {
      p->activation.minimum = (*bound)[0];
    }
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

ResId EmitSpecialOperation(Graph& graph, const std::vector<ResId>& inputs,
                           const SpecialPlan& plan, GraphConstantPool& pool) {
  const Shape& shape = plan.output.shape;
  const TosaTensor out(plan.output_type, shape), fp(DataType::float32_t, shape);
  const auto scalar = [&](float x) {
    return BroadcastScalar(graph, x, shape.size(), pool);
  };
  const ResId zero_shift = graph.AddTensorConstant(Attribute(int8_t{0}));
  auto multiply = [&](ResId a, ResId b, const TosaTensor& type) {
    return graph.AddMulOperator(a, b, zero_shift, type);
  };
  auto quantize_fp_to_int = [&](ResId val) {
    ResId scaled =
        multiply(val, scalar(static_cast<float>(1.0 / plan.output.scale)), fp);
    ResId positive = graph.AddGreaterEqualOperator(
        scaled, scalar(0), TosaTensor(DataType::bool_t, shape));
    ResId up = graph.AddFloorOperator(
        graph.AddAddOperator(scaled, scalar(.5f), fp), fp);
    ResId negative = graph.AddSubOperator(scalar(0), scaled, fp);
    ResId down = graph.AddSubOperator(
        scalar(0),
        graph.AddFloorOperator(graph.AddAddOperator(negative, scalar(.5f), fp),
                               fp),
        fp);
    scaled = graph.AddSelectOperator(positive, up, down, fp);
    if (plan.output.zero_point != 0) {
      scaled = graph.AddAddOperator(
          scaled, scalar(static_cast<float>(plan.output.zero_point)), fp);
    }
    const float q_lo =
        plan.output_type == DataType::int8_t ? -128.0f : -32768.0f;
    const float q_hi = plan.output_type == DataType::int8_t ? 127.0f : 32767.0f;
    scaled = graph.AddClampOperator(scaled, Attribute(q_lo), Attribute(q_hi),
                                    Attribute(int32_t{0}), fp);
    return graph.AddCastOperator(scaled, out);
  };
  if (plan.kind == SpecialPlan::Kind::kFullyConnected) {
    const bool wide = plan.input_type == DataType::int16_t;
    Shape expanded{plan.batch, 1, 1, plan.channels};
    ResId x = Reshape(graph, inputs[0], plan.input_type,
                      {plan.batch, 1, 1, plan.depth});
    ResId w =
        Constant(graph, plan.weights, {plan.channels, 1, 1, plan.depth}, pool);
    ResId bias;
    if (wide) {
      bias = Constant(graph, plan.bias, {plan.channels}, pool);
    } else {
      bias = Constant(graph,
                      std::vector<int32_t>(plan.bias.begin(), plan.bias.end()),
                      {plan.channels}, pool);
    }
    const DataType acc = wide ? DataType::int48_t : DataType::int32_t;
    ResId value = graph.AddConv2dOperator(
        x, w, bias,
        graph.AddTensorConstant(
            Attribute(plan.inputs[0].zero_point, plan.input_type)),
        graph.AddTensorConstant(Attribute(int8_t{0})),
        Attribute(std::vector<int32_t>{0, 0, 0, 0}),
        Attribute(std::vector<int32_t>{1, 1}),
        Attribute(std::vector<int32_t>{1, 1}), Attribute(int32_t{wide ? 4 : 1}),
        Attribute(false), TosaTensor(acc, expanded));
    value = EmitNativeRescale(graph, value, acc, plan.output_type, expanded, 0,
                              plan.output.zero_point, plan.scales, pool,
                              plan.minimum, plan.maximum);
    const int low = wide ? -32768 : -128, high = wide ? 32767 : 127;
    if (plan.activation.minimum != low || plan.activation.maximum != high) {
      value = graph.AddClampOperator(
          value, Attribute(plan.activation.minimum, plan.output_type),
          Attribute(plan.activation.maximum, plan.output_type),
          Attribute(int32_t{1}), TosaTensor(plan.output_type, expanded));
    }
    return Reshape(graph, value, plan.output_type, shape);
  }
  if (plan.kind == SpecialPlan::Kind::kMinimum ||
      plan.kind == SpecialPlan::Kind::kMaximum) {
    return graph.AddClampOperator(
        inputs[0], Attribute(static_cast<int16_t>(plan.activation.minimum)),
        Attribute(static_cast<int16_t>(plan.activation.maximum)),
        Attribute(int32_t{1}), out);
  }
  if (plan.kind == SpecialPlan::Kind::kDequantize) {
    ResId value = graph.AddCastOperator(inputs[0], fp);
    return multiply(value, scalar(static_cast<float>(plan.inputs[0].scale)),
                    fp);
  }
  if (plan.kind == SpecialPlan::Kind::kQuantize) {
    if (!plan.inputs[0].floating) {
      return EmitNativeRescale(
          graph, inputs[0], plan.input_type, plan.output_type, shape,
          plan.inputs[0].zero_point, plan.output.zero_point, plan.scales, pool,
          plan.minimum, plan.maximum);
    }
    return quantize_fp_to_int(inputs[0]);
  }
  Shape work_shape = shape;
  uint32_t gn_batch = 1, gn_spatial = 1, gn_groups = 1, gn_group_ch = 1;
  if (plan.norm_group_spatial) {
    gn_batch = shape[0];
    gn_groups = plan.norm_groups;
    gn_group_ch = shape.back() / gn_groups;
    for (size_t i = 1; i + 1 < shape.size(); ++i) gn_spatial *= shape[i];
    work_shape = {gn_batch, gn_groups, gn_spatial * gn_group_ch};
  }
  Shape row_shape = work_shape;
  row_shape.back() = 1;
  const Attribute axis(static_cast<int32_t>(work_shape.size() - 1));
  Shape channel_shape(shape.size(), 1);
  channel_shape.back() = shape.back();
  const NormWorkLayout layout{plan,       shape,         work_shape,
                              row_shape,  channel_shape, gn_batch,
                              gn_spatial, gn_groups,     gn_group_ch};
  if (plan.integer_i16_norm) {
    return EmitInt16IntegerNorm(graph, inputs, plan, layout, axis, pool);
  }
  auto to_work = [&](ResId x, DataType dt) {
    return layout.ToWork(graph, x, dt);
  };
  auto from_work = [&](ResId x, DataType dt) {
    return layout.FromWork(graph, x, dt);
  };
  const TosaTensor work_fp(DataType::float32_t, work_shape);
  const TosaTensor row(DataType::float32_t, row_shape);
  const auto work_scalar = [&](float x) {
    return BroadcastScalar(graph, x, work_shape.size(), pool);
  };
  auto mean = [&](ResId value) {
    return multiply(graph.AddReduceSumOperator(value, axis, row),
                    work_scalar(1.0f / work_shape.back()), row);
  };
  ResId data = to_work(inputs[0], plan.input_type);
  if (!plan.inputs[0].floating) {
    data = graph.AddCastOperator(data, work_fp);
    if (plan.inputs[0].zero_point != 0) {
      data = graph.AddSubOperator(
          data, work_scalar(static_cast<float>(plan.inputs[0].zero_point)),
          work_fp);
    }
    data = multiply(data, work_scalar(static_cast<float>(plan.inputs[0].scale)),
                    work_fp);
  }
  ResId centered = data;
  if (plan.kind == SpecialPlan::Kind::kLayerNorm) {
    centered = graph.AddSubOperator(data, mean(data), work_fp);
  }
  ResId square = multiply(centered, centered, work_fp);
  ResId variance = plan.kind == SpecialPlan::Kind::kL2Norm
                       ? graph.AddReduceSumOperator(square, axis, row)
                       : mean(square);
  ResId bounded = graph.AddAddOperator(
      variance, work_scalar(static_cast<float>(plan.epsilon)), row);
  ResId value = from_work(
      multiply(centered, graph.AddRsqrtOperator(bounded, row), work_fp),
      DataType::float32_t);
  const TosaTensor fp_channel(DataType::float32_t, channel_shape);
  auto channel_param = [&](size_t i) {
    ResId param = Reshape(graph, inputs[i], plan.input_types[i], channel_shape);
    if (!plan.inputs[i].floating) {
      param = graph.AddCastOperator(param, fp_channel);
      if (plan.inputs[i].zero_point != 0) {
        param = graph.AddSubOperator(
            param, scalar(static_cast<float>(plan.inputs[i].zero_point)),
            fp_channel);
      }
      param = multiply(param, scalar(static_cast<float>(plan.inputs[i].scale)),
                       fp_channel);
    }
    return param;
  };
  if (inputs.size() > 1) value = multiply(value, channel_param(1), fp);
  if (inputs.size() > 2) {
    value = graph.AddAddOperator(value, channel_param(2), fp);
  }
  return plan.output.floating ? value : quantize_fp_to_int(value);
}
}  // namespace litert::arm_vulkan_ml
