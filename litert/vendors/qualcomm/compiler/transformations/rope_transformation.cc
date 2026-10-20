// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/qualcomm/compiler/transformations/rope_transformation.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/vendors/qualcomm/compiler/transformations/orphan_cleanup_transformation.h"

namespace {

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::Unexpected;
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::GatherOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_Custom;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Not;
using litert::compiler::m_Op;
using litert::compiler::m_Options;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::SubOptions;
using litert::compiler::Tensor;

constexpr float kQ15Scale = 1.0f / 32767.0f;

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) {
    return {};
  }
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

bool IsF32(const Tensor& t) {
  return t.ElementType() == ElementType::Float32 && !t.HasQuantization();
}

auto m_F32() {
  return m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                 m_Not(m_IsQuantized()));
}

bool IsLastAxisConcat(const Op& op, int rank) {
  auto opts = GetOptionsAs<ConcatenationOptions>(op.ctx(), op.Get());
  return opts && opts->fused_activation_function == 0 &&
         (opts->axis == rank - 1 || opts->axis == -1);
}

// Shapes of the block, read from the matched root.
struct Geometry {
  int32_t b, s, h, d;  // d = full head dim.
  std::vector<int32_t> Full() const { return {b, s, h, d}; }
  std::vector<int32_t> Half() const { return {b, s, h, d / 2}; }
  std::vector<int32_t> Quarter() const { return {b, s, h, d / 4}; }
  std::vector<int32_t> Trig() const { return {b, s, 1, d / 4}; }
  std::vector<int32_t> Table() const { return {b, s, 1, d}; }
};

// Mul(slice [B,S,H,q], trig [B,S,1,q]) with operands in either order.
struct MulMatch {
  Op op;
  Tensor slice;
  Tensor trig;
};

std::optional<MulMatch> MatchMul(const Tensor& t, const Geometry& g) {
  Op mul(t.ctx(), nullptr);
  Tensor slice(t.ctx(), nullptr);
  Tensor trig(t.ctx(), nullptr);
  auto slice_m =
      m_CaptureOrSameAs(&slice, m_AllOf(m_F32(), m_Shape(g.Quarter())));
  auto trig_m = m_CaptureOrSameAs(&trig, m_AllOf(m_F32(), m_Shape(g.Trig())));
  auto mul_m = m_AllOf(
      m_HasOneUse(), m_F32(), m_Shape(g.Quarter()),
      m_CaptureOrSameAs(&mul, m_AllOf(m_Options<MulOptions>([](const auto& o) {
                                        return o.fused_activation_function == 0;
                                      }),
                                      m_CommutativeOp<kLiteRtOpCodeTflMul>(
                                          slice_m, trig_m))));
  if (!Match(t, mul_m)) {
    return std::nullopt;
  }
  return MulMatch{mul, slice, trig};
}

// Returns the begin offset along the last axis of a Slice op whose other begin
// offsets are all zero, or nullopt.
std::optional<int32_t> LastAxisSliceBegin(const Op& slice) {
  auto ins = slice.Inputs();
  if (ins.size() != 3 || !ins[1].IsConstant()) {
    return std::nullopt;
  }
  auto begin = ins[1].WeightsData<int32_t>();
  if (!begin || begin->empty()) {
    return std::nullopt;
  }
  for (size_t i = 0; i + 1 < begin->size(); ++i) {
    if ((*begin)[i] != 0) {
      return std::nullopt;
    }
  }
  return begin->back();
}

// Checks that `quarter` == in[..., half_idx * d/2 + quarter_idx * d/4 : +d/4]
// through a Slice(Slice(in)) chain, and returns `in`.
std::optional<Tensor> TraceQuarter(const Tensor& quarter, int half_idx,
                                   int quarter_idx, const Geometry& g) {
  Tensor in(quarter.ctx(), nullptr);
  auto in_m = m_CaptureOrSameAs(&in, m_AllOf(m_F32(), m_Shape(g.Full())));
  auto h_slice_m =
      m_AllOf(m_F32(), m_Shape(g.Half()), m_Custom([&](const Tensor& t) {
                auto op = t.GetDefiningOp();
                auto b = op ? LastAxisSliceBegin(*op) : std::nullopt;
                return b && *b == half_idx * (g.d / 2);
              }),
              m_Op<kLiteRtOpCodeTflSlice>(in_m, m_Any(), m_Any()));
  auto q_slice_m =
      m_AllOf(m_Custom([&](const Tensor& t) {
                auto op = t.GetDefiningOp();
                auto b = op ? LastAxisSliceBegin(*op) : std::nullopt;
                return b && *b == quarter_idx * (g.d / 4);
              }),
              m_Op<kLiteRtOpCodeTflSlice>(h_slice_m, m_Any(), m_Any()));
  if (!Match(quarter, q_slice_m)) {
    return std::nullopt;
  }
  return in;
}

// One rotated half: Concat(Sub(a*c, b*s), Add(b*c, a*s)).
struct HalfMatch {
  Tensor a, b;  // First and second quarter of the half.
  Tensor cos, sin;  // [B, S, 1, d/4].
  std::vector<LiteRtOp> ops;  // Concat, Sub, Add and the 4 Muls.
};

std::optional<HalfMatch> MatchHalf(const Tensor& half_out, const Geometry& g) {
  Op concat(half_out.ctx(), nullptr);
  Op sub(half_out.ctx(), nullptr);
  Op add(half_out.ctx(), nullptr);
  Tensor sub_lhs(half_out.ctx(), nullptr);
  Tensor sub_rhs(half_out.ctx(), nullptr);
  Tensor add_lhs(half_out.ctx(), nullptr);
  Tensor add_rhs(half_out.ctx(), nullptr);

  auto sub_m =
      m_AllOf(m_HasOneUse(),
              m_CaptureOrSameAs(
                  &sub, m_AllOf(m_Options<SubOptions>([](const auto& o) {
                                  return o.fused_activation_function == 0;
                                }),
                                m_Op<kLiteRtOpCodeTflSub>(
                                    m_CaptureOrSameAs(&sub_lhs, m_Any()),
                                    m_CaptureOrSameAs(&sub_rhs, m_Any())))));
  auto add_m =
      m_AllOf(m_HasOneUse(),
              m_CaptureOrSameAs(
                  &add, m_AllOf(m_Options<AddOptions>([](const auto& o) {
                                  return o.fused_activation_function == 0;
                                }),
                                m_Op<kLiteRtOpCodeTflAdd>(
                                    m_CaptureOrSameAs(&add_lhs, m_Any()),
                                    m_CaptureOrSameAs(&add_rhs, m_Any())))));
  auto concat_m = m_AllOf(
      m_HasOneUse(), m_Shape(g.Half()),
      m_CaptureOrSameAs(
          &concat, m_AllOf(m_Options<ConcatenationOptions>([](const auto& o) {
                             return o.fused_activation_function == 0 &&
                                    (o.axis == 3 || o.axis == -1);
                           }),
                           m_Op<kLiteRtOpCodeTflConcatenation>(sub_m, add_m))));
  if (!Match(half_out, concat_m)) {
    return std::nullopt;
  }
  // Sub: a * cos - b * sin.
  auto m0 = MatchMul(sub_lhs, g);
  auto m1 = MatchMul(sub_rhs, g);
  // Add: b * cos + a * sin (either order).
  auto m2 = MatchMul(add_lhs, g);
  auto m3 = MatchMul(add_rhs, g);
  if (!m0 || !m1 || !m2 || !m3) return std::nullopt;
  HalfMatch h{m0->slice, m1->slice, m0->trig, m1->trig, {}};
  if (h.a == h.b || h.cos == h.sin) return std::nullopt;
  auto is = [](const MulMatch& m, const Tensor& s, const Tensor& t) {
    return m.slice == s && m.trig == t;
  };
  const bool add_ok = (is(*m2, h.b, h.cos) && is(*m3, h.a, h.sin)) ||
                      (is(*m2, h.a, h.sin) && is(*m3, h.b, h.cos));
  if (!add_ok) return std::nullopt;
  h.ops = {concat.Get(), sub.Get(),    add.Get(),   m0->op.Get(),
           m1->op.Get(), m2->op.Get(), m3->op.Get()};
  return h;
}

// Shared trig tables, keyed by the (cos_lo, sin_lo, cos_hi, sin_hi) tensors.
struct TableEntry {
  std::array<LiteRtTensor, 4> key;
  LiteRtTensor cos_tab;
  LiteRtTensor sin_tab;
};

std::vector<TableEntry>& Tables() {
  static auto* tables = new std::vector<TableEntry>();
  return *tables;
}

struct IntegerTableEntry {
  LiteRtTensor position;
  LiteRtTensor frequency;
  int32_t depth;
  LiteRtTensor cos;
  LiteRtTensor sin;
};

std::vector<IntegerTableEntry>& IntegerTables() {
  static auto* tables = new std::vector<IntegerTableEntry>();
  return *tables;
}

std::vector<std::pair<LiteRtTensor, int32_t>>& IntegerPositionDepths() {
  static auto* depths = new std::vector<std::pair<LiteRtTensor, int32_t>>();
  return *depths;
}

Expected<Tensor> NewQuantizedTensor(Builder& builder, ElementType type,
                                    const std::vector<int32_t>& dims,
                                    LiteRtQuantizationPerTensor quantization) {
  RankedTensorType tensor_type(
      type, Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  return builder.BuildTensor(std::move(RankedTensorSpecBuilder(tensor_type))
                                 .WithPerTensorQuantization(quantization)
                                 .Build());
}

Expected<Tensor> NewQuantizedLike(Builder& builder, const Tensor& like,
                                  const std::vector<int32_t>& dims) {
  if (like.QTypeId() != kLiteRtQuantizationPerTensor) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  return NewQuantizedTensor(builder, like.ElementType(), dims,
                            like.PerTensorQuantization());
}

Expected<Tensor> BuildInt32Scalar(Builder& builder, int32_t value) {
  RankedTensorType type(ElementType::Int32, Layout(BuildLayout({})));
  LITERT_ASSIGN_OR_RETURN(
      auto tensor, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int32_t>(absl::MakeConstSpan(&value, 1), tensor));
  return tensor;
}

Expected<Tensor> BuildIntegerAdd(Builder& builder, const Tensor& lhs,
                                 const Tensor& rhs,
                                 const std::vector<int32_t>& dims) {
  RankedTensorType type(
      ElementType::Int32,
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  LITERT_ASSIGN_OR_RETURN(
      auto out, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflAdd, {lhs, rhs}, {out}));
  AddOptions opts;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

std::optional<int32_t> PositionDepth(const Tensor& position) {
  for (const auto& [tensor, depth] : IntegerPositionDepths()) {
    if (tensor == position.Get()) return depth;
  }
  for (const auto& position_use : position.Uses()) {
    if (position_use.user.Code() != kLiteRtOpCodeTflReshape ||
        position_use.user.Outputs().size() != 1) {
      continue;
    }
    for (const auto& reshape_use : position_use.user.Outputs()[0].Uses()) {
      if (reshape_use.user.Code() != kLiteRtOpCodeTflOneHot ||
          reshape_use.user.Inputs().size() < 2) {
        continue;
      }
      auto depth = reshape_use.user.Inputs()[1].WeightsData<int32_t>();
      if (depth && depth->size() == 1 && (*depth)[0] > 0) {
        return (*depth)[0];
      }
    }
  }
  return std::nullopt;
}

struct TrigSource {
  Tensor position;
  Tensor frequency;
  int32_t depth;
};

std::optional<TrigSource> MatchTrigSource(const Tensor& cos,
                                          const Tensor& sin) {
  auto cos_op = cos.GetDefiningOp();
  auto sin_op = sin.GetDefiningOp();
  if (!cos_op || !sin_op || cos_op->Code() != kLiteRtOpCodeTflCos ||
      sin_op->Code() != kLiteRtOpCodeTflSin || cos_op->Inputs().size() != 1 ||
      sin_op->Inputs().size() != 1 ||
      cos_op->Inputs()[0] != sin_op->Inputs()[0]) {
    return std::nullopt;
  }
  auto reshape = cos_op->Inputs()[0].GetDefiningOp();
  if (!reshape || reshape->Code() != kLiteRtOpCodeTflReshape ||
      reshape->Inputs().empty()) {
    return std::nullopt;
  }
  auto mul = reshape->Inputs()[0].GetDefiningOp();
  if (!mul || mul->Code() != kLiteRtOpCodeTflMul || mul->Inputs().size() != 2) {
    return std::nullopt;
  }
  std::optional<Tensor> frequency;
  std::optional<Tensor> cast_out;
  for (const Tensor& input : mul->Inputs()) {
    if (input.IsConstant() && input.ElementType() == ElementType::Float32) {
      frequency = input;
    } else {
      auto def = input.GetDefiningOp();
      if (def && def->Code() == kLiteRtOpCodeTflCast) cast_out = input;
    }
  }
  if (!frequency || !cast_out) return std::nullopt;
  auto cast = cast_out->GetDefiningOp();
  if (!cast || cast->Inputs().size() != 1 ||
      cast->Inputs()[0].ElementType() != ElementType::Int32) {
    return std::nullopt;
  }
  Tensor position = cast->Inputs()[0];
  auto depth = PositionDepth(position);
  if (!depth) return std::nullopt;
  return TrigSource{position, *frequency, *depth};
}

Expected<std::pair<Tensor, Tensor>> BuildIntegerTables(
    Builder& builder, const TrigSource& source) {
  for (const auto& entry : IntegerTables()) {
    if (entry.position == source.position.Get() &&
        entry.frequency == source.frequency.Get() &&
        entry.depth == source.depth) {
      return std::make_pair(Tensor(source.position.ctx(), entry.cos),
                            Tensor(source.position.ctx(), entry.sin));
    }
  }
  auto frequencies = source.frequency.WeightsData<float>();
  if (!frequencies || frequencies->empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  const int32_t width = frequencies->size();
  std::vector<int16_t> cos_data((source.depth + 1) * width);
  std::vector<int16_t> sin_data((source.depth + 1) * width * 2);
  for (int32_t table_index = 0; table_index <= source.depth; ++table_index) {
    const float position = static_cast<float>(table_index - 1);
    for (int32_t i = 0; i < width; ++i) {
      const float angle = position * (*frequencies)[i];
      cos_data[table_index * width + i] =
          static_cast<int16_t>(std::round(std::cos(angle) / kQ15Scale));
      const int16_t sin_value =
          static_cast<int16_t>(std::round(std::sin(angle) / kQ15Scale));
      sin_data[table_index * width * 2 + i] = -sin_value;
      sin_data[table_index * width * 2 + width + i] = sin_value;
    }
  }
  const LiteRtQuantizationPerTensor q15{kQ15Scale, 0};
  LITERT_ASSIGN_OR_RETURN(auto cos_table,
                          NewQuantizedTensor(builder, ElementType::Int16,
                                             {source.depth + 1, width}, q15));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int16_t>(absl::MakeConstSpan(cos_data), cos_table));
  LITERT_ASSIGN_OR_RETURN(
      auto sin_table, NewQuantizedTensor(builder, ElementType::Int16,
                                         {source.depth + 1, width * 2}, q15));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int16_t>(absl::MakeConstSpan(sin_data), sin_table));

  LITERT_ASSIGN_OR_RETURN(auto one, BuildInt32Scalar(builder, 1));
  LITERT_ASSIGN_OR_RETURN(
      auto indices,
      BuildIntegerAdd(builder, source.position, one, Dims(source.position)));
  auto gather = [&](const Tensor& table,
                    int32_t output_width) -> Expected<Tensor> {
    auto output_dims = Dims(source.position);
    output_dims.push_back(output_width);
    LITERT_ASSIGN_OR_RETURN(
        auto out,
        NewQuantizedTensor(builder, ElementType::Int16, output_dims, q15));
    LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(kLiteRtOpCodeTflGather,
                                                     {table, indices}, {out}));
    GatherOptions opts;
    opts.axis = 0;
    opts.batch_dims = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
    return out;
  };
  LITERT_ASSIGN_OR_RETURN(auto cos, gather(cos_table, width));
  LITERT_ASSIGN_OR_RETURN(auto sin, gather(sin_table, width * 2));
  IntegerTables().push_back({source.position.Get(), source.frequency.Get(),
                             source.depth, cos.Get(), sin.Get()});
  return std::make_pair(cos, sin);
}

Expected<Tensor> NewTensor(Builder& builder, const std::vector<int32_t>& dims) {
  RankedTensorType type(
      ElementType::Float32,
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  return builder.BuildTensor(RankedTensorSpecBuilder(type).Build());
}

Expected<Tensor> BuildConcat(Builder& builder, const std::vector<Tensor>& ins,
                             const std::vector<int32_t>& out_dims,
                             int32_t axis = -1) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewTensor(builder, out_dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflConcatenation, ins, {out}));
  ConcatenationOptions opts;
  opts.axis = axis >= 0 ? axis : static_cast<int32_t>(out_dims.size()) - 1;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<void> BuildBinary(Builder& builder, LiteRtOpCode code, const Tensor& a,
                           const Tensor& b, const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(code, {a, b}, {out}));
  if (code == kLiteRtOpCodeTflMul) {
    MulOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  } else {
    AddOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  }
  return {};
}

Expected<Tensor> BuildMul(Builder& builder, const Tensor& a, const Tensor& b,
                          const std::vector<int32_t>& out_dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewTensor(builder, out_dims));
  LITERT_RETURN_IF_ERROR(BuildBinary(builder, kLiteRtOpCodeTflMul, a, b, out));
  return out;
}

Expected<Tensor> BuildQuantizedSlice(Builder& builder, const Tensor& input,
                                     int32_t offset, int32_t length) {
  auto dims = Dims(input);
  if (dims.empty()) return Unexpected(kLiteRtStatusErrorInvalidArgument);
  std::vector<int32_t> begin(dims.size(), 0);
  begin.back() = offset;
  dims.back() = length;
  RankedTensorType i32_vector(
      ElementType::Int32,
      Layout(BuildLayout({static_cast<int32_t>(dims.size())})));
  auto build_vector =
      [&](const std::vector<int32_t>& values) -> Expected<Tensor> {
    LITERT_ASSIGN_OR_RETURN(
        auto tensor,
        builder.BuildTensor(RankedTensorSpecBuilder(i32_vector).Build()));
    LITERT_RETURN_IF_ERROR(
        builder.BuildWeights<int32_t>(absl::MakeConstSpan(values), tensor));
    return tensor;
  };
  LITERT_ASSIGN_OR_RETURN(auto begin_tensor, build_vector(begin));
  LITERT_ASSIGN_OR_RETURN(auto size_tensor, build_vector(dims));
  LITERT_ASSIGN_OR_RETURN(auto out, NewQuantizedLike(builder, input, dims));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(
      kLiteRtOpCodeTflSlice, {input, begin_tensor, size_tensor}, {out}));
  return out;
}

Expected<Tensor> BuildQuantizedConcat(Builder& builder,
                                      const std::vector<Tensor>& inputs,
                                      const Tensor& like, int32_t axis,
                                      const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewQuantizedLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflConcatenation, inputs, {out}));
  ConcatenationOptions opts;
  opts.axis = axis;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<Tensor> BuildQuantizedBinary(Builder& builder, LiteRtOpCode code,
                                      const Tensor& lhs, const Tensor& rhs,
                                      const Tensor& like,
                                      const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewQuantizedLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(auto op, builder.BuildOp(code, {lhs, rhs}, {out}));
  if (code == kLiteRtOpCodeTflMul) {
    MulOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  }
  return out;
}

void RegisterRopeProducers(const Tensor& tensor,
                           std::vector<LiteRtOp>& registered) {
  auto op = tensor.GetDefiningOp();
  if (!op) return;
  registered.push_back(op->Get());
  if (op->Code() == kLiteRtOpCodeTflDequantize ||
      op->Code() == kLiteRtOpCodeTflCast) {
    return;
  }
  for (const Tensor& input : op->Inputs()) {
    RegisterRopeProducers(input, registered);
  }
}

bool ShouldTileHeads(const Geometry& g, const Tensor& rope_out) {
  constexpr int64_t kDefaultMinShaLogitsBytes = 8LL * 1024 * 1024;
  if (static_cast<int64_t>(g.h) * g.s * g.s * 2 >= kDefaultMinShaLogitsBytes) {
    return true;
  }
  for (const auto& u : rope_out.Uses()) {
    if (u.user.Code() == kLiteRtOpCodeTflReshape ||
        u.user.Code() == kLiteRtOpCodeTflUnpack) {
      return true;
    }
    if (u.user.Code() == kLiteRtOpCodeTflQuantize) {
      for (const auto& q_out : u.user.Outputs()) {
        for (const auto& qu : q_out.Uses()) {
          if (qu.user.Code() == kLiteRtOpCodeTflReshape ||
              qu.user.Code() == kLiteRtOpCodeTflUnpack) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

Expected<std::pair<Tensor, Tensor>> GetOrBuildTables(
    const LiteRtCompilerContext* ctx, Builder& builder, const HalfMatch& lo,
    const HalfMatch& hi, const Geometry& g, bool tile_heads) {
  const std::array<LiteRtTensor, 4> key = {lo.cos.Get(), lo.sin.Get(),
                                           hi.cos.Get(), hi.sin.Get()};
  std::optional<Tensor> base_cos;
  std::optional<Tensor> base_sin;
  for (const auto& e : Tables()) {
    if (e.key == key) {
      base_cos = Tensor(ctx, e.cos_tab);
      base_sin = Tensor(ctx, e.sin_tab);
      break;
    }
  }
  if (!base_cos) {
    LITERT_ASSIGN_OR_RETURN(auto minus_one, NewTensor(builder, {1, 1, 1, 1}));
    constexpr std::array<float, 1> kMinusOne = {-1.0f};
    LITERT_RETURN_IF_ERROR(
        builder.BuildWeights<float>(absl::MakeConstSpan(kMinusOne), minus_one));
    LITERT_ASSIGN_OR_RETURN(auto neg_lo,
                            BuildMul(builder, lo.sin, minus_one, g.Trig()));
    LITERT_ASSIGN_OR_RETURN(auto neg_hi,
                            BuildMul(builder, hi.sin, minus_one, g.Trig()));
    LITERT_ASSIGN_OR_RETURN(
        base_cos,
        BuildConcat(builder, {lo.cos, lo.cos, hi.cos, hi.cos}, g.Table()));
    LITERT_ASSIGN_OR_RETURN(
        base_sin,
        BuildConcat(builder, {neg_lo, lo.sin, neg_hi, hi.sin}, g.Table()));
    Tables().push_back({key, base_cos->Get(), base_sin->Get()});
  }
  Tensor cos_tab = *base_cos;
  Tensor sin_tab = *base_sin;
  if (tile_heads && g.h > 1) {
    std::vector<Tensor> cos_ins(g.h, cos_tab);
    std::vector<Tensor> sin_ins(g.h, sin_tab);
    LITERT_ASSIGN_OR_RETURN(
        cos_tab, BuildConcat(builder, cos_ins, g.Full(), /*axis=*/2));
    LITERT_ASSIGN_OR_RETURN(
        sin_tab, BuildConcat(builder, sin_ins, g.Full(), /*axis=*/2));
  }
  return std::make_pair(cos_tab, sin_tab);
}

}  // namespace

extern "C" {

void ResetRopeTransformationState() {
  Tables().clear();
  IntegerTables().clear();
  IntegerPositionDepths().clear();
}

void RegisterIntegerRopePositionDepth(LiteRtTensor position, int32_t depth) {
  if (position == nullptr || depth <= 0) return;
  for (auto& [tensor, registered_depth] : IntegerPositionDepths()) {
    if (tensor == position) {
      registered_depth = depth;
      return;
    }
  }
  IntegerPositionDepths().push_back({position, depth});
}

LiteRtStatus IntegerRopeTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr,
                                       LiteRtOp op_ptr) {
  Op quantize(context, op_ptr);
  if (quantize.Code() != kLiteRtOpCodeTflQuantize ||
      quantize.Inputs().size() != 1 || quantize.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor float_out = quantize.Inputs()[0];
  Tensor int_out = quantize.Outputs()[0];
  if (float_out.ElementType() != ElementType::Float32 ||
      int_out.ElementType() != ElementType::Int16 ||
      int_out.QTypeId() != kLiteRtQuantizationPerTensor) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto root = float_out.GetDefiningOp();
  if (!root || root->Code() != kLiteRtOpCodeTflConcatenation ||
      root->Outputs().size() != 1 || root->Inputs().size() != 2) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto out_dims = Dims(float_out);
  if (out_dims.size() != 4 || out_dims[3] % 4 != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Geometry geometry{out_dims[0], out_dims[1], out_dims[2], out_dims[3]};
  auto lo = MatchHalf(root->Inputs()[0], geometry);
  auto hi = MatchHalf(root->Inputs()[1], geometry);
  if (!lo || !hi) return kLiteRtStatusPatternNoMatch;
  auto input0 = TraceQuarter(lo->a, 0, 0, geometry);
  auto input1 = TraceQuarter(lo->b, 0, 1, geometry);
  auto input2 = TraceQuarter(hi->a, 1, 0, geometry);
  auto input3 = TraceQuarter(hi->b, 1, 1, geometry);
  if (!input0 || !input1 || !input2 || !input3 || *input0 != *input1 ||
      *input0 != *input2 || *input0 != *input3) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto dequantize = input0->GetDefiningOp();
  if (!dequantize || dequantize->Code() != kLiteRtOpCodeTflDequantize ||
      dequantize->Inputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor int_input = dequantize->Inputs()[0];
  if (int_input.ElementType() != ElementType::Int16 ||
      int_input.QTypeId() != kLiteRtQuantizationPerTensor ||
      Dims(int_input) != out_dims) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto lo_trig = MatchTrigSource(lo->cos, lo->sin);
  auto hi_trig = MatchTrigSource(hi->cos, hi->sin);
  if (!lo_trig || !hi_trig ||
      lo_trig->frequency.Get() != hi_trig->frequency.Get() ||
      lo_trig->depth != hi_trig->depth) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  auto build = [&]() -> Expected<void> {
    LITERT_ASSIGN_OR_RETURN(auto lo_tables,
                            BuildIntegerTables(builder, *lo_trig));
    LITERT_ASSIGN_OR_RETURN(auto hi_tables,
                            BuildIntegerTables(builder, *hi_trig));
    auto trig_dims = Dims(lo_tables.first);
    if (trig_dims.size() != 4 || trig_dims.back() * 4 != geometry.d) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    auto half_trig_dims = trig_dims;
    half_trig_dims.back() *= 2;
    LITERT_ASSIGN_OR_RETURN(
        auto cos_lo,
        BuildQuantizedConcat(builder, {lo_tables.first, lo_tables.first},
                             lo_tables.first, 3, half_trig_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto cos_hi,
        BuildQuantizedConcat(builder, {hi_tables.first, hi_tables.first},
                             hi_tables.first, 3, half_trig_dims));
    auto full_trig_dims = trig_dims;
    full_trig_dims.back() = geometry.d;
    LITERT_ASSIGN_OR_RETURN(
        auto cos_table, BuildQuantizedConcat(builder, {cos_lo, cos_hi}, cos_lo,
                                             3, full_trig_dims));

    LITERT_ASSIGN_OR_RETURN(
        auto sin_table,
        BuildQuantizedConcat(builder, {lo_tables.second, hi_tables.second},
                             lo_tables.second, 3, full_trig_dims));

    const int32_t quarter = geometry.d / 4;
    LITERT_ASSIGN_OR_RETURN(
        auto s0, BuildQuantizedSlice(builder, int_input, 0, quarter));
    LITERT_ASSIGN_OR_RETURN(
        auto s1, BuildQuantizedSlice(builder, int_input, quarter, quarter));
    LITERT_ASSIGN_OR_RETURN(
        auto s2, BuildQuantizedSlice(builder, int_input, 2 * quarter, quarter));
    LITERT_ASSIGN_OR_RETURN(
        auto s3, BuildQuantizedSlice(builder, int_input, 3 * quarter, quarter));
    LITERT_ASSIGN_OR_RETURN(
        auto rotated, BuildQuantizedConcat(builder, {s1, s0, s3, s2}, int_input,
                                           3, out_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto cos_term,
        BuildQuantizedBinary(builder, kLiteRtOpCodeTflMul, int_input, cos_table,
                             int_out, out_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto sin_term,
        BuildQuantizedBinary(builder, kLiteRtOpCodeTflMul, rotated, sin_table,
                             int_out, out_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto add,
        builder.BuildOp(kLiteRtOpCodeTflAdd, {cos_term, sin_term}, {int_out}));
    AddOptions opts;
    opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(add, std::move(opts)));
    return {};
  };
  if (auto status = build(); !status) return status.Error().Status();

  std::vector<LiteRtOp> orphans;
  RegisterRopeProducers(float_out, orphans);
  RegisterOrphanGroup(orphans);
  builder.EraseOp(quantize);
  return kLiteRtStatusOk;
}

LiteRtStatus RopeTransformation(const LiteRtCompilerContext* context,
                                LiteRtBuilder builder_ptr, LiteRtOp op) {
  Op root(context, op);
  Tensor in_lo(context, nullptr);
  Tensor in_hi(context, nullptr);
  if (!Match(root, m_AllOf(m_Custom([](const Op& o) {
                             return IsLastAxisConcat(o, 4) &&
                                    o.Outputs().size() == 1;
                           }),
                           m_Op<kLiteRtOpCodeTflConcatenation>(
                               m_CaptureOrSameAs(&in_lo, m_Any()),
                               m_CaptureOrSameAs(&in_hi, m_Any()))))) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor out = root.Outputs()[0];
  auto out_dims = Dims(out);
  if (out_dims.size() != 4 || !IsF32(out) || out_dims[3] % 4 != 0) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Geometry g{out_dims[0], out_dims[1], out_dims[2], out_dims[3]};

  auto lo = MatchHalf(in_lo, g);
  if (!lo) return kLiteRtStatusPatternNoMatch;
  auto hi = MatchHalf(in_hi, g);
  if (!hi) return kLiteRtStatusPatternNoMatch;

  // All four quarters must come from the same input at the right offsets.
  auto in0 = TraceQuarter(lo->a, 0, 0, g);
  auto in1 = TraceQuarter(lo->b, 0, 1, g);
  auto in2 = TraceQuarter(hi->a, 1, 0, g);
  auto in3 = TraceQuarter(hi->b, 1, 1, g);
  if (!in0 || !in1 || !in2 || !in3 || *in0 != *in1 || *in0 != *in2 ||
      *in0 != *in3) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor in = *in0;

  Builder builder(context, builder_ptr);
  const bool tile_heads = ShouldTileHeads(g, out);
  auto tables = GetOrBuildTables(context, builder, *lo, *hi, g, tile_heads);
  if (!tables) return tables.Error().Status();
  auto [cos_tab, sin_tab] = *tables;

  auto in_rot = BuildConcat(builder, {lo->b, lo->a, hi->b, hi->a}, g.Full());
  if (!in_rot) return in_rot.Error().Status();
  auto term1 = BuildMul(builder, in, cos_tab, g.Full());
  if (!term1) return term1.Error().Status();
  auto term2 = BuildMul(builder, *in_rot, sin_tab, g.Full());
  if (!term2) return term2.Error().Status();
  auto add = BuildBinary(builder, kLiteRtOpCodeTflAdd, *term1, *term2, out);
  if (!add) return add.Error().Status();

  builder.EraseOp(root);
  RegisterOrphanGroup(lo->ops);
  RegisterOrphanGroup(hi->ops);
  return kLiteRtStatusOk;
}

LiteRtStatus RopeAttentionLayerTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  return RopeTransformation(context, builder_ptr, op);
}

LiteRtStatus RopeCleanupTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr, LiteRtOp op) {
  return OrphanCleanupTransformation(context, builder_ptr, op);
}

}  // extern "C"
