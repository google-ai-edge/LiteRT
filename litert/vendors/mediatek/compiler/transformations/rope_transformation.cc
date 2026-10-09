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

#include "litert/vendors/mediatek/compiler/transformations/rope_transformation.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_IsConstant;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Not;
using litert::compiler::m_Op;
using litert::compiler::m_Options;
using litert::compiler::m_Predicate;
using litert::compiler::m_Rank;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::SubOptions;
using litert::compiler::Tensor;

namespace {

// Rank of every tensor in the RoPE block ([B, S, H, d]).
constexpr int kRank = 4;

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) {
    return {};
  }
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

// Non-quantized float32 tensor.
auto m_F32() {
  return m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                 m_Not(m_IsQuantized()), "F32");
}

// Ranked tensor whose dimensions are exactly `dims`. Unlike m_Shape, -1 is
// not treated as a wildcard, so dynamic dimensions must match literally.
auto m_ExactShape(std::vector<int32_t> dims) {
  return m_Predicate<Tensor>(
      [dims = std::move(dims)](const Tensor& t) { return Dims(t) == dims; },
      "ExactShape");
}

// Op with exactly one output, which matches `m`.
template <typename M>
auto m_SoleOutput(M m) {
  return m_Predicate<Op>(
      [m = std::move(m)](const Op& op) {
        auto outs = op.Outputs();
        return outs.size() == 1 && Match(outs[0], m);
      },
      "SoleOutput");
}

// Op (or tensor's defining op) without a fused activation.
template <typename OptionsT>
auto m_NoActivation() {
  return m_Options<OptionsT>(
      [](const OptionsT& opts) { return opts.fused_activation_function == 0; },
      "NoActivation");
}

// Concatenation along the last axis of a rank-4 tensor, no fused activation.
auto m_LastAxisConcat() {
  return m_Options<ConcatenationOptions>(
      [](const ConcatenationOptions& opts) {
        return opts.fused_activation_function == 0 &&
               (opts.axis == kRank - 1 || opts.axis == -1);
      },
      "LastAxisConcat");
}

// Constant int32 Slice `begin` whose offsets are all zero except the last one,
// which must equal `last`.
auto m_LastAxisSliceBegin(int32_t last) {
  return m_AllOf(m_IsConstant(),
                 m_Predicate<Tensor>(
                     [last](const Tensor& t) {
                       auto begin = t.WeightsData<int32_t>();
                       if (!begin || begin->empty()) {
                         return false;
                       }
                       for (size_t i = 0; i + 1 < begin->size(); ++i) {
                         if ((*begin)[i] != 0) {
                           return false;
                         }
                       }
                       return begin->back() == last;
                     },
                     "LastAxisSliceBeginValues"),
                 "LastAxisSliceBegin");
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

// Matches a single-use f32 [B,S,H,d/4] Mul output and captures the Mul op
// and its slice/trig operands into `m`. If both operand orders qualify, the
// (in0, in1) = (slice, trig) order wins.
auto m_RopeMul(MulMatch* m, const Geometry& g) {
  return m_CaptureOrSameAs(
      &m->op,
      m_AllOf(m_HasOneUse(), m_F32(), m_ExactShape(g.Quarter()),
              m_CommutativeOp<kLiteRtOpCodeTflMul>(
                  m_CaptureOrSameAs(
                      &m->slice, m_AllOf(m_F32(), m_ExactShape(g.Quarter()))),
                  m_CaptureOrSameAs(&m->trig,
                                    m_AllOf(m_F32(), m_ExactShape(g.Trig())))),
              m_NoActivation<MulOptions>()));
}

// One rotated half: Concat(Sub(a*c, b*s), Add(b*c, a*s)).
struct HalfMatch {
  Op concat, sub, add;
  // Sub lhs (a*c), Sub rhs (b*s), Add lhs, Add rhs.
  std::array<MulMatch, 4> mul;

  // First and second quarter of the half.
  const Tensor& a() const { return mul[0].slice; }
  const Tensor& b() const { return mul[1].slice; }
  // [B, S, 1, d/4].
  const Tensor& cos() const { return mul[0].trig; }
  const Tensor& sin() const { return mul[1].trig; }

  // Checks the operand roles: Add is b * cos + a * sin (either order).
  bool HasRotationRoles() const {
    if (a() == b() || cos() == sin()) return false;
    auto is = [](const MulMatch& m, const Tensor& s, const Tensor& t) {
      return m.slice == s && m.trig == t;
    };
    return (is(mul[2], b(), cos()) && is(mul[3], a(), sin())) ||
           (is(mul[2], a(), sin()) && is(mul[3], b(), cos()));
  }

  // Concat, Sub, Add and the 4 Muls.
  std::vector<LiteRtOp> Ops() const {
    return {concat.Get(),    sub.Get(),       add.Get(),      mul[0].op.Get(),
            mul[1].op.Get(), mul[2].op.Get(), mul[3].op.Get()};
  }
};

// Matches one rotated half [B,S,H,d/2] and captures its ops into `h`.
auto m_RopeHalf(HalfMatch* h, const Geometry& g) {
  auto sub = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &h->sub, m_AllOf(m_Op<kLiteRtOpCodeTflSub>(m_RopeMul(&h->mul[0], g),
                                                     m_RopeMul(&h->mul[1], g)),
                           m_NoActivation<SubOptions>())));
  auto add = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &h->add, m_AllOf(m_Op<kLiteRtOpCodeTflAdd>(m_RopeMul(&h->mul[2], g),
                                                     m_RopeMul(&h->mul[3], g)),
                           m_NoActivation<AddOptions>())));
  return m_AllOf(
      m_HasOneUse(), m_ExactShape(g.Half()),
      m_CaptureOrSameAs(&h->concat, m_AllOf(m_Op<kLiteRtOpCodeTflConcatenation>(
                                                std::move(sub), std::move(add)),
                                            m_LastAxisConcat())),
      m_Predicate<Tensor>([h](const Tensor&) { return h->HasRotationRoles(); },
                          "RotationRoles"),
      "RopeHalf");
}

// Matches `in[..., half_idx * d/2 + quarter_idx * d/4 : +d/4]` through a
// Slice(Slice(in)) chain, capturing `in` (or checking it is the same tensor).
auto m_RopeQuarter(Tensor* in, int half_idx, int quarter_idx,
                   const Geometry& g) {
  auto half = m_AllOf(
      m_ExactShape(g.Half()), m_F32(),
      m_Op<kLiteRtOpCodeTflSlice>(
          m_CaptureOrSameAs(in, m_AllOf(m_ExactShape(g.Full()), m_F32())),
          m_LastAxisSliceBegin(half_idx * (g.d / 2)), m_Any()));
  return m_Op<kLiteRtOpCodeTflSlice>(
      std::move(half), m_LastAxisSliceBegin(quarter_idx * (g.d / 4)), m_Any(),
      "RopeQuarter");
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

Expected<Tensor> NewTensor(Builder& builder, const std::vector<int32_t>& dims) {
  RankedTensorType type(
      ElementType::Float32,
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  return builder.BuildTensor(RankedTensorSpecBuilder(type).Build());
}

Expected<Tensor> BuildConcat(Builder& builder, const std::vector<Tensor>& ins,
                             const std::vector<int32_t>& out_dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewTensor(builder, out_dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflConcatenation, ins, {out}));
  ConcatenationOptions opts;
  opts.axis = static_cast<int32_t>(out_dims.size()) - 1;
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

Expected<std::pair<Tensor, Tensor>> GetOrBuildTables(
    const LiteRtCompilerContext* ctx, Builder& builder, const HalfMatch& lo,
    const HalfMatch& hi, const Geometry& g) {
  const std::array<LiteRtTensor, 4> key = {lo.cos().Get(), lo.sin().Get(),
                                           hi.cos().Get(), hi.sin().Get()};
  for (const auto& e : Tables()) {
    if (e.key == key) {
      return std::make_pair(Tensor(ctx, e.cos_tab), Tensor(ctx, e.sin_tab));
    }
  }
  // -1 constant used to negate sin (MediaTek has no Neg legalization).
  LITERT_ASSIGN_OR_RETURN(auto minus_one, NewTensor(builder, {1, 1, 1, 1}));
  const float kMinusOne[1] = {-1.0f};
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<float>(absl::MakeConstSpan(kMinusOne), minus_one));
  LITERT_ASSIGN_OR_RETURN(auto neg_lo,
                          BuildMul(builder, lo.sin(), minus_one, g.Trig()));
  LITERT_ASSIGN_OR_RETURN(auto neg_hi,
                          BuildMul(builder, hi.sin(), minus_one, g.Trig()));
  LITERT_ASSIGN_OR_RETURN(
      auto cos_tab,
      BuildConcat(builder, {lo.cos(), lo.cos(), hi.cos(), hi.cos()},
                  g.Table()));
  LITERT_ASSIGN_OR_RETURN(
      auto sin_tab,
      BuildConcat(builder, {neg_lo, lo.sin(), neg_hi, hi.sin()}, g.Table()));
  Tables().push_back({key, cos_tab.Get(), sin_tab.Get()});
  return std::make_pair(cos_tab, sin_tab);
}

}  // namespace

extern "C" {

void ResetRopeTransformationState() { Tables().clear(); }

LiteRtStatus RopeTransformation(const LiteRtCompilerContext* context,
                                LiteRtBuilder builder_ptr, LiteRtOp op) {
  Op root(context, op);

  // Root: last-axis Concat of two halves into one f32 [B, S, H, d], d % 4 == 0.
  auto root_pattern = m_AllOf(
      m_Op<kLiteRtOpCodeTflConcatenation>(m_Any(), m_Any()),
      m_SoleOutput(m_AllOf(
          m_Rank(kRank), m_F32(),
          m_Predicate<Tensor>(
              [](const Tensor& t) { return Dims(t)[kRank - 1] % 4 == 0; },
              "HeadDimDivisibleBy4"))),
      m_LastAxisConcat(), "RopeRoot");
  if (!Match(root, root_pattern)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor out = root.Outputs()[0];
  const auto out_dims = Dims(out);
  const Geometry g{out_dims[0], out_dims[1], out_dims[2], out_dims[3]};

  HalfMatch lo;
  HalfMatch hi;
  if (!Match(root, m_Op<kLiteRtOpCodeTflConcatenation>(m_RopeHalf(&lo, g),
                                                       m_RopeHalf(&hi, g)))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // All four quarters must come from the same input at the right offsets.
  Tensor in;
  if (!Match(lo.a(), m_RopeQuarter(&in, 0, 0, g)) ||
      !Match(lo.b(), m_RopeQuarter(&in, 0, 1, g)) ||
      !Match(hi.a(), m_RopeQuarter(&in, 1, 0, g)) ||
      !Match(hi.b(), m_RopeQuarter(&in, 1, 1, g))) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  auto tables = GetOrBuildTables(context, builder, lo, hi, g);
  if (!tables) return tables.Error().Status();
  auto [cos_tab, sin_tab] = *tables;

  // rotate_half-like permutation: [a_lo, b_lo, a_hi, b_hi] ->
  // [b_lo, a_lo, b_hi, a_hi]; signs live in sin_tab.
  auto in_rot =
      BuildConcat(builder, {lo.b(), lo.a(), hi.b(), hi.a()}, g.Full());
  if (!in_rot) return in_rot.Error().Status();
  auto term1 = BuildMul(builder, in, cos_tab, g.Full());
  if (!term1) return term1.Error().Status();
  auto term2 = BuildMul(builder, *in_rot, sin_tab, g.Full());
  if (!term2) return term2.Error().Status();
  auto add = BuildBinary(builder, kLiteRtOpCodeTflAdd, *term1, *term2, out);
  if (!add) return add.Error().Status();

  // Only erase the root so that the new ops are spliced after all four quarter
  // slices; the rest of the old DAG is removed by OrphanCleanupTransformation.
  builder.EraseOp(root);
  RegisterOrphanGroup(lo.Ops());
  RegisterOrphanGroup(hi.Ops());
  return kLiteRtStatusOk;
}

}  // extern "C"
