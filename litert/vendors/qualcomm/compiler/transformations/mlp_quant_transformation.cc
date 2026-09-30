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

#include "litert/vendors/qualcomm/compiler/transformations/mlp_quant_transformation.h"

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

namespace {

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::CompositeOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_CompositeOp;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_Op;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::RmsNormOpts;
using litert::compiler::Tensor;

constexpr int32_t kMinSequenceFor4DFolding = 600;

std::optional<std::pair<int32_t, int32_t>> BestCroutonSpatialDims(int32_t s) {
  auto ceil8 = [](int32_t x) { return (x + 7) / 8; };
  int32_t best_tiles = ceil8(1) * ceil8(s);
  std::optional<std::pair<int32_t, int32_t>> best;
  for (int32_t h_s = 2; h_s * h_s <= s; ++h_s) {
    if (s % h_s != 0) continue;
    const int32_t w_s = s / h_s;
    const int32_t tiles = ceil8(h_s) * ceil8(w_s);
    if (tiles < best_tiles) {
      best_tiles = tiles;
      best = std::make_pair(h_s, w_s);
    }
  }
  return best;
}

bool HasShape(const Tensor& t, const std::vector<int32_t>& expected) {
  auto rtt = t.RankedTensorType();
  if (!rtt) return false;
  auto dims = rtt->Layout().Dimensions();
  if (dims.size() != expected.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (dims[i] != expected[i]) return false;
  }
  return true;
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         const std::vector<int32_t>& dims) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  auto spec = RankedTensorSpecBuilder(type);
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec =
        std::move(spec).WithPerTensorQuantization(like.PerTensorQuantization());
  }
  return builder.BuildTensor(std::move(spec).Build());
}

Expected<Tensor> ConstI32(Builder& builder, const std::vector<int32_t>& data) {
  const std::vector<int32_t> dims = {static_cast<int32_t>(data.size())};
  RankedTensorType type(ElementType::Int32,
                        Layout(BuildLayout(dims.data(), dims.data() + 1)));
  LITERT_ASSIGN_OR_RETURN(
      auto t, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int32_t>(absl::MakeConstSpan(data), t));
  return t;
}

Expected<void> BuildReshapeInto(Builder& builder, const Tensor& in,
                                const std::vector<int32_t>& dims,
                                const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape = dims;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return {};
}

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& in,
                              const Tensor& like,
                              const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, in, dims, out));
  return out;
}

// Peeks through an existing 4D->3D Reshape if its source tensor already has
// `dims_4d`, avoiding an extra Reshape pair and keeping `splice_index` local
// while `ApplyChanges` DCE cleans up the dead 4D->3D Reshape.
Expected<Tensor> As4D(Builder& builder, const Tensor& t_3d,
                      const std::vector<int32_t>& dims_4d) {
  auto def = t_3d.GetDefiningOp();
  if (def && def->Code() == kLiteRtOpCodeTflReshape && !def->Inputs().empty()) {
    Tensor src = def->Inputs()[0];
    if (HasShape(src, dims_4d) && src.ElementType() == t_3d.ElementType()) {
      return src;
    }
  }
  return BuildReshape(builder, t_3d, t_3d, dims_4d);
}

Expected<Op> CloneRmsNorm4D(const LiteRtCompilerContext* context,
                            Builder& builder, const Op& old_norm,
                            const Tensor& in_4d, const Tensor& out_4d) {
  RmsNormOpts opts;
  LITERT_RETURN_IF_ERROR(opts.InitFromOp(context, old_norm.Get()));
  Tensor gamma = old_norm.Inputs()[1];
  LITERT_ASSIGN_OR_RETURN(
      auto new_norm,
      builder.BuildOp(kLiteRtOpCodeShloComposite, {in_4d, gamma}, {out_4d}));
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_norm, std::move(opts)));
  return new_norm;
}

Expected<Op> CloneAdd4D(const LiteRtCompilerContext* context, Builder& builder,
                        const Op& old_add, const Tensor& lhs_4d,
                        const Tensor& rhs_4d, const Tensor& out_4d) {
  AddOptions opts;
  LITERT_RETURN_IF_ERROR(opts.InitFromOp(context, old_add.Get()));
  LITERT_ASSIGN_OR_RETURN(
      auto new_add,
      builder.BuildOp(kLiteRtOpCodeTflAdd, {lhs_4d, rhs_4d}, {out_4d}));
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(new_add, std::move(opts)));
  return new_add;
}

// Pattern B1: Transformer Residual-Norm trunk block:
//   fc_out (INT8 [1, S, D]) -> quant_post (INT16) -> norm_post (RmsNorm INT16)
//   -> add_op(residual_in, norm_post_out) -> add_out (INT16 [1, S, D])
//   (-> optional: norm_pre (RmsNorm INT16) -> quant_pre (INT8 [1, S, D]))
// Rewrites all matched ops to 4D `[1, H_s, W_s, D]` so QNN HTP tiles `8x8`
// Crouton spatial tiles across `(H_s, W_s)` instead of `(1, S)`.
LiteRtStatus TryFoldResidualNormBlock4D(const LiteRtCompilerContext* context,
                                        Builder& builder, const Op& add_op) {
  if (add_op.Code() != kLiteRtOpCodeTflAdd || add_op.Inputs().size() != 2 ||
      add_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor add_out = add_op.Outputs()[0];
  if (add_out.ElementType() != ElementType::Int16) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto add_type = add_out.RankedTensorType();
  if (!add_type) return kLiteRtStatusPatternNoMatch;
  auto dims = add_type->Layout().Dimensions();
  if (dims.size() != 3 || dims[0] != 1 || dims[1] < kMinSequenceFor4DFolding) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t s = dims[1];
  const int32_t d = dims[2];
  const auto spatial = BestCroutonSpatialDims(s);
  if (!spatial) return kLiteRtStatusPatternNoMatch;
  const std::vector<int32_t> dims_3d = {1, s, d};
  const std::vector<int32_t> dims_4d = {1, spatial->first, spatial->second, d};

  Op norm_post(context, nullptr);
  Op quant_post(context, nullptr);
  Tensor fc_out(context, nullptr);
  Tensor residual_in(context, nullptr);

  auto post_branch_m = m_AllOf(
      m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16), m_Shape(dims_3d),
      m_CaptureOrSameAs(
          &norm_post,
          m_CompositeOp(
              CompositeOptions::kRmsNorm,
              m_AllOf(m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
                      m_Shape(dims_3d),
                      m_CaptureOrSameAs(
                          &quant_post,
                          m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                              &fc_out,
                              m_AllOf(m_ElementType(kLiteRtElementTypeInt8),
                                      m_Shape(dims_3d)))))),
              m_Any())));
  auto res_branch_m = m_CaptureOrSameAs(
      &residual_in,
      m_AllOf(m_ElementType(kLiteRtElementTypeInt16), m_Shape(dims_3d)));
  if (!Match(add_op, m_CommutativeOp<kLiteRtOpCodeTflAdd>(post_branch_m,
                                                          res_branch_m))) {
    return kLiteRtStatusPatternNoMatch;
  }
  const bool in0_is_post = (add_op.Inputs()[0] == norm_post.Outputs()[0]);

  // Check optional pre-norm + quantize immediately consuming `add_out`.
  bool has_pre_norm = false;
  Op norm_pre(context, nullptr);
  Op quant_pre(context, nullptr);
  Tensor norm_pre_out(context, nullptr);
  Tensor quant_pre_out(context, nullptr);
  for (const auto& u : add_out.Uses()) {
    if (u.user_arg_ind == 0 && u.user.Outputs().size() == 1) {
      Tensor np_out = u.user.Outputs()[0];
      if (np_out.Uses().size() == 1) {
        Op qp = np_out.Uses()[0].user;
        auto pre_m = m_AllOf(
            m_ElementType(kLiteRtElementTypeInt8), m_Shape(dims_3d),
            m_CaptureOrSameAs(
                &quant_pre,
                m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                    &norm_pre_out,
                    m_AllOf(
                        m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
                        m_Shape(dims_3d),
                        m_CaptureOrSameAs(
                            &norm_pre, m_CompositeOp(CompositeOptions::kRmsNorm,
                                                     m_Any(), m_Any())))))));
        if (qp.Outputs().size() == 1 && Match(qp.Outputs()[0], pre_m)) {
          has_pre_norm = true;
          quant_pre_out = qp.Outputs()[0];
          break;
        }
      }
    }
  }

  auto fc_4d = As4D(builder, fc_out, dims_4d);
  if (!fc_4d) return fc_4d.Error().Status();
  auto res_4d = As4D(builder, residual_in, dims_4d);
  if (!res_4d) return res_4d.Error().Status();

  auto quant_post_4d = NewLike(builder, quant_post.Outputs()[0], dims_4d);
  if (!quant_post_4d) return quant_post_4d.Error().Status();
  auto new_quant_post =
      builder.BuildOp(kLiteRtOpCodeTflQuantize, {*fc_4d}, {*quant_post_4d});
  if (!new_quant_post) return new_quant_post.Error().Status();

  auto norm_post_4d = NewLike(builder, norm_post.Outputs()[0], dims_4d);
  if (!norm_post_4d) return norm_post_4d.Error().Status();
  auto new_norm_post = CloneRmsNorm4D(context, builder, norm_post,
                                      *quant_post_4d, *norm_post_4d);
  if (!new_norm_post) return new_norm_post.Error().Status();

  auto add_4d = NewLike(builder, add_out, dims_4d);
  if (!add_4d) return add_4d.Error().Status();
  const Tensor& lhs_4d = in0_is_post ? *norm_post_4d : *res_4d;
  const Tensor& rhs_4d = in0_is_post ? *res_4d : *norm_post_4d;
  auto new_add = CloneAdd4D(context, builder, add_op, lhs_4d, rhs_4d, *add_4d);
  if (!new_add) return new_add.Error().Status();

  auto reshape_add = BuildReshapeInto(builder, *add_4d, dims_3d, add_out);
  if (!reshape_add) return reshape_add.Error().Status();

  if (has_pre_norm) {
    auto norm_pre_4d = NewLike(builder, norm_pre_out, dims_4d);
    if (!norm_pre_4d) return norm_pre_4d.Error().Status();
    auto new_norm_pre =
        CloneRmsNorm4D(context, builder, norm_pre, *add_4d, *norm_pre_4d);
    if (!new_norm_pre) return new_norm_pre.Error().Status();

    auto quant_pre_4d = NewLike(builder, quant_pre_out, dims_4d);
    if (!quant_pre_4d) return quant_pre_4d.Error().Status();
    auto new_quant_pre = builder.BuildOp(kLiteRtOpCodeTflQuantize,
                                         {*norm_pre_4d}, {*quant_pre_4d});
    if (!new_quant_pre) return new_quant_pre.Error().Status();

    auto reshape_qpre =
        BuildReshapeInto(builder, *quant_pre_4d, dims_3d, quant_pre_out);
    if (!reshape_qpre) return reshape_qpre.Error().Status();

    builder.EraseOp(norm_pre);
    builder.EraseOp(quant_pre);
  }

  builder.EraseOp(quant_post);
  builder.EraseOp(norm_post);
  builder.EraseOp(add_op);
  return kLiteRtStatusOk;
}

// Pattern B2: Entry Add + Quantize + Layer0 Pre-Attention RmsNorm + Quantize:
//   (optional: reduce_sum1 Add(g0, g1) -> pos_emb (FP32 [1, S, D]))
//   -> entry/add(proj_out, pos_emb) -> entry_fp32 (FP32 [1, S, D])
//   -> entry/add_quantized (Quantize -> entry_int16 INT16 [1, S, D])
//   -> layer_0/pre_attention_norm (RmsNorm -> norm0_out INT16 [1, S, D])
//   -> layer_0/q_einsum/Convert (Quantize -> quant0_out INT8 [1, S, D])
LiteRtStatus TryFoldEntryAddNormBlock4D(const LiteRtCompilerContext* context,
                                        Builder& builder, const Op& add_op) {
  if (add_op.Code() != kLiteRtOpCodeTflAdd || add_op.Inputs().size() != 2 ||
      add_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor entry_fp32 = add_op.Outputs()[0];
  if (entry_fp32.ElementType() != ElementType::Float32 ||
      entry_fp32.Uses().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto entry_type = entry_fp32.RankedTensorType();
  if (!entry_type) return kLiteRtStatusPatternNoMatch;
  auto dims = entry_type->Layout().Dimensions();
  if (dims.size() != 3 || dims[0] != 1 || dims[1] < kMinSequenceFor4DFolding) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t s = dims[1];
  const int32_t d = dims[2];
  const auto spatial = BestCroutonSpatialDims(s);
  if (!spatial) return kLiteRtStatusPatternNoMatch;
  const std::vector<int32_t> dims_3d = {1, s, d};
  const std::vector<int32_t> dims_4d = {1, spatial->first, spatial->second, d};

  auto fp32_3d_m =
      m_AllOf(m_ElementType(kLiteRtElementTypeFloat32), m_Shape(dims_3d));
  if (!Match(add_op, m_Op<kLiteRtOpCodeTflAdd>(fp32_3d_m, fp32_3d_m))) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor in0 = add_op.Inputs()[0];
  Tensor in1 = add_op.Inputs()[1];

  Op quant_entry = entry_fp32.Uses()[0].user;
  if (quant_entry.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor entry_int16 = quant_entry.Outputs()[0];
  if (!Match(entry_int16,
             m_AllOf(m_ElementType(kLiteRtElementTypeInt16), m_Shape(dims_3d),
                     m_Op<kLiteRtOpCodeTflQuantize>(m_Any())))) {
    return kLiteRtStatusPatternNoMatch;
  }

  bool found_norm0 = false;
  Op norm0(context, nullptr);
  Op quant0(context, nullptr);
  Tensor norm0_out(context, nullptr);
  Tensor quant0_out(context, nullptr);
  for (const auto& u : entry_int16.Uses()) {
    if (u.user_arg_ind == 0 && u.user.Outputs().size() == 1) {
      Tensor n_out = u.user.Outputs()[0];
      if (n_out.Uses().size() == 1) {
        Op q0 = n_out.Uses()[0].user;
        auto norm0_m = m_AllOf(
            m_ElementType(kLiteRtElementTypeInt8), m_Shape(dims_3d),
            m_CaptureOrSameAs(
                &quant0,
                m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                    &norm0_out,
                    m_AllOf(
                        m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
                        m_Shape(dims_3d),
                        m_CaptureOrSameAs(
                            &norm0, m_CompositeOp(CompositeOptions::kRmsNorm,
                                                  m_Any(), m_Any())))))));
        if (q0.Outputs().size() == 1 && Match(q0.Outputs()[0], norm0_m)) {
          found_norm0 = true;
          quant0_out = q0.Outputs()[0];
          break;
        }
      }
    }
  }
  if (!found_norm0) return kLiteRtStatusPatternNoMatch;

  // Check if either input of `add_op` is `reduce_sum1` (`Add` of two 3D FP32).
  Op prev_add(context, nullptr);
  auto prev_add_m = m_AllOf(
      m_HasOneUse(), m_CaptureOrSameAs(&prev_add, m_Op<kLiteRtOpCodeTflAdd>(
                                                      fp32_3d_m, fp32_3d_m)));
  bool in1_is_prev_add = Match(in1, prev_add_m);
  bool in0_is_prev_add = !in1_is_prev_add && Match(in0, prev_add_m);

  Tensor lhs_4d(context, nullptr);
  Tensor rhs_4d(context, nullptr);
  if (in0_is_prev_add || in1_is_prev_add) {
    auto g0_4d = As4D(builder, prev_add.Inputs()[0], dims_4d);
    if (!g0_4d) return g0_4d.Error().Status();
    auto g1_4d = As4D(builder, prev_add.Inputs()[1], dims_4d);
    if (!g1_4d) return g1_4d.Error().Status();
    auto prev_4d = NewLike(builder, prev_add.Outputs()[0], dims_4d);
    if (!prev_4d) return prev_4d.Error().Status();
    auto new_prev_add =
        CloneAdd4D(context, builder, prev_add, *g0_4d, *g1_4d, *prev_4d);
    if (!new_prev_add) return new_prev_add.Error().Status();
    if (in1_is_prev_add) {
      auto other_4d = As4D(builder, in0, dims_4d);
      if (!other_4d) return other_4d.Error().Status();
      lhs_4d = *other_4d;
      rhs_4d = *prev_4d;
    } else {
      auto other_4d = As4D(builder, in1, dims_4d);
      if (!other_4d) return other_4d.Error().Status();
      lhs_4d = *prev_4d;
      rhs_4d = *other_4d;
    }
    builder.EraseOp(prev_add);
  } else {
    auto l = As4D(builder, in0, dims_4d);
    if (!l) return l.Error().Status();
    auto r = As4D(builder, in1, dims_4d);
    if (!r) return r.Error().Status();
    lhs_4d = *l;
    rhs_4d = *r;
  }

  auto entry_fp32_4d = NewLike(builder, entry_fp32, dims_4d);
  if (!entry_fp32_4d) return entry_fp32_4d.Error().Status();
  auto new_add =
      CloneAdd4D(context, builder, add_op, lhs_4d, rhs_4d, *entry_fp32_4d);
  if (!new_add) return new_add.Error().Status();

  auto entry_int16_4d = NewLike(builder, entry_int16, dims_4d);
  if (!entry_int16_4d) return entry_int16_4d.Error().Status();
  auto new_q_entry = builder.BuildOp(kLiteRtOpCodeTflQuantize, {*entry_fp32_4d},
                                     {*entry_int16_4d});
  if (!new_q_entry) return new_q_entry.Error().Status();

  auto reshape_entry =
      BuildReshapeInto(builder, *entry_int16_4d, dims_3d, entry_int16);
  if (!reshape_entry) return reshape_entry.Error().Status();

  auto norm0_4d = NewLike(builder, norm0_out, dims_4d);
  if (!norm0_4d) return norm0_4d.Error().Status();
  auto new_norm0 =
      CloneRmsNorm4D(context, builder, norm0, *entry_int16_4d, *norm0_4d);
  if (!new_norm0) return new_norm0.Error().Status();

  auto quant0_4d = NewLike(builder, quant0_out, dims_4d);
  if (!quant0_4d) return quant0_4d.Error().Status();
  auto new_q0 =
      builder.BuildOp(kLiteRtOpCodeTflQuantize, {*norm0_4d}, {*quant0_4d});
  if (!new_q0) return new_q0.Error().Status();

  auto reshape_q0 = BuildReshapeInto(builder, *quant0_4d, dims_3d, quant0_out);
  if (!reshape_q0) return reshape_q0.Error().Status();

  builder.EraseOp(add_op);
  builder.EraseOp(quant_entry);
  builder.EraseOp(norm0);
  builder.EraseOp(quant0);
  return kLiteRtStatusOk;
}

}  // namespace

extern "C" {

LiteRtStatus MLPInt8QuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr,
                                        LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  if (root_op.Code() == kLiteRtOpCodeTflAdd) {
    LiteRtStatus status = TryFoldResidualNormBlock4D(context, builder, root_op);
    if (status != kLiteRtStatusPatternNoMatch) return status;
    return TryFoldEntryAddNormBlock4D(context, builder, root_op);
  }

  Tensor fc1_out(context, nullptr);
  Tensor fc2_out(context, nullptr);
  Op gelu_op(context, nullptr);
  Op quant1_op(context, nullptr);
  Op quant2_op(context, nullptr);

  auto quant1_branch = m_AllOf(
      m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
      m_CaptureOrSameAs(&quant1_op,
                        m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                            &fc1_out, m_ElementType(kLiteRtElementTypeInt8)))));
  auto gelu_branch = m_AllOf(
      m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
      m_CaptureOrSameAs(&gelu_op, m_Op<kLiteRtOpCodeTflGelu>(quant1_branch)));

  auto quant2_branch = m_AllOf(
      m_HasOneUse(), m_ElementType(kLiteRtElementTypeInt16),
      m_CaptureOrSameAs(&quant2_op,
                        m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                            &fc2_out, m_ElementType(kLiteRtElementTypeInt8)))));

  if (!Match(root_op, m_CommutativeOp<kLiteRtOpCodeTflMul>(gelu_branch,
                                                           quant2_branch))) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto fc1_type = fc1_out.RankedTensorType();
  auto fc2_type = fc2_out.RankedTensorType();
  if (!fc1_type || !fc2_type ||
      fc1_type->Layout().Dimensions() != fc2_type->Layout().Dimensions()) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor mul_out = root_op.Outputs()[0];
  if (mul_out.ElementType() != ElementType::Int16) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto uses = mul_out.Uses();
  if (uses.size() != 1 || uses[0].user.Code() != kLiteRtOpCodeTflQuantize) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op quant3_op = uses[0].user;
  if (quant3_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor final_mul_out = quant3_op.Outputs()[0];
  if (final_mul_out.ElementType() != ElementType::Int8) {
    return kLiteRtStatusPatternNoMatch;
  }

  const auto dims = fc1_type->Layout().Dimensions();
  if (dims.size() == 3 && dims[0] == 1 && dims[1] >= kMinSequenceFor4DFolding) {
    if (const auto spatial = BestCroutonSpatialDims(dims[1])) {
      const int32_t s = dims[1];
      const int32_t f = dims[2];
      const std::vector<int32_t> dims_4d = {1, spatial->first, spatial->second,
                                            f};
      auto fc1_4d = As4D(builder, fc1_out, dims_4d);
      if (!fc1_4d) return fc1_4d.Error().Status();
      auto fc2_4d = As4D(builder, fc2_out, dims_4d);
      if (!fc2_4d) return fc2_4d.Error().Status();
      auto gelu_4d = NewLike(builder, fc1_out, dims_4d);
      if (!gelu_4d) return gelu_4d.Error().Status();
      auto new_gelu =
          builder.BuildOp(kLiteRtOpCodeTflGelu, {*fc1_4d}, {*gelu_4d});
      if (!new_gelu) return new_gelu.Error().Status();

      auto mul_4d = NewLike(builder, final_mul_out, dims_4d);
      if (!mul_4d) return mul_4d.Error().Status();
      auto new_mul =
          builder.BuildOp(kLiteRtOpCodeTflMul, {*gelu_4d, *fc2_4d}, {*mul_4d});
      if (!new_mul) return new_mul.Error().Status();
      MulOptions mul_options;
      mul_options.fused_activation_function = 0;
      auto opt_status = builder.SetOpOptions(*new_mul, std::move(mul_options));
      if (!opt_status) return opt_status.Error().Status();

      auto reshape_out =
          BuildReshapeInto(builder, *mul_4d, {1, s, f}, final_mul_out);
      if (!reshape_out) return reshape_out.Error().Status();

      builder.EraseOp(root_op);
      builder.EraseOp(gelu_op);
      builder.EraseOp(quant1_op);
      builder.EraseOp(quant2_op);
      builder.EraseOp(quant3_op);
      return kLiteRtStatusOk;
    }
  }

  // 6. Build new INT8 gelu_out tensor
  RankedTensorType gelu_int8_type = *fc1_type;
  auto new_gelu_out_spec = RankedTensorSpecBuilder(gelu_int8_type);
  if (fc1_out.QTypeId() == kLiteRtQuantizationPerTensor) {
    new_gelu_out_spec =
        std::move(new_gelu_out_spec)
            .WithPerTensorQuantization(fc1_out.PerTensorQuantization());
  }
  auto new_gelu_out_res =
      builder.BuildTensor(std::move(new_gelu_out_spec).Build());
  if (!new_gelu_out_res) {
    return new_gelu_out_res.Error().Status();
  }
  Tensor gelu_int8_out = *new_gelu_out_res;

  // 7. Build new INT8 GELU op
  auto new_gelu =
      builder.BuildOp(kLiteRtOpCodeTflGelu, {fc1_out}, {gelu_int8_out});
  if (!new_gelu) {
    return new_gelu.Error().Status();
  }

  // 8. Build new INT8 MUL op writing directly to final_mul_out
  auto new_mul = builder.BuildOp(kLiteRtOpCodeTflMul, {gelu_int8_out, fc2_out},
                                 {final_mul_out});
  if (!new_mul) {
    return new_mul.Error().Status();
  }

  MulOptions mul_options;
  mul_options.fused_activation_function = 0;
  auto opt_status = builder.SetOpOptions(*new_mul, std::move(mul_options));
  if (!opt_status) {
    return opt_status.Error().Status();
  }

  // 9. Clean up obsolete ops
  builder.EraseOp(root_op);
  builder.EraseOp(gelu_op);
  builder.EraseOp(quant1_op);
  builder.EraseOp(quant2_op);
  builder.EraseOp(quant3_op);

  return kLiteRtStatusOk;
}

}  // extern "C"
