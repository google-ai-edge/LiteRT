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

#include "litert/vendors/qualcomm/compiler/transformations/entry_embedding_transformation.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
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
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::FullyConnectedOptions;
using litert::compiler::GatherOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_AnyOf;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_ConstantValue;
using litert::compiler::m_Custom;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_IsConstant;
using litert::compiler::m_Op;
using litert::compiler::m_Options;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::OneHotOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

// Returns the dimensions of a ranked tensor, or an empty vector on failure.
std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) {
    return {};
  }
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

bool HasDims(const Tensor& t, std::initializer_list<int32_t> expected) {
  return Dims(t) == std::vector<int32_t>(expected);
}

bool HasSingleUse(const Tensor& t) { return t.Uses().size() == 1; }

template <typename T>
std::optional<T> ReadScalarConst(const Tensor& t) {
  if (!t.IsConstant()) {
    return std::nullopt;
  }
  auto data = t.WeightsData<T>();
  if (!data || data->size() != 1) {
    return std::nullopt;
  }
  return (*data)[0];
}

// RoPE angle chain driven by the same coordinate:
//   Cast(src [1, S, 1]) -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F]
//   -> {Sin, Cos}.
struct TrigChain {
  Op cast;
  Op mul;
  Op reshape;
  std::optional<Op> sin;
  std::optional<Op> cos;
  std::vector<float> freqs;
};

// One embedding branch: coord -> OneHot [-> SelectV2/Select] -> FC -> Reshape.
struct Branch {
  Op reshape;
  Op fc;
  Op one_hot;
  std::optional<Op> select;
  std::optional<Tensor> select_mask;
  std::optional<Tensor> select_fill;
  Tensor coord;
  Tensor weights;
  // Ops computing `select_mask` when it is exactly
  //   (c < 0 || c >= depth) && c != -1,
  // which allows a bool-free, select-free lookup (see header).
  std::optional<std::vector<Op>> mask_ops;
  std::optional<TrigChain> trig;
};

// Matches OneHot(coord [1, S], depth, on=1, off=0) along the last axis.
std::optional<Op> MatchOneHot(const Tensor& t, int32_t seq_len, int32_t depth) {
  Op one_hot(t.ctx(), nullptr);
  auto matcher = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &one_hot, m_AllOf(m_Options<OneHotOptions>([](const auto& o) {
                              return o.axis == -1 || o.axis == 2;
                            }),
                            m_Op<kLiteRtOpCodeTflOneHot>(
                                m_AllOf(m_ElementType(kLiteRtElementTypeInt32),
                                        m_Shape({1, seq_len})),
                                m_ConstantValue<int32_t>(depth),
                                m_ConstantValue<float>(1.0f),
                                m_ConstantValue<float>(0.0f)))));
  if (!Match(t, matcher)) {
    return std::nullopt;
  }
  return one_hot;
}

// Matches mask [1, S, 1] = Reshape((c < 0 || c >= depth) && c != -1) and
// returns the ops computing it.
std::optional<std::vector<Op>> MatchMask(const Tensor& mask,
                                         const Tensor& coord, int32_t depth) {
  Op reshape(mask.ctx(), nullptr);
  Op and_op(mask.ctx(), nullptr);
  Op or_op(mask.ctx(), nullptr);
  Op ne_op(mask.ctx(), nullptr);
  Op lt_op(mask.ctx(), nullptr);
  Op ge_op(mask.ctx(), nullptr);
  Tensor c = coord;

  auto ne_matcher =
      m_AllOf(m_HasOneUse(),
              m_CaptureOrSameAs(&ne_op, m_Op<kLiteRtOpCodeTflNotEqual>(
                                            m_CaptureOrSameAs(&c, m_Any()),
                                            m_ConstantValue<int32_t>(-1))));
  auto lt_matcher = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &lt_op, m_Op<kLiteRtOpCodeTflLess>(m_CaptureOrSameAs(&c, m_Any()),
                                             m_ConstantValue<int32_t>(0))));
  auto ge_matcher =
      m_AllOf(m_HasOneUse(),
              m_CaptureOrSameAs(&ge_op, m_Op<kLiteRtOpCodeTflGreaterEqual>(
                                            m_CaptureOrSameAs(&c, m_Any()),
                                            m_ConstantValue<int32_t>(depth))));
  auto or_matcher = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(&or_op, m_CommutativeOp<kLiteRtOpCodeTflLogicalOr>(
                                    lt_matcher, ge_matcher)));
  auto and_matcher = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(&and_op, m_CommutativeOp<kLiteRtOpCodeTflLogicalAnd>(
                                     or_matcher, ne_matcher)));
  auto reshape_matcher = m_AllOf(
      m_HasOneUse(), m_CaptureOrSameAs(&reshape, m_Op<kLiteRtOpCodeTflReshape>(
                                                     and_matcher, m_Any())));
  if (!Match(mask, reshape_matcher)) {
    return std::nullopt;
  }
  return std::vector<Op>{reshape, and_op, or_op, ne_op, lt_op, ge_op};
}

// Matches Cast -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F] -> {Sin, Cos}
// starting at `cast`.
std::optional<TrigChain> MatchTrigFromCast(const Op& cast, int32_t seq_len) {
  if (cast.Code() != kLiteRtOpCodeTflCast) return std::nullopt;
  TrigChain tc{cast, cast, cast, std::nullopt, std::nullopt, {}};
  Tensor cast_out = cast.Outputs()[0];
  if (cast_out.ElementType() != ElementType::Float32 ||
      !HasSingleUse(cast_out)) {
    return std::nullopt;
  }
  Op mul = cast_out.Uses()[0].user;
  if (mul.Code() != kLiteRtOpCodeTflMul || mul.Inputs().size() != 2 ||
      !(mul.Inputs()[0] == cast_out)) {
    return std::nullopt;
  }
  auto mul_opts =
      GetOptionsAs<litert::compiler::MulOptions>(mul.ctx(), mul.Get());
  if (!mul_opts || mul_opts->fused_activation_function != 0) {
    return std::nullopt;
  }
  Tensor freq = mul.Inputs()[1];
  auto freq_dims = Dims(freq);
  if (!freq.IsConstant() || freq.ElementType() != ElementType::Float32 ||
      freq_dims.size() != 3 || freq_dims[0] != 1 || freq_dims[1] != 1) {
    return std::nullopt;
  }
  const int32_t f = freq_dims[2];
  auto freq_data = freq.WeightsData<float>();
  if (!freq_data || freq_data->size() != static_cast<size_t>(f)) {
    return std::nullopt;
  }
  Tensor mul_out = mul.Outputs()[0];
  if (!HasSingleUse(mul_out)) return std::nullopt;
  Op reshape = mul_out.Uses()[0].user;
  if (reshape.Code() != kLiteRtOpCodeTflReshape) return std::nullopt;
  Tensor angle = reshape.Outputs()[0];
  if (!HasDims(angle, {1, seq_len, 1, f}) || angle.Uses().empty()) {
    return std::nullopt;
  }
  for (const auto& a_use : angle.Uses()) {
    if (a_use.user.Code() == kLiteRtOpCodeTflSin && !tc.sin) {
      tc.sin = a_use.user;
    } else if (a_use.user.Code() == kLiteRtOpCodeTflCos && !tc.cos) {
      tc.cos = a_use.user;
    } else {
      return std::nullopt;
    }
  }
  tc.mul = mul;
  tc.reshape = reshape;
  tc.freqs.assign(freq_data->begin(), freq_data->end());
  return tc;
}

// Finds Cast(src) -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F] -> {Sin, Cos}
// where src [1, S, 1] is the tensor that `coord` [1, S] is reshaped from.
std::optional<TrigChain> MatchTrig(const Tensor& coord, int32_t seq_len) {
  auto coord_reshape = coord.GetDefiningOp();
  if (!coord_reshape || coord_reshape->Code() != kLiteRtOpCodeTflReshape) {
    return std::nullopt;
  }
  Tensor src = coord_reshape->Inputs()[0];
  if (!HasDims(src, {1, seq_len, 1})) return std::nullopt;
  for (const auto& use : src.Uses()) {
    if (auto tc = MatchTrigFromCast(use.user, seq_len)) return tc;
  }
  return std::nullopt;
}

std::optional<Branch> MatchBranch(const Tensor& reshape_out, int32_t seq_len,
                                  int32_t emb_dim) {
  Branch b;
  Tensor fc_in(reshape_out.ctx(), nullptr);
  auto fc_matcher = m_AllOf(
      m_HasOneUse(), m_ElementType(kLiteRtElementTypeFloat32),
      m_Shape({1, seq_len, emb_dim}),
      m_CaptureOrSameAs(
          &b.fc, m_AllOf(m_Options<FullyConnectedOptions>([](const auto& o) {
                           return o.fused_activation_function == 0 &&
                                  o.weights_format == 0;
                         }),
                         m_Custom([&](const Tensor& t) {
                           auto op = t.GetDefiningOp();
                           if (!op ||
                               op->Code() != kLiteRtOpCodeTflFullyConnected) {
                             return false;
                           }
                           auto ins = op->Inputs();
                           if (ins.size() < 2 ||
                               (ins.size() > 2 && ins[2].Get() != nullptr)) {
                             return false;
                           }
                           fc_in = ins[0];
                           b.weights = ins[1];
                           return true;
                         }))));
  auto reshape_matcher = m_AllOf(
      m_HasOneUse(), m_Shape({1, 1, seq_len, emb_dim}),
      m_CaptureOrSameAs(&b.reshape,
                        m_Op<kLiteRtOpCodeTflReshape>(fc_matcher, m_Any())));
  if (!Match(reshape_out, reshape_matcher)) {
    return std::nullopt;
  }

  auto w_dims = Dims(b.weights);
  if (!b.weights.IsConstant() || b.weights.HasQuantization() ||
      b.weights.ElementType() != ElementType::Float32 || w_dims.size() != 2 ||
      w_dims[0] != emb_dim || w_dims[1] <= 0) {
    return std::nullopt;
  }
  const int32_t depth = w_dims[1];
  auto w_data = b.weights.WeightsData<float>();
  if (!w_data || w_data->size() != static_cast<size_t>(emb_dim) * depth) {
    return std::nullopt;
  }

  if (!Match(fc_in, m_AllOf(m_HasOneUse(), m_Shape({1, seq_len, depth})))) {
    return std::nullopt;
  }
  auto one_hot = MatchOneHot(fc_in, seq_len, depth);
  if (!one_hot) {
    // Optional SelectV2/Select(mask [1, S, 1], fill, one_hot) with fill in
    // {0, NaN}.
    Op select(fc_in.ctx(), nullptr);
    Tensor sel_mask(fc_in.ctx(), nullptr);
    Tensor sel_fill(fc_in.ctx(), nullptr);
    Tensor sel_one_hot(fc_in.ctx(), nullptr);
    auto mask_m = m_CaptureOrSameAs(
        &sel_mask, m_AllOf(m_ElementType(kLiteRtElementTypeBool),
                           m_Shape({1, seq_len, 1})));
    auto fill_m = m_CaptureOrSameAs(
        &sel_fill, m_AllOf(m_IsConstant(), m_Custom([](const Tensor& t) {
                             auto fill = ReadScalarConst<float>(t);
                             return fill &&
                                    (*fill == 0.0f || std::isnan(*fill));
                           })));
    auto oh_m = m_CaptureOrSameAs(&sel_one_hot, m_Any());
    auto select_m = m_CaptureOrSameAs(
        &select, m_AnyOf(m_Op<kLiteRtOpCodeTflSelectV2>(mask_m, fill_m, oh_m),
                         m_Op<kLiteRtOpCodeTflSelect>(mask_m, fill_m, oh_m)));
    if (!Match(fc_in, select_m)) {
      return std::nullopt;
    }
    one_hot = MatchOneHot(sel_one_hot, seq_len, depth);
    if (!one_hot) {
      return std::nullopt;
    }
    b.select = select;
    b.select_mask = sel_mask;
    b.select_fill = sel_fill;
  }
  b.one_hot = *one_hot;
  b.coord = one_hot->Inputs()[0];
  if (b.select) {
    b.mask_ops = MatchMask(*b.select_mask, b.coord, depth);
  }
  b.trig = MatchTrig(b.coord, seq_len);
  return b;
}

Expected<Tensor> BuildScalarConst(Builder& builder, int32_t value) {
  RankedTensorType type(ElementType::Int32, Layout(BuildLayout({})));
  auto t = builder.BuildTensor(RankedTensorSpecBuilder(type).Build());
  if (!t) {
    return t.Error();
  }
  const std::array<int32_t, 1> data = {value};
  auto w = builder.BuildWeights<int32_t>(absl::MakeConstSpan(data), *t);
  if (!w) {
    return w.Error();
  }
  return *t;
}

Expected<Tensor> BuildTensorOfType(Builder& builder, ElementType type,
                                   const std::vector<int32_t>& dims) {
  RankedTensorType t(
      type, Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  return builder.BuildTensor(RankedTensorSpecBuilder(t).Build());
}

// Builds SelectV2(coord >= 0 && coord < depth, coord, depth).
Expected<Tensor> BuildSafeCoord(Builder& builder, const Tensor& coord,
                                int32_t depth) {
  const auto dims = Dims(coord);
  LITERT_ASSIGN_OR_RETURN(auto zero_t, BuildScalarConst(builder, 0));
  LITERT_ASSIGN_OR_RETURN(auto depth_t, BuildScalarConst(builder, depth));
  LITERT_ASSIGN_OR_RETURN(auto ge_t,
                          BuildTensorOfType(builder, ElementType::Bool, dims));
  LITERT_ASSIGN_OR_RETURN(auto lt_t,
                          BuildTensorOfType(builder, ElementType::Bool, dims));
  LITERT_ASSIGN_OR_RETURN(auto ok_t,
                          BuildTensorOfType(builder, ElementType::Bool, dims));
  LITERT_ASSIGN_OR_RETURN(auto safe_t,
                          BuildTensorOfType(builder, ElementType::Int32, dims));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflGreaterEqual, {coord, zero_t}, {ge_t}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflLess, {coord, depth_t}, {lt_t}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflLogicalAnd, {ge_t, lt_t}, {ok_t}));
  LITERT_RETURN_IF_ERROR(builder.BuildOp(kLiteRtOpCodeTflSelectV2,
                                         {ok_t, coord, depth_t}, {safe_t}));
  return safe_t;
}

// Builds Gather(W^T padded with a zero row, safe_coord) -> [1, S, D].
Expected<Tensor> BuildLookup(Builder& builder, const Branch& b, int32_t seq_len,
                             int32_t emb_dim) {
  const int32_t depth = Dims(b.weights)[1];
  auto w = b.weights.WeightsData<float>();
  if (!w) {
    return w.Error();
  }
  const size_t rows = emb_dim;
  const size_t cols = depth;
  // table[c, r] = W[r, c]; table[depth, :] = 0.
  std::vector<float> table((cols + 1) * rows, 0.0f);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      table[c * rows + r] = (*w)[r * cols + c];
    }
  }
  LITERT_ASSIGN_OR_RETURN(
      auto table_t,
      BuildTensorOfType(builder, ElementType::Float32, {depth + 1, emb_dim}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<float>(absl::MakeConstSpan(table), table_t));

  LITERT_ASSIGN_OR_RETURN(auto safe_coord,
                          BuildSafeCoord(builder, b.coord, depth));
  LITERT_ASSIGN_OR_RETURN(
      auto out_t,
      BuildTensorOfType(builder, ElementType::Float32, {1, seq_len, emb_dim}));
  LITERT_ASSIGN_OR_RETURN(
      auto gather,
      builder.BuildOp(kLiteRtOpCodeTflGather, {table_t, safe_coord}, {out_t}));
  GatherOptions opts;
  opts.axis = 0;
  opts.batch_dims = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(gather, std::move(opts)));
  if (!b.select) {
    return out_t;
  }
  // SelectV2(mask [1, S, 1], fill, rows [1, S, D]) -> [1, S, D].
  LITERT_ASSIGN_OR_RETURN(
      auto sel_t,
      BuildTensorOfType(builder, ElementType::Float32, {1, seq_len, emb_dim}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflSelectV2,
                      {*b.select_mask, *b.select_fill, out_t}, {sel_t}));
  return sel_t;
}

Expected<Tensor> BuildInt32Vector(Builder& builder,
                                  const std::vector<int32_t>& values) {
  LITERT_ASSIGN_OR_RETURN(
      auto t, BuildTensorOfType(builder, ElementType::Int32,
                                {static_cast<int32_t>(values.size())}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<int32_t>(absl::MakeConstSpan(values), t));
  return t;
}

bool CanUsePaddedLookup(const Branch& b) {
  return !b.select || b.mask_ops.has_value();
}

bool CanFoldTrig(const Branch& b) {
  if (!b.trig || !b.select || !b.mask_ops) return false;
  auto fill = ReadScalarConst<float>(*b.select_fill);
  return fill && std::isnan(*fill);
}

// idx = Minimum(Maximum(c + 2, 0), depth + 2), same shape as `c`.
Expected<Tensor> BuildClampedIndex(Builder& builder, const Tensor& c,
                                   int32_t depth) {
  const auto dims = Dims(c);
  LITERT_ASSIGN_OR_RETURN(auto two_t, BuildScalarConst(builder, 2));
  LITERT_ASSIGN_OR_RETURN(auto zero_t, BuildScalarConst(builder, 0));
  LITERT_ASSIGN_OR_RETURN(auto hi_t, BuildScalarConst(builder, depth + 2));
  LITERT_ASSIGN_OR_RETURN(auto shifted,
                          BuildTensorOfType(builder, ElementType::Int32, dims));
  LITERT_ASSIGN_OR_RETURN(auto lower,
                          BuildTensorOfType(builder, ElementType::Int32, dims));
  LITERT_ASSIGN_OR_RETURN(auto idx,
                          BuildTensorOfType(builder, ElementType::Int32, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto add, builder.BuildOp(kLiteRtOpCodeTflAdd, {c, two_t}, {shifted}));
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(add, std::move(add_opts)));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflMaximum, {shifted, zero_t}, {lower}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflMinimum, {lower, hi_t}, {idx}));
  return idx;
}

Expected<Tensor> BuildGather(Builder& builder, const Tensor& table,
                             const Tensor& idx,
                             const std::vector<int32_t>& out_dims) {
  LITERT_ASSIGN_OR_RETURN(
      auto out, BuildTensorOfType(builder, ElementType::Float32, out_dims));
  LITERT_ASSIGN_OR_RETURN(auto gather, builder.BuildOp(kLiteRtOpCodeTflGather,
                                                       {table, idx}, {out}));
  GatherOptions opts;
  opts.axis = 0;
  opts.batch_dims = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(gather, std::move(opts)));
  return out;
}

// Bool-free lookup:
//   idx  = Minimum(Maximum(c + 2, 0), depth + 2)
//   rows = Gather(table [depth + 3, D], idx)
Expected<Tensor> BuildPaddedLookup(Builder& builder, const Branch& b,
                                   int32_t seq_len, int32_t emb_dim) {
  const int32_t depth = Dims(b.weights)[1];
  auto w = b.weights.WeightsData<float>();
  if (!w) {
    return w.Error();
  }
  const float fill = b.select ? *ReadScalarConst<float>(*b.select_fill) : 0.0f;
  const size_t cols = emb_dim;
  const size_t rows = depth + 3;
  std::vector<float> table(rows * cols, 0.0f);
  for (size_t d = 0; d < cols; ++d) {
    table[d] = fill;
    table[(rows - 1) * cols + d] = fill;
    for (size_t k = 0; k < static_cast<size_t>(depth); ++k) {
      table[(k + 2) * cols + d] = (*w)[d * depth + k];
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto table_t,
                          BuildTensorOfType(builder, ElementType::Float32,
                                            {static_cast<int32_t>(rows),
                                             static_cast<int32_t>(cols)}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<float>(absl::MakeConstSpan(table), table_t));
  LITERT_ASSIGN_OR_RETURN(auto idx, BuildClampedIndex(builder, b.coord, depth));
  return BuildGather(builder, table_t, idx, {1, seq_len, emb_dim});
}

// Slice(in, begin, size) writing the existing tensor `out`.
Expected<void> BuildSliceInto(Builder& builder, const Tensor& in,
                              const std::vector<int32_t>& begin,
                              const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(auto begin_t, BuildInt32Vector(builder, begin));
  LITERT_ASSIGN_OR_RETURN(auto size_t_, BuildInt32Vector(builder, Dims(out)));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflSlice, {in, begin_t, size_t_}, {out}));
  return {};
}

absl::flat_hash_map<LiteRtOp, int32_t>& TrigFoldRegistry() {
  static auto* registry = new absl::flat_hash_map<LiteRtOp, int32_t>();
  return *registry;
}

}  // namespace

extern "C" {

void ResetEntryEmbeddingTransformationState() { TrigFoldRegistry().clear(); }

LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Op root_op(context, op);
  if (root_op.Code() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto root_outs = root_op.Outputs();
  if (root_outs.size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor sum_out = root_outs[0];
  auto out_dims = Dims(sum_out);
  if (out_dims.size() != 3 || out_dims[0] != 1 ||
      sum_out.ElementType() != ElementType::Float32) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t seq_len = out_dims[1];
  const int32_t emb_dim = out_dims[2];

  // Concat of the two [1, 1, S, D] branches along axis 0 -> [2, 1, S, D],
  // reduced over axis 0 (implied by the shapes).
  Op concat(context, nullptr);
  Tensor branch0(context, nullptr);
  Tensor branch1(context, nullptr);
  auto concat_m = m_AllOf(
      m_HasOneUse(), m_Shape({2, 1, seq_len, emb_dim}),
      m_CaptureOrSameAs(&concat, m_Op<kLiteRtOpCodeTflConcatenation>(
                                     m_CaptureOrSameAs(&branch0, m_Any()),
                                     m_CaptureOrSameAs(&branch1, m_Any()))));
  if (!Match(root_op, m_Op<kLiteRtOpCodeTflSum>(concat_m, m_Any()))) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto bx = MatchBranch(branch0, seq_len, emb_dim);
  auto by = MatchBranch(branch1, seq_len, emb_dim);
  if (!bx || !by) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool padded = CanUsePaddedLookup(*bx) && CanUsePaddedLookup(*by);
  const bool allow_trig_fold = padded;

  Builder builder(context, builder_ptr);
  Tensor outs[2] = {sum_out, sum_out};
  const Branch* branches[2] = {&*bx, &*by};
  for (int i = 0; i < 2; ++i) {
    const Branch& b = *branches[i];
    auto out = padded ? BuildPaddedLookup(builder, b, seq_len, emb_dim)
                      : BuildLookup(builder, b, seq_len, emb_dim);
    if (!out) return out.Error().Status();
    outs[i] = *out;
  }

  auto add =
      builder.BuildOp(kLiteRtOpCodeTflAdd, {outs[0], outs[1]}, {sum_out});
  if (!add) return add.Error().Status();
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  if (auto s = builder.SetOpOptions(*add, std::move(add_opts)); !s) {
    return s.Error().Status();
  }

  builder.EraseOp(root_op);
  builder.EraseOp(concat);
  for (int i = 0; i < 2; ++i) {
    const Branch& b = *branches[i];
    std::vector<LiteRtOp> orphans = {b.reshape.Get(), b.fc.Get(),
                                     b.one_hot.Get()};
    if (b.select) orphans.push_back(b.select->Get());
    if (b.mask_ops && padded) {
      for (const Op& m : *b.mask_ops) orphans.push_back(m.Get());
    }
    RegisterOrphanGroup(orphans);
    if (allow_trig_fold && CanFoldTrig(b)) {
      TrigFoldRegistry()[b.trig->cast.Get()] = Dims(b.weights)[1];
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus EntryEmbeddingTrigFoldTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  auto& registry = TrigFoldRegistry();
  auto it = registry.find(op);
  if (it == registry.end()) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t depth = it->second;
  Op cast(context, op);
  if (cast.Code() != kLiteRtOpCodeTflCast) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor src = cast.Inputs()[0];
  const auto src_dims = Dims(src);
  if (src_dims.size() != 3 || src_dims[0] != 1 || src_dims[2] != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t seq_len = src_dims[1];
  auto tc = MatchTrigFromCast(cast, seq_len);
  if (!tc) {
    return kLiteRtStatusPatternNoMatch;
  }
  registry.erase(it);
  const int32_t f = static_cast<int32_t>(tc->freqs.size());
  const int32_t rows = depth + 3;
  std::vector<float> table(static_cast<size_t>(rows) * 2 * f, 0.0f);
  for (int32_t r = 1; r + 1 < rows; ++r) {
    const float coord = static_cast<float>(r - 2);
    for (int32_t j = 0; j < f; ++j) {
      const float angle = coord * tc->freqs[j];
      table[static_cast<size_t>(r) * 2 * f + j] = std::sin(angle);
      table[static_cast<size_t>(r) * 2 * f + f + j] = std::cos(angle);
    }
  }
  Builder builder(context, builder_ptr);
  auto build = [&]() -> Expected<void> {
    LITERT_ASSIGN_OR_RETURN(
        auto table_t,
        BuildTensorOfType(builder, ElementType::Float32, {rows, 2 * f}));
    LITERT_RETURN_IF_ERROR(
        builder.BuildWeights<float>(absl::MakeConstSpan(table), table_t));
    LITERT_ASSIGN_OR_RETURN(auto idx, BuildClampedIndex(builder, src, depth));
    LITERT_ASSIGN_OR_RETURN(auto rows_t, BuildGather(builder, table_t, idx,
                                                     {1, seq_len, 1, 2 * f}));
    if (tc->sin) {
      LITERT_RETURN_IF_ERROR(
          BuildSliceInto(builder, rows_t, {0, 0, 0, 0}, tc->sin->Outputs()[0]));
    }
    if (tc->cos) {
      LITERT_RETURN_IF_ERROR(
          BuildSliceInto(builder, rows_t, {0, 0, 0, f}, tc->cos->Outputs()[0]));
    }
    return {};
  };
  if (auto s = build(); !s) return s.Error().Status();
  builder.EraseOp(cast);
  builder.EraseOp(tc->mul);
  builder.EraseOp(tc->reshape);
  if (tc->sin) builder.EraseOp(*tc->sin);
  if (tc->cos) builder.EraseOp(*tc->cos);
  return kLiteRtStatusOk;
}

}  // extern "C"
