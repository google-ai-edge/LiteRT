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

#include "litert/vendors/mediatek/compiler/transformations/entry_embedding_transformation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::AddOptions;
using litert::compiler::Builder;
using litert::compiler::FullyConnectedOptions;
using litert::compiler::GatherOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_AnyOf;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_HasUsers;
using litert::compiler::m_IsConstant;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Not;
using litert::compiler::m_Op;
using litert::compiler::m_OpCode;
using litert::compiler::m_Options;
using litert::compiler::m_OpVariadic;
using litert::compiler::m_Predicate;
using litert::compiler::m_Rank;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::OneHotOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

namespace {

// Returns the dimensions of a ranked tensor, or an empty vector on failure.
std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) {
    return {};
  }
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

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

// --- Local matchers ---

// Ranked tensor whose dimensions are exactly `dims`. Unlike m_Shape, -1 is
// not treated as a wildcard: the expected dims are read from the (possibly
// dynamic) graph, so dynamic dimensions must match literally.
auto m_ExactShape(std::vector<int32_t> dims) {
  return m_Predicate<Tensor>(
      [dims = std::move(dims)](const Tensor& t) { return Dims(t) == dims; },
      "ExactShape");
}

// Constant tensor with exactly one element, equal to `value`. Unlike
// m_ConstantValue, splats with more than one element are rejected.
template <typename T>
auto m_ScalarConst(T value) {
  return m_Predicate<Tensor>(
      [value](const Tensor& t) {
        auto v = ReadScalarConst<T>(t);
        return v && *v == value;
      },
      "ScalarConst");
}

// Tensor whose defining op matches the Op matcher `m`. Adapts matchers that
// have no Tensor overload (e.g. m_OpVariadic).
template <typename M>
auto m_DefinedBy(M m) {
  return m_Predicate<Tensor>(
      [m = std::move(m)](const Tensor& t) {
        auto def = t.GetDefiningOp();
        return def && Match(*def, m);
      },
      "DefinedBy");
}

// Single-use tensor matching `m` (which must match the tensor's defining
// op); the defining op is captured into `*op`.
template <typename M>
auto m_SingleUseDef(Op* op, M m) {
  return m_CaptureOrSameAs(op, m_AllOf(m_HasOneUse(), std::move(m)));
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

// One embedding branch: coord -> OneHot [-> SelectV2] -> FC -> Reshape.
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

// Matches mask [1, S, 1] = Reshape((c < 0 || c >= depth) && c != -1) and
// returns the ops computing it.
std::optional<std::vector<Op>> MatchMask(const Tensor& mask,
                                         const Tensor& coord, int32_t depth) {
  const LiteRtCompilerContext* ctx = mask.Context();
  Op reshape(ctx, nullptr);
  Op and_op(ctx, nullptr);
  Op or_op(ctx, nullptr);
  Op ne_op(ctx, nullptr);
  Op lt_op(ctx, nullptr);
  Op ge_op(ctx, nullptr);
  // Pre-populated storage: only matches `coord` itself.
  Tensor coord_ref = coord;
  auto is_coord = m_CaptureOrSameAs(&coord_ref, m_Any());

  auto pattern = m_SingleUseDef(
      &reshape,
      m_DefinedBy(m_OpVariadic<kLiteRtOpCodeTflReshape>(m_SingleUseDef(
          &and_op,
          m_CommutativeOp<kLiteRtOpCodeTflLogicalAnd>(
              m_SingleUseDef(
                  &or_op,
                  m_CommutativeOp<kLiteRtOpCodeTflLogicalOr>(
                      m_SingleUseDef(&lt_op,
                                     m_Op<kLiteRtOpCodeTflLess>(
                                         is_coord, m_ScalarConst<int32_t>(0))),
                      m_SingleUseDef(
                          &ge_op,
                          m_Op<kLiteRtOpCodeTflGreaterEqual>(
                              is_coord, m_ScalarConst<int32_t>(depth))))),
              m_SingleUseDef(&ne_op,
                             m_Op<kLiteRtOpCodeTflNotEqual>(
                                 is_coord, m_ScalarConst<int32_t>(-1))))))));
  if (!Match(mask, pattern)) {
    return std::nullopt;
  }
  return std::vector<Op>{reshape, and_op, or_op, ne_op, lt_op, ge_op};
}

// Matches Cast -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F] -> {Sin, Cos}
// starting at `cast`. The chain is walked downstream (through users), so
// each op is located imperatively and then checked with a matcher.
std::optional<TrigChain> MatchTrigFromCast(const Op& cast, int32_t seq_len) {
  if (!Match(cast, m_OpCode<kLiteRtOpCodeTflCast>())) return std::nullopt;
  Tensor cast_out = cast.Outputs()[0];
  // Tensor::ElementType() (unlike m_ElementType) also accepts unranked
  // tensors.
  auto is_f32 = m_Predicate<Tensor>(
      [](const Tensor& t) { return t.ElementType() == ElementType::Float32; },
      "IsF32");
  if (!Match(cast_out, m_AllOf(is_f32, m_HasOneUse()))) {
    return std::nullopt;
  }

  Op mul = cast_out.Uses()[0].user;
  Tensor freq(cast.Context(), nullptr);
  auto freq_m = m_AllOf(
      m_IsConstant(), m_ElementType(kLiteRtElementTypeFloat32),
      m_Shape({1, 1, -1}),
      m_Predicate<Tensor>(
          [](const Tensor& t) {
            auto data = t.WeightsData<float>();
            return data && data->size() == static_cast<size_t>(Dims(t)[2]);
          },
          "FreqData"));
  // Pre-populated storage: only matches `cast_out` itself.
  Tensor cast_out_ref = cast_out;
  auto mul_pattern = m_AllOf(
      m_Op<kLiteRtOpCodeTflMul>(m_CaptureOrSameAs(&cast_out_ref, m_Any()),
                                m_CaptureOrSameAs(&freq, freq_m)),
      m_Options<MulOptions>([](const MulOptions& opts) {
        return opts.fused_activation_function == 0;
      }));
  if (!Match(mul, mul_pattern)) return std::nullopt;
  const int32_t f = Dims(freq)[2];

  Tensor mul_out = mul.Outputs()[0];
  if (!Match(mul_out, m_HasOneUse())) return std::nullopt;
  Op reshape = mul_out.Uses()[0].user;
  if (!Match(reshape, m_OpCode<kLiteRtOpCodeTflReshape>())) {
    return std::nullopt;
  }
  Tensor angle = reshape.Outputs()[0];
  if (!Match(angle,
             m_AllOf(m_ExactShape({1, seq_len, 1, f}), m_Not(m_HasUsers(0))))) {
    return std::nullopt;
  }
  TrigChain tc{cast, mul, reshape, std::nullopt, std::nullopt, {}};
  for (const auto& a_use : angle.Uses()) {
    if (a_use.user.Code() == kLiteRtOpCodeTflSin && !tc.sin) {
      tc.sin = a_use.user;
    } else if (a_use.user.Code() == kLiteRtOpCodeTflCos && !tc.cos) {
      tc.cos = a_use.user;
    } else {
      return std::nullopt;
    }
  }
  auto freq_data = freq.WeightsData<float>();  // Validated by `freq_m`.
  tc.freqs.assign(freq_data->begin(), freq_data->end());
  return tc;
}

// Finds Cast(src) -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F] -> {Sin, Cos}
// where src [1, S, 1] is the tensor that `coord` [1, S] is reshaped from.
std::optional<TrigChain> MatchTrig(const Tensor& coord, int32_t seq_len) {
  Tensor src(coord.Context(), nullptr);
  auto pattern = m_DefinedBy(m_OpVariadic<kLiteRtOpCodeTflReshape>(
      m_CaptureOrSameAs(&src, m_ExactShape({1, seq_len, 1}))));
  if (!Match(coord, pattern)) return std::nullopt;
  for (const auto& use : src.Uses()) {
    if (auto tc = MatchTrigFromCast(use.user, seq_len)) return tc;
  }
  return std::nullopt;
}

std::optional<Branch> MatchBranch(const Tensor& reshape_out, int32_t seq_len,
                                  int32_t emb_dim) {
  const LiteRtCompilerContext* ctx = reshape_out.Context();

  // Reshape [1, 1, S, D] <- FC [1, S, D] (f32 weights [D, depth], no bias).
  Op reshape(ctx, nullptr);
  Op fc(ctx, nullptr);
  Tensor fc_in(ctx, nullptr);
  Tensor weights(ctx, nullptr);
  auto weights_m =
      m_AllOf(m_IsConstant(), m_Not(m_IsQuantized()),
              m_ElementType(kLiteRtElementTypeFloat32), m_Rank(2),
              m_Predicate<Tensor>(
                  [emb_dim](const Tensor& t) { return Dims(t)[0] == emb_dim; },
                  "WeightsRows"));
  // Bias must be absent (optional input is either missing or null).
  auto no_bias = m_Predicate<Op>(
      [](const Op& op) {
        auto ins = op.Inputs();
        return ins.size() <= 2 || ins[2].Get() == nullptr;
      },
      "NoBias");
  auto fc_options =
      m_Options<FullyConnectedOptions>([](const FullyConnectedOptions& opts) {
        return opts.fused_activation_function == 0 && opts.weights_format == 0;
      });
  auto fc_m = m_SingleUseDef(
      &fc,
      m_AllOf(m_ExactShape({1, seq_len, emb_dim}),
              m_ElementType(kLiteRtElementTypeFloat32),
              m_DefinedBy(m_AllOf(m_OpVariadic<kLiteRtOpCodeTflFullyConnected>(
                                      m_CaptureOrSameAs(&fc_in, m_Any()),
                                      m_CaptureOrSameAs(&weights, weights_m)),
                                  no_bias, fc_options))));
  auto head = m_SingleUseDef(
      &reshape,
      m_AllOf(m_ExactShape({1, 1, seq_len, emb_dim}),
              m_DefinedBy(m_OpVariadic<kLiteRtOpCodeTflReshape>(fc_m))));
  if (!Match(reshape_out, head)) {
    return std::nullopt;
  }
  const int32_t depth = Dims(weights)[1];

  // fc_in [1, S, depth] = OneHot(coord) or SelectV2(mask, fill, OneHot(coord)).
  Op one_hot(ctx, nullptr);
  Op select(ctx, nullptr);
  Tensor coord(ctx, nullptr);
  Tensor mask(ctx, nullptr);
  Tensor fill(ctx, nullptr);
  // OneHot(coord [1, S], depth, on=1, off=0) along the last axis.
  auto one_hot_m = m_SingleUseDef(
      &one_hot,
      m_AllOf(m_Op<kLiteRtOpCodeTflOneHot>(
                  m_CaptureOrSameAs(
                      &coord, m_AllOf(m_ExactShape({1, seq_len}),
                                      m_ElementType(kLiteRtElementTypeInt32))),
                  m_ScalarConst<int32_t>(depth), m_ScalarConst<float>(1.0f),
                  m_ScalarConst<float>(0.0f)),
              m_Options<OneHotOptions>([](const OneHotOptions& opts) {
                return opts.axis == -1 || opts.axis == 2;
              })));
  // Optional SelectV2(mask [1, S, 1], fill, one_hot) with fill in {0, NaN}.
  // Row-wise, FC(fill_row) == fill for these two values (0 * W == 0 and
  // NaN * W == NaN), so the select can be re-applied after the lookup on the
  // much smaller [1, S, D] tensor.
  auto zero_or_nan = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto v = ReadScalarConst<float>(t);
        return v && (*v == 0.0f || std::isnan(*v));
      },
      "ZeroOrNaN");
  auto select_m = m_CaptureOrSameAs(
      &select, m_Op<kLiteRtOpCodeTflSelectV2>(
                   m_CaptureOrSameAs(
                       &mask, m_AllOf(m_ExactShape({1, seq_len, 1}),
                                      m_ElementType(kLiteRtElementTypeBool))),
                   m_CaptureOrSameAs(&fill, zero_or_nan), one_hot_m));
  // Captures are safe across the two alternatives: `one_hot_m` only starts
  // capturing once the tensor is known to be a OneHot output, in which case
  // `select_m` cannot match.
  auto source = m_AllOf(m_ExactShape({1, seq_len, depth}), m_HasOneUse(),
                        m_AnyOf(one_hot_m, select_m));
  if (!Match(fc_in, source)) {
    return std::nullopt;
  }

  Branch b;
  b.reshape = reshape;
  b.fc = fc;
  b.one_hot = one_hot;
  b.coord = coord;
  b.weights = weights;
  if (select.Get() != nullptr) {
    b.select = select;
    b.select_mask = mask;
    b.select_fill = fill;
    b.mask_ops = MatchMask(mask, coord, depth);
  }
  b.trig = MatchTrig(coord, seq_len);
  return b;
}

Expected<Tensor> BuildScalarConst(Builder& builder, int32_t value) {
  RankedTensorType type(ElementType::Int32, Layout(BuildLayout({})));
  auto t = builder.BuildTensor(RankedTensorSpecBuilder(type).Build());
  if (!t) {
    return t.Error();
  }
  const int32_t data[1] = {value};
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

// Folding the RoPE sin/cos into the table is exact for every coordinate with
// a finite embedding (-1 <= c < depth); all other coordinates produce NaN
// embeddings (fill == NaN), which poison the whole output anyway.
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
// Table rows: 0 <-> c <= -2, 1 <-> c == -1, 2 + k <-> c == k,
// depth + 2 <-> c >= depth.
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
    // Out-of-range rows: FC(SelectV2(mask, fill, OneHot)) == fill for
    // fill in {0, NaN}; without a select, OneHot is all-zero.
    table[d] = fill;
    table[(rows - 1) * cols + d] = fill;
    // Row 1 (c == -1) stays zero: the mask excludes -1 and OneHot is zero.
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

// Casts of RoPE trig chains whose coordinate was verified by
// EntryEmbeddingTransformation to produce NaN embeddings outside
// [-1, depth), mapped to `depth`. Consumed by EntryEmbeddingTrigFold.
absl::flat_hash_map<LiteRtOp, int32_t>& TrigFoldRegistry() {
  static auto* registry = new absl::flat_hash_map<LiteRtOp, int32_t>();
  return *registry;
}

bool EnvFlagSet(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

}  // namespace

extern "C" {

LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Op root_op(context, op);
  // Sum(concat, axis) -> sum_out [1, S, D] (f32).
  auto sum_out_m =
      m_AllOf(m_Shape({1, -1, -1}), m_ElementType(kLiteRtElementTypeFloat32));
  auto root_pattern =
      m_AllOf(m_Op<kLiteRtOpCodeTflSum>(m_Any(), m_Any()),
              m_Predicate<Op>(
                  [&sum_out_m](const Op& o) {
                    auto outs = o.Outputs();
                    return outs.size() == 1 && Match(outs[0], sum_out_m);
                  },
                  "SumOutput"));
  if (!Match(root_op, root_pattern)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor sum_out = root_op.Outputs()[0];
  const auto out_dims = Dims(sum_out);
  const int32_t seq_len = out_dims[1];
  const int32_t emb_dim = out_dims[2];

  // Concat of the two [1, 1, S, D] branches along axis 0 -> [2, 1, S, D],
  // reduced over axis 0 (implied by the shapes).
  Op concat(context, nullptr);
  Tensor x_out(context, nullptr);
  Tensor y_out(context, nullptr);
  auto concat_m =
      m_SingleUseDef(&concat, m_AllOf(m_ExactShape({2, 1, seq_len, emb_dim}),
                                      m_Op<kLiteRtOpCodeTflConcatenation>(
                                          m_CaptureOrSameAs(&x_out, m_Any()),
                                          m_CaptureOrSameAs(&y_out, m_Any()))));
  if (!Match(root_op.Inputs()[0], concat_m)) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto bx = MatchBranch(x_out, seq_len, emb_dim);
  auto by = MatchBranch(y_out, seq_len, emb_dim);
  if (!bx || !by) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool padded = CanUsePaddedLookup(*bx) && CanUsePaddedLookup(*by) &&
                      !EnvFlagSet("LITERT_MEDIATEK_EMBEDDING_LEGACY");
  const bool allow_trig_fold =
      padded && !EnvFlagSet("LITERT_MEDIATEK_EMBEDDING_NO_TRIG_FOLD");

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

  // New ops are spliced at the earliest erased op. Only Sum and Concat are
  // erased: Concat (transitively) consumes both coordinates and both masks,
  // so every input of the new ops is defined before it in any topological
  // order. The rest of the replaced sub-DAG (and the bool mask ops) becomes
  // dead and is removed by OrphanCleanupTransformation.
  builder.EraseOp(root_op);
  builder.EraseOp(concat);
  for (int i = 0; i < 2; ++i) {
    const Branch& b = *branches[i];
    std::vector<LiteRtOp> orphans = {b.reshape.Get(), b.fc.Get(),
                                     b.one_hot.Get()};
    if (b.select) orphans.push_back(b.select->Get());
    if (b.mask_ops) {
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
  Tensor src(context, nullptr);
  if (!Match(cast, m_OpVariadic<kLiteRtOpCodeTflCast>(
                       m_CaptureOrSameAs(&src, m_Shape({1, -1, 1}))))) {
    return kLiteRtStatusPatternNoMatch;
  }
  const int32_t seq_len = Dims(src)[1];
  auto tc = MatchTrigFromCast(cast, seq_len);
  if (!tc) {
    return kLiteRtStatusPatternNoMatch;
  }
  registry.erase(it);
  const int32_t f = static_cast<int32_t>(tc->freqs.size());
  // table[r] = [sin(float(r - 2) * freq), cos(float(r - 2) * freq)] for the
  // coordinates -1 <= r - 2 < depth (same float math as the CPU kernels).
  // Rows 0 and depth + 2 (out-of-range coordinates) are never observable:
  // those coordinates produce NaN embeddings that poison the whole output.
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
    // idx [1, S, 1] -> rows [1, S, 1, 2F]; Sin/Cos outputs are [1, S, 1, F].
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
  // Cast is the earliest op of the chain and its only non-constant input is
  // `src`, so splicing the new ops at the Cast is topologically valid.
  builder.EraseOp(cast);
  builder.EraseOp(tc->mul);
  builder.EraseOp(tc->reshape);
  if (tc->sin) builder.EraseOp(*tc->sin);
  if (tc->cos) builder.EraseOp(*tc->cos);
  return kLiteRtStatusOk;
}

}  // extern "C"

void ResetEntryEmbeddingTransformationState() { TrigFoldRegistry().clear(); }
