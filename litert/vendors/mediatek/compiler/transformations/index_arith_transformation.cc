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

#include "litert/vendors/mediatek/compiler/transformations/index_arith_transformation.h"

#include <algorithm>
#include <cmath>
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
#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::Builder;
using litert::compiler::DivOptions;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_AnyOf;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_ConstantValue;
using litert::compiler::m_Custom;
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
using litert::compiler::m_QType;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::OneHotOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReduceAllOptions;
using litert::compiler::SubOptions;
using litert::compiler::Tensor;

namespace {

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

// Returns the value of a constant whose elements are all equal (e.g. a scalar
// or a [1, 1] tensor).
template <typename T>
std::optional<T> ReadSplatConst(const Tensor& t) {
  if (!t.IsConstant()) return std::nullopt;
  auto data = t.WeightsData<T>();
  if (!data || data->empty()) return std::nullopt;
  for (const T& v : *data) {
    if (!(v == (*data)[0])) return std::nullopt;
  }
  return (*data)[0];
}

bool IsInt32(const Tensor& t) { return t.ElementType() == ElementType::Int32; }

bool IsPerTensor(const Tensor& t) {
  return t.QTypeId() == kLiteRtQuantizationPerTensor;
}

// Matches an int32 tensor. Unlike m_ElementType, this also accepts unranked
// tensors.
auto m_Int32() { return m_Predicate<Tensor>(IsInt32, "Int32"); }

// Matches exactly the tensor `expected`.
auto m_SameAs(const Tensor& expected) {
  return m_Predicate<Tensor>(
      [expected](const Tensor& t) { return t == expected; }, "SameAs");
}

// Matches a tensor defined by a `Code` op whose leading inputs match
// `input_matchers`. Unlike m_Op, extra trailing inputs are allowed.
template <LiteRtOpCode Code, typename... Ms>
auto m_OpPrefix(Ms... input_matchers) {
  return m_Custom(
      [m = m_OpVariadic<Code>(std::move(input_matchers)...)](
          const Tensor& t) {
        auto def = t.GetDefiningOp();
        return def && m.Match(*def);
      },
      "OpPrefix");
}

// Matches a binary elementwise op without a fused activation.
template <typename OptionsT>
auto m_NoFusedActivation() {
  return m_Options<OptionsT>(
      [](const OptionsT& o) { return o.fused_activation_function == 0; },
      "NoFusedActivation");
}

// Matches an op with exactly one output, which matches `m`.
template <typename M>
auto m_SingleOutput(M m) {
  return m_Predicate<Op>(
      [m = std::move(m)](const Op& op) {
        auto outs = op.Outputs();
        return outs.size() == 1 && m.Match(outs[0]);
      },
      "SingleOutput");
}

// Matches an op whose first output has exactly one use, by an op matching `m`.
template <typename M>
auto m_SoleConsumer(M m) {
  return m_Predicate<Op>(
      [m = std::move(m)](const Op& op) {
        auto out = op.Output(0);
        if (!out) return false;
        auto uses = out->Uses();
        return uses.size() == 1 && m.Match(uses[0].user);
      },
      "SoleConsumer");
}

// Matches Select or SelectV2 with the given inputs.
template <typename... Ms>
auto m_SelectOrSelectV2(Ms... input_matchers) {
  return m_AnyOf(m_Op<kLiteRtOpCodeTflSelect>(input_matchers...),
                 m_Op<kLiteRtOpCodeTflSelectV2>(input_matchers...));
}

Expected<Tensor> BuildTensor(Builder& builder, ElementType type,
                             const std::vector<int32_t>& dims) {
  RankedTensorType t(
      type, Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  return builder.BuildTensor(RankedTensorSpecBuilder(t).Build());
}

template <typename T>
Expected<Tensor> BuildConst(Builder& builder, ElementType type,
                            const std::vector<int32_t>& dims,
                            const std::vector<T>& values) {
  LITERT_ASSIGN_OR_RETURN(auto t, BuildTensor(builder, type, dims));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<T>(absl::MakeConstSpan(values), t));
  return t;
}

// Matched "non-empty bin" mask chain hanging off the quantized one-hot.
struct MaskChain {
  Op equal;
  Op reduce_all;
  Op logical_not;
  Tensor axes;
  bool keep_dims;
  Op quantize;
  Op mul;
};

// Returns true if the raw int value of Mul(Quantize(1.0), c) is nonzero, i.e.
// Equal(., 0) is false exactly where one_hot == 1.
bool QuantizedOneIsNonZero(const Op& quantize, const Op& mul,
                           const Tensor& mul_const) {
  const Tensor q_out = quantize.Outputs()[0];
  const Tensor m_out = mul.Outputs()[0];
  if (!IsPerTensor(q_out) || !IsPerTensor(m_out) || !IsPerTensor(mul_const)) {
    return false;
  }
  const auto qq = q_out.PerTensorQuantization();
  const auto qm = m_out.PerTensorQuantization();
  const auto qc = mul_const.PerTensorQuantization();
  if (qq.zero_point != 0 || qm.zero_point != 0 || qq.scale <= 0 ||
      qm.scale <= 0 || qc.scale <= 0) {
    return false;
  }
  std::optional<int64_t> c_raw;
  if (mul_const.ElementType() == ElementType::Int16) {
    if (auto v = ReadSplatConst<int16_t>(mul_const)) c_raw = *v;
  } else if (mul_const.ElementType() == ElementType::Int8) {
    if (auto v = ReadSplatConst<int8_t>(mul_const)) c_raw = *v;
  }
  if (!c_raw) return false;
  double q_max = 0;
  if (q_out.ElementType() == ElementType::Int16) {
    q_max = 32767.0;
  } else if (q_out.ElementType() == ElementType::Int8) {
    q_max = 127.0;
  } else {
    return false;
  }
  const double one_raw = std::min(std::round(1.0 / qq.scale), q_max);
  const double one_real = one_raw * qq.scale;
  const double c_real = static_cast<double>(*c_raw - qc.zero_point) * qc.scale;
  // Require a comfortable margin so that any rounding mode yields |raw| >= 1.
  return std::fabs(one_real * c_real / qm.scale) >= 1.0;
}

// Matches Equal(Mul(Quantize(one_hot), c), 0) -> ReduceAll -> LogicalNot.
std::optional<MaskChain> MatchMaskChain(const Op& quantize) {
  const Tensor q_out = quantize.Outputs()[0];
  // Mul(q_out, c) where c keeps the quantized 1.0 nonzero.
  auto mul_pattern = m_AllOf(
      m_Op<kLiteRtOpCodeTflMul>(m_SameAs(q_out), m_Any()),
      m_NoFusedActivation<MulOptions>(),
      m_Predicate<Op>(
          [&quantize](const Op& mul) {
            return QuantizedOneIsNonZero(quantize, mul, mul.Inputs()[1]);
          },
          "QuantizedOneIsNonZero"));
  for (const auto& q_use : q_out.Uses()) {
    const Op mul = q_use.user;
    if (!Match(mul, mul_pattern)) continue;
    const Tensor m_out = mul.Outputs()[0];
    // A per-tensor zero constant (zero point 0) of the same type as m_out.
    auto zero = m_AllOf(
        m_QType(kLiteRtQuantizationPerTensor),
        m_Predicate<Tensor>(
            [](const Tensor& t) {
              return t.PerTensorQuantization().zero_point == 0;
            },
            "ZeroPointIsZero"),
        m_Predicate<Tensor>(
            [&m_out](const Tensor& t) {
              return t.ElementType() == m_out.ElementType();
            },
            "SameElementTypeAsMulOut"),
        m_AnyOf(m_ConstantValue<int16_t>(0), m_ConstantValue<int8_t>(0)));
    for (const auto& m_use : m_out.Uses()) {
      // Fresh capture storage for every candidate.
      Op reduce_all;
      Op logical_not;
      Tensor axes;
      bool keep_dims = false;
      auto pattern = m_AllOf(
          m_Op<kLiteRtOpCodeTflEqual>(m_SameAs(m_out), zero),
          m_SoleConsumer(m_CaptureOrSameAs(
              &reduce_all,
              m_AllOf(m_Op<kLiteRtOpCodeTflReduceAll>(
                          m_Any(), m_CaptureOrSameAs(&axes, m_IsConstant())),
                      m_Options<ReduceAllOptions>(
                          [&keep_dims](const ReduceAllOptions& o) {
                            keep_dims = o.keep_dims;
                            return true;
                          },
                          "CaptureKeepDims"),
                      m_SoleConsumer(m_CaptureOrSameAs(
                          &logical_not,
                          m_OpCode<kLiteRtOpCodeTflLogicalNot>()))))));
      const Op equal = m_use.user;
      if (Match(equal, pattern)) {
        return MaskChain{equal, reduce_all, logical_not, axes,
                         keep_dims, quantize, mul};
      }
    }
  }
  return std::nullopt;
}

}  // namespace

extern "C" {

LiteRtStatus FloorDivTransformation(const LiteRtCompilerContext* context,
                                    LiteRtBuilder builder_ptr, LiteRtOp op) {
  Op select(context, op);
  Tensor cond(context, nullptr);
  Tensor q(context, nullptr);
  Tensor a(context, nullptr);
  Tensor k_t(context, nullptr);
  Op sub(context, nullptr);
  Op div(context, nullptr);

  // True branch: q - 1.
  auto q_minus_one = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(&sub, m_AllOf(m_Op<kLiteRtOpCodeTflSub>(
                                          m_CaptureOrSameAs(&q, m_Any()),
                                          m_ConstantValue<int32_t>(1)),
                                      m_NoFusedActivation<SubOptions>())),
      "QMinusOne");
  // False branch: q = Div(a, k), with k a positive int32 splat constant.
  auto positive_splat = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto k = ReadSplatConst<int32_t>(t);
        return k && *k > 0;
      },
      "PositiveSplat");
  auto q_div = m_AllOf(
      m_CaptureOrSameAs(&q, m_Any()), m_Int32(), m_HasUsers(2),
      m_CaptureOrSameAs(&div, m_AllOf(m_OpPrefix<kLiteRtOpCodeTflDiv>(
                                          m_CaptureOrSameAs(&a, m_Int32()),
                                          m_CaptureOrSameAs(&k_t,
                                                            positive_splat)),
                                      m_NoFusedActivation<DivOptions>())),
      "QDiv");
  auto select_pattern = m_AllOf(
      m_SelectOrSelectV2(m_CaptureOrSameAs(&cond, m_Any()), q_minus_one,
                         q_div),
      m_SingleOutput(m_AllOf(m_Int32(), m_Predicate<Tensor>(
                                            [&q](const Tensor& t) {
                                              return Dims(t) == Dims(q);
                                            },
                                            "SameDimsAsQ"))));
  if (!Match(select, select_pattern)) return kLiteRtStatusPatternNoMatch;

  // Condition: Sign(a) != 1 && FloorMod(a, k) != 0.
  const int32_t k = *ReadSplatConst<int32_t>(k_t);
  Op land(context, nullptr);
  Op ne_sign(context, nullptr);
  Op ne_mod(context, nullptr);
  Op sign(context, nullptr);
  Op mod(context, nullptr);
  auto sign_ne_one = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &ne_sign, m_Op<kLiteRtOpCodeTflNotEqual>(
                        m_AllOf(m_HasOneUse(),
                                m_CaptureOrSameAs(
                                    &sign, m_OpPrefix<kLiteRtOpCodeTflSign>(
                                               m_SameAs(a)))),
                        m_ConstantValue<int32_t>(1))),
      "SignNeOne");
  auto mod_ne_zero = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &ne_mod, m_Op<kLiteRtOpCodeTflNotEqual>(
                       m_AllOf(m_HasOneUse(),
                               m_CaptureOrSameAs(
                                   &mod, m_OpPrefix<kLiteRtOpCodeTflFloorMod>(
                                             m_SameAs(a),
                                             m_ConstantValue<int32_t>(k)))),
                       m_ConstantValue<int32_t>(0))),
      "ModNeZero");
  // Either operand order. The orders are mutually exclusive (an operand can't
  // be both Sign- and FloorMod-based), so captures made while trying one
  // order never affect the other.
  auto cond_pattern = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &land,
          m_AnyOf(m_OpPrefix<kLiteRtOpCodeTflLogicalAnd>(sign_ne_one,
                                                         mod_ne_zero),
                  m_OpPrefix<kLiteRtOpCodeTflLogicalAnd>(mod_ne_zero,
                                                         sign_ne_one))));
  if (!Match(cond, cond_pattern)) return kLiteRtStatusPatternNoMatch;

  Builder builder(context, builder_ptr);
  const Tensor sel_out = select.Outputs()[0];
  auto fd = builder.BuildOp(kLiteRtOpCodeTflFloorDiv, {a, k_t}, {sel_out});
  if (!fd) return fd.Error().Status();
  builder.EraseOp(select);
  builder.EraseOp(sub);
  builder.EraseOp(div);
  builder.EraseOp(land);
  builder.EraseOp(ne_sign);
  builder.EraseOp(ne_mod);
  builder.EraseOp(sign);
  builder.EraseOp(mod);
  return kLiteRtStatusOk;
}

LiteRtStatus OneHotArithTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr, LiteRtOp op) {
  constexpr int32_t kMaxDepth = 1024;
  Op one_hot(context, op);
  Tensor idx(context, nullptr);
  Tensor depth_t(context, nullptr);

  auto non_scalar = m_Predicate<Tensor>(
      [](const Tensor& t) { return !Dims(t).empty(); }, "NonScalar");
  auto valid_depth = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto depth = ReadSplatConst<int32_t>(t);
        return depth && *depth > 0 && *depth <= kMaxDepth;
      },
      "ValidDepth");
  // Only rewrite one-hots that feed quantized compute (keeps the gate tight
  // and leaves OneHot -> FullyConnected embeddings to
  // EntryEmbeddingTransformation).
  auto only_quantize_users = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto uses = t.Uses();
        return !uses.empty() &&
               std::all_of(uses.begin(), uses.end(), [](const auto& use) {
                 return Match(use.user, m_OpCode<kLiteRtOpCodeTflQuantize>());
               });
      },
      "OnlyQuantizeUsers");
  auto out_pattern = m_AllOf(
      m_ElementType(kLiteRtElementTypeFloat32), m_Not(m_IsQuantized()),
      m_Predicate<Tensor>(
          [&idx, &depth_t](const Tensor& t) {
            const auto out_dims = Dims(t);
            return out_dims.size() == Dims(idx).size() + 1 &&
                   ReadSplatConst<int32_t>(depth_t) == out_dims.back();
          },
          "OneHotDims"),
      only_quantize_users);
  auto pattern = m_AllOf(
      m_Op<kLiteRtOpCodeTflOneHot>(
          m_CaptureOrSameAs(
              &idx,
              m_AllOf(m_ElementType(kLiteRtElementTypeInt32), non_scalar)),
          m_CaptureOrSameAs(&depth_t, valid_depth),
          m_ConstantValue<float>(1.0f), m_ConstantValue<float>(0.0f)),
      m_SingleOutput(out_pattern),
      m_Options<OneHotOptions>(
          [&idx](const OneHotOptions& o) {
            return o.axis == -1 ||
                   o.axis == static_cast<int32_t>(Dims(idx).size());
          },
          "LastAxis"));
  if (!Match(one_hot, pattern)) return kLiteRtStatusPatternNoMatch;

  const Tensor out = one_hot.Outputs()[0];
  const auto idx_dims = Dims(idx);
  const auto out_dims = Dims(out);
  const int32_t depth = *ReadSplatConst<int32_t>(depth_t);
  std::optional<MaskChain> mask;
  for (const auto& use : out.Uses()) {
    mask = MatchMaskChain(use.user);
    if (mask) break;
  }

  Builder builder(context, builder_ptr);
  auto build = [&]() -> Expected<void> {
    // idx_f = Reshape(Cast(idx), idx_dims + [1])
    LITERT_ASSIGN_OR_RETURN(
        auto idx_f, BuildTensor(builder, ElementType::Float32, idx_dims));
    LITERT_RETURN_IF_ERROR(
        builder.BuildOp(kLiteRtOpCodeTflCast, {idx}, {idx_f}));
    std::vector<int32_t> col_dims = idx_dims;
    col_dims.push_back(1);
    LITERT_ASSIGN_OR_RETURN(
        auto shape_t,
        BuildConst<int32_t>(builder, ElementType::Int32,
                            {static_cast<int32_t>(col_dims.size())}, col_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto idx_col, BuildTensor(builder, ElementType::Float32, col_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto reshape,
        builder.BuildOp(kLiteRtOpCodeTflReshape, {idx_f, shape_t}, {idx_col}));
    litert::compiler::ReshapeOptions reshape_opts;
    reshape_opts.new_shape = col_dims;
    LITERT_RETURN_IF_ERROR(
        builder.SetOpOptions(reshape, std::move(reshape_opts)));

    // diff = Tile(idx_col, [1, ..., 1, D]) - iota [1, ..., 1, D]
    // (Neuron rejects Sub with both operands broadcast, hence the Tile.)
    std::vector<int32_t> multiples(out_dims.size(), 1);
    multiples.back() = depth;
    LITERT_ASSIGN_OR_RETURN(
        auto multiples_t,
        BuildConst<int32_t>(builder, ElementType::Int32,
                            {static_cast<int32_t>(multiples.size())},
                            multiples));
    LITERT_ASSIGN_OR_RETURN(
        auto idx_tiled, BuildTensor(builder, ElementType::Float32, out_dims));
    LITERT_RETURN_IF_ERROR(builder.BuildOp(
        kLiteRtOpCodeTflTile, {idx_col, multiples_t}, {idx_tiled}));
    std::vector<int32_t> iota_dims(out_dims.size(), 1);
    iota_dims.back() = depth;
    std::vector<float> iota(depth);
    for (int32_t j = 0; j < depth; ++j) iota[j] = static_cast<float>(j);
    LITERT_ASSIGN_OR_RETURN(
        auto iota_t,
        BuildConst<float>(builder, ElementType::Float32, iota_dims, iota));
    LITERT_ASSIGN_OR_RETURN(
        auto diff, BuildTensor(builder, ElementType::Float32, out_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto sub0,
        builder.BuildOp(kLiteRtOpCodeTflSub, {idx_tiled, iota_t}, {diff}));
    litert::compiler::SubOptions sub_opts;
    sub_opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sub0, std::move(sub_opts)));

    // one_hot = Relu(1 - |diff|)
    LITERT_ASSIGN_OR_RETURN(
        auto abs_t, BuildTensor(builder, ElementType::Float32, out_dims));
    LITERT_RETURN_IF_ERROR(
        builder.BuildOp(kLiteRtOpCodeTflAbs, {diff}, {abs_t}));
    LITERT_ASSIGN_OR_RETURN(
        auto one_t,
        BuildConst<float>(builder, ElementType::Float32, {}, {1.0f}));
    LITERT_ASSIGN_OR_RETURN(
        auto lin, BuildTensor(builder, ElementType::Float32, out_dims));
    LITERT_ASSIGN_OR_RETURN(
        auto sub1, builder.BuildOp(kLiteRtOpCodeTflSub, {one_t, abs_t}, {lin}));
    litert::compiler::SubOptions sub1_opts;
    sub1_opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sub1, std::move(sub1_opts)));
    LITERT_RETURN_IF_ERROR(builder.BuildOp(kLiteRtOpCodeTflRelu, {lin}, {out}));

    if (mask) {
      // mask = NotEqual(ReduceMax(one_hot, axes), 0)
      const Tensor mask_out = mask->logical_not.Outputs()[0];
      LITERT_ASSIGN_OR_RETURN(
          auto max_t,
          BuildTensor(builder, ElementType::Float32, Dims(mask_out)));
      LITERT_ASSIGN_OR_RETURN(
          auto rmax, builder.BuildOp(kLiteRtOpCodeTflReduceMax,
                                     {out, mask->axes}, {max_t}));
      litert::compiler::ReduceMaxOptions rmax_opts;
      rmax_opts.keep_dims = mask->keep_dims;
      LITERT_RETURN_IF_ERROR(builder.SetOpOptions(rmax, std::move(rmax_opts)));
      LITERT_ASSIGN_OR_RETURN(
          auto zero_t,
          BuildConst<float>(builder, ElementType::Float32, {}, {0.0f}));
      LITERT_RETURN_IF_ERROR(builder.BuildOp(kLiteRtOpCodeTflNotEqual,
                                             {max_t, zero_t}, {mask_out}));
    }
    return {};
  };
  if (auto s = build(); !s) return s.Error().Status();

  builder.EraseOp(one_hot);
  if (mask) {
    builder.EraseOp(mask->equal);
    builder.EraseOp(mask->reduce_all);
    builder.EraseOp(mask->logical_not);
    // The rescaled one-hot may now be dead (if Equal was its only user).
    RegisterOrphanGroup({mask->mul.Get(), mask->quantize.Get()});
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
