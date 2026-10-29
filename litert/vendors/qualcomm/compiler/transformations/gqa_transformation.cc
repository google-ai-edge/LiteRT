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

#include "litert/vendors/qualcomm/compiler/transformations/gqa_transformation.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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
using litert::compiler::AddOptions;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_Custom;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_HasUsers;
using litert::compiler::m_Op;
using litert::compiler::m_OpVariadic;
using litert::compiler::m_Options;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpec;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::SoftmaxOptions;
using litert::compiler::SplitOptions;
using litert::compiler::Tensor;

bool HasDims(const Tensor& t, absl::Span<const int32_t> expected) {
  auto type = t.RankedTensorType();
  return type && type->Layout().Dimensions() == expected;
}

struct PlainBatchMatmulMatch {
  Tensor lhs;
  Tensor rhs;
  LiteRtOp op = nullptr;
  bool asymmetric_quantize_input = false;
};

std::optional<PlainBatchMatmulMatch> MatchPlainBatchMatmul(const Tensor& out) {
  Op batch_matmul(out.ctx(), nullptr);
  Tensor lhs(out.ctx(), nullptr);
  Tensor rhs(out.ctx(), nullptr);
  auto matcher = m_CaptureOrSameAs(
      &batch_matmul,
      m_AllOf(m_Options<BatchMatmulOptions>([](const BatchMatmulOptions& opts) {
                return !opts.adj_x && !opts.adj_y;
              }),
              m_Op<kLiteRtOpCodeTflBatchMatmul>(
                  m_CaptureOrSameAs(&lhs, m_Any()),
                  m_CaptureOrSameAs(&rhs, m_Any()))));
  if (!Match(out, matcher) || batch_matmul.Outputs().size() != 1) {
    return std::nullopt;
  }
  auto opts = GetOptionsAs<BatchMatmulOptions>(batch_matmul.ctx(),
                                               batch_matmul.Get());
  if (!opts) return std::nullopt;
  return PlainBatchMatmulMatch{lhs, rhs, batch_matmul.Get(),
                               opts->asymmetric_quantize_input};
}

struct UnwrappedAdd {
  Tensor lhs;
  Tensor rhs;
  LiteRtOp op = nullptr;
};

// Matches a direct Add(a, b) -> out.
std::optional<UnwrappedAdd> MatchPlainAdd(const Tensor& out) {
  Op add(out.ctx(), nullptr);
  Tensor lhs(out.ctx(), nullptr);
  Tensor rhs(out.ctx(), nullptr);
  auto matcher = m_CaptureOrSameAs(
      &add,
      m_AllOf(
          m_Options<AddOptions>([](const AddOptions& opts) {
            return opts.fused_activation_function == 0;
          }),
          m_Op<kLiteRtOpCodeTflAdd>(m_CaptureOrSameAs(&lhs, m_Any()),
                                    m_CaptureOrSameAs(&rhs, m_Any()))));
  if (!Match(out, matcher) || add.Outputs().size() != 1) {
    return std::nullopt;
  }
  return UnwrappedAdd{lhs, rhs, add.Get()};
}

struct UnwrappedReshape {
  Tensor input;
  LiteRtOp op = nullptr;
};

std::optional<UnwrappedReshape> MatchReshape(const Tensor& out,
                                             std::size_t num_uses) {
  if (!Match(out, m_HasUsers(num_uses))) return std::nullopt;
  auto reshape = out.GetDefiningOp();
  if (!reshape) return std::nullopt;
  Tensor input(out.ctx(), nullptr);
  if (!Match(*reshape, m_OpVariadic<kLiteRtOpCodeTflReshape>(
                           m_CaptureOrSameAs(&input, m_Any())))) {
    return std::nullopt;
  }
  return UnwrappedReshape{input, reshape->Get()};
}

struct UnwrappedPrefillTail {
  Tensor input;
  LiteRtOp reshape = nullptr;
  LiteRtOp transpose = nullptr;
};

std::optional<UnwrappedPrefillTail> MatchPrefillTail(const Tensor& out) {
  Op transpose(out.ctx(), nullptr);
  Tensor reshape_out(out.ctx(), nullptr);
  auto perm_matcher = m_Custom([](const Tensor& perm) {
    if (!perm.IsConstant()) return false;
    auto perm_data = perm.WeightsData<int32_t>();
    return perm_data &&
           *perm_data == absl::Span<const int32_t>{0, 2, 1, 3};
  });
  auto matcher = m_AllOf(
      m_HasOneUse(),
      m_CaptureOrSameAs(
          &transpose,
          m_Op<kLiteRtOpCodeTflTranspose>(
              m_CaptureOrSameAs(&reshape_out, m_Any()), perm_matcher)));
  if (!Match(out, matcher)) return std::nullopt;
  auto reshape = MatchReshape(reshape_out, 1);
  if (!reshape) return std::nullopt;
  return UnwrappedPrefillTail{reshape->input, reshape->op, transpose.Get()};
}

struct QkConcatMatch {
  Tensor lhs;
  Tensor rhs;
  LiteRtOp op = nullptr;
};

std::optional<QkConcatMatch> MatchQkConcat(const Tensor& out,
                                           std::vector<int32_t> shape) {
  Op concat(out.ctx(), nullptr);
  Tensor lhs(out.ctx(), nullptr);
  Tensor rhs(out.ctx(), nullptr);
  auto matcher = m_AllOf(
      m_HasOneUse(), m_Shape(std::move(shape)),
      m_CaptureOrSameAs(
          &concat,
          m_AllOf(
              m_Options<ConcatenationOptions>(
                  [](const ConcatenationOptions& opts) {
                    return (opts.axis == 3 || opts.axis == -1) &&
                           opts.fused_activation_function == 0;
                  }),
              m_Op<kLiteRtOpCodeTflConcatenation>(
                  m_CaptureOrSameAs(&lhs, m_Any()),
                  m_CaptureOrSameAs(&rhs, m_Any())))));
  if (!Match(out, matcher)) return std::nullopt;
  return QkConcatMatch{lhs, rhs, concat.Get()};
}

std::optional<int32_t> SliceLastAxisBegin(const Op& slice) {
  if (slice.Code() != kLiteRtOpCodeTflSlice || slice.Inputs().size() != 3) {
    return std::nullopt;
  }
  const Tensor begin_t = slice.Inputs()[1];
  if (!begin_t.IsConstant()) return std::nullopt;
  auto begin_data = begin_t.WeightsData<int32_t>();
  if (!begin_data || begin_data->size() != 4) return std::nullopt;
  if ((*begin_data)[0] != 0 || (*begin_data)[1] != 0 || (*begin_data)[2] != 0) {
    return std::nullopt;
  }
  return (*begin_data)[3];
}

struct GqaMatch {
  Op final_reshape;
  Tensor final_out;
  Tensor q_source;
  bool q_source_is_unsplit_heads = false;
  Tensor k_cache;
  Tensor k_curr;
  Tensor v_cache;
  Tensor v_curr;
  Tensor base_mask;

  // Reference tensors for quantization parameters
  Tensor batch_matmul_k_cache_out;
  Tensor batch_matmul_k_curr_out;
  Tensor concat_qk_out;
  Tensor softmax_in;
  Tensor softmax_out;
  Tensor slice_cache_out;
  Tensor slice_curr_out;
  Tensor batch_matmul_v_cache_out;
  Tensor batch_matmul_v_curr_out;
  Tensor attn_out;

  int32_t s = 1;
  int32_t h_q = 0;
  int32_t h_kv = 0;
  int32_t g = 0;
  int32_t d = 0;
  int32_t d_v = 0;
  int32_t l_cache = 0;
  int32_t l_curr = 0;
  int32_t l_total = 0;
  float softmax_beta = 1.0f;
  bool batch_matmul_k_asym = false;
  bool batch_matmul_v_asym = false;

  std::vector<LiteRtOp> orphans;
};

std::optional<GqaMatch> MatchGqa(const LiteRtCompilerContext* ctx,
                                 LiteRtOp root_ptr) {
  Op final_reshape(ctx, root_ptr);
  if (final_reshape.Code() != kLiteRtOpCodeTflReshape ||
      final_reshape.Inputs().empty() || final_reshape.Outputs().size() != 1) {
    return std::nullopt;
  }

  GqaMatch m;
  m.final_reshape = final_reshape;
  m.final_out = final_reshape.Outputs()[0];
  auto final_out_type = m.final_out.RankedTensorType();
  if (!final_out_type) return std::nullopt;
  const auto final_out_dims = final_out_type->Layout().Dimensions();
  int32_t total_hidden = 0;
  if (final_out_dims.size() == 3 && final_out_dims[0] == 1) {
    m.s = final_out_dims[1];
    total_hidden = final_out_dims[2];
  } else if (final_out_dims.size() == 2 && final_out_dims[0] == 1) {
    m.s = 1;
    total_hidden = final_out_dims[1];
  } else {
    return std::nullopt;
  }

  Tensor attn_out = final_reshape.Inputs()[0];
  auto r_in_type = attn_out.RankedTensorType();
  if (!r_in_type) return std::nullopt;
  const auto r_in_dims = r_in_type->Layout().Dimensions();
  if (r_in_dims.size() != 4 || r_in_dims[0] != 1) {
    return std::nullopt;
  }

  // Case 1: Prefill tail Reshape([1, H_kv, G*S, D_v] -> [1, H_q, S, D_v]) ->
  // Transpose(0, 2, 1, 3) -> Reshape([1, S, H_q*D_v])
  if (auto tail = MatchPrefillTail(attn_out)) {
    m.orphans.push_back(tail->transpose);
    m.orphans.push_back(tail->reshape);
    attn_out = tail->input;
  }

  m.attn_out = attn_out;
  if (attn_out.Uses().size() != 1) return std::nullopt;
  auto attn_type = attn_out.RankedTensorType();
  if (!attn_type) return std::nullopt;
  const auto attn_dims = attn_type->Layout().Dimensions();
  if (attn_dims.size() != 4 || attn_dims[0] != 1) return std::nullopt;
  m.h_kv = attn_dims[1];
  const int32_t gs = attn_dims[2];
  m.d_v = attn_dims[3];
  if (m.s <= 0 || m.h_kv <= 0 || m.d_v <= 0 || gs <= 0 || gs % m.s != 0) {
    return std::nullopt;
  }
  m.g = gs / m.s;
  m.h_q = m.h_kv * m.g;
  if (m.h_q <= 1 || m.h_q * m.d_v != total_hidden) {
    return std::nullopt;
  }

  // Match add2 (direct Add)
  auto add2 = MatchPlainAdd(attn_out);
  if (!add2) return std::nullopt;
  m.orphans.push_back(add2->op);

  Tensor v_out0 = add2->lhs;
  Tensor v_out1 = add2->rhs;
  if (v_out0.Uses().size() != 1 || v_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto batch_matmul_v0_or = MatchPlainBatchMatmul(v_out0);
  auto batch_matmul_v1_or = MatchPlainBatchMatmul(v_out1);
  if (!batch_matmul_v0_or || !batch_matmul_v1_or) return std::nullopt;
  PlainBatchMatmulMatch batch_matmul_v0 = *batch_matmul_v0_or;
  PlainBatchMatmulMatch batch_matmul_v1 = *batch_matmul_v1_or;

  Tensor sl_out0 = batch_matmul_v0.lhs;
  Tensor sl_out1 = batch_matmul_v1.lhs;
  if (sl_out0.Uses().size() != 1 || sl_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto sl0_or = sl_out0.GetDefiningOp();
  auto sl1_or = sl_out1.GetDefiningOp();
  if (!sl0_or || !sl1_or) return std::nullopt;
  auto b0 = SliceLastAxisBegin(*sl0_or);
  auto b1 = SliceLastAxisBegin(*sl1_or);
  if (!b0 || !b1) return std::nullopt;
  if (*b0 != 0 && *b1 == 0) {
    std::swap(batch_matmul_v0, batch_matmul_v1);
    std::swap(v_out0, v_out1);
    std::swap(sl_out0, sl_out1);
    std::swap(sl0_or, sl1_or);
    std::swap(b0, b1);
  }
  if (*b0 != 0) return std::nullopt;

  m.orphans.push_back(batch_matmul_v0.op);
  m.orphans.push_back(batch_matmul_v1.op);
  m.orphans.push_back(sl0_or->Get());
  m.orphans.push_back(sl1_or->Get());

  m.batch_matmul_v_cache_out = v_out0;
  m.batch_matmul_v_curr_out = v_out1;
  m.slice_cache_out = sl_out0;
  m.slice_curr_out = sl_out1;
  m.v_cache = batch_matmul_v0.rhs;
  m.v_curr = batch_matmul_v1.rhs;

  auto sl0_type = sl_out0.RankedTensorType();
  auto sl1_type = sl_out1.RankedTensorType();
  if (!sl0_type || !sl1_type) return std::nullopt;
  const auto sl0_dims = sl0_type->Layout().Dimensions();
  const auto sl1_dims = sl1_type->Layout().Dimensions();
  if (sl0_dims.size() != 4 || sl1_dims.size() != 4) return std::nullopt;
  m.l_cache = sl0_dims[3];
  m.l_curr = sl1_dims[3];
  m.l_total = m.l_cache + m.l_curr;
  if (*b1 != m.l_cache || m.l_cache <= 0 || m.l_curr <= 0) return std::nullopt;

  if (!HasDims(m.v_cache, {1, m.h_kv, m.l_cache, m.d_v}) ||
      !HasDims(m.v_curr, {1, m.h_kv, m.l_curr, m.d_v})) {
    return std::nullopt;
  }

  m.batch_matmul_v_asym = batch_matmul_v0.asymmetric_quantize_input;

  // Softmax
  const Tensor sm_out = sl0_or->Inputs()[0];
  Op softmax(ctx, nullptr);
  Tensor sm_in(ctx, nullptr);
  auto softmax_matcher = m_AllOf(
      m_HasUsers(2),
      m_CaptureOrSameAs(
          &softmax,
          m_AllOf(
              m_Options<SoftmaxOptions>([](const SoftmaxOptions&) {
                return true;
              }),
              m_Op<kLiteRtOpCodeTflSoftmax>(
                  m_CaptureOrSameAs(&sm_in, m_Any())))));
  if (sm_out != sl1_or->Inputs()[0] || !Match(sm_out, softmax_matcher)) {
    return std::nullopt;
  }
  auto sm_opts = GetOptionsAs<SoftmaxOptions>(softmax.ctx(), softmax.Get());
  if (!sm_opts) return std::nullopt;
  m.softmax_beta = sm_opts->beta;
  m.softmax_out = sm_out;
  m.orphans.push_back(softmax.Get());

  if (sm_in.Uses().size() != 1) return std::nullopt;
  m.softmax_in = sm_in;

  // Unwrap optional reshape_mask1 before Softmax
  if (auto reshape = MatchReshape(sm_in, 1)) {
    m.orphans.push_back(reshape->op);
    sm_in = reshape->input;
    if (sm_in.Uses().size() != 1) return std::nullopt;
  }

  // Match add1 (direct Add)
  auto add1 = MatchPlainAdd(sm_in);
  if (!add1) return std::nullopt;
  m.orphans.push_back(add1->op);

  Tensor qk_cand = add1->lhs;
  Tensor mask_cand = add1->rhs;
  auto lhs_reshape = MatchReshape(qk_cand, 1);
  auto rhs_reshape = MatchReshape(mask_cand, 1);
  Tensor lhs_unwrapped = lhs_reshape ? lhs_reshape->input : qk_cand;
  Tensor rhs_unwrapped = rhs_reshape ? rhs_reshape->input : mask_cand;

  auto concat_qk = MatchQkConcat(
      lhs_unwrapped, {1, m.h_kv, gs, m.l_total});
  Tensor mask_in(ctx, nullptr);
  if (concat_qk) {
    m.concat_qk_out = lhs_unwrapped;
    mask_in = mask_cand;
    if (lhs_reshape) m.orphans.push_back(lhs_reshape->op);
  } else {
    concat_qk =
        MatchQkConcat(rhs_unwrapped, {1, m.h_kv, gs, m.l_total});
    if (!concat_qk) return std::nullopt;
    m.concat_qk_out = rhs_unwrapped;
    mask_in = qk_cand;
    if (rhs_reshape) m.orphans.push_back(rhs_reshape->op);
  }

  // Unwrap concat_mask if mask_in is [1, 1, G*S, L_total] produced by
  // Concatenation of G identical [1, 1, S, L_total] tensors along axis 2.
  auto mask_type = mask_in.RankedTensorType();
  if (!mask_type) return std::nullopt;
  const auto mask_dims = mask_type->Layout().Dimensions();
  if (mask_dims == absl::Span<const int32_t>{1, 1, gs, m.l_total} && m.g > 1) {
    auto mask_op_or = mask_in.GetDefiningOp();
    if (!mask_op_or || mask_op_or->Code() != kLiteRtOpCodeTflConcatenation ||
        static_cast<int32_t>(mask_op_or->Inputs().size()) != m.g) {
      return std::nullopt;
    }
    const Tensor first_mask = mask_op_or->Inputs()[0];
    for (const auto& inp : mask_op_or->Inputs()) {
      if (inp != first_mask) return std::nullopt;
    }
    if (!HasDims(first_mask, {1, 1, m.s, m.l_total})) {
      return std::nullopt;
    }
    m.orphans.push_back(mask_op_or->Get());
    m.base_mask = first_mask;
  } else if (mask_dims ==
                 absl::Span<const int32_t>{1, 1, m.s, m.l_total} ||
             mask_dims == absl::Span<const int32_t>{1, 1, 1, m.l_total}) {
    m.base_mask = mask_in;
  } else {
    return std::nullopt;
  }

  m.orphans.push_back(concat_qk->op);

  const Tensor k_out0 = concat_qk->lhs;
  const Tensor k_out1 = concat_qk->rhs;
  if (k_out0.Uses().size() != 1 || k_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto batch_matmul_k0 = MatchPlainBatchMatmul(k_out0);
  auto batch_matmul_k1 = MatchPlainBatchMatmul(k_out1);
  if (!batch_matmul_k0 || !batch_matmul_k1) return std::nullopt;
  m.orphans.push_back(batch_matmul_k0->op);
  m.orphans.push_back(batch_matmul_k1->op);
  m.batch_matmul_k_cache_out = k_out0;
  m.batch_matmul_k_curr_out = k_out1;

  const Tensor q_grouped = batch_matmul_k0->lhs;
  if (q_grouped != batch_matmul_k1->lhs) return std::nullopt;
  m.k_cache = batch_matmul_k0->rhs;
  m.k_curr = batch_matmul_k1->rhs;

  auto q_grouped_type = q_grouped.RankedTensorType();
  if (!q_grouped_type) return std::nullopt;
  const auto q_grouped_dims = q_grouped_type->Layout().Dimensions();
  if (q_grouped_dims.size() != 4 || q_grouped_dims[0] != 1 ||
      q_grouped_dims[1] != m.h_kv || q_grouped_dims[2] != gs) {
    return std::nullopt;
  }
  m.d = q_grouped_dims[3];
  if (m.d <= 0 || !HasDims(m.k_cache, {1, m.h_kv, m.d, m.l_cache}) ||
      !HasDims(m.k_curr, {1, m.h_kv, m.d, m.l_curr})) {
    return std::nullopt;
  }

  m.batch_matmul_k_asym = batch_matmul_k0->asymmetric_quantize_input;

  // Check if q_grouped is produced by Reshape from [1, H_q, S, D]
  m.q_source = q_grouped;
  m.q_source_is_unsplit_heads = false;
  if (auto reshape = MatchReshape(q_grouped, 2);
      reshape && HasDims(reshape->input, {1, m.h_q, m.s, m.d})) {
    m.q_source = reshape->input;
    m.q_source_is_unsplit_heads = true;
    m.orphans.push_back(reshape->op);
  }

  return m;
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         absl::Span<const int32_t> dims) {
  RankedTensorType type(
      like.ElementType(),
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  RankedTensorSpec spec{std::move(type), std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt};
  if (like.QTypeId() == kLiteRtQuantizationPerTensor) {
    spec.per_tensor_quantization = like.PerTensorQuantization();
  }
  const auto name = like.Name();
  if (!name.empty()) {
    spec.tensor_name = std::string(name);
  }
  return builder.BuildTensor(spec);
}

Expected<Tensor> ConstI32(Builder& builder, absl::Span<const int32_t> data) {
  RankedTensorType type(
      ElementType::Int32,
      Layout(BuildLayout({static_cast<int32_t>(data.size())})));
  LITERT_ASSIGN_OR_RETURN(
      auto t, builder.BuildTensor(RankedTensorSpecBuilder(type).Build()));
  LITERT_RETURN_IF_ERROR(builder.BuildWeights<int32_t>(data, t));
  return t;
}

Expected<std::vector<Tensor>> BuildSplitHeads(
    Builder& builder, const Tensor& in_4d, int32_t num_heads,
    absl::Span<const int32_t> per_head_dims) {
  if (num_heads == 1) {
    return std::vector<Tensor>{in_4d};
  }
  LITERT_ASSIGN_OR_RETURN(auto axis_t, ConstI32(builder, {1}));
  std::vector<Tensor> outs;
  outs.reserve(num_heads);
  for (int32_t i = 0; i < num_heads; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto t, NewLike(builder, in_4d, per_head_dims));
    outs.push_back(t);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSplit, {axis_t, in_4d}, outs));
  SplitOptions opts;
  opts.num_splits = num_heads;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return outs;
}

Expected<Tensor> BuildReshape(Builder& builder, const Tensor& in,
                              const Tensor& like,
                              absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape.assign(dims.begin(), dims.end());
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<void> BuildReshapeInto(Builder& builder, const Tensor& in,
                                absl::Span<const int32_t> dims,
                                const Tensor& out) {
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape.assign(dims.begin(), dims.end());
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return {};
}

Expected<Tensor> BuildBatchMatmul(Builder& builder, const Tensor& a,
                                  const Tensor& b, const Tensor& like,
                                  absl::Span<const int32_t> dims, bool asym) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, {a, b}, {out}));
  BatchMatmulOptions opts;
  opts.adj_x = false;
  opts.adj_y = false;
  opts.asymmetric_quantize_input = asym;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<Tensor> BuildConcat(Builder& builder, const std::vector<Tensor>& ins,
                             int32_t axis, const Tensor& like,
                             absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflConcatenation, ins, {out}));
  ConcatenationOptions opts;
  opts.axis = axis;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<Tensor> BuildAdd(Builder& builder, const Tensor& a, const Tensor& b,
                          const Tensor& like, absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(auto op,
                          builder.BuildOp(kLiteRtOpCodeTflAdd, {a, b}, {out}));
  AddOptions opts;
  opts.fused_activation_function = 0;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<Tensor> BuildSoftmax(Builder& builder, const Tensor& in, float beta,
                              const Tensor& like,
                              absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflSoftmax, {in}, {out}));
  SoftmaxOptions opts;
  opts.beta = beta;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
}

Expected<Tensor> BuildSlice(Builder& builder, const Tensor& in,
                            const Tensor& begin_t, const Tensor& size_t,
                            const Tensor& like,
                            absl::Span<const int32_t> dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflSlice, {in, begin_t, size_t}, {out}));
  return out;
}

Expected<void> BuildShaGqa(Builder& builder, const GqaMatch& m) {
  Tensor q_unsplit = m.q_source;
  if (!m.q_source_is_unsplit_heads) {
    LITERT_ASSIGN_OR_RETURN(
        q_unsplit,
        BuildReshape(builder, m.q_source, m.q_source, {1, m.h_q, m.s, m.d}));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto q_heads,
      BuildSplitHeads(builder, q_unsplit, m.h_q, {1, 1, m.s, m.d}));
  LITERT_ASSIGN_OR_RETURN(
      auto k_cache_heads,
      BuildSplitHeads(builder, m.k_cache, m.h_kv, {1, 1, m.d, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(
      auto k_curr_heads,
      BuildSplitHeads(builder, m.k_curr, m.h_kv, {1, 1, m.d, m.l_curr}));
  LITERT_ASSIGN_OR_RETURN(
      auto v_cache_heads,
      BuildSplitHeads(builder, m.v_cache, m.h_kv, {1, 1, m.l_cache, m.d_v}));
  LITERT_ASSIGN_OR_RETURN(
      auto v_curr_heads,
      BuildSplitHeads(builder, m.v_curr, m.h_kv, {1, 1, m.l_curr, m.d_v}));

  LITERT_ASSIGN_OR_RETURN(auto begin_cache, ConstI32(builder, {0, 0, 0, 0}));
  LITERT_ASSIGN_OR_RETURN(auto size_cache,
                          ConstI32(builder, {1, 1, m.s, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(auto begin_curr,
                          ConstI32(builder, {0, 0, 0, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(auto size_curr,
                          ConstI32(builder, {1, 1, m.s, m.l_curr}));

  std::vector<Tensor> sha_outs;
  sha_outs.reserve(m.h_q);
  for (int32_t h_kv = 0; h_kv < m.h_kv; ++h_kv) {
    for (int32_t g = 0; g < m.g; ++g) {
      const int32_t q_idx = h_kv * m.g + g;
      LITERT_ASSIGN_OR_RETURN(
          auto qk_cache,
          BuildBatchMatmul(builder, q_heads[q_idx], k_cache_heads[h_kv],
                           m.batch_matmul_k_cache_out,
                           {1, 1, m.s, m.l_cache},
                           m.batch_matmul_k_asym));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_curr,
          BuildBatchMatmul(builder, q_heads[q_idx], k_curr_heads[h_kv],
                           m.batch_matmul_k_curr_out,
                           {1, 1, m.s, m.l_curr},
                           m.batch_matmul_k_asym));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_concat, BuildConcat(builder, {qk_cache, qk_curr}, /*axis=*/3,
                                      m.concat_qk_out, {1, 1, m.s, m.l_total}));
      LITERT_ASSIGN_OR_RETURN(auto qk_masked,
                              BuildAdd(builder, qk_concat, m.base_mask,
                                       m.softmax_in, {1, 1, m.s, m.l_total}));
      LITERT_ASSIGN_OR_RETURN(
          auto probs, BuildSoftmax(builder, qk_masked, m.softmax_beta,
                                   m.softmax_out, {1, 1, m.s, m.l_total}));
      LITERT_ASSIGN_OR_RETURN(
          auto p_cache, BuildSlice(builder, probs, begin_cache, size_cache,
                                   m.slice_cache_out, {1, 1, m.s, m.l_cache}));
      LITERT_ASSIGN_OR_RETURN(
          auto p_curr, BuildSlice(builder, probs, begin_curr, size_curr,
                                  m.slice_curr_out, {1, 1, m.s, m.l_curr}));
      LITERT_ASSIGN_OR_RETURN(
          auto out_cache,
          BuildBatchMatmul(builder, p_cache, v_cache_heads[h_kv],
                           m.batch_matmul_v_cache_out,
                           {1, 1, m.s, m.d_v},
                           m.batch_matmul_v_asym));
      LITERT_ASSIGN_OR_RETURN(
          auto out_curr,
          BuildBatchMatmul(builder, p_curr, v_curr_heads[h_kv],
                           m.batch_matmul_v_curr_out,
                           {1, 1, m.s, m.d_v},
                           m.batch_matmul_v_asym));
      LITERT_ASSIGN_OR_RETURN(auto head_out,
                              BuildAdd(builder, out_cache, out_curr, m.attn_out,
                                       {1, 1, m.s, m.d_v}));
      sha_outs.push_back(head_out);
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto concat_heads,
                          BuildConcat(builder, sha_outs, /*axis=*/3, m.attn_out,
                                      {1, 1, m.s, m.h_q * m.d_v}));
  LITERT_ASSIGN_OR_RETURN(auto final_out_type, m.final_out.RankedTensorType());
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, concat_heads,
                                          final_out_type.Layout().Dimensions(),
                                          m.final_out));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(m.final_reshape));
  RegisterOrphanGroup(m.orphans);
  return {};
}

// Single-token (decode) variant: instead of one attention slice per query
// head, builds one slice per KV head whose query operand stacks the G query
// heads that share it ([1, 1, G, D]). Each KV head is read once instead of G
// times and the attention op count drops by a factor of G. The per-row math is
// unchanged, so outputs are identical to the per-head decomposition.
Expected<void> BuildGroupedGqaDecode(Builder& builder, const GqaMatch& m) {
  LITERT_ASSIGN_OR_RETURN(
      auto q_grouped,
      BuildReshape(builder, m.q_source, m.q_source, {1, m.h_kv, m.g, m.d}));

  LITERT_ASSIGN_OR_RETURN(
      auto q_groups,
      BuildSplitHeads(builder, q_grouped, m.h_kv, {1, 1, m.g, m.d}));
  LITERT_ASSIGN_OR_RETURN(
      auto k_cache_heads,
      BuildSplitHeads(builder, m.k_cache, m.h_kv, {1, 1, m.d, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(
      auto k_curr_heads,
      BuildSplitHeads(builder, m.k_curr, m.h_kv, {1, 1, m.d, m.l_curr}));
  LITERT_ASSIGN_OR_RETURN(
      auto v_cache_heads,
      BuildSplitHeads(builder, m.v_cache, m.h_kv, {1, 1, m.l_cache, m.d_v}));
  LITERT_ASSIGN_OR_RETURN(
      auto v_curr_heads,
      BuildSplitHeads(builder, m.v_curr, m.h_kv, {1, 1, m.l_curr, m.d_v}));

  // The mask has a single query row; broadcast it over the G stacked heads.
  LITERT_ASSIGN_OR_RETURN(
      auto mask_rows,
      BuildReshape(builder, m.base_mask, m.base_mask, {1, 1, 1, m.l_total}));
  std::vector<Tensor> mask_copies(m.g, mask_rows);
  LITERT_ASSIGN_OR_RETURN(
      auto mask_g, BuildConcat(builder, mask_copies, /*axis=*/2, m.base_mask,
                               {1, 1, m.g, m.l_total}));

  LITERT_ASSIGN_OR_RETURN(auto begin_cache, ConstI32(builder, {0, 0, 0, 0}));
  LITERT_ASSIGN_OR_RETURN(auto size_cache,
                          ConstI32(builder, {1, 1, m.g, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(auto begin_curr,
                          ConstI32(builder, {0, 0, 0, m.l_cache}));
  LITERT_ASSIGN_OR_RETURN(auto size_curr,
                          ConstI32(builder, {1, 1, m.g, m.l_curr}));

  std::vector<Tensor> group_outs;
  group_outs.reserve(m.h_kv);
  for (int32_t h_kv = 0; h_kv < m.h_kv; ++h_kv) {
    LITERT_ASSIGN_OR_RETURN(
        auto qk_cache,
        BuildBatchMatmul(builder, q_groups[h_kv], k_cache_heads[h_kv],
                         m.batch_matmul_k_cache_out,
                         {1, 1, m.g, m.l_cache},
                         m.batch_matmul_k_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_curr,
        BuildBatchMatmul(builder, q_groups[h_kv], k_curr_heads[h_kv],
                         m.batch_matmul_k_curr_out,
                         {1, 1, m.g, m.l_curr},
                         m.batch_matmul_k_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_concat, BuildConcat(builder, {qk_cache, qk_curr}, /*axis=*/3,
                                    m.concat_qk_out, {1, 1, m.g, m.l_total}));
    LITERT_ASSIGN_OR_RETURN(auto qk_masked,
                            BuildAdd(builder, qk_concat, mask_g, m.softmax_in,
                                     {1, 1, m.g, m.l_total}));
    LITERT_ASSIGN_OR_RETURN(
        auto probs, BuildSoftmax(builder, qk_masked, m.softmax_beta,
                                 m.softmax_out, {1, 1, m.g, m.l_total}));
    LITERT_ASSIGN_OR_RETURN(
        auto p_cache, BuildSlice(builder, probs, begin_cache, size_cache,
                                 m.slice_cache_out, {1, 1, m.g, m.l_cache}));
    LITERT_ASSIGN_OR_RETURN(
        auto p_curr, BuildSlice(builder, probs, begin_curr, size_curr,
                                m.slice_curr_out, {1, 1, m.g, m.l_curr}));
    LITERT_ASSIGN_OR_RETURN(
        auto out_cache,
        BuildBatchMatmul(builder, p_cache, v_cache_heads[h_kv],
                         m.batch_matmul_v_cache_out,
                         {1, 1, m.g, m.d_v},
                         m.batch_matmul_v_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto out_curr,
        BuildBatchMatmul(builder, p_curr, v_curr_heads[h_kv],
                         m.batch_matmul_v_curr_out,
                         {1, 1, m.g, m.d_v},
                         m.batch_matmul_v_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto group_out,
        BuildAdd(builder, out_cache, out_curr, m.attn_out, {1, 1, m.g, m.d_v}));
    group_outs.push_back(group_out);
  }

  // [1, H_kv, G, D_v] is laid out as query head (h_kv * G + g) major, which
  // matches the [1, 1, H_q * D_v] attention output.
  LITERT_ASSIGN_OR_RETURN(auto concat_groups,
                          BuildConcat(builder, group_outs, /*axis=*/1,
                                      m.attn_out, {1, m.h_kv, m.g, m.d_v}));
  LITERT_ASSIGN_OR_RETURN(auto final_out_type, m.final_out.RankedTensorType());
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, concat_groups,
                                          final_out_type.Layout().Dimensions(),
                                          m.final_out));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(m.final_reshape));
  RegisterOrphanGroup(m.orphans);
  return {};
}

}  // namespace

extern "C" {

LiteRtStatus GqaTransformation(const LiteRtCompilerContext* context,
                               LiteRtBuilder builder_ptr, LiteRtOp op_ptr) {
  auto match = MatchGqa(context, op_ptr);
  if (!match) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  auto status = (match->s == 1 && match->g > 1)
                    ? BuildGroupedGqaDecode(builder, *match)
                    : BuildShaGqa(builder, *match);
  if (!status) {
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
