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
using litert::compiler::Op;
using litert::compiler::RankedTensorSpec;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::SoftmaxOptions;
using litert::compiler::SplitOptions;
using litert::compiler::Tensor;

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
}

bool HasPerm(const Op& transpose, const std::vector<int32_t>& expected) {
  if (transpose.Code() != kLiteRtOpCodeTflTranspose ||
      transpose.Inputs().size() != 2) {
    return false;
  }
  const Tensor perm = transpose.Inputs()[1];
  if (!perm.IsConstant()) return false;
  auto perm_data = perm.WeightsData<int32_t>();
  return perm_data &&
         std::vector<int32_t>(perm_data->begin(), perm_data->end()) == expected;
}

bool IsPlainBatchMatmul(const Op& op) {
  if (op.Code() != kLiteRtOpCodeTflBatchMatmul || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1) {
    return false;
  }
  auto opts = GetOptionsAs<BatchMatmulOptions>(op.ctx(), op.Get());
  return opts && !opts->adj_x && !opts->adj_y;
}

bool IsPlainAdd(const Op& op) {
  if (op.Code() != kLiteRtOpCodeTflAdd || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1) {
    return false;
  }
  auto opts = GetOptionsAs<AddOptions>(op.ctx(), op.Get());
  return opts && opts->fused_activation_function == 0;
}

struct UnwrappedAdd {
  Tensor lhs;
  Tensor rhs;
  std::vector<LiteRtOp> ops;
};

// Matches a direct Add(a, b) -> out.
std::optional<UnwrappedAdd> MatchPlainAdd(const Tensor& out) {
  auto def_or = out.GetDefiningOp();
  if (!def_or || !IsPlainAdd(*def_or)) return std::nullopt;
  return UnwrappedAdd{
      def_or->Inputs()[0], def_or->Inputs()[1], {def_or->Get()}};
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
  Tensor bmm_k_cache_out;
  Tensor bmm_k_curr_out;
  Tensor concat_qk_out;
  Tensor softmax_in;
  Tensor softmax_out;
  Tensor slice_cache_out;
  Tensor slice_curr_out;
  Tensor bmm_v_cache_out;
  Tensor bmm_v_curr_out;
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
  bool bmm_k_asym = false;
  bool bmm_v_asym = false;

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
  const auto final_out_dims = Dims(m.final_out);
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
  const auto r_in_dims = Dims(attn_out);
  if (r_in_dims.size() != 4 || r_in_dims[0] != 1) {
    return std::nullopt;
  }

  // Case 1: Prefill tail Reshape([1, H_kv, G*S, D_v] -> [1, H_q, S, D_v]) ->
  // Transpose(0, 2, 1, 3) -> Reshape([1, S, H_q*D_v])
  if (attn_out.Uses().size() == 1) {
    if (auto tr_or = attn_out.GetDefiningOp();
        tr_or && HasPerm(*tr_or, {0, 2, 1, 3})) {
      const Tensor tr_in = tr_or->Inputs()[0];
      if (tr_in.Uses().size() != 1) return std::nullopt;
      auto r_out_or = tr_in.GetDefiningOp();
      if (!r_out_or || r_out_or->Code() != kLiteRtOpCodeTflReshape ||
          r_out_or->Inputs().empty()) {
        return std::nullopt;
      }
      m.orphans.push_back(tr_or->Get());
      m.orphans.push_back(r_out_or->Get());
      attn_out = r_out_or->Inputs()[0];
    }
  }

  m.attn_out = attn_out;
  if (attn_out.Uses().size() != 1) return std::nullopt;
  const auto attn_dims = Dims(attn_out);
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
  for (LiteRtOp op : add2->ops) m.orphans.push_back(op);

  Tensor v_out0 = add2->lhs;
  Tensor v_out1 = add2->rhs;
  if (v_out0.Uses().size() != 1 || v_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto bmm_v0_or = v_out0.GetDefiningOp();
  auto bmm_v1_or = v_out1.GetDefiningOp();
  if (!bmm_v0_or || !bmm_v1_or || !IsPlainBatchMatmul(*bmm_v0_or) ||
      !IsPlainBatchMatmul(*bmm_v1_or)) {
    return std::nullopt;
  }
  Op bmm_v0 = *bmm_v0_or;
  Op bmm_v1 = *bmm_v1_or;

  Tensor sl_out0 = bmm_v0.Inputs()[0];
  Tensor sl_out1 = bmm_v1.Inputs()[0];
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
    std::swap(bmm_v0, bmm_v1);
    std::swap(v_out0, v_out1);
    std::swap(sl_out0, sl_out1);
    std::swap(sl0_or, sl1_or);
    std::swap(b0, b1);
  }
  if (*b0 != 0) return std::nullopt;

  m.orphans.push_back(bmm_v0.Get());
  m.orphans.push_back(bmm_v1.Get());
  m.orphans.push_back(sl0_or->Get());
  m.orphans.push_back(sl1_or->Get());

  m.bmm_v_cache_out = v_out0;
  m.bmm_v_curr_out = v_out1;
  m.slice_cache_out = sl_out0;
  m.slice_curr_out = sl_out1;
  m.v_cache = bmm_v0.Inputs()[1];
  m.v_curr = bmm_v1.Inputs()[1];

  const auto sl0_dims = Dims(sl_out0);
  const auto sl1_dims = Dims(sl_out1);
  if (sl0_dims.size() != 4 || sl1_dims.size() != 4) return std::nullopt;
  m.l_cache = sl0_dims[3];
  m.l_curr = sl1_dims[3];
  m.l_total = m.l_cache + m.l_curr;
  if (*b1 != m.l_cache || m.l_cache <= 0 || m.l_curr <= 0) return std::nullopt;

  if (Dims(m.v_cache) != std::vector<int32_t>{1, m.h_kv, m.l_cache, m.d_v} ||
      Dims(m.v_curr) != std::vector<int32_t>{1, m.h_kv, m.l_curr, m.d_v}) {
    return std::nullopt;
  }

  auto bmm_v0_opts = GetOptionsAs<BatchMatmulOptions>(ctx, bmm_v0.Get());
  if (!bmm_v0_opts) return std::nullopt;
  m.bmm_v_asym = bmm_v0_opts->asymmetric_quantize_input;

  // Softmax
  const Tensor sm_out = sl0_or->Inputs()[0];
  if (sm_out != sl1_or->Inputs()[0] || sm_out.Uses().size() != 2) {
    return std::nullopt;
  }
  auto sm_or = sm_out.GetDefiningOp();
  if (!sm_or || sm_or->Code() != kLiteRtOpCodeTflSoftmax ||
      sm_or->Inputs().size() != 1) {
    return std::nullopt;
  }
  auto sm_opts = GetOptionsAs<SoftmaxOptions>(ctx, sm_or->Get());
  if (!sm_opts) return std::nullopt;
  m.softmax_beta = sm_opts->beta;
  m.softmax_out = sm_out;
  m.orphans.push_back(sm_or->Get());

  Tensor sm_in = sm_or->Inputs()[0];
  if (sm_in.Uses().size() != 1) return std::nullopt;
  m.softmax_in = sm_in;

  // Unwrap optional reshape_mask1 before Softmax
  if (auto r_m1_or = sm_in.GetDefiningOp();
      r_m1_or && r_m1_or->Code() == kLiteRtOpCodeTflReshape &&
      !r_m1_or->Inputs().empty()) {
    m.orphans.push_back(r_m1_or->Get());
    sm_in = r_m1_or->Inputs()[0];
    if (sm_in.Uses().size() != 1) return std::nullopt;
  }

  // Match add1 (direct Add)
  auto add1 = MatchPlainAdd(sm_in);
  if (!add1) return std::nullopt;
  for (LiteRtOp op : add1->ops) m.orphans.push_back(op);

  Tensor qk_cand = add1->lhs;
  Tensor mask_cand = add1->rhs;
  auto unwrap_reshape = [](const Tensor& t,
                           std::vector<LiteRtOp>& orphans) -> Tensor {
    if (t.Uses().size() == 1) {
      if (auto op_or = t.GetDefiningOp();
          op_or && op_or->Code() == kLiteRtOpCodeTflReshape &&
          !op_or->Inputs().empty()) {
        orphans.push_back(op_or->Get());
        return op_or->Inputs()[0];
      }
    }
    return t;
  };

  std::vector<LiteRtOp> lhs_r_orphans;
  std::vector<LiteRtOp> rhs_r_orphans;
  Tensor lhs_unwrapped = unwrap_reshape(qk_cand, lhs_r_orphans);
  Tensor rhs_unwrapped = unwrap_reshape(mask_cand, rhs_r_orphans);

  auto is_qk_concat = [&](const Tensor& t) {
    if (t.Uses().size() != 1) return false;
    auto op_or = t.GetDefiningOp();
    if (!op_or || op_or->Code() != kLiteRtOpCodeTflConcatenation ||
        op_or->Inputs().size() != 2) {
      return false;
    }
    return Dims(t) == std::vector<int32_t>{1, m.h_kv, gs, m.l_total};
  };

  Tensor concat_qk_out(ctx, nullptr);
  Tensor mask_in(ctx, nullptr);
  if (is_qk_concat(lhs_unwrapped)) {
    concat_qk_out = lhs_unwrapped;
    mask_in = mask_cand;
    for (LiteRtOp op : lhs_r_orphans) m.orphans.push_back(op);
  } else if (is_qk_concat(rhs_unwrapped)) {
    concat_qk_out = rhs_unwrapped;
    mask_in = qk_cand;
    for (LiteRtOp op : rhs_r_orphans) m.orphans.push_back(op);
  } else {
    return std::nullopt;
  }

  // Unwrap concat_mask if mask_in is [1, 1, G*S, L_total] produced by
  // Concatenation of G identical [1, 1, S, L_total] tensors along axis 2.
  const auto mask_dims = Dims(mask_in);
  if (mask_dims == std::vector<int32_t>{1, 1, gs, m.l_total} && m.g > 1) {
    auto mask_op_or = mask_in.GetDefiningOp();
    if (!mask_op_or || mask_op_or->Code() != kLiteRtOpCodeTflConcatenation ||
        static_cast<int32_t>(mask_op_or->Inputs().size()) != m.g) {
      return std::nullopt;
    }
    const Tensor first_mask = mask_op_or->Inputs()[0];
    for (const auto& inp : mask_op_or->Inputs()) {
      if (inp != first_mask) return std::nullopt;
    }
    if (Dims(first_mask) != std::vector<int32_t>{1, 1, m.s, m.l_total}) {
      return std::nullopt;
    }
    m.orphans.push_back(mask_op_or->Get());
    m.base_mask = first_mask;
  } else if (mask_dims == std::vector<int32_t>{1, 1, m.s, m.l_total} ||
             mask_dims == std::vector<int32_t>{1, 1, 1, m.l_total}) {
    m.base_mask = mask_in;
  } else {
    return std::nullopt;
  }

  auto concat_qk_or = concat_qk_out.GetDefiningOp();
  if (!concat_qk_or) return std::nullopt;
  m.concat_qk_out = concat_qk_out;
  m.orphans.push_back(concat_qk_or->Get());

  const Tensor k_out0 = concat_qk_or->Inputs()[0];
  const Tensor k_out1 = concat_qk_or->Inputs()[1];
  if (k_out0.Uses().size() != 1 || k_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto bmm_k0_or = k_out0.GetDefiningOp();
  auto bmm_k1_or = k_out1.GetDefiningOp();
  if (!bmm_k0_or || !bmm_k1_or || !IsPlainBatchMatmul(*bmm_k0_or) ||
      !IsPlainBatchMatmul(*bmm_k1_or)) {
    return std::nullopt;
  }
  m.orphans.push_back(bmm_k0_or->Get());
  m.orphans.push_back(bmm_k1_or->Get());
  m.bmm_k_cache_out = k_out0;
  m.bmm_k_curr_out = k_out1;

  const Tensor q_grouped = bmm_k0_or->Inputs()[0];
  if (q_grouped != bmm_k1_or->Inputs()[0]) return std::nullopt;
  m.k_cache = bmm_k0_or->Inputs()[1];
  m.k_curr = bmm_k1_or->Inputs()[1];

  const auto q_grouped_dims = Dims(q_grouped);
  if (q_grouped_dims.size() != 4 || q_grouped_dims[0] != 1 ||
      q_grouped_dims[1] != m.h_kv || q_grouped_dims[2] != gs) {
    return std::nullopt;
  }
  m.d = q_grouped_dims[3];
  if (m.d <= 0 ||
      Dims(m.k_cache) != std::vector<int32_t>{1, m.h_kv, m.d, m.l_cache} ||
      Dims(m.k_curr) != std::vector<int32_t>{1, m.h_kv, m.d, m.l_curr}) {
    return std::nullopt;
  }

  auto bmm_k0_opts = GetOptionsAs<BatchMatmulOptions>(ctx, bmm_k0_or->Get());
  if (!bmm_k0_opts) return std::nullopt;
  m.bmm_k_asym = bmm_k0_opts->asymmetric_quantize_input;

  // Check if q_grouped is produced by Reshape from [1, H_q, S, D]
  m.q_source = q_grouped;
  m.q_source_is_unsplit_heads = false;
  if (q_grouped.Uses().size() == 2) {
    if (auto rq_or = q_grouped.GetDefiningOp();
        rq_or && rq_or->Code() == kLiteRtOpCodeTflReshape &&
        !rq_or->Inputs().empty()) {
      const Tensor rq_in = rq_or->Inputs()[0];
      if (Dims(rq_in) == std::vector<int32_t>{1, m.h_q, m.s, m.d}) {
        m.q_source = rq_in;
        m.q_source_is_unsplit_heads = true;
        m.orphans.push_back(rq_or->Get());
      }
    }
  }

  return m;
}

Expected<Tensor> NewLike(Builder& builder, const Tensor& like,
                         const std::vector<int32_t>& dims) {
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

Expected<std::vector<Tensor>> BuildSplitHeads(
    Builder& builder, const Tensor& in_4d, int32_t num_heads,
    const std::vector<int32_t>& per_head_dims) {
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
                              const std::vector<int32_t>& dims) {
  LITERT_ASSIGN_OR_RETURN(auto out, NewLike(builder, like, dims));
  LITERT_ASSIGN_OR_RETURN(auto shape, ConstI32(builder, dims));
  LITERT_ASSIGN_OR_RETURN(
      auto op, builder.BuildOp(kLiteRtOpCodeTflReshape, {in, shape}, {out}));
  ReshapeOptions opts;
  opts.new_shape = dims;
  LITERT_RETURN_IF_ERROR(builder.SetOpOptions(op, std::move(opts)));
  return out;
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

Expected<Tensor> BuildBmm(Builder& builder, const Tensor& a, const Tensor& b,
                          const Tensor& like, const std::vector<int32_t>& dims,
                          bool asym) {
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
                             const std::vector<int32_t>& dims) {
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
                          const Tensor& like,
                          const std::vector<int32_t>& dims) {
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
                              const std::vector<int32_t>& dims) {
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
                            const std::vector<int32_t>& dims) {
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
          BuildBmm(builder, q_heads[q_idx], k_cache_heads[h_kv],
                   m.bmm_k_cache_out, {1, 1, m.s, m.l_cache}, m.bmm_k_asym));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_curr,
          BuildBmm(builder, q_heads[q_idx], k_curr_heads[h_kv],
                   m.bmm_k_curr_out, {1, 1, m.s, m.l_curr}, m.bmm_k_asym));
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
          BuildBmm(builder, p_cache, v_cache_heads[h_kv], m.bmm_v_cache_out,
                   {1, 1, m.s, m.d_v}, m.bmm_v_asym));
      LITERT_ASSIGN_OR_RETURN(
          auto out_curr,
          BuildBmm(builder, p_curr, v_curr_heads[h_kv], m.bmm_v_curr_out,
                   {1, 1, m.s, m.d_v}, m.bmm_v_asym));
      LITERT_ASSIGN_OR_RETURN(auto head_out,
                              BuildAdd(builder, out_cache, out_curr, m.attn_out,
                                       {1, 1, m.s, m.d_v}));
      sha_outs.push_back(head_out);
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto concat_heads,
                          BuildConcat(builder, sha_outs, /*axis=*/3, m.attn_out,
                                      {1, 1, m.s, m.h_q * m.d_v}));
  LITERT_RETURN_IF_ERROR(
      BuildReshapeInto(builder, concat_heads, Dims(m.final_out), m.final_out));

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
        BuildBmm(builder, q_groups[h_kv], k_cache_heads[h_kv],
                 m.bmm_k_cache_out, {1, 1, m.g, m.l_cache}, m.bmm_k_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_curr,
        BuildBmm(builder, q_groups[h_kv], k_curr_heads[h_kv], m.bmm_k_curr_out,
                 {1, 1, m.g, m.l_curr}, m.bmm_k_asym));
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
        BuildBmm(builder, p_cache, v_cache_heads[h_kv], m.bmm_v_cache_out,
                 {1, 1, m.g, m.d_v}, m.bmm_v_asym));
    LITERT_ASSIGN_OR_RETURN(
        auto out_curr,
        BuildBmm(builder, p_curr, v_curr_heads[h_kv], m.bmm_v_curr_out,
                 {1, 1, m.g, m.d_v}, m.bmm_v_asym));
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
  LITERT_RETURN_IF_ERROR(
      BuildReshapeInto(builder, concat_groups, Dims(m.final_out), m.final_out));

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
