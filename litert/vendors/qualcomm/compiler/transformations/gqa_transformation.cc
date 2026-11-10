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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/compiler/transformations/gqa_transformation.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/vendors/qualcomm/compiler/transformations/orphan_cleanup_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/transformation_utils.h"

namespace {

using litert::Expected;
using litert::compiler::AddOptions;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::ConcatenationOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_HasUsers;
using litert::compiler::m_Op;
using litert::compiler::m_OpVariadic;
using litert::compiler::m_Options;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::SoftmaxOptions;
using litert::compiler::SplitOptions;
using litert::compiler::Tensor;
using litert::qnn::BuildAdd;
using litert::qnn::BuildBatchMatmul;
using litert::qnn::BuildConcat;
using litert::qnn::BuildReshape;
using litert::qnn::BuildReshapeInto;
using litert::qnn::BuildSlice;
using litert::qnn::BuildSoftmax;
using litert::qnn::ConstI32;
using litert::qnn::HasDims;
using litert::qnn::HasPerm;
using litert::qnn::NewLike;
using litert::qnn::SingleOutput;
using litert::qnn::SingleUser;

std::optional<Op> MatchPlainBatchMatmul(const Tensor& out) {
  Op batch_matmul(out.ctx(), nullptr);
  auto matcher = m_CaptureOrSameAs(
      &batch_matmul,
      m_AllOf(m_Options<BatchMatmulOptions>([](const BatchMatmulOptions& opts) {
                return !opts.adj_x && !opts.adj_y &&
                       !opts.asymmetric_quantize_input;
              }),
              m_Op<kLiteRtOpCodeTflBatchMatmul>(m_Any(), m_Any())));
  if (!Match(out, matcher) || batch_matmul.Outputs().size() != 1) {
    return std::nullopt;
  }
  return batch_matmul;
}

std::optional<Op> MatchPlainAdd(const Tensor& out) {
  Op add(out.ctx(), nullptr);
  auto matcher = m_CaptureOrSameAs(
      &add,
      m_AllOf(
          m_Options<AddOptions>([](const AddOptions& opts) {
            return opts.fused_activation_function == 0;
          }),
          m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any())));
  if (!Match(out, matcher) || add.Outputs().size() != 1) {
    return std::nullopt;
  }
  return add;
}

std::optional<Op> MatchReshape(const Tensor& out, std::size_t num_uses) {
  Op reshape(out.ctx(), nullptr);
  auto matcher = m_AllOf(
      m_HasUsers(num_uses),
      m_CaptureOrSameAs(&reshape, m_OpVariadic<kLiteRtOpCodeTflReshape>(
                                      m_Any())));
  if (!Match(out, matcher)) {
    return std::nullopt;
  }
  return reshape;
}

std::optional<Op> MatchQkConcat(const Tensor& out,
                                std::vector<int32_t> shape) {
  if (shape.empty()) return std::nullopt;
  const int32_t last_axis = static_cast<int32_t>(shape.size()) - 1;

  Op concat(out.ctx(), nullptr);
  auto matcher = m_AllOf(
      m_HasOneUse(), m_Shape(std::move(shape)),
      m_CaptureOrSameAs(
          &concat,
          m_AllOf(
              m_Options<ConcatenationOptions>(
                  [last_axis](const ConcatenationOptions& opts) {
                    return (opts.axis == last_axis || opts.axis == -1) &&
                           opts.fused_activation_function == 0;
                  }),
              m_Op<kLiteRtOpCodeTflConcatenation>(m_Any(), m_Any()))));
  if (!Match(out, matcher) || concat.Inputs()[0] == concat.Inputs()[1]) {
    return std::nullopt;
  }
  return concat;
}

std::optional<Op> MatchRepeatedMaskConcat(
    const Tensor& out, int32_t group_size, int32_t grouped_sequence_length,
    int32_t sequence_length, int32_t total_context_length) {
  if (group_size <= 1) return std::nullopt;

  Op concat(out.ctx(), nullptr);
  auto matcher = m_AllOf(
      m_Shape({1, 1, grouped_sequence_length, total_context_length}),
      m_CaptureOrSameAs(
          &concat,
          m_AllOf(
              m_Options<ConcatenationOptions>(
                  [](const ConcatenationOptions& opts) {
                    return (opts.axis == 2 || opts.axis == -2) &&
                           opts.fused_activation_function == 0;
                  }),
              m_OpVariadic<kLiteRtOpCodeTflConcatenation>())));
  if (!Match(out, matcher) ||
      static_cast<int32_t>(concat.Inputs().size()) != group_size) {
    return std::nullopt;
  }

  const Tensor base_mask = concat.Inputs()[0];
  for (const Tensor& input : concat.Inputs()) {
    if (input != base_mask) return std::nullopt;
  }
  if (!HasDims(base_mask,
               {1, 1, sequence_length, total_context_length})) {
    return std::nullopt;
  }
  return concat;
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

struct SliceBatchMatmulMatch {
  Op slice;
  Op batch_matmul;
  int32_t slice_begin = 0;
};

std::optional<SliceBatchMatmulMatch> MatchSliceBatchMatmul(
    const Tensor::TensorUse& softmax_use) {
  if (softmax_use.user_arg_ind != 0) return std::nullopt;
  auto slice_begin = SliceLastAxisBegin(softmax_use.user);
  if (!slice_begin) return std::nullopt;
  auto slice_out = SingleOutput(softmax_use.user);
  if (!slice_out) return std::nullopt;
  auto batch_matmul_op = SingleUser(*slice_out, kLiteRtOpCodeTflBatchMatmul);
  if (!batch_matmul_op) return std::nullopt;
  auto batch_matmul_out = SingleOutput(*batch_matmul_op);
  if (!batch_matmul_out) return std::nullopt;
  auto batch_matmul = MatchPlainBatchMatmul(*batch_matmul_out);
  if (!batch_matmul || batch_matmul->Inputs()[0] != *slice_out) {
    return std::nullopt;
  }
  return SliceBatchMatmulMatch{softmax_use.user, *batch_matmul, *slice_begin};
}

struct OutputTailMatch {
  Op final_reshape;
  LiteRtOp reshape = nullptr;
  LiteRtOp transpose = nullptr;
};

std::optional<OutputTailMatch> MatchOutputTail(const Tensor& attn_out) {
  auto first_reshape = SingleUser(attn_out, kLiteRtOpCodeTflReshape);
  if (!first_reshape) return std::nullopt;
  auto first_out = SingleOutput(*first_reshape);
  if (!first_out) return std::nullopt;

  auto transpose = SingleUser(*first_out, kLiteRtOpCodeTflTranspose);
  if (!transpose || !HasPerm(*transpose, {0, 2, 1, 3})) {
    return OutputTailMatch{*first_reshape};
  }

  auto transpose_out = SingleOutput(*transpose);
  if (!transpose_out) return std::nullopt;
  auto final_reshape =
      SingleUser(*transpose_out, kLiteRtOpCodeTflReshape);
  if (!final_reshape) return std::nullopt;
  auto final_out = SingleOutput(*final_reshape);
  if (!final_out) return std::nullopt;
  return OutputTailMatch{*final_reshape, first_reshape->Get(),
                         transpose->Get()};
}

struct GqaMatch {
  struct Operands {
    Tensor q;
    bool q_has_unsplit_heads = false;
    Tensor k_cache;
    Tensor k_slice;
    Tensor v_cache;
    Tensor v_slice;
    Tensor mask;
  } operands;

  struct ReferenceTensors {
    Tensor qk_cache;
    Tensor qk_slice;
    Tensor qk_concat;
    Tensor attention_logits;
    Tensor attention_weights;
    Tensor p_cache;
    Tensor p_slice;
    Tensor pv_cache;
    Tensor pv_slice;
    Tensor attention_output;
  } refs;

  struct Dimensions {
    int32_t sequence_length = 1;
    int32_t q_heads = 0;
    int32_t kv_heads = 0;
    int32_t group_size = 0;
    int32_t head_dim = 0;
    int32_t cache_length = 0;
    int32_t slice_length = 0;

    int32_t ContextLength() const {
      return cache_length + slice_length;
    }
  } dims;

  struct Options {
    float softmax_beta = 1.0f;
  } options;

  Op final_reshape;
  Tensor final_output;
  std::vector<LiteRtOp> orphans;
};

struct SoftmaxMatch {
  Op op;
  float beta;
};

std::optional<SoftmaxMatch> MatchSoftmax(const LiteRtCompilerContext* ctx,
                                         LiteRtOp root_ptr) {
  Op softmax(ctx, root_ptr);
  if (!Match(softmax, m_Op<kLiteRtOpCodeTflSoftmax>(m_Any())) ||
      softmax.Outputs().size() != 1) {
    return std::nullopt;
  }

  auto softmax_opts =
      GetOptionsAs<SoftmaxOptions>(softmax.ctx(), softmax.Get());
  if (!softmax_opts) return std::nullopt;
  return SoftmaxMatch{softmax, softmax_opts->beta};
}

struct ValuePathMatch {
  GqaMatch::Dimensions dims;
  SliceBatchMatmulMatch cache_branch;
  SliceBatchMatmulMatch slice_branch;
  Tensor attention_output;
  OutputTailMatch output_tail;
  std::vector<LiteRtOp> orphans;
};

std::optional<ValuePathMatch> MatchValuePath(const Tensor& attention_weights) {
  auto softmax_uses = attention_weights.Uses();
  if (softmax_uses.size() != 2) return std::nullopt;
  auto v_branch0 = MatchSliceBatchMatmul(softmax_uses[0]);
  auto v_branch1 = MatchSliceBatchMatmul(softmax_uses[1]);
  if (!v_branch0 || !v_branch1) return std::nullopt;
  if (v_branch0->slice_begin != 0 && v_branch1->slice_begin == 0) {
    std::swap(v_branch0, v_branch1);
  }
  if (v_branch0->slice_begin != 0) return std::nullopt;

  const Tensor p_cache = v_branch0->slice.Outputs()[0];
  const Tensor p_slice = v_branch1->slice.Outputs()[0];
  auto p_cache_type = p_cache.RankedTensorType();
  auto p_slice_type = p_slice.RankedTensorType();
  if (!p_cache_type || !p_slice_type) {
    return std::nullopt;
  }
  const auto p_cache_dims = p_cache_type->Layout().Dimensions();
  const auto p_slice_dims = p_slice_type->Layout().Dimensions();
  if (p_cache_dims.size() != 4 || p_slice_dims.size() != 4) {
    return std::nullopt;
  }
  GqaMatch::Dimensions dims;
  dims.cache_length = p_cache_dims.back();
  dims.slice_length = p_slice_dims.back();
  if (v_branch1->slice_begin != dims.cache_length || dims.cache_length <= 0 ||
      dims.slice_length <= 0) {
    return std::nullopt;
  }

  const Tensor pv_cache = v_branch0->batch_matmul.Outputs()[0];
  const Tensor pv_slice = v_branch1->batch_matmul.Outputs()[0];
  auto pv_cache_out_uses = pv_cache.Uses();
  auto pv_slice_out_uses = pv_slice.Uses();
  if (pv_cache_out_uses.size() != 1 || pv_slice_out_uses.size() != 1 ||
      pv_cache_out_uses[0].user.Get() != pv_slice_out_uses[0].user.Get()) {
    return std::nullopt;
  }
  Op pv_add = pv_cache_out_uses[0].user;
  auto pv_add_matcher =
      m_AllOf(m_Options<AddOptions>([](const AddOptions& opts) {
                return opts.fused_activation_function == 0;
              }),
              m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()));
  if (!Match(pv_add, pv_add_matcher)) return std::nullopt;
  auto pv_add_out = SingleOutput(pv_add);
  if (!pv_add_out) return std::nullopt;
  const Tensor attention_output = *pv_add_out;
  auto output_tail = MatchOutputTail(attention_output);
  if (!output_tail) return std::nullopt;

  // The final Reshape is erased directly after rebuilding. Register the
  // remaining output/V path from upstream to downstream.
  std::vector<LiteRtOp> orphans;
  orphans.push_back(pv_add.Get());
  orphans.push_back(v_branch0->batch_matmul.Get());
  orphans.push_back(v_branch1->batch_matmul.Get());
  orphans.push_back(v_branch0->slice.Get());
  orphans.push_back(v_branch1->slice.Get());
  if (output_tail->transpose) orphans.push_back(output_tail->transpose);
  if (output_tail->reshape) orphans.push_back(output_tail->reshape);

  const Tensor final_out = output_tail->final_reshape.Outputs()[0];
  auto final_out_type = final_out.RankedTensorType();
  if (!final_out_type) return std::nullopt;
  const auto final_out_dims = final_out_type->Layout().Dimensions();
  int32_t hidden_size = 0;
  if (final_out_dims.size() == 3 && final_out_dims[0] == 1) {
    dims.sequence_length = final_out_dims[1];
    hidden_size = final_out_dims[2];
  } else if (final_out_dims.size() == 2 && final_out_dims[0] == 1) {
    dims.sequence_length = 1;
    hidden_size = final_out_dims[1];
  } else {
    return std::nullopt;
  }

  auto attn_type = attention_output.RankedTensorType();
  if (!attn_type) return std::nullopt;
  const auto attn_dims = attn_type->Layout().Dimensions();
  if (attn_dims.size() != 4 || attn_dims[0] != 1) return std::nullopt;
  dims.kv_heads = attn_dims[1];
  const int32_t grouped_sequence_length = attn_dims[2];
  dims.head_dim = attn_dims[3];
  if (dims.sequence_length <= 0 || dims.kv_heads <= 0 || dims.head_dim <= 0 ||
      grouped_sequence_length <= 0 ||
      grouped_sequence_length % dims.sequence_length != 0) {
    return std::nullopt;
  }
  dims.group_size = grouped_sequence_length / dims.sequence_length;
  dims.q_heads = dims.kv_heads * dims.group_size;
  if (dims.q_heads <= 1 || dims.q_heads * dims.head_dim != hidden_size) {
    return std::nullopt;
  }

  if (!HasDims(v_branch0->batch_matmul.Inputs()[1],
               {1, dims.kv_heads, dims.cache_length, dims.head_dim}) ||
      !HasDims(v_branch1->batch_matmul.Inputs()[1],
               {1, dims.kv_heads, dims.slice_length, dims.head_dim})) {
    return std::nullopt;
  }

  return ValuePathMatch{dims,         *v_branch0,
                        *v_branch1,   attention_output,
                        *output_tail, std::move(orphans)};
}

struct QkPathMatch {
  Tensor q;
  bool q_has_unsplit_heads;
  Tensor k_cache;
  Tensor k_slice;
  Tensor mask;
  Tensor qk_cache;
  Tensor qk_slice;
  Tensor qk_concat;
  std::vector<LiteRtOp> orphans;
};

std::optional<QkPathMatch> MatchQkPath(const Tensor& attention_logits,
                                       const GqaMatch::Dimensions& dims) {
  std::vector<LiteRtOp> orphans;
  if (attention_logits.Uses().size() != 1) return std::nullopt;

  auto mask_add = MatchPlainAdd(attention_logits);
  if (!mask_add) return std::nullopt;
  orphans.push_back(mask_add->Get());

  const Tensor lhs_input = mask_add->Inputs()[0];
  const Tensor rhs_input = mask_add->Inputs()[1];

  const int32_t grouped_sequence_length =
      dims.group_size * dims.sequence_length;
  const std::vector<int32_t> qk_concat_shape = {
      1, dims.kv_heads, grouped_sequence_length, dims.ContextLength()};
  auto lhs_qk_concat = MatchQkConcat(lhs_input, qk_concat_shape);
  auto rhs_qk_concat = MatchQkConcat(rhs_input, qk_concat_shape);
  if (lhs_qk_concat.has_value() == rhs_qk_concat.has_value()) {
    return std::nullopt;
  }
  std::optional<Op> qk_concat;
  Tensor qk_concat_out(attention_logits.ctx(), nullptr);
  Tensor mask_in(attention_logits.ctx(), nullptr);
  if (lhs_qk_concat.has_value()) {
    qk_concat = std::move(lhs_qk_concat);
    qk_concat_out = lhs_input;
    mask_in = rhs_input;
  } else {
    qk_concat = std::move(rhs_qk_concat);
    qk_concat_out = rhs_input;
    mask_in = lhs_input;
  }

  if (auto repeated_mask = MatchRepeatedMaskConcat(
          mask_in, dims.group_size, grouped_sequence_length,
          dims.sequence_length, dims.ContextLength())) {
    orphans.push_back(repeated_mask->Get());
    mask_in = repeated_mask->Inputs()[0];
  }
  if (!HasDims(mask_in,
               {1, 1, dims.sequence_length, dims.ContextLength()}) &&
      !HasDims(mask_in, {1, 1, 1, dims.ContextLength()})) {
    return std::nullopt;
  }

  orphans.push_back(qk_concat->Get());

  const Tensor k_out0 = qk_concat->Inputs()[0];
  const Tensor k_out1 = qk_concat->Inputs()[1];
  if (k_out0.Uses().size() != 1 || k_out1.Uses().size() != 1) {
    return std::nullopt;
  }
  auto batch_matmul_k0 = MatchPlainBatchMatmul(k_out0);
  auto batch_matmul_k1 = MatchPlainBatchMatmul(k_out1);
  if (!batch_matmul_k0 || !batch_matmul_k1) return std::nullopt;
  orphans.push_back(batch_matmul_k0->Get());
  orphans.push_back(batch_matmul_k1->Get());

  const Tensor q_grouped = batch_matmul_k0->Inputs()[0];
  const Tensor k_cache = batch_matmul_k0->Inputs()[1];
  const Tensor k_slice = batch_matmul_k1->Inputs()[1];
  if (q_grouped != batch_matmul_k1->Inputs()[0]) return std::nullopt;

  if (!HasDims(q_grouped, {1, dims.kv_heads, grouped_sequence_length,
                           dims.head_dim}) ||
      !HasDims(k_cache,
               {1, dims.kv_heads, dims.head_dim, dims.cache_length}) ||
      !HasDims(k_slice,
               {1, dims.kv_heads, dims.head_dim, dims.slice_length})) {
    return std::nullopt;
  }

  // Prefer the original per-query-head Q tensor when available, avoiding an
  // extra prefill reshape while still supporting already-grouped inputs.
  Tensor q = q_grouped;
  bool q_has_unsplit_heads = false;
  if (auto reshape = MatchReshape(q_grouped, 2);
      reshape && HasDims(reshape->Inputs()[0],
                         {1, dims.q_heads, dims.sequence_length,
                          dims.head_dim})) {
    q = reshape->Inputs()[0];
    q_has_unsplit_heads = true;
    orphans.push_back(reshape->Get());
  }

  return QkPathMatch{q,
                     q_has_unsplit_heads,
                     k_cache,
                     k_slice,
                     mask_in,
                     k_out0,
                     k_out1,
                     qk_concat_out,
                     std::move(orphans)};
}

std::vector<LiteRtOp> CollectOrphans(const SoftmaxMatch& softmax,
                                     const ValuePathMatch& value_path,
                                     const QkPathMatch& qk_path) {
  std::vector<LiteRtOp> orphans;
  orphans.reserve(1 + value_path.orphans.size() + qk_path.orphans.size());
  orphans.push_back(softmax.op.Get());
  orphans.insert(orphans.end(), value_path.orphans.begin(),
                 value_path.orphans.end());
  orphans.insert(orphans.end(), qk_path.orphans.begin(),
                 qk_path.orphans.end());
  return orphans;
}

GqaMatch AssembleGqaMatch(const SoftmaxMatch& softmax,
                          const ValuePathMatch& value_path,
                          const QkPathMatch& qk_path) {
  GqaMatch m;
  m.operands = {qk_path.q,
                qk_path.q_has_unsplit_heads,
                qk_path.k_cache,
                qk_path.k_slice,
                value_path.cache_branch.batch_matmul.Inputs()[1],
                value_path.slice_branch.batch_matmul.Inputs()[1],
                qk_path.mask};
  m.refs = {qk_path.qk_cache,
            qk_path.qk_slice,
            qk_path.qk_concat,
            softmax.op.Inputs()[0],
            softmax.op.Outputs()[0],
            value_path.cache_branch.slice.Outputs()[0],
            value_path.slice_branch.slice.Outputs()[0],
            value_path.cache_branch.batch_matmul.Outputs()[0],
            value_path.slice_branch.batch_matmul.Outputs()[0],
            value_path.attention_output};
  m.dims = value_path.dims;
  m.options = {softmax.beta};
  m.final_reshape = value_path.output_tail.final_reshape;
  m.final_output = value_path.output_tail.final_reshape.Outputs()[0];
  m.orphans = CollectOrphans(softmax, value_path, qk_path);
  return m;
}

std::optional<GqaMatch> MatchGqa(const LiteRtCompilerContext* ctx,
                                 LiteRtOp root_ptr) {
  auto softmax = MatchSoftmax(ctx, root_ptr);
  if (!softmax) return std::nullopt;
  auto value_path = MatchValuePath(softmax->op.Outputs()[0]);
  if (!value_path) return std::nullopt;
  auto qk_path = MatchQkPath(softmax->op.Inputs()[0], value_path->dims);
  if (!qk_path) return std::nullopt;
  return AssembleGqaMatch(*softmax, *value_path, *qk_path);
}

Expected<std::vector<Tensor>> BuildSplitHeadsAtAxis1(
    Builder& builder, const Tensor& in_4d, int32_t num_heads) {
  LITERT_ASSIGN_OR_RETURN(auto input_type, in_4d.RankedTensorType());
  const auto input_dims = input_type.Layout().Dimensions();
  if (input_dims.size() != 4 || num_heads <= 0 ||
      input_dims[1] != num_heads) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
  if (num_heads == 1) {
    return std::vector<Tensor>{in_4d};
  }

  std::vector<int32_t> per_head_dims(input_dims.begin(), input_dims.end());
  per_head_dims[1] = 1;
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

Expected<void> BuildGqaPerQueryHead(Builder& builder, const GqaMatch& m) {
  Tensor q_unsplit = m.operands.q;
  if (!m.operands.q_has_unsplit_heads) {
    LITERT_ASSIGN_OR_RETURN(
        q_unsplit, BuildReshape(builder, m.operands.q, m.operands.q,
                                {1, m.dims.q_heads, m.dims.sequence_length,
                                 m.dims.head_dim}));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto q_heads,
      BuildSplitHeadsAtAxis1(builder, q_unsplit, m.dims.q_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto k_cache_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.k_cache, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto k_slice_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.k_slice, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto v_cache_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.v_cache, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto v_slice_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.v_slice, m.dims.kv_heads));

  std::vector<Tensor> sha_outs;
  sha_outs.reserve(m.dims.q_heads);
  for (int32_t h_kv = 0; h_kv < m.dims.kv_heads; ++h_kv) {
    for (int32_t g = 0; g < m.dims.group_size; ++g) {
      const int32_t q_idx = h_kv * m.dims.group_size + g;
      LITERT_ASSIGN_OR_RETURN(
          auto qk_cache,
          BuildBatchMatmul(builder, q_heads[q_idx], k_cache_heads[h_kv],
                           m.refs.qk_cache,
                           {1, 1, m.dims.sequence_length,
                            m.dims.cache_length}));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_slice,
          BuildBatchMatmul(
              builder, q_heads[q_idx], k_slice_heads[h_kv],
              m.refs.qk_slice,
              {1, 1, m.dims.sequence_length, m.dims.slice_length}));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_concat,
          BuildConcat(builder, {qk_cache, qk_slice}, /*axis=*/3,
                      m.refs.qk_concat,
                      {1, 1, m.dims.sequence_length,
                       m.dims.ContextLength()}));
      LITERT_ASSIGN_OR_RETURN(
          auto qk_masked,
          BuildAdd(builder, qk_concat, m.operands.mask,
                   m.refs.attention_logits,
                   {1, 1, m.dims.sequence_length,
                    m.dims.ContextLength()}));
      LITERT_ASSIGN_OR_RETURN(
          auto p,
          BuildSoftmax(builder, qk_masked, m.options.softmax_beta,
                       m.refs.attention_weights,
                       {1, 1, m.dims.sequence_length,
                        m.dims.ContextLength()}));
      LITERT_ASSIGN_OR_RETURN(
          auto p_cache,
          BuildSlice(builder, p, {0, 0, 0, 0},
                     {1, 1, m.dims.sequence_length, m.dims.cache_length},
                     m.refs.p_cache));
      LITERT_ASSIGN_OR_RETURN(
          auto p_slice,
          BuildSlice(builder, p, {0, 0, 0, m.dims.cache_length},
                     {1, 1, m.dims.sequence_length, m.dims.slice_length},
                     m.refs.p_slice));
      LITERT_ASSIGN_OR_RETURN(
          auto out_cache,
          BuildBatchMatmul(builder, p_cache, v_cache_heads[h_kv],
                           m.refs.pv_cache,
                           {1, 1, m.dims.sequence_length, m.dims.head_dim}));
      LITERT_ASSIGN_OR_RETURN(
          auto out_slice,
          BuildBatchMatmul(builder, p_slice, v_slice_heads[h_kv],
                           m.refs.pv_slice,
                           {1, 1, m.dims.sequence_length, m.dims.head_dim}));
      LITERT_ASSIGN_OR_RETURN(
          auto head_out,
          BuildAdd(builder, out_cache, out_slice,
                   m.refs.attention_output,
                   {1, 1, m.dims.sequence_length, m.dims.head_dim}));
      sha_outs.push_back(head_out);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto concat_heads,
      BuildConcat(builder, sha_outs, /*axis=*/3, m.refs.attention_output,
                  {1, 1, m.dims.sequence_length,
                   m.dims.q_heads * m.dims.head_dim}));
  LITERT_ASSIGN_OR_RETURN(auto final_out_type,
                          m.final_output.RankedTensorType());
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, concat_heads,
                                          final_out_type.Layout().Dimensions(),
                                          m.final_output));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(m.final_reshape));
  RegisterOrphanGroup(m.orphans);
  return {};
}

// Decode stacks G query rows per KV head, reducing attention branches from
// H_q to H_kv while preserving independent row-wise attention.
Expected<void> BuildGqaPerKvHeadForDecode(Builder& builder,
                                          const GqaMatch& m) {
  LITERT_ASSIGN_OR_RETURN(
      auto q_grouped,
      BuildReshape(builder, m.operands.q, m.operands.q,
                   {1, m.dims.kv_heads, m.dims.group_size, m.dims.head_dim}));

  LITERT_ASSIGN_OR_RETURN(
      auto q_groups,
      BuildSplitHeadsAtAxis1(builder, q_grouped, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto k_cache_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.k_cache, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto k_slice_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.k_slice, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto v_cache_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.v_cache, m.dims.kv_heads));
  LITERT_ASSIGN_OR_RETURN(
      auto v_slice_heads,
      BuildSplitHeadsAtAxis1(builder, m.operands.v_slice, m.dims.kv_heads));

  // The mask has a single Q row; broadcast it over the G stacked heads.
  LITERT_ASSIGN_OR_RETURN(
      auto mask_rows, BuildReshape(builder, m.operands.mask, m.operands.mask,
                                   {1, 1, 1,
                                    m.dims.ContextLength()}));
  std::vector<Tensor> mask_copies(m.dims.group_size, mask_rows);
  LITERT_ASSIGN_OR_RETURN(
      auto mask_g,
      BuildConcat(builder, mask_copies, /*axis=*/2, m.operands.mask,
                  {1, 1, m.dims.group_size,
                   m.dims.ContextLength()}));

  std::vector<Tensor> group_outs;
  group_outs.reserve(m.dims.kv_heads);
  for (int32_t h_kv = 0; h_kv < m.dims.kv_heads; ++h_kv) {
    LITERT_ASSIGN_OR_RETURN(
        auto qk_cache,
        BuildBatchMatmul(builder, q_groups[h_kv], k_cache_heads[h_kv],
                         m.refs.qk_cache,
                         {1, 1, m.dims.group_size, m.dims.cache_length}));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_slice,
        BuildBatchMatmul(builder, q_groups[h_kv], k_slice_heads[h_kv],
                         m.refs.qk_slice,
                         {1, 1, m.dims.group_size, m.dims.slice_length}));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_concat,
        BuildConcat(builder, {qk_cache, qk_slice}, /*axis=*/3,
                    m.refs.qk_concat,
                    {1, 1, m.dims.group_size,
                     m.dims.ContextLength()}));
    LITERT_ASSIGN_OR_RETURN(
        auto qk_masked,
        BuildAdd(builder, qk_concat, mask_g, m.refs.attention_logits,
                 {1, 1, m.dims.group_size,
                  m.dims.ContextLength()}));
    LITERT_ASSIGN_OR_RETURN(
        auto p,
        BuildSoftmax(builder, qk_masked, m.options.softmax_beta,
                     m.refs.attention_weights,
                     {1, 1, m.dims.group_size,
                      m.dims.ContextLength()}));
    LITERT_ASSIGN_OR_RETURN(
        auto p_cache, BuildSlice(builder, p, {0, 0, 0, 0},
                                 {1, 1, m.dims.group_size, m.dims.cache_length},
                                 m.refs.p_cache));
    LITERT_ASSIGN_OR_RETURN(
        auto p_slice,
        BuildSlice(builder, p, {0, 0, 0, m.dims.cache_length},
                   {1, 1, m.dims.group_size, m.dims.slice_length},
                   m.refs.p_slice));
    LITERT_ASSIGN_OR_RETURN(
        auto out_cache,
        BuildBatchMatmul(builder, p_cache, v_cache_heads[h_kv],
                         m.refs.pv_cache,
                         {1, 1, m.dims.group_size, m.dims.head_dim}));
    LITERT_ASSIGN_OR_RETURN(
        auto out_slice,
        BuildBatchMatmul(builder, p_slice, v_slice_heads[h_kv],
                         m.refs.pv_slice,
                         {1, 1, m.dims.group_size, m.dims.head_dim}));
    LITERT_ASSIGN_OR_RETURN(
        auto group_out,
        BuildAdd(builder, out_cache, out_slice, m.refs.attention_output,
                 {1, 1, m.dims.group_size, m.dims.head_dim}));
    group_outs.push_back(group_out);
  }

  // [1, H_kv, G, D] is laid out as Q head (h_kv * G + g) major, which
  // matches the [1, 1, H_q * D] attention output.
  LITERT_ASSIGN_OR_RETURN(
      auto concat_groups,
      BuildConcat(builder, group_outs, /*axis=*/1, m.refs.attention_output,
                  {1, m.dims.kv_heads, m.dims.group_size, m.dims.head_dim}));
  LITERT_ASSIGN_OR_RETURN(auto final_out_type,
                          m.final_output.RankedTensorType());
  LITERT_RETURN_IF_ERROR(BuildReshapeInto(builder, concat_groups,
                                          final_out_type.Layout().Dimensions(),
                                          m.final_output));

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
  auto status = (match->dims.sequence_length == 1 && match->dims.group_size > 1)
                    ? BuildGqaPerKvHeadForDecode(builder, *match)
                    : BuildGqaPerQueryHead(builder, *match);
  if (!status) {
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
