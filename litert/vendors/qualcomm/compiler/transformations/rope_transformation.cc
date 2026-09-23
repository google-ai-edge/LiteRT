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
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

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
using litert::compiler::m_HasUsers;
using litert::compiler::m_Op;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

namespace {

constexpr int32_t kBatchSize = 1;
constexpr int32_t kSeqLen = 630;
constexpr int32_t kNumHeads = 12;
constexpr int32_t kHeadDim = 64;
constexpr int32_t kQuarterHeadDim = 16;
constexpr int32_t kHalfHeadDim = 32;

static LiteRtTensor g_cos_12 = nullptr;
static LiteRtTensor g_sin_signed_12 = nullptr;

Expected<Tensor> CreateFloat32Tensor(Builder& builder,
                                     absl::Span<const int32_t> dims) {
  RankedTensorType type(
      ElementType::Float32,
      Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  auto tensor_res = builder.BuildTensor(RankedTensorSpecBuilder(type).Build());
  if (!tensor_res) {
    return tensor_res.Error();
  }
  return *tensor_res;
}

Expected<Op> BuildConcatOp(Builder& builder, const std::vector<Tensor>& inputs,
                           Tensor& output, int32_t axis) {
  auto concat_res =
      builder.BuildOp(kLiteRtOpCodeTflConcatenation, inputs, {output});
  if (!concat_res) {
    return concat_res.Error();
  }
  ConcatenationOptions concat_options;
  concat_options.axis = axis;
  concat_options.fused_activation_function = 0;
  auto opt_status =
      builder.SetOpOptions(*concat_res, std::move(concat_options));
  if (!opt_status) {
    return opt_status.Error();
  }
  return *concat_res;
}

Expected<Op> BuildMulOp(Builder& builder, Tensor in0, Tensor in1, Tensor out) {
  auto mul_res = builder.BuildOp(kLiteRtOpCodeTflMul, {in0, in1}, {out});
  if (!mul_res) {
    return mul_res.Error();
  }
  MulOptions mul_options;
  mul_options.fused_activation_function = 0;
  auto opt_status = builder.SetOpOptions(*mul_res, std::move(mul_options));
  if (!opt_status) {
    return opt_status.Error();
  }
  return *mul_res;
}

Expected<Op> BuildAddOp(Builder& builder, Tensor in0, Tensor in1, Tensor out) {
  auto add_res = builder.BuildOp(kLiteRtOpCodeTflAdd, {in0, in1}, {out});
  if (!add_res) {
    return add_res.Error();
  }
  AddOptions add_options;
  add_options.fused_activation_function = 0;
  builder.SetOpOptions(*add_res, std::move(add_options));
  return *add_res;
}

Expected<Op> BuildNegOp(Builder& builder, Tensor in, Tensor out) {
  auto neg_res = builder.BuildOp(kLiteRtOpCodeTflNeg, {in}, {out});
  if (!neg_res) {
    return neg_res.Error();
  }
  return *neg_res;
}

}  // namespace

extern "C" {

void ResetRopeTransformationState() {
  g_cos_12 = nullptr;
  g_sin_signed_12 = nullptr;
}

LiteRtStatus RopeAttentionLayerTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Tensor s0(context, nullptr);
  Tensor s1(context, nullptr);
  Tensor s2(context, nullptr);
  Tensor s3(context, nullptr);
  Tensor cos2(context, nullptr);
  Tensor sin2(context, nullptr);
  Tensor cos3(context, nullptr);
  Tensor sin3(context, nullptr);

  auto slice_m = [](Tensor* t) {
    return m_CaptureOrSameAs(
        t, m_Shape({kBatchSize, kSeqLen, kNumHeads, kQuarterHeadDim}));
  };
  auto trig_m = [](Tensor* t) {
    return m_CaptureOrSameAs(
        t, m_Shape({kBatchSize, kSeqLen, 1, kQuarterHeadDim}));
  };
  auto mul_m = [&](Tensor* slice, Tensor* trig) {
    return m_CommutativeOp<kLiteRtOpCodeTflMul>(slice_m(slice), trig_m(trig));
  };
  auto sub_m = [&](Tensor* s_a, Tensor* trig_a, Tensor* s_b, Tensor* trig_b) {
    return m_Op<kLiteRtOpCodeTflSub>(mul_m(s_a, trig_a), mul_m(s_b, trig_b));
  };
  auto add_m = []() { return m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()); };
  auto concat_pair_m = [](auto&& sub_branch, auto&& add_branch) {
    return m_AnyOf(m_Op<kLiteRtOpCodeTflConcatenation>(sub_branch, add_branch),
                   m_Op<kLiteRtOpCodeTflConcatenation>(add_branch, sub_branch));
  };

  auto root_pattern = m_Op<kLiteRtOpCodeTflConcatenation>(
      concat_pair_m(sub_m(&s0, &cos2, &s1, &sin2), add_m()),
      concat_pair_m(sub_m(&s2, &cos3, &s3, &sin3), add_m()));

  if (!Match(root_op, root_pattern)) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor out_tensor = root_op.Outputs()[0];
  if (!Match(out_tensor,
             m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                     m_Shape({kBatchSize, kSeqLen, kNumHeads, kHeadDim})))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Trace in_tensor from s0
  Tensor in_tensor(context, nullptr);
  auto in_matcher = m_CaptureOrSameAs(
      &in_tensor, m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                          m_Shape({kBatchSize, kSeqLen, kNumHeads, kHeadDim})));
  auto slice_chain_matcher = m_Op<kLiteRtOpCodeTflSlice>(
      m_Op<kLiteRtOpCodeTflSlice>(in_matcher, m_Any(), m_Any()), m_Any(),
      m_Any());
  if (!Match(s0, slice_chain_matcher)) {
    return kLiteRtStatusPatternNoMatch;
  }

  // 8. Prepare global trig tensors lazily on first block
  bool is_first_block = (g_cos_12 == nullptr);
  if (is_first_block) {
    // neg_sin2
    auto neg_sin2_t =
        CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, 1, kQuarterHeadDim});
    if (!neg_sin2_t) return neg_sin2_t.Error().Status();
    auto neg_sin2_op = BuildNegOp(builder, sin2, *neg_sin2_t);
    if (!neg_sin2_op) return neg_sin2_op.Error().Status();

    // neg_sin3
    auto neg_sin3_t =
        CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, 1, kQuarterHeadDim});
    if (!neg_sin3_t) return neg_sin3_t.Error().Status();
    auto neg_sin3_op = BuildNegOp(builder, sin3, *neg_sin3_t);
    if (!neg_sin3_op) return neg_sin3_op.Error().Status();

    // cos_64: Concat({cos2, cos2, cos3, cos3}, axis=3)
    auto cos_64_t =
        CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, 1, kHeadDim});
    if (!cos_64_t) return cos_64_t.Error().Status();
    auto cos_64_op =
        BuildConcatOp(builder, {cos2, cos2, cos3, cos3}, *cos_64_t, /*axis=*/3);
    if (!cos_64_op) return cos_64_op.Error().Status();

    // cos_12: Concat(12 x cos_64, axis=2)
    auto cos_12_t = CreateFloat32Tensor(
        builder, {kBatchSize, kSeqLen, kNumHeads, kHeadDim});
    if (!cos_12_t) return cos_12_t.Error().Status();
    std::vector<Tensor> cos_12_ins(kNumHeads, *cos_64_t);
    auto cos_12_op = BuildConcatOp(builder, cos_12_ins, *cos_12_t, /*axis=*/2);
    if (!cos_12_op) return cos_12_op.Error().Status();

    // sin_signed_64: Concat({neg_sin2, sin2, neg_sin3, sin3}, axis=3)
    auto sin_signed_64_t =
        CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, 1, kHeadDim});
    if (!sin_signed_64_t) return sin_signed_64_t.Error().Status();
    auto sin_signed_64_op = BuildConcatOp(
        builder, {*neg_sin2_t, sin2, *neg_sin3_t, sin3}, *sin_signed_64_t,
        /*axis=*/3);
    if (!sin_signed_64_op) return sin_signed_64_op.Error().Status();

    // sin_signed_12: Concat(12 x sin_signed_64, axis=2)
    auto sin_signed_12_t = CreateFloat32Tensor(
        builder, {kBatchSize, kSeqLen, kNumHeads, kHeadDim});
    if (!sin_signed_12_t) return sin_signed_12_t.Error().Status();
    std::vector<Tensor> sin_12_ins(kNumHeads, *sin_signed_64_t);
    auto sin_signed_12_op =
        BuildConcatOp(builder, sin_12_ins, *sin_signed_12_t, /*axis=*/2);
    if (!sin_signed_12_op) return sin_signed_12_op.Error().Status();

    g_cos_12 = cos_12_t->Get();
    g_sin_signed_12 = sin_signed_12_t->Get();
  }

  Tensor cos_12_tensor(context, g_cos_12);
  Tensor sin_signed_12_tensor(context, g_sin_signed_12);

  // 9. Build in_rot: Concat({s1, s0, s3, s2}, axis=3)
  auto in_rot_t =
      CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, kNumHeads, kHeadDim});
  if (!in_rot_t) return in_rot_t.Error().Status();
  auto in_rot_op =
      BuildConcatOp(builder, {s1, s0, s3, s2}, *in_rot_t, /*axis=*/3);
  if (!in_rot_op) return in_rot_op.Error().Status();

  // 10. Build term1: Mul(in_tensor, cos_12)
  auto term1_t =
      CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, kNumHeads, kHeadDim});
  if (!term1_t) return term1_t.Error().Status();
  auto term1_op = BuildMulOp(builder, in_tensor, cos_12_tensor, *term1_t);
  if (!term1_op) return term1_op.Error().Status();

  // 11. Build term2: Mul(in_rot, sin_signed_12)
  auto term2_t =
      CreateFloat32Tensor(builder, {kBatchSize, kSeqLen, kNumHeads, kHeadDim});
  if (!term2_t) return term2_t.Error().Status();
  auto term2_op =
      BuildMulOp(builder, *in_rot_t, sin_signed_12_tensor, *term2_t);
  if (!term2_op) return term2_op.Error().Status();

  // 12. Build out: Add(term1, term2) -> out_tensor
  auto add_op = BuildAddOp(builder, *term1_t, *term2_t, out_tensor);
  if (!add_op) return add_op.Error().Status();

  // 13. Erase obsolete root op.
  // Crucial: Only erase root_op here so that builder.ApplyChanges splices the
  // new ops precisely at root_op's position, strictly after all input slices
  // (s0, s1, s2, s3). The orphaned child ops (concat_lo, concat_hi, sub0, add0,
  // sub1, add1, and muls) are safely eliminated by RopeCleanupTransformation.
  builder.EraseOp(root_op);

  return kLiteRtStatusOk;
}

LiteRtStatus RopeCleanupTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr,
                                       LiteRtOp op_ptr) {
  Op op(context, op_ptr);
  if (op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor out_tensor = op.Outputs()[0];
  if (!Match(out_tensor,
             m_AllOf(m_ElementType(kLiteRtElementTypeFloat32),
                     m_Shape({kBatchSize, kSeqLen, kNumHeads, kHalfHeadDim}),
                     m_HasUsers(0)))) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op sub_op(context, nullptr);
  Op add_op(context, nullptr);
  auto cleanup_pattern =
      m_AnyOf(m_Op<kLiteRtOpCodeTflConcatenation>(
                  m_CaptureOrSameAs(
                      &sub_op, m_Op<kLiteRtOpCodeTflSub>(m_Any(), m_Any())),
                  m_CaptureOrSameAs(
                      &add_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any()))),
              m_Op<kLiteRtOpCodeTflConcatenation>(
                  m_CaptureOrSameAs(
                      &add_op, m_Op<kLiteRtOpCodeTflAdd>(m_Any(), m_Any())),
                  m_CaptureOrSameAs(
                      &sub_op, m_Op<kLiteRtOpCodeTflSub>(m_Any(), m_Any()))));

  if (!Match(op, cleanup_pattern)) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  builder.EraseOp(op);
  builder.EraseOp(sub_op);
  builder.EraseOp(add_op);

  for (const auto& in : sub_op.Inputs()) {
    if (auto def_op = in.GetDefiningOp()) {
      if (def_op->Code() == kLiteRtOpCodeTflMul) {
        builder.EraseOp(*def_op);
      }
    }
  }
  for (const auto& in : add_op.Inputs()) {
    if (auto def_op = in.GetDefiningOp()) {
      if (def_op->Code() == kLiteRtOpCodeTflMul) {
        builder.EraseOp(*def_op);
      }
    }
  }

  return kLiteRtStatusOk;
}

}  // extern "C"
