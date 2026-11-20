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

#include "litert/vendors/mediatek/compiler/transformations/attention_mask_transformation.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
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

using litert::BuildLayout;
using litert::ElementType;
using litert::Expected;
using litert::Layout;
using litert::RankedTensorType;
using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::m_AllOf;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_ConstantValue;
using litert::compiler::m_ElementType;
using litert::compiler::m_HasOneUse;
using litert::compiler::m_Op;
using litert::compiler::m_Predicate;
using litert::compiler::m_Rank;
using litert::compiler::m_Shape;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReshapeOptions;
using litert::compiler::SubOptions;
using litert::compiler::Tensor;

namespace {

// Matches an op with exactly one output tensor that satisfies `m`.
template <typename M>
auto HasSingleOutput(M m) {
  return m_Predicate<Op>(
      [m = std::move(m)](const Op& op) {
        auto outs = op.Outputs();
        return outs.size() == 1 && m.Match(outs[0]);
      },
      "HasSingleOutput");
}

auto IsF32() { return m_ElementType(kLiteRtElementTypeFloat32); }

template <typename T>
Expected<Tensor> BuildConst(Builder& builder, ElementType type,
                            const std::vector<int32_t>& dims,
                            const std::vector<T>& values) {
  RankedTensorType t_type(
      type, Layout(BuildLayout(dims.data(), dims.data() + dims.size())));
  LITERT_ASSIGN_OR_RETURN(
      auto t, builder.BuildTensor(RankedTensorSpecBuilder(t_type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildWeights<T>(absl::MakeConstSpan(values), t));
  return t;
}

}  // namespace

extern "C" {

LiteRtStatus AttentionMaskTransformation(const LiteRtCompilerContext* context,
                                         LiteRtBuilder builder_ptr,
                                         LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op current_op(context, op);

  // Pattern 1: Rewrite rank-mismatched outer product Mul to BatchMatMul.
  // MDLA rejects elementwise Mul between [1, S] and [1, S, 1].
  Tensor mask_2d(context, nullptr);
  Tensor col_mask_3d(context, nullptr);
  auto is_square_static = m_Predicate<Tensor>(
      [](const Tensor& t) {
        auto type = t.RankedTensorType();
        const auto& dims = type->Layout().Dimensions();
        return dims[1] > 0 && dims[1] == dims[2];
      },
      "SquareStatic");
  auto outer_product =
      m_AllOf(m_CommutativeOp<kLiteRtOpCodeTflMul>(
                  m_CaptureOrSameAs(&mask_2d, m_AllOf(IsF32(), m_Rank(2))),
                  m_CaptureOrSameAs(&col_mask_3d, m_AllOf(IsF32(), m_Rank(3)))),
              HasSingleOutput(
                  m_AllOf(IsF32(), m_Shape({1, -1, -1}), is_square_static)));

  if (Match(current_op, outer_product)) {
    Tensor mul_out = current_op.Outputs()[0];
    auto mul_out_type = mul_out.RankedTensorType();
    const int32_t S = mul_out_type->Layout().Dimensions()[1];
    if (!Match(mask_2d, m_Shape({1, S})) ||
        !Match(col_mask_3d, m_Shape({1, S, 1}))) {
      return kLiteRtStatusPatternNoMatch;
    }

    // Build row_mask_3d [1, 1, S] = Reshape(mask_2d [1, S], [1, 1, S]).
    std::vector<int32_t> row_dims = {1, 1, S};
    LITERT_ASSIGN_OR_RETURN(
        auto shape_const,
        BuildConst<int32_t>(builder, ElementType::Int32, {3}, row_dims));

    RankedTensorType row_type(
        ElementType::Float32,
        Layout(
            BuildLayout(row_dims.data(), row_dims.data() + row_dims.size())));
    LITERT_ASSIGN_OR_RETURN(
        auto row_mask_3d,
        builder.BuildTensor(RankedTensorSpecBuilder(row_type).Build()));

    LITERT_ASSIGN_OR_RETURN(
        auto row_reshape,
        builder.BuildOp(kLiteRtOpCodeTflReshape, {mask_2d, shape_const},
                        {row_mask_3d}));
    ReshapeOptions reshape_opts;
    reshape_opts.new_shape = row_dims;
    LITERT_RETURN_IF_ERROR(
        builder.SetOpOptions(row_reshape, std::move(reshape_opts)));

    // Build BatchMatMul(col_mask_3d [1, S, 1], row_mask_3d [1, 1, S])
    // -> mul_out [1, S, S].
    LITERT_ASSIGN_OR_RETURN(
        auto bmm_op, builder.BuildOp(kLiteRtOpCodeTflBatchMatmul,
                                     {col_mask_3d, row_mask_3d}, {mul_out}));
    BatchMatmulOptions bmm_opts;
    bmm_opts.adj_x = false;
    bmm_opts.adj_y = false;
    bmm_opts.asymmetric_quantize_input = false;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(bmm_op, std::move(bmm_opts)));

    builder.EraseOp(current_op);
    LITERT_LOG(LITERT_INFO,
               "AttentionMaskTransformation: rewritten outer-product Mul to "
               "BatchMatMul");
    return kLiteRtStatusOk;
  }

  // Pattern 2: Rewrite Cast(LogicalNot(NotEqual(x, 0.0f))) -> Sub(1.0f, x)
  // NotEqual operand selection prefers a zero RHS (x = LHS); a zero LHS
  // (x = RHS) is only considered when the RHS is not zero.
  Tensor x(context, nullptr);
  Op lnot_op(context, nullptr);
  Op ne_op(context, nullptr);
  auto is_zero = m_ConstantValue<float>(0.0f, "Zero");
  auto ne_x_zero = m_AllOf(
      m_Op<kLiteRtOpCodeTflNotEqual>(m_Any(), is_zero),
      m_Op<kLiteRtOpCodeTflNotEqual>(m_CaptureOrSameAs(&x, IsF32()), m_Any()));
  auto ne_zero_x = m_AllOf(
      m_Op<kLiteRtOpCodeTflNotEqual>(is_zero, m_Not(is_zero)),
      m_Op<kLiteRtOpCodeTflNotEqual>(m_Any(), m_CaptureOrSameAs(&x, IsF32())));
  auto mask_invert = m_AllOf(
      m_Op<kLiteRtOpCodeTflCast>(m_AllOf(
          m_HasOneUse(),
          m_CaptureOrSameAs(
              &lnot_op,
              m_Op<kLiteRtOpCodeTflLogicalNot>(m_AllOf(
                  m_HasOneUse(),
                  m_CaptureOrSameAs(&ne_op, m_AnyOf(ne_x_zero, ne_zero_x))))))),
      HasSingleOutput(IsF32()));

  if (Match(current_op, mask_invert)) {
    // Rewrite: Cast(LogicalNot(NotEqual(x, 0))) -> Sub(1.0f, x)
    LITERT_ASSIGN_OR_RETURN(
        auto one_const,
        BuildConst<float>(builder, ElementType::Float32, {}, {1.0f}));

    Tensor cast_out = current_op.Outputs()[0];
    LITERT_ASSIGN_OR_RETURN(
        auto sub_op,
        builder.BuildOp(kLiteRtOpCodeTflSub, {one_const, x}, {cast_out}));

    SubOptions sub_opts;
    sub_opts.fused_activation_function = 0;
    LITERT_RETURN_IF_ERROR(builder.SetOpOptions(sub_op, std::move(sub_opts)));

    builder.EraseOp(current_op);
    builder.EraseOp(lnot_op);
    builder.EraseOp(ne_op);

    LITERT_LOG(
        LITERT_INFO,
        "AttentionMaskTransformation: replaced Cast->LogicalNot->NotEqual with "
        "Sub(1.0, x)");
    return kLiteRtStatusOk;
  }

  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
