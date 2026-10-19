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

#include "litert/vendors/qualcomm/compiler/transformations/mask_simplification_transformation.h"

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
using litert::compiler::ConcatenationOptions;
using litert::compiler::GetOptionsAs;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

std::vector<int32_t> Dims(const Tensor& t) {
  auto type = t.RankedTensorType();
  if (!type) return {};
  auto dims = type->Layout().Dimensions();
  return std::vector<int32_t>(dims.begin(), dims.end());
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

Expected<void> RewriteBoolCastAdd(Builder& builder, const Op& add_op,
                                  const Op& cast0, const Op& cast1,
                                  const Tensor& b0, const Tensor& b1) {
  const Tensor out = add_op.Outputs()[0];
  const auto out_dims = Dims(out);
  RankedTensorType bool_type(
      ElementType::Bool,
      Layout(BuildLayout(out_dims.data(), out_dims.data() + out_dims.size())));
  LITERT_ASSIGN_OR_RETURN(
      auto or_out,
      builder.BuildTensor(RankedTensorSpecBuilder(bool_type).Build()));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflLogicalOr, {b0, b1}, {or_out}));
  LITERT_RETURN_IF_ERROR(
      builder.BuildOp(kLiteRtOpCodeTflCast, {or_out}, {out}));

  LITERT_RETURN_IF_ERROR(builder.EraseOp(add_op));
  RegisterOrphanGroup({cast0.Get(), cast1.Get()});
  return {};
}

// Simplifies Reshape(in) -> Add(reshaped_in, mask) -> Reshape(out) when
// Dims(in) == Dims(out).
LiteRtStatus TryRewriteReshapeMaskAddReshape(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    const Op& post_reshape) {
  if (post_reshape.Code() != kLiteRtOpCodeTflReshape ||
      post_reshape.Inputs().empty() || post_reshape.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  const Tensor add_out = post_reshape.Inputs()[0];
  if (add_out.Uses().size() != 1) return kLiteRtStatusPatternNoMatch;
  auto add_or = add_out.GetDefiningOp();
  if (!add_or || add_or->Code() != kLiteRtOpCodeTflAdd ||
      add_or->Inputs().size() != 2 || add_or->Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto add_opts = GetOptionsAs<AddOptions>(context, add_or->Get());
  if (!add_opts || add_opts->fused_activation_function != 0) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor pre_out = add_or->Inputs()[0];
  const Tensor mask_in = add_or->Inputs()[1];
  if (pre_out.Uses().size() != 1) return kLiteRtStatusPatternNoMatch;
  auto pre_or = pre_out.GetDefiningOp();
  if (!pre_or || pre_or->Code() != kLiteRtOpCodeTflReshape ||
      pre_or->Inputs().empty()) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor add_input = pre_or->Inputs()[0];
  const Tensor final_out = post_reshape.Outputs()[0];
  const auto input_dims = Dims(add_input);
  const auto out_dims = Dims(final_out);
  auto mask_dims = Dims(mask_in);
  if (input_dims.empty() || input_dims != out_dims ||
      mask_dims.size() != input_dims.size()) {
    return kLiteRtStatusPatternNoMatch;
  }

  int broadcast_axis = -1;
  int32_t broadcast_size = 1;
  for (size_t i = 0; i < mask_dims.size(); ++i) {
    const int32_t m_dim = mask_dims[i];
    const int32_t in_dim = input_dims[i];
    if (m_dim == in_dim || m_dim == 1 || in_dim == 1) continue;
    if (m_dim <= 0 || in_dim % m_dim != 0 || broadcast_axis != -1) {
      return kLiteRtStatusPatternNoMatch;
    }
    broadcast_axis = static_cast<int>(i);
    broadcast_size = in_dim / m_dim;
  }

  Builder builder(context, builder_ptr);
  Tensor effective_mask = mask_in;
  if (broadcast_axis != -1) {
    std::vector<int32_t> new_mask_dims = mask_dims;
    new_mask_dims[broadcast_axis] = input_dims[broadcast_axis];
    auto concat_out = NewLike(builder, mask_in, new_mask_dims);
    if (!concat_out) return concat_out.Error().Status();
    std::vector<Tensor> concat_ins(broadcast_size, mask_in);
    auto concat_op = builder.BuildOp(kLiteRtOpCodeTflConcatenation, concat_ins,
                                     {*concat_out});
    if (!concat_op) return concat_op.Error().Status();
    ConcatenationOptions concat_opts;
    concat_opts.axis = broadcast_axis;
    concat_opts.fused_activation_function = 0;
    auto opt_s = builder.SetOpOptions(*concat_op, std::move(concat_opts));
    if (!opt_s) return opt_s.Error().Status();
    effective_mask = *concat_out;
  }

  auto new_add = builder.BuildOp(kLiteRtOpCodeTflAdd,
                                 {add_input, effective_mask}, {final_out});
  if (!new_add) return new_add.Error().Status();
  AddOptions new_add_opts;
  new_add_opts.fused_activation_function = 0;
  auto add_s = builder.SetOpOptions(*new_add, std::move(new_add_opts));
  if (!add_s) return add_s.Error().Status();

  auto erase_s = builder.EraseOp(post_reshape);
  if (!erase_s) return erase_s.Error().Status();
  RegisterOrphanGroup({add_or->Get(), pre_or->Get()});
  return kLiteRtStatusOk;
}

}  // namespace

extern "C" {

LiteRtStatus SimplifyMaskingAddTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op_ptr) {
  Op op(context, op_ptr);
  if (op.Code() == kLiteRtOpCodeTflReshape) {
    return TryRewriteReshapeMaskAddReshape(context, builder_ptr, op);
  }
  if (op.Code() != kLiteRtOpCodeTflAdd || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto add_opts = GetOptionsAs<AddOptions>(context, op_ptr);
  if (!add_opts || add_opts->fused_activation_function != 0) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor lhs = op.Inputs()[0];
  const Tensor rhs = op.Inputs()[1];
  if (lhs.Uses().size() != 1 || rhs.Uses().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto cast0_or = lhs.GetDefiningOp();
  auto cast1_or = rhs.GetDefiningOp();
  if (!cast0_or || !cast1_or || cast0_or->Code() != kLiteRtOpCodeTflCast ||
      cast1_or->Code() != kLiteRtOpCodeTflCast ||
      cast0_or->Inputs().size() != 1 || cast1_or->Inputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  const Tensor b0 = cast0_or->Inputs()[0];
  const Tensor b1 = cast1_or->Inputs()[0];
  if (b0.ElementType() != ElementType::Bool ||
      b1.ElementType() != ElementType::Bool) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (Dims(op.Outputs()[0]).empty()) {
    return kLiteRtStatusPatternNoMatch;
  }

  Builder builder(context, builder_ptr);
  auto status = RewriteBoolCastAdd(builder, op, *cast0_or, *cast1_or, b0, b1);
  if (!status) {
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
