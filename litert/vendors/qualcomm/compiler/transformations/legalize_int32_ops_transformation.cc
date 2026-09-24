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

#include "litert/vendors/qualcomm/compiler/transformations/legalize_int32_ops_transformation.h"

#include <utility>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

using litert::ElementType;
using litert::RankedTensorType;
using litert::compiler::Builder;
using litert::compiler::m_Any;
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_ElementType;
using litert::compiler::m_Op;
using litert::compiler::Match;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::ReduceMaxOptions;
using litert::compiler::Tensor;

extern "C" {

LiteRtStatus LegalizeInt32SignTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Tensor in_tensor(context, nullptr);
  if (!Match(root_op,
             m_Op<kLiteRtOpCodeTflSign>(m_CaptureOrSameAs(
                 &in_tensor, m_ElementType(kLiteRtElementTypeInt32)))) ||
      root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor out_tensor = root_op.Outputs()[0];

  auto in_ranked_type = in_tensor.RankedTensorType();
  if (!in_ranked_type) {
    return in_ranked_type.Error().Status();
  }

  RankedTensorType f32_type = *in_ranked_type;
  f32_type.SetElementType(ElementType::Float32);

  auto f32_in_res =
      builder.BuildTensor(RankedTensorSpecBuilder(f32_type).Build());
  if (!f32_in_res) {
    return f32_in_res.Error().Status();
  }
  Tensor f32_in = *f32_in_res;

  auto f32_out_res =
      builder.BuildTensor(RankedTensorSpecBuilder(f32_type).Build());
  if (!f32_out_res) {
    return f32_out_res.Error().Status();
  }
  Tensor f32_out = *f32_out_res;

  // 1. Cast(in_tensor [i32] -> f32_in [f32])
  auto cast_in = builder.BuildOp(kLiteRtOpCodeTflCast, {in_tensor}, {f32_in});
  if (!cast_in) {
    return cast_in.Error().Status();
  }

  // 2. Sign(f32_in [f32] -> f32_out [f32])
  auto sign_f32 = builder.BuildOp(kLiteRtOpCodeTflSign, {f32_in}, {f32_out});
  if (!sign_f32) {
    return sign_f32.Error().Status();
  }

  // 3. Cast(f32_out [f32] -> out_tensor [i32])
  auto cast_out =
      builder.BuildOp(kLiteRtOpCodeTflCast, {f32_out}, {out_tensor});
  if (!cast_out) {
    return cast_out.Error().Status();
  }

  builder.EraseOp(root_op);
  return kLiteRtStatusOk;
}

LiteRtStatus LegalizeInt32ReduceMaxTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Tensor in_tensor(context, nullptr);
  Tensor axis_tensor(context, nullptr);
  if (!Match(root_op,
             m_Op<kLiteRtOpCodeTflReduceMax>(
                 m_CaptureOrSameAs(&in_tensor,
                                   m_ElementType(kLiteRtElementTypeInt32)),
                 m_CaptureOrSameAs(&axis_tensor, m_Any()))) ||
      root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor out_tensor = root_op.Outputs()[0];

  auto in_ranked_type = in_tensor.RankedTensorType();
  if (!in_ranked_type) {
    return in_ranked_type.Error().Status();
  }
  auto out_ranked_type = out_tensor.RankedTensorType();
  if (!out_ranked_type) {
    return out_ranked_type.Error().Status();
  }

  RankedTensorType f32_in_type = *in_ranked_type;
  f32_in_type.SetElementType(ElementType::Float32);

  RankedTensorType f32_out_type = *out_ranked_type;
  f32_out_type.SetElementType(ElementType::Float32);

  auto f32_in_res =
      builder.BuildTensor(RankedTensorSpecBuilder(f32_in_type).Build());
  if (!f32_in_res) {
    return f32_in_res.Error().Status();
  }
  Tensor f32_in = *f32_in_res;

  auto f32_out_res =
      builder.BuildTensor(RankedTensorSpecBuilder(f32_out_type).Build());
  if (!f32_out_res) {
    return f32_out_res.Error().Status();
  }
  Tensor f32_out = *f32_out_res;

  // 1. Cast(in_tensor [i32] -> f32_in [f32])
  auto cast_in = builder.BuildOp(kLiteRtOpCodeTflCast, {in_tensor}, {f32_in});
  if (!cast_in) {
    return cast_in.Error().Status();
  }

  // 2. ReduceMax(f32_in [f32], axis_tensor -> f32_out [f32])
  auto new_reduce = builder.BuildOp(kLiteRtOpCodeTflReduceMax,
                                    {f32_in, axis_tensor}, {f32_out});
  if (!new_reduce) {
    return new_reduce.Error().Status();
  }

  // Copy keep_dims option from root_op if present
  ReduceMaxOptions reduce_options;
  if (reduce_options.InitFromOp(context, root_op.Get()) == kLiteRtStatusOk) {
    auto opt_status =
        builder.SetOpOptions(*new_reduce, std::move(reduce_options));
    if (!opt_status) {
      return opt_status.Error().Status();
    }
  }

  // 3. Cast(f32_out [f32] -> out_tensor [i32])
  auto cast_out =
      builder.BuildOp(kLiteRtOpCodeTflCast, {f32_out}, {out_tensor});
  if (!cast_out) {
    return cast_out.Error().Status();
  }

  builder.EraseOp(root_op);
  return kLiteRtStatusOk;
}

}  // extern "C"
