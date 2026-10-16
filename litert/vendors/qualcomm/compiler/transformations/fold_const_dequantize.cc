// Copyright 2025 Google LLC.
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

#include "litert/vendors/qualcomm/compiler/transformations/fold_const_dequantize.h"

#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_quantization.h"

using litert::ElementType;
using litert::RankedTensorType;
using litert::compiler::Builder;
using litert::compiler::DequantizeWeights;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;
using litert::compiler::m_AllOf;
using litert::compiler::m_IsConstant;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Op;
using litert::compiler::Match;

extern "C" {

// Constant-folds tfl.dequantize when its sole input is a quantized constant.
//
// Pattern:  const_q (quantized) -> tfl.dequantize -> out (float32)
// Result:   new_const_f32 (float32 weights) -> tfl.cast(f32->f32) -> out
//
// The cast is a float32->float32 identity that lets downstream ops keep their
// existing reference to `out`.  QNN constant-propagation eliminates it.
LiteRtStatus FoldConstDequantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  if (root_op.Inputs().size() != 1 || root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (root_op.Code() != kLiteRtOpCodeTflDequantize) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor in_tensor = root_op.Inputs()[0];
  if (!in_tensor.IsConstant() || !in_tensor.HasQuantization()) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor out_tensor = root_op.Outputs()[0];
  auto out_ranked_type = out_tensor.RankedTensorType();
  if (!out_ranked_type) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (static_cast<LiteRtElementType>(out_ranked_type->ElementType()) !=
      kLiteRtElementTypeFloat32) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Compute float32 values by dequantizing the constant weights.
  auto dequantized = DequantizeWeights(in_tensor);
  if (!dequantized) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Build a new float32 tensor (same shape as out_tensor) with the
  // pre-dequantized weights embedded.
  RankedTensorType float_type = *out_ranked_type;
  float_type.SetElementType(ElementType::Float32);

  auto new_const_res =
      builder.BuildTensor(RankedTensorSpecBuilder(float_type).Build());
  if (!new_const_res) {
    return new_const_res.Error().Status();
  }

  auto weights_res = builder.BuildWeights(
      absl::MakeConstSpan(dequantized->data(), dequantized->size()),
      *new_const_res);
  if (!weights_res) {
    return weights_res.Error().Status();
  }

  // Rebuild the tensor with the weights attached.
  auto new_const_with_w =
      builder.BuildTensor(RankedTensorSpecBuilder(float_type)
                              .WithWeights(*weights_res)
                              .Build());
  if (!new_const_with_w) {
    return new_const_with_w.Error().Status();
  }

  // Insert a float32->float32 cast so that `out_tensor` (referenced by
  // downstream ops) retains a valid producer after the dequantize is erased.
  auto cast_op =
      builder.BuildOp(kLiteRtOpCodeTflCast, {*new_const_with_w}, {out_tensor});
  if (!cast_op) {
    return cast_op.Error().Status();
  }

  builder.EraseOp(root_op);
  return kLiteRtStatusOk;
}

}  // extern "C"
