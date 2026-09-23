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

#include "litert/vendors/qualcomm/compiler/transformations/mlp_quant_transformation.h"

#include <utility>
#include <vector>

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
using litert::compiler::m_CaptureOrSameAs;
using litert::compiler::m_CommutativeOp;
using litert::compiler::m_ElementType;
using litert::compiler::m_Op;
using litert::compiler::Match;
using litert::compiler::MulOptions;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;

extern "C" {

LiteRtStatus MLPInt8QuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr,
                                        LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);

  Tensor fc1_out(context, nullptr);
  Tensor fc2_out(context, nullptr);
  Op gelu_op(context, nullptr);
  Op quant1_op(context, nullptr);
  Op quant2_op(context, nullptr);

  auto gelu_branch = m_CaptureOrSameAs(
      &gelu_op,
      m_Op<kLiteRtOpCodeTflGelu>(m_CaptureOrSameAs(
          &quant1_op, m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                          &fc1_out, m_ElementType(kLiteRtElementTypeInt8))))));

  auto quant2_branch = m_CaptureOrSameAs(
      &quant2_op, m_Op<kLiteRtOpCodeTflQuantize>(m_CaptureOrSameAs(
                      &fc2_out, m_ElementType(kLiteRtElementTypeInt8))));

  if (!Match(root_op, m_CommutativeOp<kLiteRtOpCodeTflMul>(gelu_branch,
                                                           quant2_branch))) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (quant1_op.Outputs().size() != 1 ||
      quant1_op.Outputs()[0].Uses().size() != 1 ||
      quant2_op.Outputs().size() != 1 ||
      quant2_op.Outputs()[0].Uses().size() != 1 ||
      gelu_op.Outputs().size() != 1 ||
      gelu_op.Outputs()[0].Uses().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto fc1_type = fc1_out.RankedTensorType();
  auto fc2_type = fc2_out.RankedTensorType();
  if (!fc1_type || !fc2_type ||
      fc1_type->Layout().Dimensions() != fc2_type->Layout().Dimensions()) {
    return kLiteRtStatusPatternNoMatch;
  }

  if (root_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor mul_out = root_op.Outputs()[0];
  if (mul_out.ElementType() != ElementType::Int16) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto uses = mul_out.Uses();
  if (uses.size() != 1 || uses[0].user.Code() != kLiteRtOpCodeTflQuantize) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op quant3_op = uses[0].user;
  if (quant3_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  Tensor final_mul_out = quant3_op.Outputs()[0];
  if (final_mul_out.ElementType() != ElementType::Int8) {
    return kLiteRtStatusPatternNoMatch;
  }

  // 6. Build new INT8 gelu_out tensor
  RankedTensorType gelu_int8_type = *fc1_type;
  auto new_gelu_out_spec = RankedTensorSpecBuilder(gelu_int8_type);
  if (fc1_out.QTypeId() == kLiteRtQuantizationPerTensor) {
    new_gelu_out_spec =
        std::move(new_gelu_out_spec)
            .WithPerTensorQuantization(fc1_out.PerTensorQuantization());
  }
  auto new_gelu_out_res =
      builder.BuildTensor(std::move(new_gelu_out_spec).Build());
  if (!new_gelu_out_res) {
    return new_gelu_out_res.Error().Status();
  }
  Tensor gelu_int8_out = *new_gelu_out_res;

  // 7. Build new INT8 GELU op
  auto new_gelu =
      builder.BuildOp(kLiteRtOpCodeTflGelu, {fc1_out}, {gelu_int8_out});
  if (!new_gelu) {
    return new_gelu.Error().Status();
  }

  // 8. Build new INT8 MUL op writing directly to final_mul_out
  auto new_mul = builder.BuildOp(kLiteRtOpCodeTflMul, {gelu_int8_out, fc2_out},
                                 {final_mul_out});
  if (!new_mul) {
    return new_mul.Error().Status();
  }

  MulOptions mul_options;
  mul_options.fused_activation_function = 0;
  auto opt_status = builder.SetOpOptions(*new_mul, std::move(mul_options));
  if (!opt_status) {
    return opt_status.Error().Status();
  }

  // 9. Clean up obsolete ops
  builder.EraseOp(root_op);
  builder.EraseOp(gelu_op);
  builder.EraseOp(quant1_op);
  builder.EraseOp(quant2_op);
  builder.EraseOp(quant3_op);

  return kLiteRtStatusOk;
}

}  // extern "C"
