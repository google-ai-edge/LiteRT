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

#include "litert/vendors/qualcomm/compiler/transformations/dequantize_quantize_transformation.h"

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"

using litert::ElementType;
using litert::compiler::Builder;
using litert::compiler::Op;
using litert::compiler::Tensor;

extern "C" {

LiteRtStatus DequantizeQuantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Op quantize(context, op);
  if (quantize.Code() != kLiteRtOpCodeTflQuantize ||
      quantize.Inputs().size() != 1 || quantize.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor dequantize_output = quantize.Inputs()[0];
  if (dequantize_output.ElementType() != ElementType::Float32) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto dequantize_result = dequantize_output.GetDefiningOp();
  if (!dequantize_result) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op dequantize = *dequantize_result;
  if (dequantize.Code() != kLiteRtOpCodeTflDequantize ||
      dequantize.Inputs().size() != 1 || dequantize.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor pattern_input = dequantize.Inputs()[0];
  Tensor pattern_output = quantize.Outputs()[0];
  const bool is_int8_to_int16 =
      pattern_input.ElementType() == ElementType::Int8 &&
      pattern_output.ElementType() == ElementType::Int16;
  const bool is_int16_to_int8 =
      pattern_input.ElementType() == ElementType::Int16 &&
      pattern_output.ElementType() == ElementType::Int8;
  if ((!is_int8_to_int16 && !is_int16_to_int8) ||
      !pattern_input.HasQuantization() || !pattern_output.HasQuantization()) {
    return kLiteRtStatusPatternNoMatch;
  }

  const bool dequantize_has_single_use = dequantize_output.Uses().size() == 1;

  Builder builder(context, builder_ptr);
  auto new_quantize = builder.BuildOp(kLiteRtOpCodeTflQuantize, {pattern_input},
                                      {pattern_output});
  if (!new_quantize) {
    return new_quantize.Error().Status();
  }
  auto erase_status = builder.EraseOp(quantize);
  if (!erase_status) {
    return erase_status.Error().Status();
  }

  if (dequantize_has_single_use) {
    erase_status = builder.EraseOp(dequantize);
    if (!erase_status) {
      return erase_status.Error().Status();
    }
  }

  return kLiteRtStatusOk;
}

}  // extern "C"
