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

#include <cstddef>
#include <cstdint>
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
#include "tensor/internal/fp16.h"

using litert::ElementType;
using litert::RankedTensorType;
using litert::compiler::Builder;
using litert::compiler::DequantizeWeights;
using litert::compiler::Op;
using litert::compiler::RankedTensorSpecBuilder;
using litert::compiler::Tensor;
using litert::compiler::m_AllOf;
using litert::compiler::m_ElementType;
using litert::compiler::m_IsConstant;
using litert::compiler::m_IsQuantized;
using litert::compiler::m_Op;
using litert::compiler::Match;
using litert::tensor::fp16_ieee_to_fp32_value;

extern "C" {

// Folds a constant through a tfl.dequantize op into a float32 constant,
// erasing the dequantize. Two input cases are handled:
//
// Case A — float16 constant (no quantization params):
//   const (float16) -> tfl.dequantize -> output (float32)
//   Each fp16 value is converted to fp32 with fp16_ieee_to_fp32_value().
//
// Case B — quantized integer constant (int8/uint8/int16 with scale/zp):
//   const (quantized int) -> tfl.dequantize -> output (float32)
//   Values are dequantized via DequantizeWeights() from litert_quantization.h.
//
// In both cases a new float32 constant tensor is built with the folded
// weights and all consumers of the dequantize output are rewired to use it
// via builder.SetOpInput(). The dequantize op is then erased.
LiteRtStatus FoldConstDequantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op dequant_op(context, op);

  if (dequant_op.Inputs().size() != 1 || dequant_op.Outputs().size() != 1) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (dequant_op.Code() != kLiteRtOpCodeTflDequantize) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor input = dequant_op.Inputs()[0];
  if (!input.IsConstant()) {
    return kLiteRtStatusPatternNoMatch;
  }

  Tensor output = dequant_op.Outputs()[0];
  auto out_ranked_type = output.RankedTensorType();
  if (!out_ranked_type) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (static_cast<LiteRtElementType>(out_ranked_type->ElementType()) !=
      kLiteRtElementTypeFloat32) {
    return kLiteRtStatusPatternNoMatch;
  }

  std::vector<float> fp32_data;

  const auto in_element_type =
      static_cast<LiteRtElementType>(input.ElementType());

  if (in_element_type == kLiteRtElementTypeFloat16) {
    // Case A: float16 constant — convert via IEEE bit-cast, no q-params needed.
    if (!input.HasWeights()) {
      return kLiteRtStatusPatternNoMatch;
    }
    absl::Span<const uint8_t> raw_bytes = input.Weights().Bytes();
    const size_t num_elements = raw_bytes.size() / sizeof(uint16_t);
    const uint16_t* fp16_ptr =
        reinterpret_cast<const uint16_t*>(raw_bytes.data());
    fp32_data.reserve(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      fp32_data.push_back(fp16_ieee_to_fp32_value(fp16_ptr[i]));
    }
  } else {
    // Case B: quantized integer constant — use scale/zero_point dequantization.
    if (!input.HasQuantization()) {
      return kLiteRtStatusPatternNoMatch;
    }
    auto dequantized = DequantizeWeights(input);
    if (!dequantized) {
      return kLiteRtStatusPatternNoMatch;
    }
    fp32_data = std::move(*dequantized);
  }

  // Build a new float32 tensor with the folded constant weights.
  auto new_tensor_spec =
      RankedTensorSpecBuilder(*out_ranked_type)
          .WithTensorName(std::string(output.Name()))
          .Build();
  auto new_tensor = builder.BuildTensor(new_tensor_spec);
  if (!new_tensor) {
    return new_tensor.Error().Status();
  }

  auto weights =
      builder.BuildWeights<float>(absl::MakeSpan(fp32_data), *new_tensor);
  if (!weights) {
    return weights.Error().Status();
  }

  // Rewire every consumer of the dequantize output to use the new tensor.
  for (const auto& use : output.Uses()) {
    auto set_res =
        builder.SetOpInput(use.user, use.user_arg_ind, *new_tensor);
    if (!set_res) {
      return set_res.Error().Status();
    }
  }

  LITERT_RETURN_IF_ERROR(builder.EraseOp(dequant_op));
  return kLiteRtStatusOk;
}

}  // extern "C"
