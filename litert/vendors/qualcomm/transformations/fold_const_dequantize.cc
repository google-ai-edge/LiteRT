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

#include "litert/vendors/qualcomm/transformations/fold_const_dequantize.h"

#include <cstdint>
#include <vector>

#include "fp16.h"  // from @FP16
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"

using litert::compiler::Builder;
using litert::compiler::Op;
using litert::compiler::Tensor;

extern "C" {

// Folds a float16 constant through a Dequantize op, producing a float32
// constant. The Dequantize op is erased; its output tensor is reused with
// new float32 weights.
LiteRtStatus FoldConstDequantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op dequant_op(context, op);

  // Match: Dequantize whose input is a float16 constant.
  Tensor input;
  if (!litert::compiler::Match(
          dequant_op,
          litert::compiler::m_Op<kLiteRtOpCodeTflDequantize>(
              litert::compiler::m_CaptureOrSameAs(
                  &input, litert::compiler::m_AllOf(
                              litert::compiler::m_IsConstant(),
                              litert::compiler::m_ElementType(
                                  kLiteRtElementTypeFloat16)))))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Read the float16 weights as raw bytes, then reinterpret as uint16_t.
  // WeightsData<uint16_t>() cannot be used because GetElementType<uint16_t>()
  // returns UInt16, not Float16.
  if (!input.HasWeights()) {
    return kLiteRtStatusPatternNoMatch;
  }
  absl::Span<const uint8_t> raw_bytes = input.Weights().Bytes();

  // Convert each fp16 value to fp32.
  const size_t num_elements = raw_bytes.size() / sizeof(uint16_t);
  const uint16_t* fp16_ptr =
      reinterpret_cast<const uint16_t*>(raw_bytes.data());
  std::vector<float> fp32_data;
  fp32_data.reserve(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    fp32_data.push_back(fp16_ieee_to_fp32_value(fp16_ptr[i]));
  }

  // Attach float32 weights directly to the Dequantize output tensor so that
  // downstream ops see a float32 constant without any intervening op.
  Tensor output = dequant_op.Outputs()[0];
  auto weights = builder.BuildWeights<float>(absl::MakeSpan(fp32_data), output);
  if (!weights) {
    return weights.Error().Status();
  }

  // Erase the Dequantize op; its output tensor now carries the folded weights.
  LITERT_RETURN_IF_ERROR(builder.EraseOp(dequant_op));

  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(const LiteRtCompilerContext* context,
                                 LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
