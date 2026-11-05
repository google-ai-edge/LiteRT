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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_FOLD_CONST_DEQUANTIZE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_FOLD_CONST_DEQUANTIZE_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constant-folds a tfl.dequantize whose input is a quantized constant tensor.
//
// Before:
//   const_tensor (int4/int8/uint8/int16, per-tensor or per-channel quantized)
//          |
//   +------v----------+
//   |  tfl.dequantize |
//   +------┬----------+
//          |
//   out_tensor (float32)
//
// After:
//   new_const_tensor (float32, pre-dequantized values embedded as weights)
//          |
//   out_tensor (float32)  [new_const_tensor wired directly as out_tensor]
//
// The transformation erases the dequantize op and its quantized input; the
// dequantized float values are materialized as a new constant tensor.
LiteRtStatus FoldConstDequantizeTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_FOLD_CONST_DEQUANTIZE_H_
