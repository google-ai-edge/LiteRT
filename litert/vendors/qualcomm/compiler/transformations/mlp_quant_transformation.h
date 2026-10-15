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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MLP_QUANT_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MLP_QUANT_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transforms Vision Encoder MLP blocks from
// quantize(i8->i16) -> gelu(i16) -> quantize(i8->i16) -> mul(i16) ->
// quantize(i16->i8) to native INT8 gelu -> mul, saving ~5-6ms.
//
// Before:
//        fc1_out (int8)               fc2_out (int8)
//              |                            |
//       +------v------+              +------v------+
//       |   quantize  | (i8 -> i16)  |   quantize  | (i8 -> i16)
//       +------┬------+              +------┬------+
//              |                            |
//       +------v------+                     |
//       |     gelu    | (int16)             |
//       +------┬------+                     |
//              \                            /
//               +─────────────┬────────────+
//                             |
//                      +──────v──────+
//                      |     mul     | (int16)
//                      +──────┬──────+
//                             | mul_out (int16)
//                      +──────v──────+
//                      |   quantize  | (i16 -> i8)
//                      +──────┬──────+
//                             |
//                     final_mul_out (int8)
//
// After:
//        fc1_out (int8)               fc2_out (int8)
//              |                            |
//       +------v------+                     |
//       |     gelu    | (int8)              |
//       +------┬------+                     |
//              \                            /
//               +─────────────┬────────────+
//                             |
//                      +──────v──────+
//                      |     mul     | (int8)
//                      +──────┬──────+
//                             |
//                     final_mul_out (int8)
LiteRtStatus MLPInt8QuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MLP_QUANT_TRANSFORMATION_H_
