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

// Transforms Vision Encoder MLP and Transformer Trunk blocks:
// 1. MLP Gating Block (3D or 4D when S >= 600):
//    Replaces quantize(i8->i16) -> gelu(i16) -> quantize(i8->i16) -> mul(i16)
//    -> quantize(i16->i8) with native INT8 gelu -> mul. When S >= 600 and S
//    factors into (H_s, W_s) with fewer 8x8 Crouton tiles than (1, S), folds
//    the 3D [1, S, F] activations to 4D [1, H_s, W_s, F].
// 2. Transformer Residual + RmsNorm Trunk Block (when S >= 600):
//    Folds 3D [1, S, D] post-projection Quantize(i8->i16) -> RmsNorm(i16) ->
//    Add(i16) (-> optional pre-norm RmsNorm(i16) -> Quantize(i16->i8)) to 4D
//    [1, H_s, W_s, D] so QNN HTP tiles 8x8 Crouton spatial tiles across
//    (H_s, W_s) instead of (1, S).
//
// Before (MLP Gating):
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
// After (MLP Gating, folded to [1, H_s, W_s, F] when S >= 600):
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
//
// Before / After (Residual-Norm Trunk when S >= 600):
//   fc_out [1,S,D] -> Quantize -> RmsNorm \
//                                           Add -> [1,S,D] -> RmsNorm ->
//                                           Quantize
//   residual [1,S,D] ---------------------/
//   ==> Folded to 4D [1, H_s, W_s, D] with boundary Reshapes.
LiteRtStatus MLPInt8QuantTransformation(const LiteRtCompilerContext* context,
                                        LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MLP_QUANT_TRANSFORMATION_H_
