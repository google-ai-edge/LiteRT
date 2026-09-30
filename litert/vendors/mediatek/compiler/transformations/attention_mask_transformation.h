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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_MASK_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_MASK_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transforms text encoder attention mask computation to native NPU arithmetic:
// 1. Rewrites rank-mismatched outer product Mul([1, S], [1, S, 1]) ->
//    BatchMatMul([1, S, 1], Reshape([1, S], [1, 1, S])) -> [1, S, S] to avoid
//    MediaTek MDLA's rejection of elementwise Mul between rank 2 and rank 3.
// 2. Rewrites Cast(LogicalNot(NotEqual(x, 0.0f))) -> Sub(1.0f, x)
//    for x in {0.0, 1.0}.
//
// Pattern 1 (Mul -> Reshape + BatchMatMul):
// Before:
//      mask_2d [1, S] (f32)       col_mask [1, S, 1] (f32)
//                 \                     /
//                  MUL [1, S, S]  <-- MDLA rejected (rank 2 vs rank 3)
// After:
//      mask_2d [1, S] (f32)       col_mask [1, S, 1] (f32)
//            |                          |
//      RESHAPE [1, 1, S]                |
//            \                         /
//             BATCH_MATMUL [1, S, S]
//
// Pattern 2 (Cast -> Sub):
// Before:
//      x (f32)    0.0f (f32 const)
//         \       /
//        NOT_EQUAL (bool)
//            |
//       LOGICAL_NOT (bool)
//            |
//          CAST (f32)
// After:
//      1.0f (f32 const)   x (f32)
//            \            /
//                 SUB (f32)
LiteRtStatus AttentionMaskTransformation(const LiteRtCompilerContext* context,
                                         LiteRtBuilder builder_ptr,
                                         LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_MASK_TRANSFORMATION_H_
