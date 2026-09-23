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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

extern "C" {

// Resets global cached trig tensors (g_cos_12, g_sin_signed_12) across runs.
void ResetRopeTransformationState();

// Fuses Rotary Position Embedding (RoPE) in Vision Encoder attention layers.
//
// Before:
//                       in_tensor [1, 630, 12, 64]
//                      /         |         |         \
//                   (Slice)   (Slice)   (Slice)   (Slice)
//                     s0        s1        s2        s3
//                    /  \      /  \      /  \      /  \
//    cos2, sin2 --->|    |<-->|    |    |    |<-->|    |<--- cos3, sin3
//                   v    v    v    v    v    v    v    v
//                  Mul  Mul  Mul  Mul  Mul  Mul  Mul  Mul
//                    \  /      \  /      \  /      \  /
//                     Sub       Add       Sub       Add
//                      \        /          \        /
//                      Concat(lo)          Concat(hi)
//                    [1,630,12,32]       [1,630,12,32]
//                           \                 /
//                            +───────┬───────+
//                                    |
//                           Concat (root op)
//                                    |
//                       out_tensor [1, 630, 12, 64]
//
// After:
//                                in_tensor [1, 630, 12, 64]
//                                /                        \
//          s1, s0, s3, s2 <-----+ (Slices)                 |
//                |              \                         |
//      +---------v---------+                              |
//      |  Concat (axis=3)  |                              |
//      | [1, 630, 12, 64]  |                              |
//      +---------┬---------+                              |
//                |     sin_signed_12                      |     cos_12
//                |   [1, 630, 12, 64]                     |   [1, 630, 12, 64]
//                \            /                           \            /
//                 +─────┬────+                             +─────┬────+
//                       |                                        |
//                 +─────v─────+                            +─────v─────+
//                 |    Mul    |                            |    Mul    |
//                 |  (term2)  |                            |  (term1)  |
//                 +─────┬─────+                            +─────┬─────+
//                       \                                        /
//                        +──────────────────┬───────────────────+
//                                           |
//                                     +─────v─────+
//                                     |    Add    |
//                                     +─────┬─────+
//                                           |
//                                out_tensor [1, 630, 12, 64]
LiteRtStatus RopeAttentionLayerTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Erases orphaned child ops left behind by RopeAttentionLayerTransformation:
//
// Before (dead branch with 0 users):
//      Mul     Mul           Mul     Mul
//        \     /               \     /
//      +──v───v──+           +──v───v──+
//      |   Sub   |           |   Add   | [1, 630, 12, 16]
//      +────┬────+           +────┬────+
//            \                   /
//             +────────┬────────+
//                      |
//              +───────v───────+
//              | Concatenation | [1, 630, 12, 32] (0 users)
//              +───────────────+
//
// After:
//      (Concatenation, Sub, Add, and upstream Mul ops are erased;
//       dead branches completely eliminated from the subgraph)
LiteRtStatus RopeCleanupTransformation(const LiteRtCompilerContext* context,
                                       LiteRtBuilder builder_ptr, LiteRtOp op);

}  // extern "C"

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_
