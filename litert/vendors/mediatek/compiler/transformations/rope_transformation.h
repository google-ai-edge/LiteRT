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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resets the cached (shared) trig tables built by RopeTransformation.
void ResetRopeTransformationState();

// Fuses a 2D (x/y) rotary position embedding applied on a [B, S, H, Dh]
// tensor. Each half of the head dim is rotated with its own cos/sin pair of
// shape [B, S, 1, Dh/4], which is broadcast over heads.
//
// Before (per Q/K tensor, per layer: 4 Slice + 8 Mul + 2 Sub + 2 Add +
// 3 Concat on [B, S, H, Dh/4] pieces):
//
//                       in_tensor [B, S, H, Dh]
//                      /         |         |         \
//                   (Slice)   (Slice)   (Slice)   (Slice)
//                     s0        s1        s2        s3
//                    /  \      /  \      /  \      /  \
//     cos2, sin2 -->|    |<-->|    |    |    |<-->|    |<-- cos3, sin3
//                   v    v    v    v    v    v    v    v
//                  Mul  Mul  Mul  Mul  Mul  Mul  Mul  Mul
//                    \  /      \  /      \  /      \  /
//                     Sub       Add       Sub       Add
//                      \        /          \        /
//                      Concat(lo)          Concat(hi)
//                            \                 /
//                             +-------+-------+
//                                     |
//                              Concat (root op)
//                                     |
//                           out_tensor [B, S, H, Dh]
//
// After (1 Concat + 2 Mul + 1 Add; the trig tables [B, S, 1, Dh] are built
// once and shared by every RoPE block that uses the same cos/sin tensors):
//
//   cos_tab  = Concat(cos2, cos2, cos3, cos3)             [B, S, 1, Dh]
//   sin_tab  = Concat(-sin2, sin2, -sin3, sin3)            [B, S, 1, Dh]
//
//                  in_tensor [B, S, H, Dh]
//                 /                       \
//     Concat(s1, s0, s3, s2)               |
//              |                           |
//        Mul(., sin_tab)            Mul(., cos_tab)   (broadcast over H)
//               \                         /
//                +---------- Add --------+
//                             |
//                   out_tensor [B, S, H, Dh]
//
// Gating: exact slice offsets (0, Dh/4, Dh/2, 3Dh/4) traced back to a common
// f32 input, exact Sub/Add operand roles and Concat order, shared trig shapes
// [B, S, 1, Dh/4]. The negation uses Mul by -1 (no Neg op), which MediaTek NPU
// supports.
LiteRtStatus RopeTransformation(const LiteRtCompilerContext* context,
                                LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ROPE_TRANSFORMATION_H_
