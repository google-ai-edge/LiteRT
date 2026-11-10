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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites a 2D coordinate embedding implemented as
//   OneHot(coord, depth) [-> SelectV2(mask, 0, one_hot)] -> FC(W[D, depth])
// for the x and y coordinates, followed by Reshape + Concat + Sum, into a
// table lookup: Gather(W^T[depth + 1, D], safe_coord) + Add (or padded lookup).
//
// Before:
//   coord_x [1, S]                      coord_y [1, S]
//          |                                   |
//   +------v------+                     +------v------+
//   |   OneHot    |                     |   OneHot    |
//   | (SelectV2)  |                     | (SelectV2)  |
//   +------+------+                     +------+------+
//          | [1, S, depth]                     | [1, S, depth]
//   +------v------+                     +------v------+
//   | FC [D,depth]| <-- w_x             | FC [D,depth]| <-- w_y
//   +------+------+                     +------+------+
//          |                                   |
//   +------v------+                     +------v------+
//   |   Reshape   |                     |   Reshape   |
//   | [1,1,S,D]   |                     | [1,1,S,D]   |
//   +------+------+                     +------+------+
//           \                                 /
//            +----------------+--------------+
//                             |
//                    +--------v--------+
//                    |  Concat (ax 0)  | [2, 1, S, D]
//                    +--------+--------+
//                             |
//                    +--------v--------+
//                    |   Sum (ax 0)    |
//                    +--------+--------+
//                             |
//                     sum_out [1, S, D]
//
// After (padded lookup, default):
//   idx  = Minimum(Maximum(c + 2, 0), depth + 2)
//   rows = Gather(table [depth + 3, D], idx)
//   sum_out = Add(rows_x, rows_y)
//
// Only Sum and Concat are erased directly; the rest of the replaced sub-DAG
// (including OneHot, SelectV2, FC, Reshape, and bool mask ops) is registered
// with OrphanCleanupTransformation.
LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

// Replaces a RoPE Sin/Cos chain registered by EntryEmbeddingTransformation
// with an exact table lookup:
//
// Before:                                 After:
//   src [1, S, 1] int32                     src [1, S, 1] int32
//        |                                       |
//   Cast -> Mul(freq [1, 1, F])             Add(+2) -> Maximum(0)
//        |                                  -> Minimum(depth + 2)
//   Reshape [1, S, 1, F]                         |
//      /        \                           Gather(table [depth + 3, 2F])
//    Sin        Cos                              | [1, S, 1, 2F]
//     |          |                          Slice [.., :F]  Slice [.., F:]
//  sin_out    cos_out                            |               |
//                                             sin_out         cos_out
LiteRtStatus EntryEmbeddingTrigFoldTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

// Clears the trig chains registered for EntryEmbeddingTrigFoldTransformation
// (call when transformations are registered for a new compilation).
void ResetEntryEmbeddingTransformationState();

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
