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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites a 2D coordinate embedding implemented as
//   OneHot(coord, depth) [-> SelectV2(mask, 0, one_hot)] -> FC(W[D, depth])
// for the x and y coordinates, followed by Reshape + Concat + Sum, into a
// table lookup: Gather(W^T[depth + 1, D], safe_coord) + Add.
//
// FC(OneHot(c)) == W[:, c] for 0 <= c < depth and 0 otherwise. To keep that
// exact semantic (and avoid out-of-bounds gathers for padding coordinates),
// an all-zero row is appended to the transposed table at index `depth` and
// out-of-range coordinates are remapped to it:
//   safe_coord = SelectV2(c >= 0 && c < depth, c, depth)
//
// The pattern is gated on: constant f32 weights [D, depth] without bias,
// OneHot on/off values of 1/0 along the last axis, an optional
// SelectV2(mask [1, S, 1], fill, one_hot) whose fill is the constant 0 or NaN,
// and consistent [1, S] / [1, S, D] shapes (S is read from the graph, so any
// sequence length is supported).
//
// For fill in {0, NaN}, FC(fill_row) == fill row-wise, so the optional
// SelectV2 is re-applied on the gathered [1, S, D] rows with the original
// mask and fill, preserving exact semantics (e.g. NaN for invalid coords).
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
// After:
//   coord_x [1, S]                          coord_y [1, S]
//          |                                       |
//   +------v-------+                        +------v-------+
//   | range check  |  (GreaterEqual, Less,  | range check  |
//   | + SelectV2   |   LogicalAnd, SelectV2)| + SelectV2   |
//   +------+-------+                        +------+-------+
//          | safe_x                                | safe_y
//   +------v---------------+         +-------------v--------+
//   | Gather(w_x_t, safe_x)|         | Gather(w_y_t, safe_y)|
//   | w_x_t [depth + 1, D] |         | w_y_t [depth + 1, D] |
//   +------+---------------+         +-------------+--------+
//          | [1, S, D]                             | [1, S, D]
//   +------v---------------+         +-------------v--------+
//   | (SelectV2(mask_x,    |         | (SelectV2(mask_y,    |
//   |   fill, rows))       |         |   fill, rows))       |
//   +------+---------------+         +-------------+--------+
//           \                                     /
//            +------------------+----------------+
//                               |
//                        +------v------+
//                        |     Add     |
//                        +------+------+
//                               |
//                       sum_out [1, S, D]
//
// Bool-free path (default when every SelectV2 mask is exactly
//   mask = (c < 0 || c >= depth) && c != -1,
// or there is no SelectV2; opt out with LITERT_MEDIATEK_EMBEDDING_LEGACY=1):
// MediaTek NPUs have no bool support, so the range check, the mask and the
// SelectV2 are folded into a padded table indexed by clamped integer math:
//   idx  = Minimum(Maximum(c + 2, 0), depth + 2)            (int32, CPU)
//   rows = Gather(table [depth + 3, D], idx)
// with table rows: 0 <-> c <= -2 (fill), 1 <-> c == -1 (zeros),
// 2 + k <-> c == k (W^T[k]), depth + 2 <-> c >= depth (fill).
//
// Only Sum and Concat are erased directly (new ops are spliced at the
// earliest erased op, and Concat depends on every coordinate and mask); the
// rest of the replaced sub-DAG, including the bool mask ops, is registered
// with OrphanCleanupTransformation.
//
// When fill == NaN and the coordinate also drives a RoPE angle chain
//   Cast(src [1, S, 1]) -> Mul(freq [1, 1, F]) -> Reshape [1, S, 1, F]
//   -> {Sin, Cos}
// (src being the tensor `c` is reshaped from), the Cast is registered for
// EntryEmbeddingTrigFoldTransformation below, unless
// LITERT_MEDIATEK_EMBEDDING_NO_TRIG_FOLD=1.
LiteRtStatus EntryEmbeddingTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

// Replaces a RoPE Sin/Cos chain registered by EntryEmbeddingTransformation
// with an exact table lookup. MediaTek NPUs have no Sin/Cos, and evaluating
// them with polynomials in fp16 is inaccurate for large angles. Since the
// coordinates are integers, sin/cos(float(c) * freq) are precomputed in f32
// (same math as the CPU kernels) for every coordinate -1 <= c < depth; all
// other coordinates produce NaN embeddings that poison the whole output, so
// their rows are unobservable.
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

#ifdef __cplusplus
}

// Clears the trig chains registered for EntryEmbeddingTrigFoldTransformation
// (call when transformations are registered for a new compilation).
void ResetEntryEmbeddingTransformationState();
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ENTRY_EMBEDDING_TRANSFORMATION_H_
