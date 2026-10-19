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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MASK_SIMPLIFICATION_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MASK_SIMPLIFICATION_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Simplifies attention mask construction and application patterns for Qualcomm
// HTP:
//
// Pattern 1 — Boolean Cast + Add to LogicalOr + Cast (root = tfl.add):
//   Replaces the sum of two Cast(bool -> int/float) tensors (used when
//   combining causal and padding masks) with a single boolean LogicalOr
//   followed by one Cast.
//
//   Before:                                 After:
//     b0 (bool)         b1 (bool)             b0 (bool)     b1 (bool)
//         |                 |                      \           /
//   +-----v-----+     +-----v-----+               +─v─────────v─+
//   | tfl.cast  |     | tfl.cast  |               |tfl.logical_or| (bool)
//   +-----+-----+     +-----+-----+               +──────┬──────+
//          \               /                             |
//           +──────┬──────+                       +──────v──────+
//                  |                              |  tfl.cast   |
//           +──────v──────+                       +──────┬──────+
//           |   tfl.add   | (root)                       |
//           +──────┬──────+                             out
//                  |
//                 out
//
// Pattern 2 — Rank-5 Broadcast Reshape-Add-Reshape Elimination
// (root = post-tfl.reshape):
//   Eliminates a 4D -> 5D -> 4D Reshape wrapper around an attention-mask Add
//   where the input and output 4D shapes are identical (e.g. [1, H_kv, G*S, L])
//   and the 4D mask [1, 1, S, L] was being broadcast across G via a 5D
//   [1, H_kv, G, S, L] intermediate. Instead, tiles the mask G times along the
//   broadcast axis via Concatenation and adds directly in 4D.
//
//   Before:
//     add_input [1, H_kv, G*S, L]
//                 |
//          +──────v──────+
//          | tfl.reshape |
//          +──────┬──────+
//                 | pre_out [1, H_kv, G, S, L]    mask_in [1, 1, S, L]
//                  \                                   /
//                   +────────────────┬────────────────+
//                                    |
//                             +──────v──────+
//                             |   tfl.add   | [1, H_kv, G, S, L]
//                             +──────┬──────+
//                                    |
//                             +──────v──────+
//                             | tfl.reshape | (root)
//                             +──────┬──────+
//                                    |
//                       final_out [1, H_kv, G*S, L]
//
//   After:
//     add_input [1, H_kv, G*S, L]         mask_in [1, 1, S, L]
//                 |                                |
//                 |                     +──────────v──────────+
//                 |                     |  tfl.concatenation  | (G copies,
//                 |                     +──────────┬──────────+  axis = 2)
//                 |                                | [1, 1, G*S, L]
//                  \                              /
//                   +──────────────┬─────────────+
//                                  |
//                           +──────v──────+
//                           |   tfl.add   |
//                           +──────┬──────+
//                                  |
//                     final_out [1, H_kv, G*S, L]
LiteRtStatus SimplifyMaskingAddTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_MASK_SIMPLIFICATION_TRANSFORMATION_H_
