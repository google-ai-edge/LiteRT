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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resets the chunking configuration to auto mode and clears the
// per-compilation state. Called once per compilation when transformations are
// registered: only attention whose logits tensor (H x S x K x bytes) is >=
// 8 MiB and S >= 2 is rewritten, with one head per chunk (H chunks).
// Single-head attention is split along the query axis into the fewest chunks of
// <= 4 MiB of logits.
void ResetAttentionChunkTransformationState();

// Overrides the configuration (used by tests). `head_chunks` / `query_chunks`
// are the number of chunks along the head / query axis; 1 x 1 disables the
// transformation. Enables tail absorption and turns auto mode off.
void SetAttentionChunkConfig(int head_chunks, int query_chunks);

// Overrides the epilogue absorption (used by tests).
void SetAttentionChunkAbsorbTail(bool absorb_tail);

// Splits a self-attention core into independent chunks along the head axis
// (and/or the query axis) so that the per-chunk logits/probabilities are small
// enough to stay on-chip (e.g. in 8 MB VTCM on SM8850). Each softmax row is
// computed exactly as before; with the epilogue absorbed (default) the result
// is bit-exact.
//
// Before (root = first BatchMatmul; H heads, S queries, K keys, D/Dv dims;
// the epilogue is optional and only absorbed when present):
//
//      Q [H, S, D]      Kt [H, D, K]
//            \            /
//           BatchMatmul (root)          (adj_x = adj_y = false)
//                  |  logits [H, S, K]
//               Reshape  [1, H, S, K]
//                  |
//               Softmax  [1, H, S, K]
//                  |
//               Reshape  [H, S, K]       V [H, K, Dv]
//                   \                    /
//                    +-- BatchMatmul ---+
//                             |  out [H, S, Dv] (int16)
//         - - - - - - - - - - | - - - - - - - - - epilogue - - -
//                          Reshape            [1, H, S, Dv]
//                             |
//                          Transpose(0,2,1,3) [1, S, H, Dv]
//                             |
//                          Quantize           [1, S, H, Dv] (int8)
//
// After (Qualcomm MHA -> SHA mode when H chunks of 1 head are selected and the
// 4D -> 3D Q/K/V prologue is present; when S >= 600, Q and Softmax/Out are
// additionally folded into 4D [1, H_s, W_s, D] while K/V use [1, 1, K, D] to
// broadcast across H_s and maximize 8x8 Crouton tile utilization):
//
//   q_4d [1, S, H, D]   k_4d [1, K, H, D]   v_4d [1, K, H, Dv]
//      | Reshape           | Reshape           | Reshape
//   [1, S, H*D]         [1, K, H*D]         [1, K, H*Dv]
//      | Split(axis=2)     | Split(axis=2)     | Split(axis=2)
//      v                   v                   v
//   q_i [1, S, D]       k_i [1, K, D]       v_i [1, K, Dv]    i = 0 .. H-1
//        \                /                    |
//       BatchMatmul(adj_y=true) [1, S, K]      |
//                |                             |
//             Softmax           [1, S, K]      |
//                 \                           /
//                  +----- BatchMatmul -------+  o_i [1, S, Dv] (int16)
//                              |
//                           Quantize            y_i [1, S, Dv] (int8)
//                              |
//   y_0 ... y_(H-1) -----> Concat(axis=2) ----> [1, S, H*Dv] (int8)
LiteRtStatus AttentionChunkTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_
