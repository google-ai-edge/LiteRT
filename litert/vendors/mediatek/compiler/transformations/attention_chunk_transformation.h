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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// (Re-)reads the chunking configuration from the environment and clears the
// per-compilation state. Called once per compilation when transformations are
// registered:
//   LITERT_MEDIATEK_ATTN_CHUNKS unset      (default) auto mode: only attention
//                                          whose logits tensor
//                                          (H x S x K x bytes) is >= 8 MiB and
//                                          S >= 2 is rewritten, with one head
//                                          per chunk (H chunks). Single-head
//                                          attention is split along the query
//                                          axis into the fewest chunks of
//                                          <= 4 MiB of logits.
//   LITERT_MEDIATEK_ATTN_CHUNKS=0          disable the transformation.
//   LITERT_MEDIATEK_ATTN_CHUNKS=N          explicit number of chunks (N must
//                                          divide the axis).
//   LITERT_MEDIATEK_ATTN_CHUNK_AXIS=query  split the query axis (default), or
//                                  =head   split the head axis.
//   LITERT_MEDIATEK_ATTN_HEAD_CHUNKS=M     (query mode only) additionally split
//                                          the heads into M groups.
//   LITERT_MEDIATEK_ATTN_CHUNK_TAIL=0      disable output epilogue absorption.
void ResetAttentionChunkTransformationState();

// Overrides the configuration (used by tests). `head_chunks` / `query_chunks`
// are the number of chunks along the head / query axis; 1 x 1 disables the
// transformation. Enables tail absorption and turns auto mode off.
void SetAttentionChunkConfig(int head_chunks, int query_chunks);

// Overrides the epilogue absorption (used by tests).
void SetAttentionChunkAbsorbTail(bool absorb_tail);

// Splits a self-attention core into independent chunks along the head axis
// (and/or the query axis) so that the per-chunk logits/probabilities are small
// enough to stay on-chip. Each softmax row is computed exactly as before; with
// the epilogue absorbed (default) the result is bit-exact.
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
// After (default auto mode: head axis, N = H chunks of h = H / N heads; query
// mode is identical with Q sliced along axis 1, K/V shared, and the Concat
// along axis 1 of the epilogue output / axis 1 of `out`):
//
//     Q [H, S, D]    Kt [H, D, K]    V [H, K, Dv]
//      |  Slice        |  Slice        |  Slice       (axis 0, heads i*h..)
//      v               v               v
//     Qi [h, S, D]   Kti [h, D, K]   Vi [h, K, Dv]      i = 0 .. N-1
//        \            /                |
//        BatchMatmul      [h, S, K]    |
//             |                        |
//          Reshape        [1, h, S, K] |
//             |                        |
//          Softmax        [1, h, S, K] |
//             |                        |
//          Reshape        [h, S, K]    |
//              \                      /
//               +--- BatchMatmul ----+  Oi [h, S, Dv] (int16)
//                         |
//                      Reshape            [1, h, S, Dv]      epilogue, per
//                         |                                  chunk (tail)
//                      Transpose(0,2,1,3) [1, S, h, Dv]
//                         |
//                      Quantize           [1, S, h, Dv] (int8)
//                         |
//     Y0 ... Y(N-1) ---> Concat(axis = 2) ---> [1, S, H, Dv] (int8)
//
// Without an epilogue (or with LITERT_MEDIATEK_ATTN_CHUNK_TAIL=0) the chunk
// outputs Oi are concatenated directly into `out` (axis 0 for heads, axis 1
// for queries) and the original epilogue is left in place.
//
// The epilogue (Reshape -> Transpose -> Quantize) is absorbed into each chunk
// so that downstream compiler fusions (such as fused accumulator
// requantization) occur per chunk before concatenation, avoiding double
// rounding and materialization of the unquantized concatenated intermediate.
//
// Every new tensor inherits the per-tensor quantization of the tensor it
// replaces (Slice outputs inherit Q's / Kt's / V's), so int16 kernels see the
// exact same scales. Only the last replaced op (the Quantize, or the second
// BatchMatmul without epilogue) is erased by this rewrite, so the new ops are
// spliced after every input; the dead rest of the old chain is registered with
// and removed by OrphanCleanupTransformation.
//
// Gating: auto mode (default) requires >= 8 MiB of logits and S >= 2, explicit
// chunk counts come from the environment (see above), CHUNKS=0 or the
// `no_attention_chunk` filter token disables it; rank-3 BatchMatmuls with
// adj_x = adj_y = false and matching H/S/K/D dims; the logits -> Reshape ->
// Softmax -> Reshape -> BatchMatmul(P as LHS) chain has single uses and the
// exact [1, H, S, K] / [H, S, K] shapes; all tensors are either float32
// without quantization or int8/int16 per-tensor quantized; the chunk count
// divides the split axis. The epilogue is only absorbed if `out` has a single
// Reshape user with the exact shapes above, a constant (0, 2, 1, 3)
// permutation and a per-tensor Quantize. Chunk ops produced by this rewrite
// (and the original root) never re-match.
LiteRtStatus AttentionChunkTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_MEDIATEK_COMPILER_TRANSFORMATIONS_ATTENTION_CHUNK_TRANSFORMATION_H_
