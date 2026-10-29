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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decomposes a Grouped Query Attention (GQA) block with split past/current KV
// caches into per-head / per-KV-group attention slices along the head axis so
// that QNN HTP executes compact rank-4 [1, 1, ...] BatchMatmuls and Softmaxes
// in VTCM without multi-head rank-4 broadcast transpose overhead.
//
// Let:
//   H_kv    = number of KV heads
//   G       = query group size (H_q = H_kv * G query heads)
//   S       = query sequence length (S = 1 for decode, S > 1 for prefill)
//   D, D_v  = head dim for Q/K and V
//   L_cache = past KV cache capacity, L_curr = S, L_total = L_cache + L_curr
//
// Before (root = final Reshape to [1, S, H_q * D_v]):
//
//   q_source [1, H_q, S, D]
//         | Reshape
//         v
//   q_grouped [1, H_kv, G*S, D]
//        /                     \
//       |  k_cache              |  k_curr
//       |  [1,H_kv,D,L_cache]   |  [1,H_kv,D,L_curr]
//       v                       v
//    BatchMatmul             BatchMatmul
//   [1,H_kv,G*S,L_cache]    [1,H_kv,G*S,L_curr]
//        \                     /
//         +── Concat(axis=3) ─+
//                   | [1, H_kv, G*S, L_total]
//                   |                              base_mask [1, 1, S, L_total]
//                   |                                 (or Concat x G on axis 2)
//                   +────────────── Add <──────────────────────+
//                                    | [1, H_kv, G*S, L_total]
//                                 Softmax
//                                 /     \
//         Slice [.., :L_cache]  /         \  Slice [.., L_cache:]
//        [1, H_kv, G*S, L_cache]           [1, H_kv, G*S, L_curr]
//                 |                                  |
//                 | v_cache [1,H_kv,L_cache,D_v]     | v_curr
//                 [1,H_kv,L_curr,D_v] v                                  v
//            BatchMatmul                        BatchMatmul
//                 \                                  /
//                  +────────────── Add ─────────────+
//                                   | attn_out [1, H_kv, G*S, D_v]
//                     (optional prefill epilogue:
//                      Reshape [1, H_q, S, D_v] -> Transpose(0,2,1,3))
//                                   |
//                            Reshape (root)
//                                   |
//                       final_out [1, S, H_q * D_v]
//
// After — Case A: Single-token Decode (S == 1, G > 1, "Grouped GQA Decode"):
//   Stacks the G query heads sharing each KV head into a single [1, 1, G, D]
//   slice (H_kv slices total instead of H_q), reading each KV head once and
//   reducing attention op count by a factor of G:
//
//   q_source  -> Reshape [1, H_kv, G, D] -> Split(ax=1, H_kv) -> q_g [1, 1, G,
//   D] k_cache   -> Split(ax=1, H_kv)                            -> kc  [1, 1,
//   D, L_cache] k_curr    -> Split(ax=1, H_kv)                            -> kn
//   [1, 1, D, 1] v_cache   -> Split(ax=1, H_kv)                            ->
//   vc  [1, 1, L_cache, D_v] v_curr    -> Split(ax=1, H_kv) -> vn  [1, 1, 1,
//   D_v] base_mask -> Reshape [1, 1, 1, L_total] -> Concat(ax=2, G)-> m_g [1,
//   1, G, L_total]
//
//   For each KV head h_kv in 0 .. H_kv - 1:
//     BMM(q_g, kc) [1,1,G,L_cache]   BMM(q_g, kn) [1,1,G,1]
//                 \                 /
//                  Concat(axis=3)  [1, 1, G, L_total]
//                        |
//                  Add(., m_g)     [1, 1, G, L_total]
//                        |
//                     Softmax      [1, 1, G, L_total]
//                     /     \
//       Slice(:L_cache)     Slice(L_cache:)
//              |                   |
//         BMM(., vc)          BMM(., vn)
//              \                   /
//               +────── Add ──────+  --> group_out[h_kv] [1, 1, G, D_v]
//
//   Concat(group_out[0..H_kv-1], axis=1) [1, H_kv, G, D_v]
//     --> Reshape --> final_out [1, 1, H_q * D_v]
//
// After — Case B: Multi-token Prefill (S > 1, "Single-Head Attention / SHA"):
//   Splits Q into H_q single-head slices [1, 1, S, D] and K/V into H_kv slices,
//   running H_q independent [1, 1, S, ...] attention branches (where query head
//   q_idx = h_kv * G + g reuses KV head h_kv and `base_mask` [1, 1, S, L_total]
//   directly) and concatenating along axis=3 into [1, 1, S, H_q * D_v] ->
//   Reshape -> final_out [1, S, H_q * D_v].
LiteRtStatus GqaTransformation(const LiteRtCompilerContext* context,
                               LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_
