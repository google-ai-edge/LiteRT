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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decomposes a Grouped Query Attention (GQA) block with separate cache/slice
// K/V inputs into per-head / per-KV-group attention slices along the head axis
// so that QNN HTP executes compact rank-4 [1, 1, ...] BatchMatmuls and
// Softmaxes in VTCM without multi-head rank-4 broadcast transpose overhead.
//
// Let:
//   H_q       = q_heads
//   H_kv      = kv_heads
//   G         = group_size (H_q = H_kv * G)
//   S         = sequence_length (S = 1 for decode, S > 1 for prefill)
//   D         = head_dim for Q/K/V
//   L_cache   = cache_length
//   L_slice   = slice_length
//   L_context = L_cache + L_slice
//
// Before (root = final Reshape to [1, S, H_q * D]):
//
//   q [1, H_q, S, D]
//         | Reshape
//         v
//   q_grouped [1, H_kv, G*S, D]
//        /                     \
//       |  k_cache              |  k_slice
//       |  [1,H_kv,D,L_cache]   |  [1,H_kv,D,L_slice]
//       v                       v
//    BatchMatmul             BatchMatmul
//   qk_cache                qk_slice
//   [1,H_kv,G*S,L_cache]    [1,H_kv,G*S,L_slice]
//        \                     /
//         +── Concat(axis=3) ─+
//                   | qk_concat [1, H_kv, G*S, L_context]
//                   |                              mask [1, 1, S, L_context]
//                   |                                 (or Concat x G on axis 2)
//                   +────────────── Add <──────────────────────+
//                                    | attention_logits
//                                    | [1, H_kv, G*S, L_context]
//                                 Softmax
//                                    | attention_weights
//                                 /     \
//         Slice [.., :L_cache]  /         \  Slice [.., L_cache:]
//        p_cache                            p_slice
//        [1, H_kv, G*S, L_cache]           [1, H_kv, G*S, L_slice]
//                 |                                  |
//                 | v_cache [1,H_kv,L_cache,D]       | v_slice
//                 |                                  | [1,H_kv,L_slice,D]
//                 v                                  v
//            BatchMatmul                        BatchMatmul
//              pv_cache                           pv_slice
//                 \                                  /
//                  +────────────── Add ─────────────+
//                                   | attention_output [1, H_kv, G*S, D]
//                     (optional prefill epilogue:
//                      Reshape [1, H_q, S, D] -> Transpose(0,2,1,3))
//                                   |
//                            Reshape (root)
//                                   |
//                       final_output [1, S, H_q * D]
//
// After — Case A: Single-token Decode (S == 1, G > 1, "Grouped GQA Decode"):
//   Stacks the G query heads sharing each KV head into a single [1, 1, G, D]
//   slice (H_kv slices total instead of H_q), reading each KV head once and
//   reducing attention op count by a factor of G:
//
//   q       -> Reshape [1, H_kv, G, D] -> Split(ax=1, H_kv) -> q_g
//              [1, 1, G, D]
//   k_cache -> Split(ax=1, H_kv) -> kc [1, 1, D, L_cache]
//   k_slice -> Split(ax=1, H_kv) -> ks [1, 1, D, L_slice]
//   v_cache -> Split(ax=1, H_kv) -> vc [1, 1, L_cache, D]
//   v_slice -> Split(ax=1, H_kv) -> vs [1, 1, L_slice, D]
//   mask    -> Reshape [1, 1, 1, L_context] -> Concat(ax=2, G)
//              -> m_g [1, 1, G, L_context]
//
//   For each KV head h_kv in 0 .. H_kv - 1:
//     BMM(q_g, kc) [1,1,G,L_cache]   BMM(q_g, ks) [1,1,G,L_slice]
//                 \                 /
//                  Concat(axis=3)  [1, 1, G, L_context]
//                        |
//                  Add(., m_g)     [1, 1, G, L_context]
//                        |
//                     Softmax      [1, 1, G, L_context]
//                     /     \
//       Slice(:L_cache)     Slice(L_cache:)
//              |                   |
//         BMM(., vc)          BMM(., vs)
//              \                   /
//               +────── Add ──────+  --> group_out[h_kv] [1, 1, G, D]
//
//   Concat(group_out[0..H_kv-1], axis=1) [1, H_kv, G, D]
//     --> Reshape --> final_output [1, 1, H_q * D]
//
// After — Case B: Multi-token Prefill (S > 1, "Single-Head Attention / SHA"):
//   Splits Q into H_q single-head slices [1, 1, S, D] and K/V into H_kv slices,
//   running H_q independent [1, 1, S, ...] attention branches (where query head
//   q_idx = h_kv * G + g reuses KV head h_kv and mask [1, 1, S, L_context]
//   directly) and concatenating along axis=3 into [1, 1, S, H_q * D] ->
//   Reshape -> final_output [1, S, H_q * D].
LiteRtStatus GqaTransformation(const LiteRtCompilerContext* context,
                               LiteRtBuilder builder_ptr, LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_GQA_TRANSFORMATION_H_
