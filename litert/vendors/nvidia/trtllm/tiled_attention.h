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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_TILED_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_TILED_ATTENTION_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Tensor-core attention for prefill: many query rows against the keys of a KV
// cache followed by the keys of the chunk being prefilled,
//   scores[h][r][j] = q[h][r] . key[h][j]    (scores where the mask is false
//                                             become `fill`)
//   out[h][r]       = softmax_j(scores[h][r]) . value[h]
// q and out are [heads, rows, depth] in FP16 (`q_bf16` false) or BF16. Keys
// [0, cache_len) are the rows of k and v ([heads, cache_len, depth] FP16), keys
// [cache_len, cache_len + new_len) those of k_new and v_new ([heads, new_len,
// depth] FP16; new_len may be 0). mask is [mask_rows, cache_len + new_len]
// bool, or null when every key is visible; query row r of every head uses
// mask row r % mask_rows.
//
// A block owns 32768 / depth query rows of one head and walks the keys in
// tiles of 16 with an online softmax, so no [rows, keys] tensor is
// materialized. It stages each tile of K, then V, in shared memory and skips
// the tiles none of its rows can see. Keys no row can see must therefore not
// matter to the result, which holds when exp(fill - max score) underflows (any
// customary mask fill does); a row that sees no key at all still gets the
// softmax over all keys. The value products accumulate in FP16 over at most
// eight tiles, with weights relative to the largest score of their row so
// far, and in FP32 beyond. The scores of FP16 queries accumulate in FP32;
// those of BF16 queries, which are only precise to 8 bits, accumulate in FP16
// over at most 128 dims and in FP32 beyond.
struct LiteRtNvidiaAttentionShape {
  int32_t heads;
  int32_t rows;  // per head
  int32_t mask_rows;
  int32_t depth;
  int32_t cache_len;
  int32_t new_len;
};

// Depth 256 or 512, rows and mask_rows multiples of 32768 / depth, mask_rows
// a divisor of rows, cache_len a multiple of 16 when new_len > 0, and at most
// 262144 keys.
extern "C" bool LiteRtNvidiaTiledAttentionSupports(
    const LiteRtNvidiaAttentionShape* shape);

// Whether the current device can run the kernels (compute capability 8.0 or
// newer with about 100 KB of shared memory per block).
extern "C" bool LiteRtNvidiaTiledAttentionAvailable();

// Depends only on the shape.
extern "C" size_t LiteRtNvidiaTiledAttentionWorkspaceBytes(
    const LiteRtNvidiaAttentionShape* shape);

extern "C" cudaError_t LiteRtNvidiaLaunchTiledAttention(
    const LiteRtNvidiaAttentionShape* shape, const void* q, bool q_bf16,
    const void* k, const void* v, const void* k_new, const void* v_new,
    const bool* mask, float fill, void* out, void* workspace,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_TILED_ATTENTION_H_
