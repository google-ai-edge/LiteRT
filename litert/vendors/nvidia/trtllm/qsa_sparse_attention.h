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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_SPARSE_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_SPARSE_ATTENTION_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"

// Visibility from the row model of the attention kernels instead of a mask
// (LiteRtNvidiaQsaRows, qsa_indexer.h).
inline constexpr int32_t kLiteRtNvidiaQsaMaskRows = 2;

// odml.sparse_sdpa_transposed, Qwen 4's sparse attention: query row r attends
// with softmax(scale * q . k) over the keys of its selected blocks (topk
// [rows, block_topk]: block b holds keys 4b .. 4b + 3, entries outside
// [0, cache_len / 4) are ignored, a repeated entry counts once) and of every
// block it sees some but not all keys of (the incomplete block at its end,
// and the cache_len % 4 keys past the last whole block), restricted to the
// keys it sees. A row that sees none outputs zero.
//
// q is [q_heads, rows, head_dim] and out [rows, q_heads, head_dim], both in
// q_type; k is [kv_heads, cache_len, head_dim] and v the same, or
// [kv_heads, head_dim, cache_len] with value_columns, in cache_type; query
// head h reads KV head h / (q_heads / kv_heads). The keys a row sees come
// from mask [rows, cache_len] (bool true, or float exactly 0) or, with
// mask_type kLiteRtNvidiaQsaMaskRows, from `row_model`. At most 64 blocks
// beyond the selected ones are taken per row (causal and sliding-window
// masks have one or two).
//
// One launch lists each row's blocks in ascending order (a pass over its
// mask row), one attends over them split into chunks, and with more than one
// chunk a third combines them. A single row runs on CUDA cores in FP32, the
// query heads of a KV head sharing each key and value load. Several rows
// over FP16 caches holding values as rows run on tensor cores: tiles of up to
// 16 rows (with the row model, consecutive rows of one slot, each slot's
// rows consecutive) walk the union of their blocks, each key and value
// loaded once for all of them, with the queries and the probabilities in
// FP16 and FP32 accumulation. The result does not depend on the device.
struct LiteRtNvidiaQsaAttentionParams {
  int32_t rows;
  int32_t q_heads;
  int32_t kv_heads;
  int32_t head_dim;  // 64, 128 or 256
  int32_t cache_len;
  int32_t block_topk;  // at most 4096
  int32_t q_type;      // LiteRtNvidiaQsaType of q and out
  int32_t cache_type;  // of k and v: float32 or float16
  int32_t value_columns;
  int32_t mask_type;  // LiteRtNvidiaQsaMask or kLiteRtNvidiaQsaMaskRows
  float scale;
};

extern "C" bool LiteRtNvidiaQsaAttentionSupports(
    const LiteRtNvidiaQsaAttentionParams* params);

extern "C" size_t LiteRtNvidiaQsaAttentionWorkspaceBytes(
    const LiteRtNvidiaQsaAttentionParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchQsaAttention(
    const LiteRtNvidiaQsaAttentionParams* params, const void* q, const void* k,
    const void* v, const int32_t* topk, const void* mask,
    const LiteRtNvidiaQsaRows* row_model, void* out, void* workspace,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_SPARSE_ATTENTION_H_
