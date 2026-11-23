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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_INDEXER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_INDEXER_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Element types of the tensors the QSA kernels read and write.
enum LiteRtNvidiaQsaType : int32_t {
  kLiteRtNvidiaQsaFloat32 = 0,
  kLiteRtNvidiaQsaFloat16 = 1,
  kLiteRtNvidiaQsaBFloat16 = 2,
};

// How the QSA kernels read a mask [rows, mask_len]: bool (true sees the key)
// or float (additive; see the visibility rules below).
enum LiteRtNvidiaQsaMask : int32_t {
  kLiteRtNvidiaQsaMaskBool = 0,
  kLiteRtNvidiaQsaMaskFloat = 1,
};

// Tokens per compressed block of Qwen 4's sparse attention (QSA): the only
// compress ratio the composites define.
inline constexpr int32_t kLiteRtNvidiaQsaBlockTokens = 4;

// Qwen 4's sparse-attention indexer, odml.qsa_indexer_update: one step of
// `rows` tokens at positions start_pos + t, which
//  1. writes the incomplete block's raw keys into the ring [4, index_dim];
//  2. compresses every block the step completes into the pool
//     [max_blocks, index_dim]: the mean of its four keys, RMS-normalized
//     (x * rsqrt(mean(x^2) + eps) * norm_weight) and rotated (NeoX halves
//     over the first rotary_dim dims, cos/sin [max_blocks, rotary_dim] of the
//     block);
//  3. scores each row's query heads q [rows, index_heads, index_dim] against
//     the blocks below clamp((start_pos + rows) / 4, 0, max_blocks):
//     scale * sum_h relu(q_h . pool[b]) when all four of the block's keys are
//     visible to the row, none otherwise (and 0 for every complete block when
//     no more than block_topk blocks exist);
//  4. writes each row's block_topk best blocks into topk [rows, block_topk],
//     ties to the lower block, in ascending block order, padded with -1.
// The rows taken into the ring and pool are those up to the last one whose
// own key the mask shows (right-padded prefill chunks); a completed block
// clears the ring. Visibility in the mask: bool true; a float entry is
// visible for the ring (its diagonal) when above -1 and for the scores when
// exactly 0 (the GPU delegate's rules).
//
// ring and pool are updated in place (no other block of the launch reads
// what one writes). Every float is accumulated in FP32 and the result does
// not depend on the launch configuration.
struct LiteRtNvidiaQsaIndexerParams {
  int32_t rows;
  int32_t index_heads;  // at most 32
  int32_t index_dim;    // a multiple of 4, at most 256
  int32_t rotary_dim;   // even, at most index_dim
  int32_t max_blocks;
  int32_t block_topk;  // at most 4096
  int32_t mask_len;    // at least 4 * max_blocks
  int32_t q_type;      // LiteRtNvidiaQsaType of q and k_new
  int32_t state_type;  // of ring and pool: float32 or float16
  int32_t mask_type;   // LiteRtNvidiaQsaMask
  float rms_norm_eps;
  float scale;
};

extern "C" bool LiteRtNvidiaQsaIndexerSupports(
    const LiteRtNvidiaQsaIndexerParams* params);

// The scratch the launch needs: the scores of a chunk of rows.
extern "C" size_t LiteRtNvidiaQsaIndexerWorkspaceBytes(
    const LiteRtNvidiaQsaIndexerParams* params);

// q [rows, index_heads, index_dim] and k_new [rows, index_dim] in q_type;
// norm_weight [index_dim], cos and sin [max_blocks, rotary_dim] in FP32;
// start_pos a device int32 (negative counts as 0); mask [rows, mask_len].
extern "C" cudaError_t LiteRtNvidiaLaunchQsaIndexerUpdate(
    const LiteRtNvidiaQsaIndexerParams* params, const void* q,
    const void* k_new, void* ring, void* pool, const float* norm_weight,
    const float* cos, const float* sin, const int32_t* start_pos,
    const void* mask, int32_t* topk, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_INDEXER_H_
