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

#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/qsa_common.cuh"

namespace {

using litert::nvidia::qsa::BlockReduceMax;
using litert::nvidia::qsa::BlockReduceSum;
using litert::nvidia::qsa::kBlock;
using litert::nvidia::qsa::LoadAsFloat;
using litert::nvidia::qsa::LoadEightAsFloat;
using litert::nvidia::qsa::StoreFromFloat;
using litert::nvidia::qsa::ToFloat;
using litert::nvidia::qsa::WarpSum;

constexpr int kMaxIndexDim = 256;
constexpr int kMaxIndexHeads = 32;
constexpr int kMaxTopk = 4096;
// Scores of at most this many bytes are kept at once; longer prefill chunks
// score and select their rows in chunks.
constexpr size_t kScoreBytes = size_t{32} << 20;
// The score of a block a row cannot see, and the bound below which a score
// marks such a block (the GPU delegate's sentinels).
constexpr float kHiddenScore = -1.0e4f;
constexpr float kHiddenBelow = -9999.0f;

// The diagonal entry of a mask row: whether the step's row t holds a real
// token (right padding hides its own key).
__device__ __forceinline__ bool DiagonalVisible(const void* mask, int32_t type,
                                                int64_t index) {
  if (type == kLiteRtNvidiaQsaMaskBool) {
    return static_cast<const uint8_t*>(mask)[index] != 0;
  }
  return static_cast<const float*>(mask)[index] > -1.0f;
}

// Whether all four keys of block `block` are visible to a row whose mask row
// starts at `row`.
__device__ __forceinline__ bool BlockComplete(const void* mask, int32_t type,
                                              int64_t row, int32_t block) {
  const int64_t at = row + int64_t{4} * block;
  if (type == kLiteRtNvidiaQsaMaskBool) {
    const uint8_t* m = static_cast<const uint8_t*>(mask) + at;
    return m[0] != 0 && m[1] != 0 && m[2] != 0 && m[3] != 0;
  }
  const float* m = static_cast<const float*>(mask) + at;
  return m[0] == 0.0f && m[1] == 0.0f && m[2] == 0.0f && m[3] == 0.0f;
}

// The step's tokens up to the last one whose own key the mask shows.
__device__ int32_t ValidRows(const LiteRtNvidiaQsaIndexerParams& p,
                             const void* mask, int32_t pos0, float* scratch) {
  int32_t valid = 0;
  for (int32_t t = threadIdx.x; t < p.rows; t += blockDim.x) {
    const int64_t column = int64_t{pos0} + t;
    if (column < p.mask_len &&
        DiagonalVisible(mask, p.mask_type,
                        int64_t{t} * p.mask_len + column)) {
      valid = t + 1;
    }
  }
  return static_cast<int32_t>(
      BlockReduceMax(static_cast<float>(valid), scratch));
}

// The slot of row r of the row model (0 without slots), and whether r is
// the first row of its slot in the launch.
__device__ __forceinline__ int32_t SlotOf(const LiteRtNvidiaQsaRows& model,
                                          int32_t r) {
  return model.slot != nullptr ? model.slot[r] : 0;
}
__device__ __forceinline__ bool FirstOfSlot(const LiteRtNvidiaQsaRows& model,
                                            int32_t r) {
  return r == 0 || (model.slot != nullptr && model.slot[r - 1] != model.slot[r]);
}

// One CTA per block the step can complete, starting with the block of
// start_pos, which alone reads the incoming ring and writes the new one.
// With the row model (model.context_len set), the same per slot: the CTAs
// of row blockIdx.y write the rows of its slot when it is the slot's first,
// a linear chain from the slot's context_len.
template <typename State>
__global__ void __launch_bounds__(kBlock) QsaPoolUpdateKernel(
    LiteRtNvidiaQsaIndexerParams p, const void* __restrict__ k_new,
    State* ring, State* pool, const float* __restrict__ norm_weight,
    const float* __restrict__ cos, const float* __restrict__ sin,
    const int32_t* __restrict__ start_pos, const void* __restrict__ mask,
    LiteRtNvidiaQsaRows model, int64_t ring_stride, int64_t pool_stride) {
  __shared__ float old_ring[kLiteRtNvidiaQsaBlockTokens][kMaxIndexDim];
  __shared__ float pooled[kMaxIndexDim];
  __shared__ float scratch[32];
  const int32_t dim = p.index_dim;
  const bool row_model = model.context_len != nullptr;
  const int32_t first_row = static_cast<int32_t>(blockIdx.y);
  if (row_model && !FirstOfSlot(model, first_row)) return;
  if (row_model) {
    const int32_t slot = SlotOf(model, first_row);
    ring += slot * ring_stride;
    pool += slot * pool_stride;
  }
  const int32_t pos0 =
      max(row_model ? model.context_len[first_row] : start_pos[0], 0);
  const int32_t start_blk = pos0 / kLiteRtNvidiaQsaBlockTokens;
  const int32_t block = start_blk + static_cast<int32_t>(blockIdx.x);
  const bool owns_ring = blockIdx.x == 0;

  if (owns_ring) {
    for (int32_t i = threadIdx.x; i < kLiteRtNvidiaQsaBlockTokens * dim;
         i += blockDim.x) {
      old_ring[i / dim][i % dim] = ToFloat(ring[i]);
    }
  }
  int32_t valid = 0;
  if (row_model) {
    valid = 1;
    while (first_row + valid < p.rows &&
           !FirstOfSlot(model, first_row + valid)) {
      ++valid;
    }
    __syncthreads();
  } else {
    valid = ValidRows(p, mask, pos0, scratch);  // synchronizes
  }
  const int32_t end_pos = pos0 + valid;
  const int32_t end_blk = end_pos / kLiteRtNvidiaQsaBlockTokens;

  if (block < end_blk && block < p.max_blocks) {
    const int32_t base = block * kLiteRtNvidiaQsaBlockTokens;
    float square = 0.0f;
    for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
      float sum = 0.0f;
      for (int r = 0; r < kLiteRtNvidiaQsaBlockTokens; ++r) {
        const int32_t token = base + r;
        sum += token >= pos0
                   ? LoadAsFloat(k_new, p.q_type,
                                 int64_t{first_row + token - pos0} * dim + d)
                   : old_ring[r][d];
      }
      const float mean = sum * (1.0f / kLiteRtNvidiaQsaBlockTokens);
      pooled[d] = mean;
      square += mean * mean;
    }
    const float total = BlockReduceSum(square, scratch);
    const float inv_rms =
        rsqrtf(total * (1.0f / static_cast<float>(dim)) + p.rms_norm_eps);
    for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
      pooled[d] = pooled[d] * inv_rms * norm_weight[d];
    }
    __syncthreads();
    const int32_t half = p.rotary_dim / 2;
    const float* c = cos + int64_t{block} * p.rotary_dim;
    const float* s = sin + int64_t{block} * p.rotary_dim;
    State* out = pool + int64_t{block} * dim;
    for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
      float value = pooled[d];
      if (d < p.rotary_dim) {
        const float partner = d < half ? -pooled[d + half] : pooled[d - half];
        value = value * c[d] + partner * s[d];
      }
      StoreFromFloat(value, out + d);
    }
  }

  if (owns_ring && valid > 0) {
    const int32_t base = end_blk * kLiteRtNvidiaQsaBlockTokens;
    for (int32_t i = threadIdx.x; i < kLiteRtNvidiaQsaBlockTokens * dim;
         i += blockDim.x) {
      const int32_t r = i / dim;
      const int32_t d = i % dim;
      const int32_t token = base + r;
      float value = 0.0f;
      if (token >= pos0 && token < end_pos) {
        value = LoadAsFloat(k_new, p.q_type,
                            int64_t{first_row + token - pos0} * dim + d);
      } else if (token < pos0 && end_blk == start_blk) {
        value = old_ring[r][d];
      }
      StoreFromFloat(value, ring + i);
    }
  }
}

template <typename State>
__device__ __forceinline__ int32_t StateType() {
  return sizeof(State) == 2 ? kLiteRtNvidiaQsaFloat16 : kLiteRtNvidiaQsaFloat32;
}

// The scores of rows [row0, row0 + chunk) against the blocks, as a tiled
// product in FP32: a CTA takes kTileBlocks pool rows and the query vectors
// (rows x heads, at most kTileVectors) of a group of rows, staged through
// shared memory kTileDims dims at a time, each thread accumulating 4 pool
// rows x 4 vectors; then each row's relu-sum over its heads, in head order.
constexpr int kTileBlocks = 64;
constexpr int kTileVectors = 64;
constexpr int kTileDims = 32;
constexpr int kTilePad = 4;

// The relu'd products of `vectors` query vectors (from q_vector0) with the
// pool rows [block0, block0 + kTileBlocks) below `bound`, into
// shared[vector][block] (kTileBlocks per vector): the tiled product of
// QsaScoreKernel (which keeps its own copy: as a call it takes more registers
// and runs slower), staged kTileDims dims at a time.
template <typename State>
__device__ __forceinline__ void TileReluProducts(
    const LiteRtNvidiaQsaIndexerParams& p, const void* __restrict__ q,
    int64_t q_vector0, int32_t vectors, const State* __restrict__ pool,
    int32_t block0, int32_t bound, float* shared) {
  float (*q_tile)[kTileVectors + kTilePad] =
      reinterpret_cast<float (*)[kTileVectors + kTilePad]>(shared);
  float (*k_tile)[kTileBlocks + kTilePad] =
      reinterpret_cast<float (*)[kTileBlocks + kTilePad]>(
          shared + kTileDims * (kTileVectors + kTilePad));
  float (*products)[kTileBlocks] =
      reinterpret_cast<float (*)[kTileBlocks]>(shared);
  const int32_t dim = p.index_dim;
  const int tid = threadIdx.x;
  const int tb = tid % (kTileBlocks / 4);
  const int tv = tid / (kTileBlocks / 4);
  float acc[4][4] = {};
  for (int32_t d0 = 0; d0 < dim; d0 += kTileDims) {
    // Eight dims at a time (index_dim % 8 == 0 keeps the loads aligned).
    if (dim % 8 == 0) {
      for (int i = tid * 8; i < kTileVectors * kTileDims; i += kBlock * 8) {
        const int v = i / kTileDims;
        const int d = i % kTileDims;
        float eight[8] = {};
        if (v < vectors && d0 + d < dim) {
          LoadEightAsFloat(q, p.q_type,
                           (q_vector0 + v) * dim + d0 + d,
                           eight);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) q_tile[d + j][v] = eight[j];
      }
      for (int i = tid * 8; i < kTileBlocks * kTileDims; i += kBlock * 8) {
        const int b = i / kTileDims;
        const int d = i % kTileDims;
        const int32_t block = block0 + b;
        float eight[8] = {};
        if (block < bound && d0 + d < dim) {
          LoadEightAsFloat(pool, StateType<State>(),
                           int64_t{block} * dim + d0 + d, eight);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) k_tile[d + j][b] = eight[j];
      }
    } else {
      for (int i = tid; i < kTileVectors * kTileDims; i += kBlock) {
        const int v = i / kTileDims;
        const int d = i % kTileDims;
        q_tile[d][v] =
            v < vectors && d0 + d < dim
                ? LoadAsFloat(q, p.q_type,
                              (q_vector0 + v) * dim + d0 + d)
                : 0.0f;
      }
      for (int i = tid; i < kTileBlocks * kTileDims; i += kBlock) {
        const int b = i / kTileDims;
        const int d = i % kTileDims;
        const int32_t block = block0 + b;
        k_tile[d][b] = block < bound && d0 + d < dim
                           ? ToFloat(pool[int64_t{block} * dim + d0 + d])
                           : 0.0f;
      }
    }
    __syncthreads();
#pragma unroll 8
    for (int d = 0; d < kTileDims; ++d) {
      const float4 kv = *reinterpret_cast<const float4*>(&k_tile[d][tb * 4]);
      const float4 qv = *reinterpret_cast<const float4*>(&q_tile[d][tv * 4]);
      const float ks[4] = {kv.x, kv.y, kv.z, kv.w};
      const float qs[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(ks[a], qs[b], acc[a][b]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int b = 0; b < 4; ++b) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      products[tv * 4 + b][tb * 4 + a] = fmaxf(acc[a][b], 0.0f);
    }
  }
  __syncthreads();
}

template <typename State>
__global__ void __launch_bounds__(kBlock) QsaScoreKernel(
    LiteRtNvidiaQsaIndexerParams p, int32_t row0, int32_t group_rows,
    const void* __restrict__ q, const State* __restrict__ pool,
    const int32_t* __restrict__ start_pos, const void* __restrict__ mask,
    float* __restrict__ scores) {
  // The staged operands, then the relu'd products [vector][block].
  __shared__ __align__(16) float shared[2 * kTileDims *
                                        (kTileVectors + kTilePad)];
  float (*q_tile)[kTileVectors + kTilePad] =
      reinterpret_cast<float (*)[kTileVectors + kTilePad]>(shared);
  float (*k_tile)[kTileBlocks + kTilePad] =
      reinterpret_cast<float (*)[kTileBlocks + kTilePad]>(
          shared + kTileDims * (kTileVectors + kTilePad));
  float (*products)[kTileBlocks] =
      reinterpret_cast<float (*)[kTileBlocks]>(shared);
  const int32_t heads = p.index_heads;
  const int32_t dim = p.index_dim;
  const int32_t pos0 = max(start_pos[0], 0);
  const int32_t bound = min(
      max((pos0 + p.rows) / kLiteRtNvidiaQsaBlockTokens, 0), p.max_blocks);
  const bool dense = bound <= p.block_topk;
  const int32_t block0 = static_cast<int32_t>(blockIdx.x) * kTileBlocks;
  const int32_t first = row0 + static_cast<int32_t>(blockIdx.y) * group_rows;
  const int32_t rows = min(group_rows, p.rows - first);
  if (block0 >= bound || rows <= 0) return;
  const int32_t vectors = rows * heads;
  const int tid = threadIdx.x;
  const int tb = tid % (kTileBlocks / 4);
  const int tv = tid / (kTileBlocks / 4);

  if (!dense) {
    float acc[4][4] = {};
    for (int32_t d0 = 0; d0 < dim; d0 += kTileDims) {
      // Eight dims at a time (index_dim % 8 == 0 keeps the loads aligned).
      if (dim % 8 == 0) {
        for (int i = tid * 8; i < kTileVectors * kTileDims; i += kBlock * 8) {
          const int v = i / kTileDims;
          const int d = i % kTileDims;
          float eight[8] = {};
          if (v < vectors && d0 + d < dim) {
            LoadEightAsFloat(q, p.q_type,
                             (int64_t{first} * heads + v) * dim + d0 + d,
                             eight);
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) q_tile[d + j][v] = eight[j];
        }
        for (int i = tid * 8; i < kTileBlocks * kTileDims; i += kBlock * 8) {
          const int b = i / kTileDims;
          const int d = i % kTileDims;
          const int32_t block = block0 + b;
          float eight[8] = {};
          if (block < bound && d0 + d < dim) {
            LoadEightAsFloat(pool, StateType<State>(),
                             int64_t{block} * dim + d0 + d, eight);
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) k_tile[d + j][b] = eight[j];
        }
      } else {
        for (int i = tid; i < kTileVectors * kTileDims; i += kBlock) {
          const int v = i / kTileDims;
          const int d = i % kTileDims;
          q_tile[d][v] =
              v < vectors && d0 + d < dim
                  ? LoadAsFloat(q, p.q_type,
                                (int64_t{first} * heads + v) * dim + d0 + d)
                  : 0.0f;
        }
        for (int i = tid; i < kTileBlocks * kTileDims; i += kBlock) {
          const int b = i / kTileDims;
          const int d = i % kTileDims;
          const int32_t block = block0 + b;
          k_tile[d][b] = block < bound && d0 + d < dim
                             ? ToFloat(pool[int64_t{block} * dim + d0 + d])
                             : 0.0f;
        }
      }
      __syncthreads();
#pragma unroll 8
      for (int d = 0; d < kTileDims; ++d) {
        const float4 kv = *reinterpret_cast<const float4*>(&k_tile[d][tb * 4]);
        const float4 qv = *reinterpret_cast<const float4*>(&q_tile[d][tv * 4]);
        const float ks[4] = {kv.x, kv.y, kv.z, kv.w};
        const float qs[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
        for (int a = 0; a < 4; ++a) {
#pragma unroll
          for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(ks[a], qs[b], acc[a][b]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        products[tv * 4 + b][tb * 4 + a] = fmaxf(acc[a][b], 0.0f);
      }
    }
    __syncthreads();
  }
  for (int i = tid; i < rows * kTileBlocks; i += kBlock) {
    const int r = i / kTileBlocks;
    const int b = i % kTileBlocks;
    const int32_t block = block0 + b;
    if (block >= bound) continue;
    const int32_t row = first + r;
    float score = kHiddenScore;
    if (BlockComplete(mask, p.mask_type, int64_t{row} * p.mask_len, block)) {
      float sum = 0.0f;
      if (!dense) {
        for (int32_t h = 0; h < heads; ++h) sum += products[r * heads + h][b];
      }
      score = sum * p.scale;
    }
    scores[int64_t{row - row0} * p.max_blocks + block] = score;
  }
}

// ---------------------------------------------------------------------------
// The row model (LiteRtNvidiaLaunchQsaIndexerRows)
// ---------------------------------------------------------------------------

// Row r of the row model: its slot, its slot's committed tokens, the rows of
// the step before it on its path, and (linear chains) its slot's first row.
struct RowView {
  int32_t slot;
  int32_t context;
  int32_t rank;
  int32_t first;
};

__device__ __forceinline__ RowView ViewRow(const LiteRtNvidiaQsaRows& model,
                                           int32_t r) {
  RowView view;
  view.slot = SlotOf(model, r);
  view.context = max(model.context_len[r], 0);
  if (model.ancestors != nullptr) {
    view.first = -1;
    view.rank = __popcll(model.ancestors[r] & ~(uint64_t{1} << r));
  } else {
    int32_t first = r;
    while (!FirstOfSlot(model, first)) --first;
    view.first = first;
    view.rank = r - first;
  }
  return view;
}

// The row of the step at path position `rank` (< view.rank + 1) of row r.
__device__ __forceinline__ int32_t PathRow(const LiteRtNvidiaQsaRows& model,
                                           const RowView& view, int32_t r,
                                           int32_t rank) {
  if (model.ancestors == nullptr) return view.first + rank;
  uint64_t bits = model.ancestors[r] | (uint64_t{1} << r);
  for (int32_t i = 0; i < rank; ++i) bits &= bits - 1;
  return __ffsll(static_cast<long long>(bits)) - 1;
}

// The blocks below a row's position that its path completes (those from its
// slot's context on): a CTA per block and row pools the four keys (the
// slot's ring before the context, the path's rows after), as the pool would
// hold them, and scores it. Also writes each row's bound (the blocks complete
// at its position).
template <typename State>
__global__ void __launch_bounds__(kBlock) QsaRowsPathKernel(
    LiteRtNvidiaQsaIndexerParams p, LiteRtNvidiaQsaRows model,
    int64_t ring_stride, const void* __restrict__ q,
    const void* __restrict__ k_new, const State* __restrict__ ring,
    const float* __restrict__ norm_weight, const float* __restrict__ cos,
    const float* __restrict__ sin, float* __restrict__ scores,
    int32_t* __restrict__ bounds) {
  __shared__ float pooled[kMaxIndexDim];
  __shared__ float head_sums[kMaxIndexHeads];
  __shared__ float scratch[32];
  const int32_t r = static_cast<int32_t>(blockIdx.y);
  const RowView view = ViewRow(model, r);
  const int32_t position = view.context + view.rank;
  const int32_t bound =
      min((position + 1) / kLiteRtNvidiaQsaBlockTokens, p.max_blocks);
  if (blockIdx.x == 0 && threadIdx.x == 0) bounds[r] = bound;
  const int32_t block =
      view.context / kLiteRtNvidiaQsaBlockTokens + static_cast<int32_t>(blockIdx.x);
  if (block >= bound) return;
  float* out = scores + int64_t{r} * p.max_blocks + block;
  if (bound <= p.block_topk) {
    if (threadIdx.x == 0) *out = 0.0f;
    return;
  }
  const int32_t dim = p.index_dim;
  const State* slot_ring = ring + view.slot * ring_stride;
  float square = 0.0f;
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t < kLiteRtNvidiaQsaBlockTokens; ++t) {
      const int32_t token = block * kLiteRtNvidiaQsaBlockTokens + t;
      sum += token < view.context
                 ? ToFloat(slot_ring[int64_t{t} * dim + d])
                 : LoadAsFloat(k_new, p.q_type,
                               int64_t{PathRow(model, view, r,
                                               token - view.context)} *
                                       dim +
                                   d);
    }
    const float mean = sum * (1.0f / kLiteRtNvidiaQsaBlockTokens);
    pooled[d] = mean;
    square += mean * mean;
  }
  const float total = BlockReduceSum(square, scratch);
  const float inv_rms =
      rsqrtf(total * (1.0f / static_cast<float>(dim)) + p.rms_norm_eps);
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    pooled[d] = pooled[d] * inv_rms * norm_weight[d];
  }
  __syncthreads();
  // Rotated and rounded as the pool holds a block (QsaPoolUpdateKernel).
  const int32_t half = p.rotary_dim / 2;
  const float* c = cos + int64_t{block} * p.rotary_dim;
  const float* s = sin + int64_t{block} * p.rotary_dim;
  float rotated[kMaxIndexDim / kBlock + 1];
  int n = 0;
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x, ++n) {
    float value = pooled[d];
    if (d < p.rotary_dim) {
      const float partner = d < half ? -pooled[d + half] : pooled[d - half];
      value = value * c[d] + partner * s[d];
    }
    State held;
    StoreFromFloat(value, &held);
    rotated[n] = ToFloat(held);
  }
  __syncthreads();
  n = 0;
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x, ++n) {
    pooled[d] = rotated[n];
  }
  __syncthreads();
  // A warp per head: relu(q_h . block), summed in head order.
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  for (int32_t h = warp; h < p.index_heads; h += kBlock / 32) {
    float dot = 0.0f;
    for (int32_t d = lane; d < dim; d += 32) {
      dot += LoadAsFloat(q, p.q_type,
                         (int64_t{r} * p.index_heads + h) * dim + d) *
             pooled[d];
    }
    dot = WarpSum(dot);
    if (lane == 0) head_sums[h] = fmaxf(dot, 0.0f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int32_t h = 0; h < p.index_heads; ++h) sum += head_sums[h];
    *out = sum * p.scale;
  }
}

// The slot's pooled blocks below its context: the tiled product of
// QsaScoreKernel over groups of up to group_rows consecutive rows of one
// slot (a CTA per group start: its row's offset from the slot's first row a
// multiple of group_rows).
template <typename State>
__global__ void __launch_bounds__(kBlock) QsaRowsScoreKernel(
    LiteRtNvidiaQsaIndexerParams p, LiteRtNvidiaQsaRows model,
    int64_t pool_stride, int32_t group_rows, const void* __restrict__ q,
    const State* __restrict__ pool, float* __restrict__ scores) {
  __shared__ __align__(16) float shared[2 * kTileDims *
                                        (kTileVectors + kTilePad)];
  float (*products)[kTileBlocks] =
      reinterpret_cast<float (*)[kTileBlocks]>(shared);
  const int32_t first = static_cast<int32_t>(blockIdx.y);
  int32_t slot_first = first;
  while (!FirstOfSlot(model, slot_first)) --slot_first;
  if ((first - slot_first) % group_rows != 0) return;
  int32_t rows = 1;
  while (rows < group_rows && first + rows < p.rows &&
         !FirstOfSlot(model, first + rows)) {
    ++rows;
  }
  // The rows of a slot share its context; the largest bounds the tiles.
  int32_t committed = 0;
  for (int32_t r = 0; r < rows; ++r) {
    committed = max(committed,
                    min(max(model.context_len[first + r], 0) /
                            kLiteRtNvidiaQsaBlockTokens,
                        p.max_blocks));
  }
  const int32_t block0 = static_cast<int32_t>(blockIdx.x) * kTileBlocks;
  if (block0 >= committed) return;
  const int32_t heads = p.index_heads;
  const State* slot_pool = pool + SlotOf(model, first) * pool_stride;
  // Dense when a row has no more blocks than it selects: every score 0.
  bool any_sparse = false;
  for (int32_t r = 0; r < rows; ++r) {
    const RowView view = ViewRow(model, first + r);
    any_sparse |= min((view.context + view.rank + 1) /
                          kLiteRtNvidiaQsaBlockTokens,
                      p.max_blocks) > p.block_topk;
  }
  if (any_sparse) {
    TileReluProducts<State>(p, q, int64_t{first} * heads, rows * heads,
                            slot_pool, block0, committed, shared);
  }
  for (int i = threadIdx.x; i < rows * kTileBlocks; i += kBlock) {
    const int r = i / kTileBlocks;
    const int b = i % kTileBlocks;
    const int32_t block = block0 + b;
    const RowView view = ViewRow(model, first + r);
    const int32_t row_committed =
        min(view.context / kLiteRtNvidiaQsaBlockTokens, p.max_blocks);
    if (block >= row_committed) continue;
    const bool dense = min((view.context + view.rank + 1) /
                               kLiteRtNvidiaQsaBlockTokens,
                           p.max_blocks) <= p.block_topk;
    float sum = 0.0f;
    if (!dense) {
      for (int32_t h = 0; h < heads; ++h) sum += products[r * heads + h][b];
    }
    scores[int64_t{first + r} * p.max_blocks + block] = sum * p.scale;
  }
}

// An order-preserving key of a score: larger scores have larger keys, -0
// and +0 the same one.
__device__ __forceinline__ uint32_t ScoreKey(float score) {
  const uint32_t bits = __float_as_uint(score + 0.0f);
  return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

constexpr int kSelectThreads = 1024;

// Exclusive prefix sum over the CTA, in thread order; `total` gets the sum.
__device__ int32_t BlockExclusiveScan(int32_t value, int32_t* warp_sums,
                                      int32_t* total) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  int32_t inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int32_t other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  __syncthreads();
  if (lane == 31) warp_sums[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int warps = blockDim.x / 32;
    int32_t sum = lane < warps ? warp_sums[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int32_t other = __shfl_up_sync(0xffffffffu, sum, offset);
      if (lane >= offset) sum += other;
    }
    if (lane < warps) warp_sums[lane] = sum;  // inclusive over warps
  }
  __syncthreads();
  const int32_t before = warp == 0 ? 0 : warp_sums[warp - 1];
  *total = warp_sums[blockDim.x / 32 - 1];
  return before + inclusive - value;
}

// A row's top blocks by radix selection of the k-th largest key (four passes
// of eight bits), then their indices in ascending order: every block above
// the threshold and the lowest-indexed ones equal to it.
__global__ void __launch_bounds__(kSelectThreads) QsaSelectKernel(
    LiteRtNvidiaQsaIndexerParams p, int32_t row0,
    const int32_t* __restrict__ start_pos, const int32_t* __restrict__ bounds,
    const float* __restrict__ scores, int32_t* __restrict__ topk) {
  __shared__ int32_t histogram[256];
  __shared__ int32_t warp_sums[32];
  __shared__ int32_t shared_digit;
  __shared__ int32_t shared_remaining;
  const int32_t row = row0 + static_cast<int32_t>(blockIdx.x);
  // The row model's rows have bounds of their own.
  const int32_t bound =
      bounds != nullptr
          ? bounds[row]
          : min(max((max(start_pos[0], 0) + p.rows) /
                        kLiteRtNvidiaQsaBlockTokens,
                    0),
                p.max_blocks);
  const float* s = scores + int64_t{blockIdx.x} * p.max_blocks;
  int32_t* out = topk + int64_t{row} * p.block_topk;

  int32_t valid_local = 0;
  for (int32_t b = threadIdx.x; b < bound; b += blockDim.x) {
    valid_local += s[b] > kHiddenBelow ? 1 : 0;
  }
  int32_t valid = 0;
  BlockExclusiveScan(valid_local, warp_sums, &valid);
  const int32_t target = min(valid, p.block_topk);

  uint32_t prefix = 0;
  uint32_t prefix_mask = 0;
  int32_t remaining = target;  // keys equal to the threshold still to take
  if (target > 0) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = threadIdx.x; i < 256; i += blockDim.x) histogram[i] = 0;
      __syncthreads();
      // Whole warps step together so the ballot sees all their lanes.
      for (int32_t base = 0; base < bound; base += blockDim.x) {
        const int32_t b = base + static_cast<int32_t>(threadIdx.x);
        const float score = b < bound ? s[b] : kHiddenScore;
        const uint32_t key = ScoreKey(score);
        const bool counted =
            score > kHiddenBelow && (key & prefix_mask) == prefix;
        const uint32_t digit = (key >> shift) & 0xFFu;
        // One shared atomic per distinct digit of the warp: the scores of a
        // row crowd into few digits.
        const uint32_t active = __ballot_sync(0xffffffffu, counted);
        if (counted) {
          const uint32_t same = __match_any_sync(active, digit);
          if (static_cast<int>(threadIdx.x % 32) == __ffs(same) - 1) {
            atomicAdd(&histogram[digit], __popc(same));
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int32_t above = 0;
        int32_t digit = 255;
        for (; digit > 0; --digit) {
          if (above + histogram[digit] >= remaining) break;
          above += histogram[digit];
        }
        shared_digit = digit;
        shared_remaining = remaining - above;
      }
      __syncthreads();
      prefix |= static_cast<uint32_t>(shared_digit) << shift;
      prefix_mask |= 0xFFu << shift;
      remaining = shared_remaining;
      __syncthreads();
    }
  }
  const uint32_t threshold = prefix;

  // Contiguous runs of blocks per thread keep the output in block order.
  const int32_t per = (bound + kSelectThreads - 1) / kSelectThreads;
  const int32_t lo = min(bound, static_cast<int32_t>(threadIdx.x) * per);
  const int32_t hi = min(bound, lo + per);
  int32_t above = 0;
  int32_t equal = 0;
  if (target > 0) {
    for (int32_t b = lo; b < hi; ++b) {
      const float score = s[b];
      if (score <= kHiddenBelow || !(score == score)) continue;
      const uint32_t key = ScoreKey(score);
      above += key > threshold ? 1 : 0;
      equal += key == threshold ? 1 : 0;
    }
  }
  int32_t unused = 0;
  const int32_t equal_before = BlockExclusiveScan(equal, warp_sums, &unused);
  const int32_t take_equal =
      min(max(remaining - equal_before, 0), equal);
  int32_t offset = BlockExclusiveScan(above + take_equal, warp_sums, &unused);
  if (target > 0) {
    int32_t equal_left = take_equal;
    for (int32_t b = lo; b < hi; ++b) {
      const float score = s[b];
      if (score <= kHiddenBelow || !(score == score)) continue;
      const uint32_t key = ScoreKey(score);
      if (key > threshold || (key == threshold && equal_left > 0)) {
        if (key == threshold) --equal_left;
        out[offset++] = b;
      }
    }
  }
  for (int32_t i = target + threadIdx.x; i < p.block_topk; i += blockDim.x) {
    out[i] = -1;
  }
}

// The rows of up to this many blocks per thread (32 x 1024: a 128k context)
// keep every score of the row in registers.
constexpr int kKeysPerThread = 32;

// Sums three counts (each at most 32768 over the CTA) over the CTA, for every
// thread, behind one barrier; `sums` holds 128 ints (alternate calls use
// alternate halves, so the barrier also separates a call's writes from the
// reads of the call before).
__device__ __forceinline__ void BlockSumThreeAlternating(int32_t* a,
                                                         int32_t* b,
                                                         int32_t* c,
                                                         int32_t* sums,
                                                         int parity) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  // A warp's counts stay below 65536: two share a word.
  const uint32_t ab = __reduce_add_sync(
      0xffffffffu,
      static_cast<uint32_t>(*a) | (static_cast<uint32_t>(*b) << 16));
  const uint32_t cc =
      __reduce_add_sync(0xffffffffu, static_cast<uint32_t>(*c));
  int32_t* half = sums + parity * 64;
  if (lane == 0) {
    half[warp] = static_cast<int32_t>(ab);
    half[32 + warp] = static_cast<int32_t>(cc);
  }
  __syncthreads();
  const bool mine = lane < static_cast<int>(blockDim.x / 32);
  const uint32_t ab_lane = mine ? static_cast<uint32_t>(half[lane]) : 0u;
  const uint32_t c_lane = mine ? static_cast<uint32_t>(half[32 + lane]) : 0u;
  *a = static_cast<int32_t>(__reduce_add_sync(0xffffffffu, ab_lane & 0xFFFFu));
  *b = static_cast<int32_t>(__reduce_add_sync(0xffffffffu, ab_lane >> 16));
  *c = static_cast<int32_t>(__reduce_add_sync(0xffffffffu, c_lane));
}

// QsaSelectKernel for rows of at most kKeysPerThread blocks per thread,
// with the row's scores held in registers: the k-th largest key two bits at
// a time (16 counting passes over the registers behind one barrier each, no
// shared-memory histogram whose atomics the scores of a row crowd into a few
// bins), then the indices in ascending order from contiguous runs per
// thread.
__global__ void __launch_bounds__(kSelectThreads) QsaSelectInRegistersKernel(
    LiteRtNvidiaQsaIndexerParams p, int32_t row0,
    const int32_t* __restrict__ start_pos, const int32_t* __restrict__ bounds,
    const float* __restrict__ scores, int32_t* __restrict__ topk) {
  __shared__ int32_t sums[128];
  __shared__ int32_t warp_sums[32];
  const int32_t row = row0 + static_cast<int32_t>(blockIdx.x);
  // The row model's rows have bounds of their own.
  const int32_t bound =
      bounds != nullptr
          ? bounds[row]
          : min(max((max(start_pos[0], 0) + p.rows) /
                        kLiteRtNvidiaQsaBlockTokens,
                    0),
                p.max_blocks);
  const float* s = scores + int64_t{blockIdx.x} * p.max_blocks;
  int32_t* out = topk + int64_t{row} * p.block_topk;
  // Runs of a multiple of 4 blocks, so that a run starts 16-byte aligned.
  const int32_t per =
      ((bound + 4 * static_cast<int32_t>(blockDim.x) - 1) /
       (4 * static_cast<int32_t>(blockDim.x))) * 4;
  const int32_t first = static_cast<int32_t>(threadIdx.x) * per;
  // Key 0 marks a block that is no candidate (every candidate's key is
  // above it).
  uint32_t key[kKeysPerThread];
  int32_t valid_local = 0;
#pragma unroll
  for (int j = 0; j < kKeysPerThread; j += 4) {
    const int32_t b = first + j;
    float4 four = make_float4(kHiddenScore, kHiddenScore, kHiddenScore,
                              kHiddenScore);
    if (j < per && b + 3 < bound && p.max_blocks % 4 == 0) {
      four = *reinterpret_cast<const float4*>(s + b);
    } else if (j < per && b < bound) {
      four.x = s[b];
      if (b + 1 < bound) four.y = s[b + 1];
      if (b + 2 < bound) four.z = s[b + 2];
      if (b + 3 < bound) four.w = s[b + 3];
    }
    const float values[4] = {four.x, four.y, four.z, four.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bool valid = values[i] > kHiddenBelow;
      key[j + i] = valid ? ScoreKey(values[i]) : 0u;
      valid_local += valid ? 1 : 0;
    }
  }
  int32_t valid = valid_local;
  int32_t unused_a = 0;
  int32_t unused_b = 0;
  BlockSumThreeAlternating(&unused_a, &unused_b, &valid, sums, 0);
  const int32_t target = min(valid, p.block_topk);
  uint32_t threshold = 0;
  int parity = 1;
  if (target > 0) {
    for (int bit = 30; bit >= 0; bit -= 2) {
      // count(key >= threshold | d << bit) for d = 1, 2, 3: two packed sums.
      const uint32_t c1 = threshold | (1u << bit);
      const uint32_t c2 = threshold | (2u << bit);
      const uint32_t c3 = threshold | (3u << bit);
      int32_t n1 = 0;
      int32_t n2 = 0;
      int32_t n3 = 0;
#pragma unroll
      for (int j = 0; j < kKeysPerThread; ++j) {
        n1 += key[j] >= c1 ? 1 : 0;
        n2 += key[j] >= c2 ? 1 : 0;
        n3 += key[j] >= c3 ? 1 : 0;
      }
      BlockSumThreeAlternating(&n1, &n2, &n3, sums, parity);
      parity ^= 1;
      threshold = n3 >= target   ? c3
                  : n2 >= target ? c2
                  : n1 >= target ? c1
                                 : threshold;
    }
  }
  int32_t above = 0;
  int32_t equal = 0;
  if (target > 0) {
#pragma unroll
    for (int j = 0; j < kKeysPerThread; ++j) {
      above += key[j] > threshold ? 1 : 0;
      equal += key[j] == threshold ? 1 : 0;
    }
  }
  int32_t above_total = 0;
  BlockExclusiveScan(above, warp_sums, &above_total);
  int32_t unused = 0;
  const int32_t equal_before = BlockExclusiveScan(equal, warp_sums, &unused);
  const int32_t remaining = target - above_total;
  const int32_t take_equal = min(max(remaining - equal_before, 0), equal);
  int32_t offset = BlockExclusiveScan(above + take_equal, warp_sums, &unused);
  if (target > 0) {
    int32_t equal_left = take_equal;
#pragma unroll
    for (int j = 0; j < kKeysPerThread; ++j) {
      const bool taken = key[j] > threshold ||
                         (key[j] == threshold && equal_left > 0);
      if (key[j] == threshold && taken) --equal_left;
      if (taken) out[offset++] = first + j;
    }
  }
  for (int32_t i = target + threadIdx.x; i < p.block_topk; i += blockDim.x) {
    out[i] = -1;
  }
}

// Threads of the register selection: enough warps for runs of at most
// kKeysPerThread blocks.
int32_t SelectThreads(const LiteRtNvidiaQsaIndexerParams& p) {
  const int32_t threads = (p.max_blocks + kKeysPerThread - 1) / kKeysPerThread;
  return std::min(kSelectThreads, std::max(32, (threads + 31) / 32 * 32));
}

bool ValidTypes(const LiteRtNvidiaQsaIndexerParams& p) {
  return (p.q_type == kLiteRtNvidiaQsaFloat32 ||
          p.q_type == kLiteRtNvidiaQsaFloat16 ||
          p.q_type == kLiteRtNvidiaQsaBFloat16) &&
         (p.state_type == kLiteRtNvidiaQsaFloat32 ||
          p.state_type == kLiteRtNvidiaQsaFloat16) &&
         (p.mask_type == kLiteRtNvidiaQsaMaskBool ||
          p.mask_type == kLiteRtNvidiaQsaMaskFloat);
}

int32_t ChunkRows(const LiteRtNvidiaQsaIndexerParams& p) {
  const size_t row_bytes = static_cast<size_t>(p.max_blocks) * sizeof(float);
  const size_t rows = std::max<size_t>(1, kScoreBytes / row_bytes);
  return static_cast<int32_t>(std::min<size_t>(rows, p.rows));
}

// Rows per CTA of the score kernel: as many as kTileVectors query vectors
// hold.
int32_t GroupRows(const LiteRtNvidiaQsaIndexerParams& p) {
  return std::max(1, kTileVectors / p.index_heads);
}

template <typename State>
cudaError_t Launch(const LiteRtNvidiaQsaIndexerParams& p, const void* q,
                   const void* k_new, State* ring, State* pool,
                   const float* norm_weight, const float* cos, const float* sin,
                   const int32_t* start_pos, const void* mask, int32_t* topk,
                   float* scores, cudaStream_t stream) {
  const int32_t candidates = std::max(
      1, std::min(p.max_blocks, p.rows / kLiteRtNvidiaQsaBlockTokens + 2));
  QsaPoolUpdateKernel<State><<<candidates, kBlock, 0, stream>>>(
      p, k_new, ring, pool, norm_weight, cos, sin, start_pos, mask,
      LiteRtNvidiaQsaRows{}, 0, 0);
  const int32_t chunk = ChunkRows(p);
  const int32_t group = GroupRows(p);
  // A CTA per tile of the pool (the tiles past the blocks a step can see,
  // only known on the device, return at once).
  const int32_t tiles = (p.max_blocks + kTileBlocks - 1) / kTileBlocks;
  for (int32_t row0 = 0; row0 < p.rows; row0 += chunk) {
    const int32_t rows = std::min(chunk, p.rows - row0);
    const dim3 grid(tiles, (rows + group - 1) / group);
    QsaScoreKernel<State><<<grid, kBlock, 0, stream>>>(
        p, row0, group, q, pool, start_pos, mask, scores);
    if (p.max_blocks <= kKeysPerThread * kSelectThreads) {
      QsaSelectInRegistersKernel<<<rows, SelectThreads(p), 0, stream>>>(
          p, row0, start_pos, nullptr, scores, topk);
    } else {
      QsaSelectKernel<<<rows, kSelectThreads, 0, stream>>>(
          p, row0, start_pos, nullptr, scores, topk);
    }
  }
  return cudaPeekAtLastError();
}


// The rows of a launch of the row model: their scores fit the workspace's
// one chunk.
template <typename State>
cudaError_t LaunchRows(const LiteRtNvidiaQsaIndexerParams& p,
                       const LiteRtNvidiaQsaRows& model, int64_t ring_stride,
                       int64_t pool_stride, const void* q, const void* k_new,
                       State* ring, State* pool, const float* norm_weight,
                       const float* cos, const float* sin, int32_t* topk,
                       float* scores, int32_t* bounds, cudaStream_t stream) {
  if (q != nullptr) {
    // Path blocks: a step of n rows completes at most n / 4 + 2 blocks from
    // a context.
    const int32_t path_blocks = std::max(
        1, std::min(p.max_blocks, p.rows / kLiteRtNvidiaQsaBlockTokens + 2));
    QsaRowsPathKernel<State><<<dim3(path_blocks, p.rows), kBlock, 0, stream>>>(
        p, model, ring_stride, q, k_new, ring, norm_weight, cos, sin, scores,
        bounds);
    const int32_t tiles = (p.max_blocks + kTileBlocks - 1) / kTileBlocks;
    QsaRowsScoreKernel<State><<<dim3(tiles, p.rows), kBlock, 0, stream>>>(
        p, model, pool_stride, GroupRows(p), q, pool, scores);
    if (p.max_blocks <= kKeysPerThread * kSelectThreads) {
      QsaSelectInRegistersKernel<<<p.rows, SelectThreads(p), 0, stream>>>(
          p, 0, nullptr, bounds, scores, topk);
    } else {
      QsaSelectKernel<<<p.rows, kSelectThreads, 0, stream>>>(
          p, 0, nullptr, bounds, scores, topk);
    }
  }
  if (model.ancestors == nullptr) {
    const int32_t candidates = std::max(
        1, std::min(p.max_blocks, p.rows / kLiteRtNvidiaQsaBlockTokens + 2));
    QsaPoolUpdateKernel<State><<<dim3(candidates, p.rows), kBlock, 0, stream>>>(
        p, k_new, ring, pool, norm_weight, cos, sin, nullptr, nullptr, model,
        ring_stride, pool_stride);
  }
  return cudaPeekAtLastError();
}
}  // namespace

extern "C" bool LiteRtNvidiaQsaIndexerSupports(
    const LiteRtNvidiaQsaIndexerParams* params) {
  if (params == nullptr) return false;
  const auto& p = *params;
  return ValidTypes(p) && p.rows >= 1 && p.index_heads >= 1 &&
         p.index_heads <= kMaxIndexHeads && p.index_dim >= 4 &&
         p.index_dim <= kMaxIndexDim && p.index_dim % 4 == 0 &&
         p.rotary_dim >= 2 && p.rotary_dim <= p.index_dim &&
         p.rotary_dim % 2 == 0 && p.max_blocks >= 1 && p.block_topk >= 1 &&
         p.block_topk <= kMaxTopk &&
         int64_t{p.mask_len} >=
             int64_t{p.max_blocks} * kLiteRtNvidiaQsaBlockTokens;
}

extern "C" size_t LiteRtNvidiaQsaIndexerWorkspaceBytes(
    const LiteRtNvidiaQsaIndexerParams* params) {
  if (!LiteRtNvidiaQsaIndexerSupports(params)) return 0;
  return static_cast<size_t>(ChunkRows(*params)) * params->max_blocks *
         sizeof(float);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQsaIndexerUpdate(
    const LiteRtNvidiaQsaIndexerParams* params, const void* q,
    const void* k_new, void* ring, void* pool, const float* norm_weight,
    const float* cos, const float* sin, const int32_t* start_pos,
    const void* mask, int32_t* topk, void* workspace, cudaStream_t stream) {
  if (!LiteRtNvidiaQsaIndexerSupports(params) || q == nullptr ||
      k_new == nullptr || ring == nullptr || pool == nullptr ||
      norm_weight == nullptr || cos == nullptr || sin == nullptr ||
      start_pos == nullptr || mask == nullptr || topk == nullptr ||
      workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  float* scores = static_cast<float*>(workspace);
  if (params->state_type == kLiteRtNvidiaQsaFloat16) {
    return Launch(*params, q, k_new, static_cast<__half*>(ring),
                  static_cast<__half*>(pool), norm_weight, cos, sin, start_pos,
                  mask, topk, scores, stream);
  }
  return Launch(*params, q, k_new, static_cast<float*>(ring),
                static_cast<float*>(pool), norm_weight, cos, sin, start_pos,
                mask, topk, scores, stream);
}

extern "C" bool LiteRtNvidiaQsaIndexerRowsSupports(
    const LiteRtNvidiaQsaIndexerParams* params,
    const LiteRtNvidiaQsaRows* rows) {
  if (params == nullptr || rows == nullptr || rows->context_len == nullptr) {
    return false;
  }
  LiteRtNvidiaQsaIndexerParams p = *params;
  p.mask_type = kLiteRtNvidiaQsaMaskBool;
  p.mask_len = p.max_blocks * kLiteRtNvidiaQsaBlockTokens;
  return LiteRtNvidiaQsaIndexerSupports(&p) &&
         (rows->ancestors == nullptr || p.rows <= 64) &&
         ChunkRows(p) == p.rows;
}

extern "C" size_t LiteRtNvidiaQsaIndexerRowsWorkspaceBytes(
    const LiteRtNvidiaQsaIndexerParams* params) {
  if (params == nullptr) return 0;
  const size_t scores = static_cast<size_t>(params->rows) *
                        std::max(params->max_blocks, 0) * sizeof(float);
  return (scores + 255) / 256 * 256 +
         static_cast<size_t>(std::max(params->rows, 0)) * sizeof(int32_t);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQsaIndexerRows(
    const LiteRtNvidiaQsaIndexerParams* params,
    const LiteRtNvidiaQsaRows* rows, int64_t ring_stride, int64_t pool_stride,
    const void* q, const void* k_new, void* ring, void* pool,
    const float* norm_weight, const float* cos, const float* sin,
    int32_t* topk, void* workspace, cudaStream_t stream) {
  if (!LiteRtNvidiaQsaIndexerRowsSupports(params, rows) || k_new == nullptr ||
      ring == nullptr || pool == nullptr || norm_weight == nullptr ||
      cos == nullptr || sin == nullptr || (q == nullptr) != (topk == nullptr) ||
      (q != nullptr && workspace == nullptr) ||
      (q == nullptr && rows->ancestors != nullptr)) {
    return cudaErrorInvalidValue;
  }
  const auto& p = *params;
  float* scores = static_cast<float*>(workspace);
  int32_t* bounds = reinterpret_cast<int32_t*>(
      static_cast<char*>(workspace) +
      (static_cast<size_t>(p.rows) * p.max_blocks * sizeof(float) + 255) /
          256 * 256);
  if (p.state_type == kLiteRtNvidiaQsaFloat16) {
    return LaunchRows(p, *rows, ring_stride, pool_stride, q, k_new,
                      static_cast<__half*>(ring), static_cast<__half*>(pool),
                      norm_weight, cos, sin, topk, scores, bounds, stream);
  }
  return LaunchRows(p, *rows, ring_stride, pool_stride, q, k_new,
                    static_cast<float*>(ring), static_cast<float*>(pool),
                    norm_weight, cos, sin, topk, scores, bounds, stream);
}
