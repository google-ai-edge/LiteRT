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

#include "litert/vendors/nvidia/trtllm/qsa_sparse_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/qsa_common.cuh"

namespace {

using litert::nvidia::qsa::LoadAsFloat;
using litert::nvidia::qsa::LoadEightAsFloat;
using litert::nvidia::qsa::StoreFromFloat;
using litert::nvidia::qsa::ToFloat;
using litert::nvidia::qsa::WarpSum;

constexpr int kMaxExtraBlocks = 64;
constexpr int kMaxTopk = 4096;
constexpr int64_t kMaxCacheLen = int64_t{1} << 20;
constexpr int kListThreads = 1024;
// The attention CTA: four warps, each walking one row's block list (or a
// quarter of it for a single row).
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
// CTAs a launch aims for, and the fewest blocks a chunk of a list holds.
constexpr int kTargetCtas = 256;
constexpr int kMinBlocksPerSplit = 16;

// Which of a block's four keys row `row` sees, as bits 0..3.
struct Visibility {
  int32_t type;
  const void* mask;
  LiteRtNvidiaQsaRows model;
  int32_t rows;
  int32_t cache_len;

  __device__ bool Token(int32_t row, int32_t token) const {
    if (token < model.context_len[row]) return true;
    if (model.ancestors != nullptr) {
      uint64_t bits = model.ancestors[row];
      while (bits != 0) {
        const int j = __ffsll(static_cast<long long>(bits)) - 1;
        bits &= bits - 1;
        if (j < rows && model.write_index[j] == token) return true;
      }
      return false;
    }
    const int32_t slot = model.slot != nullptr ? model.slot[row] : 0;
    for (int32_t j = 0; j <= row; ++j) {
      if (model.write_index[j] == token &&
          (model.slot == nullptr || model.slot[j] == slot)) {
        return true;
      }
    }
    return false;
  }

  __device__ uint32_t Block(int32_t row, int32_t block) const {
    const int64_t at = int64_t{row} * cache_len + int64_t{4} * block;
    // Whole blocks of a cache of a multiple of 4 keys are aligned: one load.
    if (cache_len % 4 == 0 && type == kLiteRtNvidiaQsaMaskBool) {
      const uint32_t bytes =
          *reinterpret_cast<const uint32_t*>(
              static_cast<const uint8_t*>(mask) + at);
      return ((bytes & 0xFFu) != 0 ? 1u : 0u) |
             ((bytes & 0xFF00u) != 0 ? 2u : 0u) |
             ((bytes & 0xFF0000u) != 0 ? 4u : 0u) |
             ((bytes & 0xFF000000u) != 0 ? 8u : 0u);
    }
    if (cache_len % 4 == 0 && type == kLiteRtNvidiaQsaMaskFloat) {
      const float4 m =
          *reinterpret_cast<const float4*>(static_cast<const float*>(mask) + at);
      return (m.x == 0.0f ? 1u : 0u) | (m.y == 0.0f ? 2u : 0u) |
             (m.z == 0.0f ? 4u : 0u) | (m.w == 0.0f ? 8u : 0u);
    }
    // Keys past the cache (the short last block) are not seen.
    uint32_t bits = 0;
    for (int t = 0; t < 4; ++t) {
      const int32_t token = 4 * block + t;
      if (token >= cache_len) break;
      bool seen;
      if (type == kLiteRtNvidiaQsaMaskBool) {
        seen = static_cast<const uint8_t*>(mask)[at + t] != 0;
      } else if (type == kLiteRtNvidiaQsaMaskFloat) {
        seen = static_cast<const float*>(mask)[at + t] == 0.0f;
      } else {
        seen = Token(row, token);
      }
      if (seen) bits |= 1u << t;
    }
    return bits;
  }
};

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
    if (lane < warps) warp_sums[lane] = sum;
  }
  __syncthreads();
  const int32_t before = warp == 0 ? 0 : warp_sums[warp - 1];
  *total = warp_sums[blockDim.x / 32 - 1];
  return before + inclusive - value;
}

// Writes the set bits of `bitmap` (`words` words in shared memory) as
// ascending indices into `list`, at most `capacity`, and their count (capped)
// into `count`. Every thread of the CTA takes part.
__device__ void CompactBitmap(const uint32_t* bitmap, int32_t words,
                              int32_t capacity, int32_t* list, int32_t* count,
                              int32_t* warp_sums) {
  const int32_t per = (words + blockDim.x - 1) / blockDim.x;
  const int32_t lo = min(words, static_cast<int32_t>(threadIdx.x) * per);
  const int32_t hi = min(words, lo + per);
  int32_t mine = 0;
  for (int32_t w = lo; w < hi; ++w) mine += __popc(bitmap[w]);
  int32_t total = 0;
  int32_t offset = BlockExclusiveScan(mine, warp_sums, &total);
  for (int32_t w = lo; w < hi && offset < capacity; ++w) {
    uint32_t bits = bitmap[w];
    while (bits != 0 && offset < capacity) {
      const int bit = __ffs(bits) - 1;
      bits &= bits - 1;
      list[offset++] = w * 32 + bit;
    }
  }
  if (threadIdx.x == 0) *count = min(total, capacity);
}

// A row's blocks: the selected ones and those it sees partly, as a bitmap
// over the cache's blocks built in shared memory, written out (`bitmaps`,
// for the tensor-core path) or listed in ascending order (`lists`).
__global__ void __launch_bounds__(kListThreads) QsaListKernel(
    LiteRtNvidiaQsaAttentionParams p, const int32_t* __restrict__ topk,
    Visibility visibility, int32_t capacity, int32_t* __restrict__ lists,
    int32_t* __restrict__ counts, uint32_t* __restrict__ bitmaps) {
  extern __shared__ uint32_t bitmap[];
  __shared__ int32_t warp_sums[32];
  const int32_t row = blockIdx.x;
  // Selectable blocks; a short last block (cache_len % 4 keys) is seen
  // partly whenever any of its keys is.
  const int32_t blocks = p.cache_len / 4;
  const int32_t all_blocks = (p.cache_len + 3) / 4;
  const int32_t words = (all_blocks + 31) / 32;
  for (int32_t w = threadIdx.x; w < words; w += blockDim.x) bitmap[w] = 0;
  __syncthreads();
  for (int32_t b = threadIdx.x; b < all_blocks; b += blockDim.x) {
    const uint32_t seen = visibility.Block(row, b);
    if (seen != 0 && seen != 0xFu) {
      atomicOr(&bitmap[b / 32], 1u << (b % 32));
    }
  }
  for (int32_t i = threadIdx.x; i < p.block_topk; i += blockDim.x) {
    const int32_t b = topk[int64_t{row} * p.block_topk + i];
    if (b >= 0 && b < blocks) atomicOr(&bitmap[b / 32], 1u << (b % 32));
  }
  __syncthreads();
  if (bitmaps != nullptr) {
    for (int32_t w = threadIdx.x; w < words; w += blockDim.x) {
      bitmaps[int64_t{row} * words + w] = bitmap[w];
    }
    return;
  }
  CompactBitmap(bitmap, words, capacity, lists + int64_t{row} * capacity,
                counts + row, warp_sums);
}

// The blocks some row of a tile of rows attends to, in ascending order (the
// union of the rows' bitmaps), each with the rows that attend to it (bit r
// for row r of the tile, in `members`).
__global__ void __launch_bounds__(kListThreads) QsaTileListKernel(
    LiteRtNvidiaQsaAttentionParams p, int32_t tile_rows,
    const uint32_t* __restrict__ bitmaps, int32_t capacity,
    int32_t* __restrict__ lists, uint32_t* __restrict__ members,
    int32_t* __restrict__ counts) {
  extern __shared__ uint32_t bitmap[];
  __shared__ int32_t warp_sums[32];
  const int32_t tile = blockIdx.x;
  const int32_t row0 = tile * tile_rows;
  const int32_t rows = min(tile_rows, p.rows - row0);
  const int32_t words = ((p.cache_len + 3) / 4 + 31) / 32;
  for (int32_t w = threadIdx.x; w < words; w += blockDim.x) {
    uint32_t any = 0;
    for (int32_t r = 0; r < rows; ++r) {
      any |= bitmaps[int64_t{row0 + r} * words + w];
    }
    bitmap[w] = any;
  }
  __syncthreads();
  int32_t* list = lists + int64_t{tile} * capacity;
  CompactBitmap(bitmap, words, capacity, list, counts + tile, warp_sums);
  __syncthreads();
  uint32_t* member = members + int64_t{tile} * capacity;
  for (int32_t i = threadIdx.x; i < counts[tile]; i += blockDim.x) {
    const int32_t block = list[i];
    uint32_t rows_in = 0;
    for (int32_t r = 0; r < rows; ++r) {
      rows_in |= ((bitmaps[int64_t{row0 + r} * words + block / 32] >>
                   (block % 32)) &
                  1u)
                 << r;
    }
    member[i] = rows_in;
  }
}

template <typename Cache>
struct Vec;
// Loads kN (2, 4 or 8) consecutive cache values as FP32, in the widest
// aligned vectors: a lane's slice starts at a multiple of kN.
template <>
struct Vec<float> {
  template <int kN>
  __device__ static void Load(const float* at, float (&out)[kN]) {
    if constexpr (kN % 4 == 0) {
#pragma unroll
      for (int i = 0; i < kN; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(at + i);
        out[i] = v.x;
        out[i + 1] = v.y;
        out[i + 2] = v.z;
        out[i + 3] = v.w;
      }
    } else {
#pragma unroll
      for (int i = 0; i < kN; i += 2) {
        const float2 v = *reinterpret_cast<const float2*>(at + i);
        out[i] = v.x;
        out[i + 1] = v.y;
      }
    }
  }
};
template <>
struct Vec<__half> {
  template <int kN>
  __device__ static void Load(const __half* at, float (&out)[kN]) {
    if constexpr (kN % 8 == 0) {
#pragma unroll
      for (int i = 0; i < kN; i += 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(at + i);
        const __half2* pairs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 v = __half22float2(pairs[j]);
          out[i + 2 * j] = v.x;
          out[i + 2 * j + 1] = v.y;
        }
      }
    } else if constexpr (kN % 4 == 0) {
      const uint2 raw = *reinterpret_cast<const uint2*>(at);
      const __half2* pairs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float2 v = __half22float2(pairs[j]);
        out[2 * j] = v.x;
        out[2 * j + 1] = v.y;
      }
    } else {
      const float2 v = __half22float2(*reinterpret_cast<const __half2*>(at));
      out[0] = v.x;
      out[1] = v.y;
    }
  }
};

// Clang may retain loops despite the kernel's best-effort unroll hints, and
// reports the failed transformation at the kernel's declaration.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
// Attention of up to kHeads query heads of one KV head over chunk `split` of
// the block lists of up to four rows. Lane l holds dims [l * kPerLane,
// (l + 1) * kPerLane) of every head's query, accumulator, keys and values.
template <int kDim, int kHeads, typename Cache>
__global__ void __launch_bounds__(kThreads) QsaAttendKernel(
    LiteRtNvidiaQsaAttentionParams p, int32_t head_chunks,
    int32_t rows_per_cta, int32_t splits, const void* __restrict__ q,
    const Cache* __restrict__ k, const Cache* __restrict__ v,
    const int32_t* __restrict__ lists, const int32_t* __restrict__ counts,
    int32_t capacity, Visibility visibility, void* __restrict__ out,
    float* __restrict__ partial) {
  constexpr int kPerLane = kDim / 32;
  extern __shared__ float shared[];  // queries, then the warps' merge
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int32_t group = p.q_heads / p.kv_heads;
  const int32_t kv_head = blockIdx.x / head_chunks;
  const int32_t head0 = kv_head * group + (blockIdx.x % head_chunks) * kHeads;
  const int32_t heads =
      min(kHeads, kv_head * group + group - head0);  // of this CTA
  const int32_t split = blockIdx.z;
  const int warps_per_row = kWarps / rows_per_cta;
  const int slot_in_cta = warp / warps_per_row;
  const int sub = warp % warps_per_row;
  const int32_t row =
      static_cast<int32_t>(blockIdx.y) * rows_per_cta + slot_in_cta;
  const bool active = row < p.rows;

  // Stage the queries of this CTA's rows as FP32, eight at a time.
  float* staged = shared;  // [rows_per_cta][kHeads][kDim]
#pragma unroll 2
  for (int32_t i = threadIdx.x * 8; i < rows_per_cta * kHeads * kDim;
       i += blockDim.x * 8) {
    const int32_t r = static_cast<int32_t>(blockIdx.y) * rows_per_cta +
                      i / (kHeads * kDim);
    const int32_t h = (i / kDim) % kHeads;
    const int32_t d = i % kDim;
    float eight[8] = {};
    if (r < p.rows && h < heads) {
      LoadEightAsFloat(q, p.q_type,
                       (int64_t{head0 + h} * p.rows + r) * kDim + d, eight);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) staged[i + j] = eight[j];
  }
  __syncthreads();
  const float* my_q = staged + slot_in_cta * kHeads * kDim + lane * kPerLane;

  float acc[kHeads][kPerLane];
  float m[kHeads];
  float l[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) acc[h][i] = 0.0f;
  }

  int64_t slot_offset = 0;
  if (active && p.mask_type == kLiteRtNvidiaQsaMaskRows &&
      visibility.model.slot != nullptr) {
    slot_offset = int64_t{visibility.model.slot[row]} *
                  visibility.model.slot_stride;
  }
  const int64_t head_size = int64_t{p.cache_len} * kDim;
  const Cache* kh = k + slot_offset + kv_head * head_size;
  const Cache* vh = v + slot_offset + kv_head * head_size;
  const int32_t count = active ? counts[row] : 0;
  const int32_t per = (count + splits - 1) / splits;
  const int32_t lo = min(count, split * per);
  const int32_t hi = min(count, lo + per);
  const int32_t* list = lists + int64_t{row} * capacity;

  for (int32_t i = lo + sub; i < hi; i += warps_per_row) {
    const int32_t block = list[i];
    const uint32_t seen = visibility.Block(row, block);
    if (seen == 0) continue;
    const int32_t token0 = 4 * block;
    // Every load at once; a short last block reads its last key again for
    // the keys past the cache, which the row does not see.
    int32_t tokens[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) tokens[t] = min(token0 + t, p.cache_len - 1);
    float key[4][kPerLane];
    float value[4][kPerLane];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      Vec<Cache>::Load(kh + int64_t{tokens[t]} * kDim + lane * kPerLane,
                       key[t]);
    }
    if (p.value_columns) {
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const Cache* column = vh + int64_t{lane * kPerLane + j} * p.cache_len;
#pragma unroll
        for (int t = 0; t < 4; ++t) value[t][j] = ToFloat(column[tokens[t]]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        Vec<Cache>::Load(vh + int64_t{tokens[t]} * kDim + lane * kPerLane,
                         value[t]);
      }
    }
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      if (h >= heads) break;
      float qh[kPerLane];
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) qh[j] = my_q[h * kDim + j];
      float score[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) dot = fmaf(qh[j], key[t][j], dot);
        score[t] = (seen >> t) & 1u ? WarpSum(dot) * p.scale : -INFINITY;
      }
      const float block_max =
          fmaxf(fmaxf(score[0], score[1]), fmaxf(score[2], score[3]));
      const float m_new = fmaxf(m[h], block_max);
      const float alpha = expf(m[h] - m_new);
      l[h] *= alpha;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) acc[h][j] *= alpha;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (((seen >> t) & 1u) == 0) continue;
        const float weight = expf(score[t] - m_new);
        l[h] += weight;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
          acc[h][j] = fmaf(weight, value[t][j], acc[h][j]);
        }
      }
      m[h] = m_new;
    }
  }

  // Merge the warps of a row in warp order, through shared memory.
  float* merge = shared;  // [kWarps][kHeads][kDim + 2]
  constexpr int kStride = kDim + 2;
  if (warps_per_row > 1) {
    __syncthreads();  // the staged queries are no longer read
    float* mine = merge + warp * kHeads * kStride;
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        mine[h * kStride + lane * kPerLane + j] = acc[h][j];
      }
      if (lane == 0) {
        mine[h * kStride + kDim] = m[h];
        mine[h * kStride + kDim + 1] = l[h];
      }
    }
    __syncthreads();
    if (sub != 0) return;
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
      float total_m = -INFINITY;
      for (int s = 0; s < warps_per_row; ++s) {
        total_m = fmaxf(total_m, merge[((warp + s) * kHeads + h) * kStride +
                                       kDim]);
      }
      float total_l = 0.0f;
      float merged[kPerLane];
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) merged[j] = 0.0f;
      for (int s = 0; s < warps_per_row; ++s) {
        const float* other = merge + ((warp + s) * kHeads + h) * kStride;
        const float ms = other[kDim];
        if (ms == -INFINITY) continue;
        const float weight = expf(ms - total_m);
        total_l += weight * other[kDim + 1];
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
          merged[j] = fmaf(weight, other[lane * kPerLane + j], merged[j]);
        }
      }
      m[h] = total_m;
      l[h] = total_l;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) acc[h][j] = merged[j];
    }
  }
  if (!active) return;

  for (int h = 0; h < heads && h < kHeads; ++h) {
    const int32_t q_head = head0 + h;
    if (splits == 1) {
      const float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
      const int64_t base = (int64_t{row} * p.q_heads + q_head) * kDim +
                           lane * kPerLane;
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const float value = acc[h][j] * inv;
        if (p.q_type == kLiteRtNvidiaQsaFloat16) {
          StoreFromFloat(value, static_cast<__half*>(out) + base + j);
        } else if (p.q_type == kLiteRtNvidiaQsaBFloat16) {
          StoreFromFloat(value, static_cast<__nv_bfloat16*>(out) + base + j);
        } else {
          StoreFromFloat(value, static_cast<float*>(out) + base + j);
        }
      }
    } else {
      float* part =
          partial +
          ((int64_t{row} * p.q_heads + q_head) * splits + split) * (kDim + 4);
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) part[lane * kPerLane + j] = acc[h][j];
      if (lane == 0) {
        part[kDim] = m[h];
        part[kDim + 1] = l[h];
      }
    }
  }
}
#pragma clang diagnostic pop

// The tensor-core path for steps of several rows (verify, prefill chunks)
// over FP16 caches holding values as rows: a CTA takes the query vectors
// (rows x heads, at most kMaxTileVectors) of a tile of rows on one KV head,
// a warp per 16 of them, and walks chunk `split` of the union of the rows'
// blocks kChunkTokens keys at a time, each key and value loaded once for all
// of them. Products on tensor cores (FP16 in, FP32 accumulation): a first
// pass finds each row's largest score, a second takes the probabilities (in
// FP32 for the sums, FP16 for the products with the values) and accumulates
// the values, so the accumulators are never rescaled. A key counts for a
// row when the row lists its block (its bitmap) and sees it.
constexpr int kChunkTokens = 16;
constexpr int kMaxTileVectors = 96;
constexpr int kMaxTileRows = 16;

// Clang may retain loops despite the kernel's best-effort unroll hints, and
// reports the failed transformation at the kernel's declaration.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
template <int kDim>
__global__ void __launch_bounds__(kMaxTileVectors / 16 * 32)
    QsaAttendTileKernel(LiteRtNvidiaQsaAttentionParams p, int32_t tile_rows,
                        int32_t m_pad, int32_t splits,
                        const void* __restrict__ q,
                        const __half* __restrict__ k,
                        const __half* __restrict__ v,
                        const uint32_t* __restrict__ tile_members,
                        const int32_t* __restrict__ tile_lists,
                        const int32_t* __restrict__ tile_counts,
                        int32_t tile_capacity, Visibility visibility,
                        void* __restrict__ out, float* __restrict__ partial) {
  namespace wmma = nvcuda::wmma;
  constexpr int kLd = kDim + 8;                 // halves per staged row
  constexpr int kScoreLd = kChunkTokens + 4;    // floats
  constexpr int kProbLd = kChunkTokens + 8;     // halves
  extern __shared__ __align__(32) unsigned char shared_bytes[];
  // Warps past m_pad / 16 only help to load.
  __half* q_s = reinterpret_cast<__half*>(shared_bytes);
  __half* k_s = q_s + m_pad * kLd;
  __half* v_s = k_s + kChunkTokens * kLd;
  float* s_s = reinterpret_cast<float*>(v_s + kChunkTokens * kLd);
  __half* p_s = reinterpret_cast<__half*>(s_s + m_pad * kScoreLd);
  float* m_s = reinterpret_cast<float*>(p_s + m_pad * kProbLd);
  float* l_s = m_s + m_pad;
  uint8_t* allowed = reinterpret_cast<uint8_t*>(l_s + m_pad);
  int32_t* chunk_blocks =
      reinterpret_cast<int32_t*>(allowed + kMaxTileRows * kChunkTokens);
  uint32_t* chunk_members = reinterpret_cast<uint32_t*>(chunk_blocks + 4);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool computes = warp < m_pad / 16;
  const int32_t group = p.q_heads / p.kv_heads;
  const int32_t kv_head = blockIdx.x;
  const int32_t tile = blockIdx.y;
  const int32_t split = blockIdx.z;
  const int32_t row0 = tile * tile_rows;
  const int32_t rows = min(tile_rows, p.rows - row0);
  const int32_t vectors = rows * group;  // vector m: row m / group, head m % group

#pragma unroll 2
  for (int32_t i = threadIdx.x * 8; i < m_pad * kDim; i += blockDim.x * 8) {
    const int32_t m = i / kDim;
    const int32_t d = i % kDim;
    float eight[8] = {};
    if (m < vectors) {
      const int64_t q_row =
          int64_t{kv_head * group + m % group} * p.rows + row0 + m / group;
      LoadEightAsFloat(q, p.q_type, q_row * kDim + d, eight);
    }
    __align__(16) __half halves[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) halves[j] = __float2half(eight[j]);
    *reinterpret_cast<uint4*>(q_s + m * kLd + d) =
        *reinterpret_cast<const uint4*>(halves);
  }
  for (int32_t i = threadIdx.x; i < m_pad; i += blockDim.x) {
    m_s[i] = -INFINITY;
    l_s[i] = 0.0f;
  }
  for (int32_t i = threadIdx.x; i < kMaxTileRows * kChunkTokens;
       i += blockDim.x) {
    allowed[i] = 0;
  }
  const int32_t count = tile_counts[tile];
  // Chunks of a multiple of four blocks per split.
  const int32_t per = ((count + splits - 1) / splits + 3) / 4 * 4;
  const int32_t lo = min(count, split * per);
  const int32_t hi = min(count, lo + per);
  const int32_t* list = tile_lists + int64_t{tile} * tile_capacity;
  const uint32_t* members = tile_members + int64_t{tile} * tile_capacity;
  const int64_t head = int64_t{kv_head} * p.cache_len * kDim;
  __syncthreads();

  // The keys (and values) of the four blocks list[c .. c + 4) and the keys
  // each row may attend to.
  const auto stage = [&](int32_t c, bool with_values) {
    if (threadIdx.x < 4) {
      const bool in = c + static_cast<int32_t>(threadIdx.x) < hi;
      chunk_blocks[threadIdx.x] = in ? list[c + threadIdx.x] : -1;
      chunk_members[threadIdx.x] = in ? members[c + threadIdx.x] : 0u;
    }
    __syncthreads();
    constexpr int kVectorsPerRow = kDim / 8;
#pragma unroll 4
    for (int i = threadIdx.x; i < kChunkTokens * kVectorsPerRow;
         i += blockDim.x) {
      const int t = i / kVectorsPerRow;
      const int d8 = i % kVectorsPerRow;
      const int32_t block = chunk_blocks[t / 4];
      uint4 key = make_uint4(0, 0, 0, 0);
      uint4 value = make_uint4(0, 0, 0, 0);
      if (block >= 0 && block * 4 + t % 4 < p.cache_len) {
        const int64_t at = head + (int64_t{block} * 4 + t % 4) * kDim + d8 * 8;
        key = *reinterpret_cast<const uint4*>(k + at);
        if (with_values) value = *reinterpret_cast<const uint4*>(v + at);
      }
      *reinterpret_cast<uint4*>(k_s + t * kLd + d8 * 8) = key;
      if (with_values) {
        *reinterpret_cast<uint4*>(v_s + t * kLd + d8 * 8) = value;
      }
    }
    for (int i = threadIdx.x; i < rows * 4; i += blockDim.x) {
      const int r = i / 4;
      const int j = i % 4;
      const int32_t block = chunk_blocks[j];
      uint32_t seen = 0;
      if (block >= 0 && ((chunk_members[j] >> r) & 1u) != 0) {
        seen = visibility.Block(row0 + r, block);
      }
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        allowed[r * kChunkTokens + j * 4 + t] = (seen >> t) & 1u;
      }
    }
    __syncthreads();
  };

  // This warp's 16 vectors against the chunk's keys, into s_s.
  const auto score = [&]() {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int kk = 0; kk < kDim / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
      wmma::load_matrix_sync(a, q_s + warp * 16 * kLd + kk * 16, kLd);
      wmma::load_matrix_sync(b, k_s + kk * 16, kLd);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(s_s + warp * 16 * kScoreLd, acc, kScoreLd,
                            wmma::mem_row_major);
    __syncwarp();
  };

  // Two lanes per vector, eight keys each.
  const int32_t m = computes ? warp * 16 + lane / 2 : 0;
  const int t0 = (lane % 2) * 8;
  const bool real = computes && m < vectors;
  const uint8_t* row_allowed = allowed + (real ? m / group : 0) * kChunkTokens;

  for (int32_t c = lo; c < hi; c += 4) {
    stage(c, false);
    if (!computes) continue;
    score();
    float largest = -INFINITY;
    if (real) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        if (row_allowed[t0 + t]) {
          largest = fmaxf(largest, s_s[m * kScoreLd + t0 + t] * p.scale);
        }
      }
    }
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 1));
    if (lane % 2 == 0) m_s[m] = fmaxf(m_s[m], largest);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[kDim / 16];
#pragma unroll
  for (int j = 0; j < kDim / 16; ++j) wmma::fill_fragment(o[j], 0.0f);
  for (int32_t c = lo; c < hi; c += 4) {
    stage(c, true);
    if (!computes) continue;
    score();
    const float row_max = m_s[m];
    float sum = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      float weight = 0.0f;
      if (real && row_allowed[t0 + t] && row_max != -INFINITY) {
        weight = expf(s_s[m * kScoreLd + t0 + t] * p.scale - row_max);
      }
      p_s[m * kProbLd + t0 + t] = __float2half(weight);
      sum += weight;
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    if (lane % 2 == 0) l_s[m] += sum;
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
    wmma::load_matrix_sync(pa, p_s + warp * 16 * kProbLd, kProbLd);
#pragma unroll
    for (int j = 0; j < kDim / 16; ++j) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
      wmma::load_matrix_sync(vb, v_s + j * 16, kLd);
      wmma::mma_sync(o[j], pa, vb, o[j]);
    }
  }

  if (!computes) return;
  // Out through this warp's score rows, 16 x 16 at a time.
  float* scratch = s_s + warp * 16 * kScoreLd;
#pragma unroll
  for (int j = 0; j < kDim / 16; ++j) {
    __syncwarp();
    wmma::store_matrix_sync(scratch, o[j], kScoreLd, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      const int32_t vector = warp * 16 + e / 16;
      if (vector >= vectors) continue;
      const int32_t row = row0 + vector / group;
      const int32_t q_head = kv_head * group + vector % group;
      const float value = scratch[(e / 16) * kScoreLd + e % 16];
      const int32_t d = j * 16 + e % 16;
      if (splits == 1) {
        const float total = l_s[vector];
        const float result = total > 0.0f ? value / total : 0.0f;
        const int64_t at = (int64_t{row} * p.q_heads + q_head) * kDim + d;
        if (p.q_type == kLiteRtNvidiaQsaFloat16) {
          StoreFromFloat(result, static_cast<__half*>(out) + at);
        } else if (p.q_type == kLiteRtNvidiaQsaBFloat16) {
          StoreFromFloat(result, static_cast<__nv_bfloat16*>(out) + at);
        } else {
          StoreFromFloat(result, static_cast<float*>(out) + at);
        }
      } else {
        partial[((int64_t{row} * p.q_heads + q_head) * splits + split) *
                    (kDim + 4) +
                d] = value;
      }
    }
  }
  if (splits > 1 && lane % 2 == 0 && real) {
    const int32_t row = row0 + m / group;
    const int32_t q_head = kv_head * group + m % group;
    float* part = partial +
                  ((int64_t{row} * p.q_heads + q_head) * splits + split) *
                      (kDim + 4);
    part[kDim] = m_s[m];
    part[kDim + 1] = l_s[m];
  }
}
#pragma clang diagnostic pop

// The chunks of a row and head into its output, in chunk order: the first
// warp weighs the chunks (at most 64), then every thread sums its dims.
__global__ void QsaCombineKernel(LiteRtNvidiaQsaAttentionParams p,
                                 int32_t splits,
                                 const float* __restrict__ partial,
                                 void* __restrict__ out) {
  __shared__ float weights[64];
  __shared__ float inv_total;
  const int32_t q_head = blockIdx.x;
  const int32_t row = blockIdx.y;
  const int32_t dim = p.head_dim;
  const int32_t stride = dim + 4;
  const float* parts =
      partial + (int64_t{row} * p.q_heads + q_head) * splits * stride;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    float m[2];
    float l[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int32_t s = lane + 32 * i;
      m[i] = s < splits ? parts[s * stride + dim] : -INFINITY;
      l[i] = s < splits ? parts[s * stride + dim + 1] : 0.0f;
    }
    float total_m = fmaxf(m[0], m[1]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_m = fmaxf(total_m, __shfl_xor_sync(0xffffffffu, total_m, offset));
    }
    float total_l = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float weight = m[i] == -INFINITY ? 0.0f : expf(m[i] - total_m);
      if (lane + 32 * i < splits) weights[lane + 32 * i] = weight;
      total_l += weight * l[i];
    }
    // The sum in lane order (the same for every launch).
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_l += __shfl_xor_sync(0xffffffffu, total_l, offset);
    }
    if (lane == 0) inv_total = total_l > 0.0f ? 1.0f / total_l : 0.0f;
  }
  __syncthreads();
  const int64_t base = (int64_t{row} * p.q_heads + q_head) * dim;
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float sum = 0.0f;
#pragma unroll 4
    for (int32_t s = 0; s < splits; ++s) {
      sum = fmaf(weights[s], parts[s * stride + d], sum);
    }
    const float value = sum * inv_total;
    if (p.q_type == kLiteRtNvidiaQsaFloat16) {
      StoreFromFloat(value, static_cast<__half*>(out) + base + d);
    } else if (p.q_type == kLiteRtNvidiaQsaBFloat16) {
      StoreFromFloat(value, static_cast<__nv_bfloat16*>(out) + base + d);
    } else {
      StoreFromFloat(value, static_cast<float*>(out) + base + d);
    }
  }
}

int32_t Capacity(const LiteRtNvidiaQsaAttentionParams& p) {
  return std::min((p.cache_len + 3) / 4, p.block_topk + kMaxExtraBlocks);
}

int32_t HeadsPerCta(int32_t group) { return group <= 2 ? 2 : (group <= 4 ? 4 : 8); }

int32_t RowsPerCta(int32_t rows) { return rows == 1 ? 1 : (rows == 2 ? 2 : 4); }

struct Plan {
  int32_t capacity;
  int32_t heads_per_cta;
  int32_t head_chunks;
  int32_t rows_per_cta;
  int32_t row_groups;
  int32_t splits;
  // The tensor-core path (QsaAttendTileKernel).
  bool tiled;
  int32_t tile_rows;
  int32_t tiles;
  int32_t tile_capacity;
  int32_t m_pad;
};

Plan MakePlan(const LiteRtNvidiaQsaAttentionParams& p) {
  Plan plan = {};
  plan.capacity = Capacity(p);
  const int32_t group = p.q_heads / p.kv_heads;
  // The row model's slots may give the rows of a tile caches of their own:
  // it takes the per-row kernel.
  plan.tiled = p.rows >= 2 && p.cache_type == kLiteRtNvidiaQsaFloat16 &&
               p.value_columns == 0 && group <= kMaxTileVectors &&
               p.mask_type != kLiteRtNvidiaQsaMaskRows;
  int64_t base = 0;
  int64_t useful = 0;
  if (plan.tiled) {
    plan.tile_rows = std::min({p.rows, kMaxTileRows,
                               std::max(1, kMaxTileVectors / group)});
    plan.tiles = (p.rows + plan.tile_rows - 1) / plan.tile_rows;
    plan.tile_capacity = static_cast<int32_t>(std::min<int64_t>(
        (p.cache_len + 3) / 4, int64_t{plan.tile_rows} * plan.capacity));
    plan.m_pad = (plan.tile_rows * group + 15) / 16 * 16;
    base = int64_t{p.kv_heads} * plan.tiles;
    // As little as two chunks of four blocks per split.
    useful = std::max<int64_t>(1, plan.capacity / 8);
  } else {
    plan.heads_per_cta = HeadsPerCta(group);
    plan.head_chunks = (group + plan.heads_per_cta - 1) / plan.heads_per_cta;
    plan.rows_per_cta = RowsPerCta(p.rows);
    plan.row_groups = (p.rows + plan.rows_per_cta - 1) / plan.rows_per_cta;
    base = int64_t{p.kv_heads} * plan.head_chunks * plan.row_groups;
    useful = std::max<int64_t>(1, plan.capacity / kMinBlocksPerSplit);
  }
  const int64_t wanted = (kTargetCtas + base - 1) / base;
  plan.splits = static_cast<int32_t>(
      std::max<int64_t>(1, std::min<int64_t>({wanted, useful, 64})));
  return plan;
}

int32_t Words(const LiteRtNvidiaQsaAttentionParams& p) {
  return ((p.cache_len + 3) / 4 + 31) / 32;
}

// The lists (or bitmaps and tile lists) of the workspace, 256-byte aligned.
size_t ListBytes(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan) {
  size_t bytes = 0;
  if (plan.tiled) {
    // Bitmaps, then the tiles' lists, members and counts.
    bytes = static_cast<size_t>(p.rows) * Words(p) * sizeof(uint32_t) +
            (2 * static_cast<size_t>(plan.tiles) * plan.tile_capacity +
             plan.tiles) *
                sizeof(int32_t);
  } else {
    bytes = (static_cast<size_t>(p.rows) * plan.capacity + p.rows) *
            sizeof(int32_t);
  }
  return (bytes + 255) / 256 * 256;
}

size_t PartialBytes(const LiteRtNvidiaQsaAttentionParams& p,
                    const Plan& plan) {
  if (plan.splits == 1) return 0;
  return static_cast<size_t>(p.rows) * p.q_heads * plan.splits *
         (p.head_dim + 4) * sizeof(float);
}

template <int kDim>
size_t TileSharedBytes(int32_t m_pad) {
  const size_t ld = kDim + 8;
  return static_cast<size_t>(m_pad) * ld * 2 +           // queries
         2 * static_cast<size_t>(kChunkTokens) * ld * 2 +  // keys, values
         static_cast<size_t>(m_pad) * (kChunkTokens + 4) * 4 +  // scores
         static_cast<size_t>(m_pad) * (kChunkTokens + 8) * 2 +  // weights
         static_cast<size_t>(m_pad) * 2 * 4 +                  // max, sum
         kMaxTileRows * kChunkTokens + 8 * sizeof(int32_t);
}

template <int kDim>
void LaunchTile(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan,
                const void* q, const void* k, const void* v,
                const uint32_t* tile_members, const int32_t* tile_lists,
                const int32_t* tile_counts, const Visibility& visibility,
                void* out, float* partial, cudaStream_t stream) {
  const size_t shared = TileSharedBytes<kDim>(plan.m_pad);
  auto kernel = QsaAttendTileKernel<kDim>;
  if (shared > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared));
  }
  const dim3 grid(p.kv_heads, plan.tiles, plan.splits);
  const int threads = std::max(4, plan.m_pad / 16) * 32;
  kernel<<<grid, threads, shared, stream>>>(
      p, plan.tile_rows, plan.m_pad, plan.splits, q,
      static_cast<const __half*>(k),
      static_cast<const __half*>(v), tile_members, tile_lists, tile_counts,
      plan.tile_capacity, visibility, out, partial);
}

template <int kDim, int kHeads, typename Cache>
void LaunchAttend(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan,
                  const void* q, const void* k, const void* v,
                  const int32_t* lists, const int32_t* counts,
                  const Visibility& visibility, void* out, float* partial,
                  cudaStream_t stream) {
  const size_t staged = static_cast<size_t>(plan.rows_per_cta) * kHeads *
                        kDim * sizeof(float);
  const size_t merge =
      static_cast<size_t>(kWarps) * kHeads * (kDim + 2) * sizeof(float);
  const size_t shared = std::max(staged, merge);
  auto kernel = QsaAttendKernel<kDim, kHeads, Cache>;
  if (shared > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared));
  }
  const dim3 grid(p.kv_heads * plan.head_chunks, plan.row_groups, plan.splits);
  kernel<<<grid, kThreads, shared, stream>>>(
      p, plan.head_chunks, plan.rows_per_cta, plan.splits, q,
      static_cast<const Cache*>(k), static_cast<const Cache*>(v), lists,
      counts, plan.capacity, visibility, out, partial);
}

template <int kDim, typename Cache>
void LaunchForHeads(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan,
                    const void* q, const void* k, const void* v,
                    const int32_t* lists, const int32_t* counts,
                    const Visibility& visibility, void* out, float* partial,
                    cudaStream_t stream) {
  switch (plan.heads_per_cta) {
    case 2:
      return LaunchAttend<kDim, 2, Cache>(p, plan, q, k, v, lists, counts,
                                          visibility, out, partial, stream);
    case 4:
      return LaunchAttend<kDim, 4, Cache>(p, plan, q, k, v, lists, counts,
                                          visibility, out, partial, stream);
    default:
      return LaunchAttend<kDim, 8, Cache>(p, plan, q, k, v, lists, counts,
                                          visibility, out, partial, stream);
  }
}

template <typename Cache>
void LaunchForDim(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan,
                  const void* q, const void* k, const void* v,
                  const int32_t* lists, const int32_t* counts,
                  const Visibility& visibility, void* out, float* partial,
                  cudaStream_t stream) {
  switch (p.head_dim) {
    case 64:
      return LaunchForHeads<64, Cache>(p, plan, q, k, v, lists, counts,
                                       visibility, out, partial, stream);
    case 128:
      return LaunchForHeads<128, Cache>(p, plan, q, k, v, lists, counts,
                                        visibility, out, partial, stream);
    default:
      return LaunchForHeads<256, Cache>(p, plan, q, k, v, lists, counts,
                                        visibility, out, partial, stream);
  }
}

}  // namespace

extern "C" bool LiteRtNvidiaQsaAttentionSupports(
    const LiteRtNvidiaQsaAttentionParams* params) {
  if (params == nullptr) return false;
  const auto& p = *params;
  return p.rows >= 1 && p.kv_heads >= 1 && p.q_heads >= p.kv_heads &&
         p.q_heads % p.kv_heads == 0 &&
         (p.head_dim == 64 || p.head_dim == 128 || p.head_dim == 256) &&
         p.cache_len >= 1 && p.cache_len <= kMaxCacheLen &&
         p.block_topk >= 1 &&
         p.block_topk <= kMaxTopk &&
         (p.q_type == kLiteRtNvidiaQsaFloat32 ||
          p.q_type == kLiteRtNvidiaQsaFloat16 ||
          p.q_type == kLiteRtNvidiaQsaBFloat16) &&
         (p.cache_type == kLiteRtNvidiaQsaFloat32 ||
          p.cache_type == kLiteRtNvidiaQsaFloat16) &&
         (p.mask_type == kLiteRtNvidiaQsaMaskBool ||
          p.mask_type == kLiteRtNvidiaQsaMaskFloat ||
          (p.mask_type == kLiteRtNvidiaQsaMaskRows && p.rows <= 64));
}

extern "C" size_t LiteRtNvidiaQsaAttentionWorkspaceBytes(
    const LiteRtNvidiaQsaAttentionParams* params) {
  if (!LiteRtNvidiaQsaAttentionSupports(params)) return 0;
  const Plan plan = MakePlan(*params);
  return ListBytes(*params, plan) + PartialBytes(*params, plan);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQsaAttention(
    const LiteRtNvidiaQsaAttentionParams* params, const void* q, const void* k,
    const void* v, const int32_t* topk, const void* mask,
    const LiteRtNvidiaQsaRows* row_model, void* out, void* workspace,
    cudaStream_t stream) {
  if (!LiteRtNvidiaQsaAttentionSupports(params) || q == nullptr ||
      k == nullptr || v == nullptr || topk == nullptr || out == nullptr ||
      workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  const auto& p = *params;
  const bool rows_model = p.mask_type == kLiteRtNvidiaQsaMaskRows;
  if (rows_model ? (row_model == nullptr || row_model->context_len == nullptr ||
                    row_model->write_index == nullptr)
                 : mask == nullptr) {
    return cudaErrorInvalidValue;
  }
  const Plan plan = MakePlan(p);
  Visibility visibility{};
  visibility.type = p.mask_type;
  visibility.mask = mask;
  if (rows_model) visibility.model = *row_model;
  visibility.rows = p.rows;
  visibility.cache_len = p.cache_len;

  float* partial = reinterpret_cast<float*>(static_cast<char*>(workspace) +
                                            ListBytes(p, plan));
  const size_t bitmap_bytes = static_cast<size_t>(Words(p)) * sizeof(uint32_t);
  if (bitmap_bytes > 48 * 1024) {
    cudaFuncSetAttribute(QsaListKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(bitmap_bytes));
    cudaFuncSetAttribute(QsaTileListKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(bitmap_bytes));
  }
  if (plan.tiled) {
    uint32_t* bitmaps = static_cast<uint32_t*>(workspace);
    int32_t* tile_lists = reinterpret_cast<int32_t*>(
        bitmaps + static_cast<size_t>(p.rows) * Words(p));
    uint32_t* tile_members = reinterpret_cast<uint32_t*>(
        tile_lists + static_cast<size_t>(plan.tiles) * plan.tile_capacity);
    int32_t* tile_counts = reinterpret_cast<int32_t*>(
        tile_members + static_cast<size_t>(plan.tiles) * plan.tile_capacity);
    QsaListKernel<<<p.rows, kListThreads, bitmap_bytes, stream>>>(
        p, topk, visibility, plan.capacity, nullptr, nullptr, bitmaps);
    QsaTileListKernel<<<plan.tiles, kListThreads, bitmap_bytes, stream>>>(
        p, plan.tile_rows, bitmaps, plan.tile_capacity, tile_lists,
        tile_members, tile_counts);
    switch (p.head_dim) {
      case 64:
        LaunchTile<64>(p, plan, q, k, v, tile_members, tile_lists, tile_counts,
                       visibility, out, partial, stream);
        break;
      case 128:
        LaunchTile<128>(p, plan, q, k, v, tile_members, tile_lists,
                        tile_counts, visibility, out, partial, stream);
        break;
      default:
        LaunchTile<256>(p, plan, q, k, v, tile_members, tile_lists,
                        tile_counts, visibility, out, partial, stream);
        break;
    }
  } else if (p.cache_type == kLiteRtNvidiaQsaFloat16) {
    int32_t* lists = static_cast<int32_t*>(workspace);
    int32_t* counts = lists + static_cast<size_t>(p.rows) * plan.capacity;
    QsaListKernel<<<p.rows, kListThreads, bitmap_bytes, stream>>>(
        p, topk, visibility, plan.capacity, lists, counts, nullptr);
    LaunchForDim<__half>(p, plan, q, k, v, lists, counts, visibility, out,
                         partial, stream);
  } else {
    int32_t* lists = static_cast<int32_t*>(workspace);
    int32_t* counts = lists + static_cast<size_t>(p.rows) * plan.capacity;
    QsaListKernel<<<p.rows, kListThreads, bitmap_bytes, stream>>>(
        p, topk, visibility, plan.capacity, lists, counts, nullptr);
    LaunchForDim<float>(p, plan, q, k, v, lists, counts, visibility, out,
                        partial, stream);
  }
  if (plan.splits > 1) {
    const dim3 grid(p.q_heads, p.rows);
    QsaCombineKernel<<<grid, std::min(256, p.head_dim), 0, stream>>>(
        p, plan.splits, partial, out);
  }
  return cudaPeekAtLastError();
}
