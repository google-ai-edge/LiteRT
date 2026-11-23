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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/qsa_common.cuh"

namespace {

using litert::nvidia::qsa::LoadAsFloat;
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
    if (type == kLiteRtNvidiaQsaMaskBool) {
      // cache_len % 4 == 0 keeps the four bytes aligned.
      const uint32_t bytes =
          *reinterpret_cast<const uint32_t*>(
              static_cast<const uint8_t*>(mask) + at);
      return ((bytes & 0xFFu) != 0 ? 1u : 0u) |
             ((bytes & 0xFF00u) != 0 ? 2u : 0u) |
             ((bytes & 0xFF0000u) != 0 ? 4u : 0u) |
             ((bytes & 0xFF000000u) != 0 ? 8u : 0u);
    }
    if (type == kLiteRtNvidiaQsaMaskFloat) {
      const float4 m =
          *reinterpret_cast<const float4*>(static_cast<const float*>(mask) + at);
      return (m.x == 0.0f ? 1u : 0u) | (m.y == 0.0f ? 2u : 0u) |
             (m.z == 0.0f ? 4u : 0u) | (m.w == 0.0f ? 8u : 0u);
    }
    uint32_t bits = 0;
    for (int t = 0; t < 4; ++t) {
      if (Token(row, 4 * block + t)) bits |= 1u << t;
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

// A row's blocks in ascending order: the selected ones and those it sees
// partly, through a bitmap over the cache's blocks in shared memory.
__global__ void __launch_bounds__(kListThreads) QsaListKernel(
    LiteRtNvidiaQsaAttentionParams p, const int32_t* __restrict__ topk,
    Visibility visibility, int32_t capacity, int32_t* __restrict__ lists,
    int32_t* __restrict__ counts) {
  extern __shared__ uint32_t bitmap[];
  __shared__ int32_t warp_sums[32];
  const int32_t row = blockIdx.x;
  const int32_t blocks = p.cache_len / 4;
  const int32_t words = (blocks + 31) / 32;
  for (int32_t w = threadIdx.x; w < words; w += blockDim.x) bitmap[w] = 0;
  __syncthreads();
  for (int32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
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
  const int32_t per = (words + kListThreads - 1) / kListThreads;
  const int32_t lo = min(words, static_cast<int32_t>(threadIdx.x) * per);
  const int32_t hi = min(words, lo + per);
  int32_t mine = 0;
  for (int32_t w = lo; w < hi; ++w) mine += __popc(bitmap[w]);
  int32_t total = 0;
  int32_t offset = BlockExclusiveScan(mine, warp_sums, &total);
  int32_t* list = lists + int64_t{row} * capacity;
  for (int32_t w = lo; w < hi && offset < capacity; ++w) {
    uint32_t bits = bitmap[w];
    while (bits != 0 && offset < capacity) {
      const int bit = __ffs(bits) - 1;
      bits &= bits - 1;
      list[offset++] = w * 32 + bit;
    }
  }
  if (threadIdx.x == 0) counts[row] = min(total, capacity);
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

  // Stage the queries of this CTA's rows as FP32.
  float* staged = shared;  // [rows_per_cta][kHeads][kDim]
  for (int32_t i = threadIdx.x; i < rows_per_cta * kHeads * kDim;
       i += blockDim.x) {
    const int32_t r = static_cast<int32_t>(blockIdx.y) * rows_per_cta +
                      i / (kHeads * kDim);
    const int32_t h = (i / kDim) % kHeads;
    const int32_t d = i % kDim;
    staged[i] = (r < p.rows && h < heads)
                    ? LoadAsFloat(q, p.q_type,
                                  (int64_t{head0 + h} * p.rows + r) * kDim + d)
                    : 0.0f;
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
    float key[4][kPerLane];
    float value[4][kPerLane];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      Vec<Cache>::Load(kh + int64_t{token0 + t} * kDim + lane * kPerLane,
                       key[t]);
    }
    if (p.value_columns) {
#pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const Cache* column =
            vh + int64_t{lane * kPerLane + j} * p.cache_len + token0;
#pragma unroll
        for (int t = 0; t < 4; ++t) value[t][j] = ToFloat(column[t]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        Vec<Cache>::Load(vh + int64_t{token0 + t} * kDim + lane * kPerLane,
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

// The chunks of a row and head into its output, in chunk order.
__global__ void QsaCombineKernel(LiteRtNvidiaQsaAttentionParams p,
                                 int32_t splits,
                                 const float* __restrict__ partial,
                                 void* __restrict__ out) {
  const int32_t q_head = blockIdx.x;
  const int32_t row = blockIdx.y;
  const int32_t dim = p.head_dim;
  const float* parts =
      partial + (int64_t{row} * p.q_heads + q_head) * splits * (dim + 4);
  float total_m = -INFINITY;
  for (int32_t s = 0; s < splits; ++s) {
    total_m = fmaxf(total_m, parts[s * (dim + 4) + dim]);
  }
  const int64_t base = (int64_t{row} * p.q_heads + q_head) * dim;
  for (int32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    float total_l = 0.0f;
    float sum = 0.0f;
    for (int32_t s = 0; s < splits; ++s) {
      const float* part = parts + s * (dim + 4);
      const float ms = part[dim];
      if (ms == -INFINITY) continue;
      const float weight = expf(ms - total_m);
      total_l += weight * part[dim + 1];
      sum = fmaf(weight, part[d], sum);
    }
    const float value = total_l > 0.0f ? sum / total_l : 0.0f;
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
  return std::min(p.cache_len / 4, p.block_topk + kMaxExtraBlocks);
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
};

Plan MakePlan(const LiteRtNvidiaQsaAttentionParams& p) {
  Plan plan;
  plan.capacity = Capacity(p);
  const int32_t group = p.q_heads / p.kv_heads;
  plan.heads_per_cta = HeadsPerCta(group);
  plan.head_chunks = (group + plan.heads_per_cta - 1) / plan.heads_per_cta;
  plan.rows_per_cta = RowsPerCta(p.rows);
  plan.row_groups = (p.rows + plan.rows_per_cta - 1) / plan.rows_per_cta;
  const int64_t base =
      int64_t{p.kv_heads} * plan.head_chunks * plan.row_groups;
  const int64_t wanted = (kTargetCtas + base - 1) / base;
  const int64_t useful =
      std::max<int64_t>(1, plan.capacity / kMinBlocksPerSplit);
  plan.splits = static_cast<int32_t>(
      std::max<int64_t>(1, std::min<int64_t>({wanted, useful, 64})));
  return plan;
}

size_t ListBytes(const LiteRtNvidiaQsaAttentionParams& p, const Plan& plan) {
  const size_t bytes =
      (static_cast<size_t>(p.rows) * plan.capacity + p.rows) * sizeof(int32_t);
  return (bytes + 255) / 256 * 256;
}

size_t PartialBytes(const LiteRtNvidiaQsaAttentionParams& p,
                    const Plan& plan) {
  if (plan.splits == 1) return 0;
  return static_cast<size_t>(p.rows) * p.q_heads * plan.splits *
         (p.head_dim + 4) * sizeof(float);
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
         p.cache_len >= 4 && p.cache_len % 4 == 0 &&
         p.cache_len <= kMaxCacheLen && p.block_topk >= 1 &&
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

  int32_t* lists = static_cast<int32_t*>(workspace);
  int32_t* counts = lists + static_cast<size_t>(p.rows) * plan.capacity;
  float* partial = reinterpret_cast<float*>(static_cast<char*>(workspace) +
                                            ListBytes(p, plan));
  const size_t bitmap_bytes =
      static_cast<size_t>((p.cache_len / 4 + 31) / 32) * sizeof(uint32_t);
  if (bitmap_bytes > 48 * 1024) {
    cudaFuncSetAttribute(QsaListKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(bitmap_bytes));
  }
  QsaListKernel<<<p.rows, kListThreads, bitmap_bytes, stream>>>(
      p, topk, visibility, plan.capacity, lists, counts);
  if (p.cache_type == kLiteRtNvidiaQsaFloat16) {
    LaunchForDim<__half>(p, plan, q, k, v, lists, counts, visibility, out,
                         partial, stream);
  } else {
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
