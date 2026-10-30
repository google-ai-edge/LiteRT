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

#include "litert/vendors/nvidia/trtllm/global_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

namespace {

// The kernel walks the keys in tiles with an online softmax (running row
// maximum and sum) and computes the two products with m16n8k16 tensor-core
// operations that accumulate in fp32. Probabilities are the fp16 "A" operand
// of the value product, stored scaled by 2^14 so that small ones keep
// precision; the row sums use the same rounded values.
constexpr float kProbScale = 16384.0f;
constexpr int kDepth = 512;
constexpr int kSlices = kDepth / 16;        // k-slices of the score product
constexpr int kVectorsPerRow = kDepth / 8;  // 16-byte vectors per cache row
constexpr int kMaxSegments = 64;

// Streaming kernel (decode): one block owns 16 query rows and reads each key
// and value once, straight into tensor-core operands. Prefill, with 16 rows
// per prompt token, runs on the tiled kernel (tiled_attention.h), whose
// blocks share every key and value they read among 64 rows.
constexpr int kStreamRows = 16;
constexpr int kStreamTile = 64;
constexpr int kStreamThreads = 256;
constexpr int kStreamWarps = kStreamThreads / 32;
constexpr int kMaxStreamSplits = 336;
constexpr int kMinTiledRows = 128;

__device__ __forceinline__ void Mma(float& d0, float& d1, float& d2, float& d3,
                                    uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t PackHalves(float lo, float hi) {
  const __half2 packed = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ __forceinline__ uint32_t PackBf16(float lo, float hi) {
  const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

// Two BF16 values in one word -> two FP16 values in one word.
__device__ __forceinline__ uint32_t Bf16WordToHalfWord(uint32_t word) {
  const float2 values =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word));
  return PackHalves(values.x, values.y);
}

template <typename T>
struct IsBf16 {
  static constexpr bool kValue = false;
};
template <>
struct IsBf16<__nv_bfloat16> {
  static constexpr bool kValue = true;
};

template <typename T>
__device__ __forceinline__ int4 LoadQueryVector(const T* row, int vector) {
  int4 loaded = reinterpret_cast<const int4*>(row)[vector];
  if (IsBf16<T>::kValue) {
    loaded.x = Bf16WordToHalfWord(loaded.x);
    loaded.y = Bf16WordToHalfWord(loaded.y);
    loaded.z = Bf16WordToHalfWord(loaded.z);
    loaded.w = Bf16WordToHalfWord(loaded.w);
  }
  return loaded;
}

template <typename T>
__device__ __forceinline__ uint32_t PackOutput(float lo, float hi) {
  return IsBf16<T>::kValue ? PackBf16(lo, hi) : PackHalves(lo, hi);
}

// summary[(mask_row * segments + segment) * 2 + {0, 1}] = {first masked key,
// end of the visible keys} of the segment (seq and 0 when it has none).
__global__ void __launch_bounds__(kStreamThreads)
    MaskSummaryKernel(const bool* __restrict__ mask, int seq, int segments,
                      int* __restrict__ summary) {
  __shared__ int first_s[kStreamWarps];
  __shared__ int end_s[kStreamWarps];
  const int segment = blockIdx.x;
  const int mask_row = blockIdx.y;
  const int begin =
      static_cast<int>(static_cast<int64_t>(seq) * segment / segments);
  const int end =
      static_cast<int>(static_cast<int64_t>(seq) * (segment + 1) / segments);
  const bool* row = mask + static_cast<size_t>(mask_row) * seq;
  int first_masked = seq;
  int visible_end = 0;
  for (int j = begin + threadIdx.x; j < end; j += kStreamThreads) {
    const bool visible = row[j];
    first_masked = visible ? first_masked : min(first_masked, j);
    visible_end = visible ? max(visible_end, j + 1) : visible_end;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    first_masked =
        min(first_masked, __shfl_xor_sync(0xffffffffu, first_masked, offset));
    visible_end =
        max(visible_end, __shfl_xor_sync(0xffffffffu, visible_end, offset));
  }
  const int warp = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    first_s[warp] = first_masked;
    end_s[warp] = visible_end;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int w = 1; w < kStreamWarps; ++w) {
      first_masked = min(first_masked, first_s[w]);
      visible_end = max(visible_end, end_s[w]);
    }
    int* entry =
        summary + (static_cast<size_t>(mask_row) * segments + segment) * 2;
    entry[0] = first_masked;
    entry[1] = visible_end;
  }
}

// Writes, for thread `tid` < block_rows, the first masked key and the end of
// the visible keys of query row r0 + tid, and resets its softmax state.
__device__ __forceinline__ void InitializeRow(
    int tid, int r0, const bool* mask, const int* summary, int mask_rows,
    int segments, int seq, int* first_s, int* end_s, float* max_s,
    float* sum_s) {
  int first_masked = seq;
  int visible_end = mask == nullptr ? seq : 0;
  if (mask != nullptr) {
    const int* entry =
        summary + static_cast<size_t>((r0 + tid) % mask_rows) * segments * 2;
    for (int s = 0; s < segments; ++s) {
      first_masked = min(first_masked, entry[2 * s]);
      visible_end = max(visible_end, entry[2 * s + 1]);
    }
  }
  first_s[tid] = first_masked;
  end_s[tid] = visible_end;
  max_s[tid] = -INFINITY;
  sum_s[tid] = 0.0f;
}

// Tiles [0, full_end) are visible to every row of the block; tiles from
// tile_end on are visible to none. A row that sees no key takes the softmax
// over the whole sequence (every score is `fill`), like the unfused graph.
__device__ __forceinline__ void TileRange(const int* first_s, const int* end_s,
                                          int block_rows, int tile, int seq,
                                          int* full_end, int* tile_end) {
  const int tiles = (seq + tile - 1) / tile;
  int full = seq / tile;  // a trailing partial tile takes the masked path
  int end = 0;
  for (int r = 0; r < block_rows; ++r) {
    full = min(full, first_s[r] / tile);
    end = max(end, end_s[r] == 0 ? tiles : (end_s[r] + tile - 1) / tile);
  }
  *full_end = full;
  *tile_end = end;
}

// grid (splits, rows / 16). Block (split, row tile) visits key tiles split,
// split + splits, ... With one split the block writes `out`; otherwise its
// partial sums go to the workspace (acc [blocks][16][depth], row maximum and
// row sum [blocks][16]) for CombineKernel.
//
// Lane (g, t) of a warp (g = lane / 4) holds rows g and g + 8 of a 16-row
// operand. Dims are assigned to lanes so that loads are contiguous: in the
// score product lane t reads the 16-byte vectors 4m + t of a key row (the
// four lanes of a group read 64 contiguous bytes per load), in the value
// product the lanes of group g read dims 8g .. 8g + 7 of the warp's 64.
template <typename T>
__global__ void __launch_bounds__(kStreamThreads, 3)
    StreamingAttentionKernel(const T* __restrict__ q,
                             const __half* __restrict__ k,
                             const __half* __restrict__ v,
                             const bool* __restrict__ mask,
                             const int* __restrict__ summary, int mask_rows,
                             int segments, int seq, float fill,
                             float* __restrict__ ws_acc,
                             float* __restrict__ ws_m,
                             float* __restrict__ ws_l, T* __restrict__ out) {
  constexpr int kWarpDims = kDepth / kStreamWarps;
  constexpr int kNSlices = kWarpDims / 8;
  __shared__ __align__(16) float scores_s[kStreamRows][kStreamTile];
  __shared__ __align__(16) uint16_t probs_s[kStreamRows][kStreamTile];
  __shared__ float max_s[kStreamRows];
  __shared__ float sum_s[kStreamRows];
  __shared__ float scale_s[kStreamRows];
  __shared__ int first_s[kStreamRows];
  __shared__ int end_s[kStreamRows];
  // query_s[(slice * 4 + i) * 32 + lane]: operand word i of `slice` for the
  // lane, so every load reads 32 consecutive words.
  __shared__ uint32_t query_s[kSlices * 4 * 32];

  const int splits = gridDim.x;
  const int split = blockIdx.x;
  const int r0 = blockIdx.y * kStreamRows;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  if (tid < kStreamRows) {
    InitializeRow(tid, r0, mask, summary, mask_rows, segments, seq, first_s,
                  end_s, max_s, sum_s);
  }
  for (int i = tid; i < kStreamRows * kVectorsPerRow; i += kStreamThreads) {
    const int row = i / kVectorsPerRow;
    const int vector = i % kVectorsPerRow;
    const int4 loaded = LoadQueryVector(
        q + static_cast<size_t>(r0 + row) * kDepth, vector);
    const int owner = (row & 7) * 4 + (vector & 3);
    const int slice = 2 * (vector >> 2);
    const int high = row >> 3;
    query_s[(slice * 4 + high) * 32 + owner] = loaded.x;
    query_s[(slice * 4 + 2 + high) * 32 + owner] = loaded.y;
    query_s[((slice + 1) * 4 + high) * 32 + owner] = loaded.z;
    query_s[((slice + 1) * 4 + 2 + high) * 32 + owner] = loaded.w;
  }
  float acc[kNSlices][4];
#pragma unroll
  for (int ns = 0; ns < kNSlices; ++ns) {
    acc[ns][0] = acc[ns][1] = acc[ns][2] = acc[ns][3] = 0.0f;
  }
  __syncthreads();
  int full_end;
  int tile_end;
  TileRange(first_s, end_s, kStreamRows, kStreamTile, seq, &full_end,
            &tile_end);

  for (int c = split; c < tile_end; c += splits) {
    const int j0 = c * kStreamTile;
    // Scores for rows (g, g + 8) x keys (8 * warp + 2t, +1).
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    {
      const int key = min(j0 + warp * 8 + g, seq - 1);
      const int4* k_row =
          reinterpret_cast<const int4*>(k + static_cast<size_t>(key) * kDepth) +
          t;
      constexpr int kBatch = 4;
#pragma unroll
      for (int m0 = 0; m0 < kSlices / 2; m0 += kBatch) {
        int4 kv[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
          kv[b] = k_row[4 * (m0 + b)];
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
          const uint32_t* a = &query_s[(2 * (m0 + b)) * 4 * 32 + lane];
          Mma(s0, s1, s2, s3, a[0], a[32], a[64], a[96], kv[b].x, kv[b].y);
          Mma(u0, u1, u2, u3, a[128], a[160], a[192], a[224], kv[b].z,
              kv[b].w);
        }
      }
    }
    *reinterpret_cast<float2*>(&scores_s[g][warp * 8 + 2 * t]) =
        make_float2(s0 + u0, s1 + u1);
    *reinterpret_cast<float2*>(&scores_s[g + 8][warp * 8 + 2 * t]) =
        make_float2(s2 + u2, s3 + u3);
    __syncthreads();

    // Online softmax: thread handles row tid / 16, keys 4 * (tid % 16) .. +3.
    {
      const int row = tid >> 4;
      const int kq = (tid & 15) * 4;
      float sc[4];
      *reinterpret_cast<float4*>(&sc[0]) =
          *reinterpret_cast<const float4*>(&scores_s[row][kq]);
      if (c >= full_end) {
        const bool* mask_row =
            mask == nullptr
                ? nullptr
                : mask + static_cast<size_t>((r0 + row) % mask_rows) * seq;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          // Keys past the end of the cache do not exist: no weight, even for
          // a row whose scores are all `fill`.
          const int key = j0 + kq + e;
          const bool visible =
              mask_row == nullptr || mask_row[min(key, seq - 1)];
          sc[e] = key < seq ? (visible ? sc[e] : fill) : -INFINITY;
        }
      }
      float tile_max = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        tile_max =
            fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
      }
      const float old_max = max_s[row];
      const float new_max = fmaxf(old_max, tile_max);
      const bool finite = new_max > -INFINITY;
      const __half2 p01 = __floats2half2_rn(
          finite ? __expf(sc[0] - new_max) * kProbScale : 0.0f,
          finite ? __expf(sc[1] - new_max) * kProbScale : 0.0f);
      const __half2 p23 = __floats2half2_rn(
          finite ? __expf(sc[2] - new_max) * kProbScale : 0.0f,
          finite ? __expf(sc[3] - new_max) * kProbScale : 0.0f);
      const float2 r01 = __half22float2(p01);
      const float2 r23 = __half22float2(p23);
      float tile_sum = r01.x + r01.y + r23.x + r23.y;
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset);
      }
      *reinterpret_cast<__half2*>(&probs_s[row][kq]) = p01;
      *reinterpret_cast<__half2*>(&probs_s[row][kq + 2]) = p23;
      if ((tid & 15) == 0) {
        const float scale =
            old_max > -INFINITY ? __expf(old_max - new_max) : 0.0f;
        max_s[row] = new_max;
        sum_s[row] = sum_s[row] * scale + tile_sum;
        scale_s[row] = scale;
      }
    }
    __syncthreads();

    // Values: the warp owns dims [warp * kWarpDims, +kWarpDims); lane (g, t)
    // reads dims 8g .. 8g + 7 of them from keys 2t, 2t + 1, 2t + 8, 2t + 9 of
    // each 16-key slice.
    {
      const float scale_lo = scale_s[g];
      const float scale_hi = scale_s[g + 8];
      if (scale_lo != 1.0f || scale_hi != 1.0f) {
#pragma unroll
        for (int ns = 0; ns < kNSlices; ++ns) {
          acc[ns][0] *= scale_lo;
          acc[ns][1] *= scale_lo;
          acc[ns][2] *= scale_hi;
          acc[ns][3] *= scale_hi;
        }
      }
#pragma unroll
      for (int ks = 0; ks < kStreamTile / 16; ++ks) {
        const uint32_t a0 =
            *reinterpret_cast<const uint32_t*>(&probs_s[g][16 * ks + 2 * t]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            &probs_s[g + 8][16 * ks + 2 * t]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            &probs_s[g][16 * ks + 8 + 2 * t]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            &probs_s[g + 8][16 * ks + 8 + 2 * t]);
        const int key = j0 + 16 * ks + 2 * t;
        const size_t column = static_cast<size_t>(warp) * kWarpDims + g * 8;
        uint32_t words[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int key_i = min(key + (i & 1) + (i >> 1) * 8, seq - 1);
          const int4 loaded = *reinterpret_cast<const int4*>(
              v + static_cast<size_t>(key_i) * kDepth + column);
          words[i][0] = loaded.x;
          words[i][1] = loaded.y;
          words[i][2] = loaded.z;
          words[i][3] = loaded.w;
        }
#pragma unroll
        for (int ns = 0; ns < kNSlices; ++ns) {
          const int word = ns >> 1;
          const int shift = (ns & 1) * 16;
          const uint32_t b0 = ((words[0][word] >> shift) & 0xffffu) |
                              ((words[1][word] >> shift) << 16);
          const uint32_t b1 = ((words[2][word] >> shift) & 0xffffu) |
                              ((words[3][word] >> shift) << 16);
          Mma(acc[ns][0], acc[ns][1], acc[ns][2], acc[ns][3], a0, a1, a2, a3,
              b0, b1);
        }
      }
    }
  }
  __syncthreads();

  // acc[ns][{0, 1}] are row g, dims 16t + {0, 8} + ns of the warp's range.
  const size_t column = static_cast<size_t>(warp) * kWarpDims + 16 * t;
  if (splits == 1) {
    const float sum_lo = sum_s[g];
    const float sum_hi = sum_s[g + 8];
    const float inv_lo = sum_lo > 0.0f ? 1.0f / sum_lo : 0.0f;
    const float inv_hi = sum_hi > 0.0f ? 1.0f / sum_hi : 0.0f;
    T* out_lo = out + static_cast<size_t>(r0 + g) * kDepth + column;
    T* out_hi = out + static_cast<size_t>(r0 + g + 8) * kDepth + column;
#pragma unroll
    for (int ns = 0; ns < kNSlices; ns += 2) {
      *reinterpret_cast<uint32_t*>(out_lo + ns) =
          PackOutput<T>(acc[ns][0] * inv_lo, acc[ns + 1][0] * inv_lo);
      *reinterpret_cast<uint32_t*>(out_lo + 8 + ns) =
          PackOutput<T>(acc[ns][1] * inv_lo, acc[ns + 1][1] * inv_lo);
      *reinterpret_cast<uint32_t*>(out_hi + ns) =
          PackOutput<T>(acc[ns][2] * inv_hi, acc[ns + 1][2] * inv_hi);
      *reinterpret_cast<uint32_t*>(out_hi + 8 + ns) =
          PackOutput<T>(acc[ns][3] * inv_hi, acc[ns + 1][3] * inv_hi);
    }
    return;
  }
  const size_t part = static_cast<size_t>(blockIdx.y) * splits + split;
  float* acc_lo = ws_acc + (part * kStreamRows + g) * kDepth + column;
  float* acc_hi = ws_acc + (part * kStreamRows + g + 8) * kDepth + column;
#pragma unroll
  for (int ns = 0; ns < kNSlices; ++ns) {
    acc_lo[ns] = acc[ns][0];
    acc_lo[8 + ns] = acc[ns][1];
    acc_hi[ns] = acc[ns][2];
    acc_hi[8 + ns] = acc[ns][3];
  }
  if (tid < kStreamRows) {
    ws_m[part * kStreamRows + tid] = max_s[tid];
    ws_l[part * kStreamRows + tid] = sum_s[tid];
  }
}

// One block per query row: merges the partial sums of the splits.
constexpr int kCombineThreads = 256;
constexpr int kCombineWarps = kCombineThreads / 32;

template <typename T>
__global__ void __launch_bounds__(kCombineThreads)
    CombineKernel(const float* __restrict__ ws_acc,
                  const float* __restrict__ ws_m,
                  const float* __restrict__ ws_l, int splits,
                  T* __restrict__ out) {
  constexpr int kElems = kDepth / kCombineThreads;  // dims per thread
  __shared__ float max_s[kCombineWarps];
  const int row = blockIdx.x;
  const int r = row % kStreamRows;
  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(row / kStreamRows) * splits;
  float m = -INFINITY;
  for (int s = tid; s < splits; s += kCombineThreads) {
    m = fmaxf(m, ws_m[(base + s) * kStreamRows + r]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
  }
  if ((tid & 31) == 0) {
    max_s[tid >> 5] = m;
  }
  __syncthreads();
  m = max_s[0];
#pragma unroll
  for (int w = 1; w < kCombineWarps; ++w) {
    m = fmaxf(m, max_s[w]);
  }
  float l = 0.0f;
  float acc[kElems];
#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    acc[e] = 0.0f;
  }
  for (int s = 0; s < splits; ++s) {
    const size_t idx = (base + s) * kStreamRows + r;
    const float ms = ws_m[idx];
    const float w = ms > -INFINITY ? __expf(ms - m) : 0.0f;
    l = fmaf(w, ws_l[idx], l);
    const float* part = ws_acc + idx * kDepth + tid * kElems;
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      acc[e] = fmaf(w, part[e], acc[e]);
    }
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  static_assert(kElems == 2, "one packed word per thread");
  *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(row) * kDepth +
                               tid * kElems) =
      PackOutput<T>(acc[0] * inv, acc[1] * inv);
}

// The split count bounds the workspace, which is sized from the shapes alone
// (the plan may be built on another device than the one that runs it).
int MaxSplits(int rows, int seq) {
  const int tiles = (seq + kStreamTile - 1) / kStreamTile;
  const int wanted = kMaxStreamSplits / (rows / kStreamRows);
  return wanted < 1 ? 1 : (wanted < tiles ? wanted : tiles);
}

int SegmentCount(int mask_rows) {
  const int wanted = 128 / mask_rows;
  return wanted < 1 ? 1 : (wanted > kMaxSegments ? kMaxSegments : wanted);
}

size_t SummaryBytes(int mask_rows) {
  const size_t bytes =
      static_cast<size_t>(mask_rows) * SegmentCount(mask_rows) * 2 * sizeof(int);
  return (bytes + 255) / 256 * 256;
}

size_t StreamingWorkspaceBytes(int rows, int mask_rows, int seq) {
  const size_t parts = static_cast<size_t>(rows) * MaxSplits(rows, seq);
  return SummaryBytes(mask_rows) +
         parts * (static_cast<size_t>(kDepth) + 2) * sizeof(float);
}

// The shape for the tiled kernel, if it takes it.
bool TiledShape(int rows, int mask_rows, int seq,
                LiteRtNvidiaAttentionShape* shape) {
  *shape = {/*heads=*/1, rows, mask_rows, kDepth, /*cache_len=*/seq,
            /*new_len=*/0};
  return rows >= kMinTiledRows && LiteRtNvidiaTiledAttentionSupports(shape);
}

template <typename T>
cudaError_t Launch(const T* q, const __half* k, const __half* v,
                   const bool* mask, int mask_rows, int rows, int seq,
                   float fill, T* out, void* workspace, cudaStream_t stream) {
  int device = 0;
  int multiprocessors = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&multiprocessors,
                                  cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  // About two blocks per multiprocessor, within the workspace bound.
  const int row_tiles = rows / kStreamRows;
  const int max_splits = MaxSplits(rows, seq);
  int splits = (2 * multiprocessors + row_tiles - 1) / row_tiles;
  splits = splits < max_splits ? splits : max_splits;
  const int segments = SegmentCount(mask_rows);
  int* summary = static_cast<int*>(workspace);
  float* ws_acc = reinterpret_cast<float*>(static_cast<char*>(workspace) +
                                           SummaryBytes(mask_rows));
  const size_t parts = static_cast<size_t>(rows) * splits;
  float* ws_m = ws_acc + parts * kDepth;
  float* ws_l = ws_m + parts;
  if (mask != nullptr) {
    MaskSummaryKernel<<<dim3(segments, mask_rows), kStreamThreads, 0, stream>>>(
        mask, seq, segments, summary);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  StreamingAttentionKernel<T>
      <<<dim3(splits, row_tiles), kStreamThreads, 0, stream>>>(
          q, k, v, mask, summary, mask_rows, segments, seq, fill, ws_acc, ws_m,
          ws_l, out);
  status = cudaGetLastError();
  if (status != cudaSuccess || splits == 1) return status;
  CombineKernel<T><<<rows, kCombineThreads, 0, stream>>>(ws_acc, ws_m, ws_l,
                                                         splits, out);
  return cudaGetLastError();
}

}  // namespace

extern "C" bool LiteRtNvidiaGlobalAttentionSupports(int32_t rows,
                                                    int32_t mask_rows,
                                                    int32_t depth) {
  return depth == kDepth && rows > 0 && rows % kStreamRows == 0 &&
         mask_rows > 0 && rows % mask_rows == 0;
}

extern "C" size_t LiteRtNvidiaGlobalAttentionWorkspaceBytes(int32_t rows,
                                                            int32_t mask_rows,
                                                            int32_t seq,
                                                            int32_t depth) {
  if (!LiteRtNvidiaGlobalAttentionSupports(rows, mask_rows, depth) ||
      seq <= 0) {
    return 0;
  }
  // The streaming kernel also serves the shapes of the tiled kernel on
  // devices that cannot run it.
  const size_t streaming = StreamingWorkspaceBytes(rows, mask_rows, seq);
  LiteRtNvidiaAttentionShape shape;
  if (!TiledShape(rows, mask_rows, seq, &shape)) {
    return streaming;
  }
  const size_t tiled = LiteRtNvidiaTiledAttentionWorkspaceBytes(&shape);
  return streaming > tiled ? streaming : tiled;
}

extern "C" cudaError_t LiteRtNvidiaLaunchGlobalAttention(
    const void* q, bool q_bf16, const void* k, const void* v, const bool* mask,
    int32_t mask_rows, int32_t rows, int32_t seq, int32_t depth, float fill,
    void* out, void* workspace, cudaStream_t stream) {
  if (mask == nullptr) {
    mask_rows = 1;
  }
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      workspace == nullptr || seq <= 0 ||
      !LiteRtNvidiaGlobalAttentionSupports(rows, mask_rows, depth)) {
    return cudaErrorInvalidValue;
  }
  // Without a mask the workspace may have been sized for any mask_rows.
  LiteRtNvidiaAttentionShape shape;
  if (mask != nullptr && TiledShape(rows, mask_rows, seq, &shape) &&
      LiteRtNvidiaTiledAttentionAvailable()) {
    return LiteRtNvidiaLaunchTiledAttention(
        &shape, q, q_bf16, k, v, /*k_new=*/nullptr, /*v_new=*/nullptr, mask,
        fill, out, workspace, stream);
  }
  if (q_bf16) {
    return Launch(static_cast<const __nv_bfloat16*>(q),
                  static_cast<const __half*>(k), static_cast<const __half*>(v),
                  mask, mask_rows, rows, seq, fill,
                  static_cast<__nv_bfloat16*>(out), workspace, stream);
  }
  return Launch(static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v), mask, mask_rows, rows, seq,
                fill, static_cast<__half*>(out), workspace, stream);
}
