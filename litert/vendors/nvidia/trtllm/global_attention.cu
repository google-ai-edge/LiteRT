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

// Where the 16 rows of a row tile are among the query rows, for tiles of
// chunk = 1 << chunk_shift tokens. Rows in order (chunk 16): tile t holds
// rows 16 t to 16 t + 15. Rows of `tokens` tokens per query head (row =
// head * tokens + token): tile t holds tokens t * chunk to t * chunk +
// chunk - 1 of each of its 16 / chunk heads, in the order a launch over
// those tokens alone holds them.
struct RowTiles {
  int tokens;
  int chunk_shift;
  __host__ __device__ int Row(int tile, int r) const {
    return (r >> chunk_shift) * tokens + (tile << chunk_shift) +
           (r & ((1 << chunk_shift) - 1));
  }
};

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
// the visible keys of its query row, whose mask row is `mask_row`, and resets
// its softmax state.
__device__ __forceinline__ void InitializeRow(
    int tid, int mask_row, const bool* mask, const int* summary, int segments,
    int seq, int* first_s, int* end_s, float* max_s, float* sum_s) {
  int first_masked = seq;
  int visible_end = mask == nullptr ? seq : 0;
  if (mask != nullptr) {
    const int* entry =
        summary + static_cast<size_t>(mask_row) * segments * 2;
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

// grid (splits, blocks). Block (split, y) owns kTiles row tiles, from tile
// kTiles * y on (those below `row_tiles`), and visits key tiles split,
// split + splits, ... With one split the block writes `out`; otherwise its
// partial sums go to the workspace (acc [row tiles][splits][16][depth], row
// maximum and row sum [row tiles][splits][16]) for CombineKernel.
//
// A block of two row tiles reads each key and value once for both. A long
// cache is read at the memory's bandwidth whatever the rows, so two tiles
// then cost what one does; each tile's sums are those of a block that owns
// it alone (the same operations in the same order, over the same key tiles).
//
// Lane (g, t) of a warp (g = lane / 4) holds rows g and g + 8 of a 16-row
// operand. Dims are assigned to lanes so that loads are contiguous: in the
// score product lane t reads the 16-byte vectors 4m + t of a key row (the
// four lanes of a group read 64 contiguous bytes per load), in the value
// product the lanes of group g read dims 8g .. 8g + 7 of the warp's 64.
template <typename T, int kTiles>
__global__ void __launch_bounds__(kStreamThreads, kTiles == 1 ? 3 : 2)
    StreamingAttentionKernel(const T* __restrict__ q,
                             const __half* __restrict__ k,
                             const __half* __restrict__ v,
                             const bool* __restrict__ mask,
                             const int* __restrict__ summary, int mask_rows,
                             int segments, int seq, float fill,
                             RowTiles layout, int row_tiles,
                             float* __restrict__ ws_acc,
                             float* __restrict__ ws_m,
                             float* __restrict__ ws_l, T* __restrict__ out) {
  constexpr int kWarpDims = kDepth / kStreamWarps;
  constexpr int kNSlices = kWarpDims / 8;
  __shared__ __align__(16) float scores_s[kTiles][kStreamRows][kStreamTile];
  __shared__ __align__(16) uint16_t probs_s[kTiles][kStreamRows][kStreamTile];
  __shared__ float max_s[kTiles][kStreamRows];
  __shared__ float sum_s[kTiles][kStreamRows];
  __shared__ float scale_s[kTiles][kStreamRows];
  __shared__ int first_s[kTiles][kStreamRows];
  __shared__ int end_s[kTiles][kStreamRows];
  // query_s[tile][(slice * 4 + i) * 32 + lane]: operand word i of `slice` for
  // the lane, so every load reads 32 consecutive words.
  __shared__ uint32_t query_s[kTiles][kSlices * 4 * 32];

  const int splits = gridDim.x;
  const int split = blockIdx.x;
  const int tile0 = blockIdx.y * kTiles;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  float acc[kTiles][kNSlices][4];
  // The row tiles the block owns (the last block of an odd number owns one),
  // and the mask row of the row whose softmax the thread computes (tid / 16).
  bool owns[kTiles];
  const bool* mask_row[kTiles];
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
#pragma unroll
    for (int ns = 0; ns < kNSlices; ++ns) {
      acc[i][ns][0] = acc[i][ns][1] = acc[i][ns][2] = acc[i][ns][3] = 0.0f;
    }
    owns[i] = kTiles == 1 || tile0 + i < row_tiles;
    mask_row[i] = nullptr;
    if (!owns[i]) {
      continue;
    }
    if (tid < kStreamRows) {
      InitializeRow(tid, layout.Row(tile0 + i, tid) % mask_rows, mask, summary,
                    segments, seq, first_s[i], end_s[i], max_s[i], sum_s[i]);
    }
    if (mask != nullptr) {
      mask_row[i] =
          mask +
          static_cast<size_t>(layout.Row(tile0 + i, tid >> 4) % mask_rows) *
              seq;
    }
    for (int j = tid; j < kStreamRows * kVectorsPerRow; j += kStreamThreads) {
      const int row = j / kVectorsPerRow;
      const int vector = j % kVectorsPerRow;
      const int4 loaded = LoadQueryVector(
          q + static_cast<size_t>(layout.Row(tile0 + i, row)) * kDepth, vector);
      const int owner = (row & 7) * 4 + (vector & 3);
      const int slice = 2 * (vector >> 2);
      const int high = row >> 3;
      query_s[i][(slice * 4 + high) * 32 + owner] = loaded.x;
      query_s[i][(slice * 4 + 2 + high) * 32 + owner] = loaded.y;
      query_s[i][((slice + 1) * 4 + high) * 32 + owner] = loaded.z;
      query_s[i][((slice + 1) * 4 + 2 + high) * 32 + owner] = loaded.w;
    }
  }
  __syncthreads();
  // Each row tile's key tiles are those its rows see, as in a block of its
  // own; the block walks up to the last of them.
  int full_end[kTiles];
  int tile_end[kTiles];
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    full_end[i] = 0;
    tile_end[i] = 0;
    if (owns[i]) {
      TileRange(first_s[i], end_s[i], kStreamRows, kStreamTile, seq,
                &full_end[i], &tile_end[i]);
    }
  }
  int last = tile_end[0];
#pragma unroll
  for (int i = 1; i < kTiles; ++i) {
    last = max(last, tile_end[i]);
  }

  for (int c = split; c < last; c += splits) {
    const int j0 = c * kStreamTile;
    bool visits[kTiles];
    float s[kTiles][4];
    float u[kTiles][4];
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      visits[i] = kTiles == 1 || c < tile_end[i];
      s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
      u[i][0] = u[i][1] = u[i][2] = u[i][3] = 0.0f;
    }
    // Scores for rows (g, g + 8) x keys (8 * warp + 2t, +1).
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
#pragma unroll
          for (int i = 0; i < kTiles; ++i) {
            if (!visits[i]) {
              continue;
            }
            const uint32_t* a = &query_s[i][(2 * (m0 + b)) * 4 * 32 + lane];
            Mma(s[i][0], s[i][1], s[i][2], s[i][3], a[0], a[32], a[64], a[96],
                kv[b].x, kv[b].y);
            Mma(u[i][0], u[i][1], u[i][2], u[i][3], a[128], a[160], a[192],
                a[224], kv[b].z, kv[b].w);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      if (!visits[i]) {
        continue;
      }
      *reinterpret_cast<float2*>(&scores_s[i][g][warp * 8 + 2 * t]) =
          make_float2(s[i][0] + u[i][0], s[i][1] + u[i][1]);
      *reinterpret_cast<float2*>(&scores_s[i][g + 8][warp * 8 + 2 * t]) =
          make_float2(s[i][2] + u[i][2], s[i][3] + u[i][3]);
    }
    __syncthreads();

    // Online softmax: thread handles row tid / 16, keys 4 * (tid % 16) .. +3.
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      if (!visits[i]) {
        continue;
      }
      const int row = tid >> 4;
      const int kq = (tid & 15) * 4;
      float sc[4];
      *reinterpret_cast<float4*>(&sc[0]) =
          *reinterpret_cast<const float4*>(&scores_s[i][row][kq]);
      if (c >= full_end[i]) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          // Keys past the end of the cache do not exist: no weight, even for
          // a row whose scores are all `fill`.
          const int key = j0 + kq + e;
          const bool visible =
              mask_row[i] == nullptr || mask_row[i][min(key, seq - 1)];
          sc[e] = key < seq ? (visible ? sc[e] : fill) : -INFINITY;
        }
      }
      float tile_max = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        tile_max =
            fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
      }
      const float old_max = max_s[i][row];
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
      *reinterpret_cast<__half2*>(&probs_s[i][row][kq]) = p01;
      *reinterpret_cast<__half2*>(&probs_s[i][row][kq + 2]) = p23;
      if ((tid & 15) == 0) {
        const float scale =
            old_max > -INFINITY ? __expf(old_max - new_max) : 0.0f;
        max_s[i][row] = new_max;
        sum_s[i][row] = sum_s[i][row] * scale + tile_sum;
        scale_s[i][row] = scale;
      }
    }
    __syncthreads();

    // Values: the warp owns dims [warp * kWarpDims, +kWarpDims); lane (g, t)
    // reads dims 8g .. 8g + 7 of them from keys 2t, 2t + 1, 2t + 8, 2t + 9 of
    // each 16-key slice.
    {
#pragma unroll
      for (int i = 0; i < kTiles; ++i) {
        if (!visits[i]) {
          continue;
        }
        const float scale_lo = scale_s[i][g];
        const float scale_hi = scale_s[i][g + 8];
        if (scale_lo != 1.0f || scale_hi != 1.0f) {
#pragma unroll
          for (int ns = 0; ns < kNSlices; ++ns) {
            acc[i][ns][0] *= scale_lo;
            acc[i][ns][1] *= scale_lo;
            acc[i][ns][2] *= scale_hi;
            acc[i][ns][3] *= scale_hi;
          }
        }
      }
#pragma unroll
      for (int ks = 0; ks < kStreamTile / 16; ++ks) {
        uint32_t a[kTiles][4];
#pragma unroll
        for (int i = 0; i < kTiles; ++i) {
          if (!visits[i]) {
            continue;
          }
          a[i][0] = *reinterpret_cast<const uint32_t*>(
              &probs_s[i][g][16 * ks + 2 * t]);
          a[i][1] = *reinterpret_cast<const uint32_t*>(
              &probs_s[i][g + 8][16 * ks + 2 * t]);
          a[i][2] = *reinterpret_cast<const uint32_t*>(
              &probs_s[i][g][16 * ks + 8 + 2 * t]);
          a[i][3] = *reinterpret_cast<const uint32_t*>(
              &probs_s[i][g + 8][16 * ks + 8 + 2 * t]);
        }
        const int key = j0 + 16 * ks + 2 * t;
        const size_t column = static_cast<size_t>(warp) * kWarpDims + g * 8;
        uint32_t words[4][4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const int key_w = min(key + (w & 1) + (w >> 1) * 8, seq - 1);
          const int4 loaded = *reinterpret_cast<const int4*>(
              v + static_cast<size_t>(key_w) * kDepth + column);
          words[w][0] = loaded.x;
          words[w][1] = loaded.y;
          words[w][2] = loaded.z;
          words[w][3] = loaded.w;
        }
#pragma unroll
        for (int ns = 0; ns < kNSlices; ++ns) {
          const int word = ns >> 1;
          const int shift = (ns & 1) * 16;
          const uint32_t b0 = ((words[0][word] >> shift) & 0xffffu) |
                              ((words[1][word] >> shift) << 16);
          const uint32_t b1 = ((words[2][word] >> shift) & 0xffffu) |
                              ((words[3][word] >> shift) << 16);
#pragma unroll
          for (int i = 0; i < kTiles; ++i) {
            if (!visits[i]) {
              continue;
            }
            Mma(acc[i][ns][0], acc[i][ns][1], acc[i][ns][2], acc[i][ns][3],
                a[i][0], a[i][1], a[i][2], a[i][3], b0, b1);
          }
        }
      }
    }
  }
  __syncthreads();

  // acc[i][ns][{0, 1}] are row g, dims 16t + {0, 8} + ns of the warp's range.
  const size_t column = static_cast<size_t>(warp) * kWarpDims + 16 * t;
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    if (!owns[i]) {
      continue;
    }
    if (splits == 1) {
      const float sum_lo = sum_s[i][g];
      const float sum_hi = sum_s[i][g + 8];
      const float inv_lo = sum_lo > 0.0f ? 1.0f / sum_lo : 0.0f;
      const float inv_hi = sum_hi > 0.0f ? 1.0f / sum_hi : 0.0f;
      T* out_lo =
          out + static_cast<size_t>(layout.Row(tile0 + i, g)) * kDepth + column;
      T* out_hi = out +
                  static_cast<size_t>(layout.Row(tile0 + i, g + 8)) * kDepth +
                  column;
#pragma unroll
      for (int ns = 0; ns < kNSlices; ns += 2) {
        *reinterpret_cast<uint32_t*>(out_lo + ns) = PackOutput<T>(
            acc[i][ns][0] * inv_lo, acc[i][ns + 1][0] * inv_lo);
        *reinterpret_cast<uint32_t*>(out_lo + 8 + ns) = PackOutput<T>(
            acc[i][ns][1] * inv_lo, acc[i][ns + 1][1] * inv_lo);
        *reinterpret_cast<uint32_t*>(out_hi + ns) = PackOutput<T>(
            acc[i][ns][2] * inv_hi, acc[i][ns + 1][2] * inv_hi);
        *reinterpret_cast<uint32_t*>(out_hi + 8 + ns) = PackOutput<T>(
            acc[i][ns][3] * inv_hi, acc[i][ns + 1][3] * inv_hi);
      }
      continue;
    }
    const size_t part = static_cast<size_t>(tile0 + i) * splits + split;
    float* acc_lo = ws_acc + (part * kStreamRows + g) * kDepth + column;
    float* acc_hi = ws_acc + (part * kStreamRows + g + 8) * kDepth + column;
#pragma unroll
    for (int ns = 0; ns < kNSlices; ++ns) {
      acc_lo[ns] = acc[i][ns][0];
      acc_lo[8 + ns] = acc[i][ns][1];
      acc_hi[ns] = acc[i][ns][2];
      acc_hi[8 + ns] = acc[i][ns][3];
    }
    if (tid < kStreamRows) {
      ws_m[part * kStreamRows + tid] = max_s[i][tid];
      ws_l[part * kStreamRows + tid] = sum_s[i][tid];
    }
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
                  RowTiles layout, T* __restrict__ out) {
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
  *reinterpret_cast<uint32_t*>(
      out + static_cast<size_t>(layout.Row(row / kStreamRows, r)) * kDepth +
      tid * kElems) = PackOutput<T>(acc[0] * inv, acc[1] * inv);
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

// `split_rows`: the rows of the launch whose split count this one takes
// (Launch).
size_t StreamingWorkspaceBytes(int rows, int split_rows, int mask_rows,
                               int seq) {
  const size_t parts = static_cast<size_t>(rows) * MaxSplits(split_rows, seq);
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

// `split_rows` is `rows`, or 16 for row tiles that are each computed as a
// 16-row launch of their own computes them: the splits are then that
// launch's, and a block owns two row tiles.
template <typename T>
cudaError_t Launch(const T* q, const __half* k, const __half* v,
                   const bool* mask, int mask_rows, int rows, int seq,
                   float fill, RowTiles layout, int split_rows, T* out,
                   void* workspace, cudaStream_t stream) {
  int device = 0;
  int multiprocessors = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&multiprocessors,
                                  cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  // About two blocks per multiprocessor, within the workspace bound.
  const int row_tiles = rows / kStreamRows;
  const int split_tiles = split_rows / kStreamRows;
  const int max_splits = MaxSplits(split_rows, seq);
  int splits = (2 * multiprocessors + split_tiles - 1) / split_tiles;
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
  if (split_rows == rows) {
    StreamingAttentionKernel<T, 1>
        <<<dim3(splits, row_tiles), kStreamThreads, 0, stream>>>(
            q, k, v, mask, summary, mask_rows, segments, seq, fill, layout,
            row_tiles, ws_acc, ws_m, ws_l, out);
  } else {
    StreamingAttentionKernel<T, 2>
        <<<dim3(splits, (row_tiles + 1) / 2), kStreamThreads, 0, stream>>>(
            q, k, v, mask, summary, mask_rows, segments, seq, fill, layout,
            row_tiles, ws_acc, ws_m, ws_l, out);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess || splits == 1) return status;
  CombineKernel<T><<<rows, kCombineThreads, 0, stream>>>(
      ws_acc, ws_m, ws_l, splits, layout, out);
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
  const size_t streaming =
      StreamingWorkspaceBytes(rows, rows, mask_rows, seq);
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
  const RowTiles in_order = {/*tokens=*/0, /*chunk_shift=*/4};
  if (q_bf16) {
    return Launch(static_cast<const __nv_bfloat16*>(q),
                  static_cast<const __half*>(k), static_cast<const __half*>(v),
                  mask, mask_rows, rows, seq, fill, in_order, rows,
                  static_cast<__nv_bfloat16*>(out), workspace, stream);
  }
  return Launch(static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v), mask, mask_rows, rows, seq,
                fill, in_order, rows, static_cast<__half*>(out), workspace,
                stream);
}

extern "C" bool LiteRtNvidiaGlobalAttentionTokensSupports(int32_t rows,
                                                          int32_t tokens,
                                                          int32_t mask_rows,
                                                          int32_t depth) {
  if (depth != kDepth || rows <= 0 || tokens <= 0 || rows % tokens != 0) {
    return false;
  }
  // A row tile holds 16 / heads tokens of each query head.
  const int32_t heads = rows / tokens;
  return heads <= kStreamRows && (heads & (heads - 1)) == 0 &&
         tokens % (kStreamRows / heads) == 0 &&
         (mask_rows == 1 || mask_rows == tokens);
}

extern "C" size_t LiteRtNvidiaGlobalAttentionTokensWorkspaceBytes(
    int32_t rows, int32_t tokens, int32_t mask_rows, int32_t seq,
    int32_t depth) {
  if (!LiteRtNvidiaGlobalAttentionTokensSupports(rows, tokens, mask_rows,
                                                 depth) ||
      seq <= 0) {
    return 0;
  }
  return StreamingWorkspaceBytes(rows, kStreamRows, mask_rows, seq);
}

extern "C" cudaError_t LiteRtNvidiaLaunchGlobalAttentionTokens(
    const void* q, bool q_bf16, const void* k, const void* v, const bool* mask,
    int32_t mask_rows, int32_t tokens, int32_t rows, int32_t seq, int32_t depth,
    float fill, void* out, void* workspace, cudaStream_t stream) {
  if (mask == nullptr) {
    mask_rows = 1;
  }
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      workspace == nullptr || seq <= 0 ||
      !LiteRtNvidiaGlobalAttentionTokensSupports(rows, tokens, mask_rows,
                                                 depth)) {
    return cudaErrorInvalidValue;
  }
  int chunk_shift = 0;
  while ((rows / tokens) << chunk_shift < kStreamRows) {
    ++chunk_shift;
  }
  const RowTiles by_token = {tokens, chunk_shift};
  if (q_bf16) {
    return Launch(static_cast<const __nv_bfloat16*>(q),
                  static_cast<const __half*>(k), static_cast<const __half*>(v),
                  mask, mask_rows, rows, seq, fill, by_token, kStreamRows,
                  static_cast<__nv_bfloat16*>(out), workspace, stream);
  }
  return Launch(static_cast<const __half*>(q), static_cast<const __half*>(k),
                static_cast<const __half*>(v), mask, mask_rows, rows, seq,
                fill, by_token, kStreamRows, static_cast<__half*>(out),
                workspace, stream);
}
