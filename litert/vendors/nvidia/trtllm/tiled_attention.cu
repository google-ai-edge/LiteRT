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

#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kTile = 16;  // keys per tile
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlockElements = 32768;  // query rows x depth of a block
constexpr int kMaxBitmapWords = 512;   // 16384 tiles
constexpr int kMaxSplits = 8;
constexpr size_t kMaxPartialBytes = size_t{144} << 20;

__device__ __forceinline__ void Mma(float& d0, float& d1, float& d2, float& d3,
                                    uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// m16n8k16 accumulating in fp16: twice the throughput of fp32 accumulation.
__device__ __forceinline__ void MmaHalf(uint32_t& d0, uint32_t& d1,
                                        uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d0), "+r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ldmatrix.x4: lanes 0-7 pass the addresses of the rows of matrix 0 (eight
// 16-bit elements each), lanes 8-15 those of matrix 1, and so on. Every lane
// receives its operand fragment of each matrix.
// The same without an addend: d = a . b.
__device__ __forceinline__ void MmaHalfProduct(uint32_t& d0, uint32_t& d1,
                                               uint32_t a0, uint32_t a1,
                                               uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  const uint32_t zero = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero));
}

__device__ __forceinline__ void LoadMatrixX4(uint32_t& r0, uint32_t& r1,
                                             uint32_t& r2, uint32_t& r3,
                                             const void* shared_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(address));
}

__device__ __forceinline__ void LoadMatrixX4Trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3,
                                                  const void* shared_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(address));
}

__device__ __forceinline__ uint32_t PackHalves(float lo, float hi) {
  const __half2 packed = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ __forceinline__ uint32_t PackBf16(float lo, float hi) {
  const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ __forceinline__ float2 UnpackHalves(uint32_t word) {
  return __half22float2(*reinterpret_cast<const __half2*>(&word));
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

// A block of eight warps owns kRows query rows in kRowGroups groups of 16.
// kGroupWarps warps share a row group: each computes the scores over
// kWarpSlices k-slices (16 dims each) and the values of kWarpDims dims.
//
// BF16 queries carry a relative rounding error of 2^-9 into every score, more
// than fp16 accumulation over chains of kChain k-slices adds, so their scores
// accumulate in fp16 (twice the tensor-core throughput) and the chains are
// added up in fp32. The scores of FP16 queries accumulate in fp32.
template <int kD, bool kHalfScores>
struct Geometry {
  static constexpr int kRows = kBlockElements / kD;          // 64 | 128
  static constexpr int kRowGroups = kRows / 16;              // 4 | 8
  static constexpr int kGroupWarps = kWarps / kRowGroups;    // 2 | 1
  static constexpr int kSlices = kD / 16;                    // 32 | 16
  static constexpr int kWarpSlices = kSlices / kGroupWarps;  // 16
  static constexpr int kChain = 8;
  static constexpr int kWarpChains = kWarpSlices / kChain;   // 2
  // Partial sums per score in shared memory: fp16 chains or fp32 sums.
  static constexpr int kPartials =
      kHalfScores ? kGroupWarps * kWarpChains : kGroupWarps;
  static constexpr int kPartialBytes = kHalfScores ? 2 : 4;
  static constexpr int kWarpDims = kD / kGroupWarps;         // 256
  static constexpr int kNSlices = kWarpDims / 8;             // 32
  static constexpr int kVectorsPerRow = kD / 8;  // 16-byte vectors per row
  static constexpr int kQueryVectors = kRows * kVectorsPerRow / kThreads;
  static constexpr int kStageVectors = kTile * kVectorsPerRow / kThreads;
  static constexpr int kRowThreads = kThreads / kRows;       // 4 | 2
  static constexpr int kThreadKeys = kTile / kRowThreads;    // 4 | 8

  // Row strides are padded so that the eight rows of one ldmatrix fall in
  // distinct shared-memory banks.
  static constexpr int kRowBytes = kD * 2 + 16;
  static constexpr int kProbRowBytes = kTile * 2 + 16;
  static constexpr size_t kQuery = 0;
  static constexpr size_t kCache =
      kQuery + static_cast<size_t>(kRows) * kRowBytes;
  static constexpr size_t kScores =
      kCache + static_cast<size_t>(kTile) * kRowBytes;
  static constexpr size_t kProbs =
      kScores + static_cast<size_t>(kPartials) * kRows * kTile * kPartialBytes;
  static constexpr size_t kState =
      kProbs + static_cast<size_t>(kRows) * kProbRowBytes;
  // Row maximum, sum, scale and factor, then a flag per softmax warp.
  static constexpr size_t kBitmaps =
      kState + static_cast<size_t>(kRows) * 4 * sizeof(float) +
      kWarps * sizeof(int);
  static constexpr size_t kBytes =
      kBitmaps + 2 * kMaxBitmapWords * sizeof(uint32_t);
};

constexpr size_t kSharedBytes =
    Geometry<256, false>::kBytes > Geometry<512, false>::kBytes
        ? Geometry<256, false>::kBytes
        : Geometry<512, false>::kBytes;
static_assert(Geometry<256, true>::kBytes == Geometry<256, false>::kBytes &&
                  Geometry<512, true>::kBytes == Geometry<512, false>::kBytes,
              "both score precisions take the same shared memory");

// row_bits[(mask_row * 2 + {0, 1}) * words + w]: bit c is set when the mask
// row sees a key of tile 32 w + c / when it sees all 16 keys of the tile.
__global__ void __launch_bounds__(kThreads)
    RowBitmapKernel(const bool* __restrict__ mask, int seq, int words,
                    uint32_t* __restrict__ row_bits) {
  const int word = blockIdx.x * kWarps + threadIdx.x / 32;
  const int j0 = (word * 32 + threadIdx.x % 32) * kTile;
  const int mask_row = blockIdx.y;
  const bool* row = mask + static_cast<size_t>(mask_row) * seq;
  bool any = false;
  bool all = j0 + kTile <= seq;
  for (int j = j0; j < min(j0 + kTile, seq); ++j) {
    const bool visible = row[j];
    any = any || visible;
    all = all && visible;
  }
  const uint32_t any_word = __ballot_sync(0xffffffffu, any);
  const uint32_t all_word = __ballot_sync(0xffffffffu, all);
  if (threadIdx.x % 32 == 0 && word < words) {
    uint32_t* bits = row_bits + static_cast<size_t>(mask_row) * 2 * words;
    bits[word] = any_word;
    bits[words + word] = all_word;
  }
}

// One block per `block_rows` consecutive mask rows: the tiles some row of
// the block sees and the tiles all of its rows see entirely. A row that sees
// no key takes the softmax over all keys (every score is `fill`), like the
// unfused graph, so its block visits every tile.
__global__ void __launch_bounds__(kThreads)
    BlockBitmapKernel(const uint32_t* __restrict__ row_bits, int block_rows,
                      int words, int tiles, uint32_t* __restrict__ block_bits) {
  constexpr int kThreadWords = kMaxBitmapWords / kThreads;
  __shared__ int sees_s[kBlockElements / 256];
  const int first_row = blockIdx.x * block_rows;
  if (threadIdx.x < block_rows) {
    sees_s[threadIdx.x] = 0;
  }
  __syncthreads();
  uint32_t any_words[kThreadWords];
  uint32_t all_words[kThreadWords];
#pragma unroll
  for (int i = 0; i < kThreadWords; ++i) {
    any_words[i] = 0;
    all_words[i] = 0xffffffffu;
    const int w = threadIdx.x + i * kThreads;
    if (w >= words) continue;
    for (int r = 0; r < block_rows; ++r) {
      const uint32_t* bits =
          row_bits + static_cast<size_t>(first_row + r) * 2 * words;
      const uint32_t any = bits[w];
      any_words[i] |= any;
      all_words[i] &= bits[words + w];
      if (any != 0) {
        sees_s[r] = 1;  // every writer stores the same value
      }
    }
  }
  __syncthreads();
  bool blind = false;
  for (int r = 0; r < block_rows; ++r) {
    blind = blind || sees_s[r] == 0;
  }
  uint32_t* out = block_bits + static_cast<size_t>(blockIdx.x) * 2 * words;
#pragma unroll
  for (int i = 0; i < kThreadWords; ++i) {
    const int w = threadIdx.x + i * kThreads;
    if (w >= words) continue;
    const int valid = min(max(tiles - w * 32, 0), 32);
    const uint32_t valid_bits =
        valid == 32 ? 0xffffffffu : ((1u << valid) - 1u);
    out[w] = blind ? valid_bits : any_words[i];
    out[words + w] = blind ? 0u : all_words[i];
  }
}

struct Sources {
  const __half* k;
  const __half* v;
  const __half* k_new;
  const __half* v_new;
  int cache_len;
  int new_len;
};

// grid (splits, heads * rows / kRows). Block (split, y) owns kRows query rows
// of one head and visits the visible tiles congruent to `split` modulo
// `splits`. With one split the block writes `out`; otherwise its partial sums
// go to the workspace (acc [blocks][kRows][depth], row maximum and row sum
// [blocks][kRows]) for CombineKernel.
//
// Each thread loads the next tile into registers while its warp computes and
// stores it to shared memory once the block is done with the tile it
// replaces. The scores accumulate in fp32. The value product of a tile is one
// fp16 sum over its 16 keys per output, with weights relative to the largest
// of their row in the tile and divided by the tile size, so that the sum
// cannot overflow and small weights keep their precision; the tiles of a row
// are combined in fp32.
template <typename T, int kD>
__global__ void __launch_bounds__(kThreads, 1)
    TiledAttentionKernel(const T* __restrict__ q, Sources src,
                         const bool* __restrict__ mask,
                         const uint32_t* __restrict__ block_bits,
                         int mask_rows, int rows, float fill,
                         float* __restrict__ ws_acc, float* __restrict__ ws_m,
                         float* __restrict__ ws_l, T* __restrict__ out) {
  constexpr bool kHalfScores = IsBf16<T>::kValue;
  using G = Geometry<kD, kHalfScores>;
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* query_s = smem + G::kQuery;
  unsigned char* cache_s = smem + G::kCache;
  float* scores_s = reinterpret_cast<float*>(smem + G::kScores);
  uint32_t* chains_s = reinterpret_cast<uint32_t*>(smem + G::kScores);
  unsigned char* probs_s = smem + G::kProbs;
  float* max_s = reinterpret_cast<float*>(smem + G::kState);
  float* sum_s = max_s + G::kRows;
  float* scale_s = sum_s + G::kRows;
  float* factor_s = scale_s + G::kRows;
  int* rescaled_s = reinterpret_cast<int*>(factor_s + G::kRows);
  uint32_t* any_s = reinterpret_cast<uint32_t*>(smem + G::kBitmaps);
  uint32_t* all_s = any_s + kMaxBitmapWords;

  const int splits = gridDim.x;
  const int split = blockIdx.x;
  const int row_tiles = rows / G::kRows;
  const int head = blockIdx.y / row_tiles;
  const int r0 = (blockIdx.y % row_tiles) * G::kRows;
  const int seq = src.cache_len + src.new_len;
  const int tiles = (seq + kTile - 1) / kTile;
  const int words = (tiles + 31) / 32;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row_group = warp / G::kGroupWarps;
  const int part = warp % G::kGroupWarps;

  const T* q_head = q + static_cast<size_t>(head) * rows * kD;
  const __half* k_head = src.k + static_cast<size_t>(head) * src.cache_len * kD;
  const __half* v_head = src.v + static_cast<size_t>(head) * src.cache_len * kD;
  const __half* k_new_head =
      src.k_new + static_cast<size_t>(head) * src.new_len * kD;
  const __half* v_new_head =
      src.v_new + static_cast<size_t>(head) * src.new_len * kD;

  if (tid < G::kRows) {
    max_s[tid] = -INFINITY;
    sum_s[tid] = 0.0f;
  }
  for (int w = tid; w < words; w += kThreads) {
    if (block_bits == nullptr) {
      const int valid = min(tiles - w * 32, 32);
      any_s[w] = valid == 32 ? 0xffffffffu : ((1u << valid) - 1u);
      const int full = min(max(seq / kTile - w * 32, 0), 32);
      all_s[w] = full == 32 ? 0xffffffffu : ((1u << full) - 1u);
    } else {
      const uint32_t* bits =
          block_bits +
          static_cast<size_t>((r0 % mask_rows) / G::kRows) * 2 * words;
      any_s[w] = bits[w];
      all_s[w] = bits[words + w];
    }
  }
#pragma unroll
  for (int j = 0; j < G::kQueryVectors; ++j) {
    const int i = tid + j * kThreads;
    const int row = i / G::kVectorsPerRow;
    const int vector = i % G::kVectorsPerRow;
    *reinterpret_cast<int4*>(query_s + row * G::kRowBytes + vector * 16) =
        LoadQueryVector(q_head + static_cast<size_t>(r0 + row) * kD, vector);
  }
  float acc[G::kNSlices][4];
#pragma unroll
  for (int ns = 0; ns < G::kNSlices; ++ns) {
    acc[ns][0] = acc[ns][1] = acc[ns][2] = acc[ns][3] = 0.0f;
  }
  __syncthreads();

  // First visible tile at or after `c` among split, split + splits, ...
  auto next_tile = [&](int c) {
    while (c < tiles) {
      const uint32_t rest = any_s[c >> 5] >> (c & 31);
      if (rest & 1u) {
        return c;
      }
      if (rest == 0) {
        const int next_word = ((c >> 5) + 1) << 5;
        c += ((next_word - c + splits - 1) / splits) * splits;
      } else {
        c += splits;
      }
    }
    return tiles;
  };

  // Per-lane operand addresses. A 16x16 "A" operand is {rows 0-7, rows 8-15}
  // x {cols 0-7, cols 8-15}.
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_col = (lane >> 4) * 8;
  const unsigned char* query_lane =
      query_s + (row_group * 16 + a_row) * G::kRowBytes +
      (part * G::kWarpSlices * 16 + a_col) * 2;
  const unsigned char* probs_lane =
      probs_s + (row_group * 16 + a_row) * G::kProbRowBytes + a_col * 2;
  // Scores "B" (dims x keys): matrices {0, 1} are keys 0-7 at dims {0-7,
  // 8-15} of the slice, matrices {2, 3} keys 8-15.
  const unsigned char* key_lane =
      cache_s + ((lane & 7) + (lane >> 4) * 8) * G::kRowBytes +
      (part * G::kWarpSlices * 16 + ((lane >> 3) & 1) * 8) * 2;
  // Values "B" (keys x dims), loaded transposed: matrices {0, 1} are keys
  // {0-7, 8-15} at dims 0-7 of a pair of n-slices, matrices {2, 3} the same
  // keys at dims 8-15.
  const unsigned char* value_lane =
      cache_s + ((lane & 7) + ((lane >> 3) & 1) * 8) * G::kRowBytes +
      (part * G::kWarpDims + (lane >> 4) * 8) * 2;

  int4 staged[G::kStageVectors];
  auto load_tile = [&](bool values, int tile) {
    const int j0 = tile * kTile;
    const bool fresh = j0 >= src.cache_len;
    const __half* base = fresh ? (values ? v_new_head : k_new_head)
                               : (values ? v_head : k_head);
    const int first = fresh ? j0 - src.cache_len : j0;
    const int last = (fresh ? src.new_len : src.cache_len) - 1;
#pragma unroll
    for (int j = 0; j < G::kStageVectors; ++j) {
      const int i = tid + j * kThreads;
      const int key = min(first + i / G::kVectorsPerRow, last);
      staged[j] = reinterpret_cast<const int4*>(
          base + static_cast<size_t>(key) * kD)[i % G::kVectorsPerRow];
    }
  };
  auto store_tile = [&]() {
#pragma unroll
    for (int j = 0; j < G::kStageVectors; ++j) {
      const int i = tid + j * kThreads;
      *reinterpret_cast<int4*>(cache_s +
                               (i / G::kVectorsPerRow) * G::kRowBytes +
                               (i % G::kVectorsPerRow) * 16) = staged[j];
    }
  };

  int c = next_tile(split);
  if (c < tiles) {
    load_tile(/*values=*/false, c);
  }
  while (c < tiles) {
    const int j0 = c * kTile;
    store_tile();
    __syncthreads();
    load_tile(/*values=*/true, c);

    // Scores for rows (g, g + 8) of the row group x keys 8 kg + 2t, + 1 over
    // the warp's k-slices.
    if (kHalfScores) {
      uint32_t chain[G::kWarpChains][2][2];
#pragma unroll
      for (int s = 0; s < G::kWarpSlices; ++s) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        LoadMatrixX4(a0, a1, a2, a3, query_lane + s * 32);
        LoadMatrixX4(b0, b1, b2, b3, key_lane + s * 32);
        uint32_t* sums = chain[s / G::kChain][0];
        if (s % G::kChain == 0) {
          MmaHalfProduct(sums[0], sums[1], a0, a1, a2, a3, b0, b1);
          MmaHalfProduct(sums[2], sums[3], a0, a1, a2, a3, b2, b3);
        } else {
          MmaHalf(sums[0], sums[1], a0, a1, a2, a3, b0, b1);
          MmaHalf(sums[2], sums[3], a0, a1, a2, a3, b2, b3);
        }
      }
#pragma unroll
      for (int ch = 0; ch < G::kWarpChains; ++ch) {
        uint32_t* lo = chains_s +
                       ((static_cast<size_t>(part) * G::kWarpChains + ch) *
                            G::kRows +
                        row_group * 16 + g) *
                           (kTile / 2) +
                       t;
        uint32_t* hi = lo + 8 * (kTile / 2);
        lo[0] = chain[ch][0][0];
        lo[4] = chain[ch][1][0];
        hi[0] = chain[ch][0][1];
        hi[4] = chain[ch][1][1];
      }
    } else {
      float sc[2][4];
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        sc[kg][0] = sc[kg][1] = sc[kg][2] = sc[kg][3] = 0.0f;
      }
#pragma unroll
      for (int s = 0; s < G::kWarpSlices; ++s) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        LoadMatrixX4(a0, a1, a2, a3, query_lane + s * 32);
        LoadMatrixX4(b0, b1, b2, b3, key_lane + s * 32);
        Mma(sc[0][0], sc[0][1], sc[0][2], sc[0][3], a0, a1, a2, a3, b0, b1);
        Mma(sc[1][0], sc[1][1], sc[1][2], sc[1][3], a0, a1, a2, a3, b2, b3);
      }
      float* lo = scores_s +
                  (static_cast<size_t>(part) * G::kRows + row_group * 16 + g) *
                      kTile +
                  2 * t;
      float* hi = lo + 8 * kTile;
#pragma unroll
      for (int kg = 0; kg < 2; ++kg) {
        *reinterpret_cast<float2*>(lo + 8 * kg) =
            make_float2(sc[kg][0], sc[kg][1]);
        *reinterpret_cast<float2*>(hi + 8 * kg) =
            make_float2(sc[kg][2], sc[kg][3]);
      }
    }
    __syncthreads();

    // Softmax of the tile: thread handles row tid / kRowThreads and
    // kThreadKeys keys. The value tile replaces the key tile in shared memory
    // meanwhile.
    store_tile();
    {
      const int row = tid / G::kRowThreads;
      const int kq = (tid % G::kRowThreads) * G::kThreadKeys;
      float sc[G::kThreadKeys];
#pragma unroll
      for (int e = 0; e < G::kThreadKeys; e += 2) {
        float2 sum = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int p = 0; p < G::kPartials; ++p) {
          const size_t pair = (static_cast<size_t>(p) * G::kRows + row) *
                                  (kTile / 2) +
                              (kq + e) / 2;
          const float2 partial =
              kHalfScores
                  ? UnpackHalves(chains_s[pair])
                  : *reinterpret_cast<const float2*>(scores_s + 2 * pair);
          sum.x += partial.x;
          sum.y += partial.y;
        }
        sc[e] = sum.x;
        sc[e + 1] = sum.y;
      }
      if (((all_s[c >> 5] >> (c & 31)) & 1u) == 0) {
        const bool* mask_row =
            mask == nullptr
                ? nullptr
                : mask + static_cast<size_t>((r0 + row) % mask_rows) * seq;
#pragma unroll
        for (int e = 0; e < G::kThreadKeys; ++e) {
          // Keys past the end of the sequence do not exist: no weight, even
          // for a row whose scores are all `fill`.
          const int key = j0 + kq + e;
          const bool visible =
              mask_row == nullptr || mask_row[min(key, seq - 1)];
          sc[e] = key < seq ? (visible ? sc[e] : fill) : -INFINITY;
        }
      }
      float tile_max = sc[0];
#pragma unroll
      for (int e = 1; e < G::kThreadKeys; ++e) {
        tile_max = fmaxf(tile_max, sc[e]);
      }
#pragma unroll
      for (int offset = G::kRowThreads / 2; offset > 0; offset >>= 1) {
        tile_max =
            fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
      }
      const bool finite = tile_max > -INFINITY;
      float tile_sum = 0.0f;
      uint32_t* prob_words = reinterpret_cast<uint32_t*>(
          probs_s + row * G::kProbRowBytes + kq * 2);
#pragma unroll
      for (int e = 0; e < G::kThreadKeys; e += 2) {
        const __half2 packed = __floats2half2_rn(
            finite ? __expf(sc[e] - tile_max) * (1.0f / kTile) : 0.0f,
            finite ? __expf(sc[e + 1] - tile_max) * (1.0f / kTile) : 0.0f);
        // The row sums use the rounded weights the value product sees.
        const float2 rounded = __half22float2(packed);
        tile_sum += rounded.x + rounded.y;
        prob_words[e / 2] = *reinterpret_cast<const uint32_t*>(&packed);
      }
#pragma unroll
      for (int offset = G::kRowThreads / 2; offset > 0; offset >>= 1) {
        tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset);
      }
      // The rows of a warp are those of one row group; the running sums of a
      // row group are only rescaled when the maximum of one of its rows grew.
      float scale = 1.0f;
      if (tid % G::kRowThreads == 0) {
        const float old_max = max_s[row];
        const float new_max = fmaxf(old_max, tile_max);
        scale = old_max >= new_max ? 1.0f : __expf(old_max - new_max);
        const float factor = finite ? __expf(tile_max - new_max) : 0.0f;
        max_s[row] = new_max;
        sum_s[row] = sum_s[row] * scale + tile_sum * factor;
        scale_s[row] = scale;
        factor_s[row] = factor;
      }
      const bool rescaled = __any_sync(0xffffffffu, scale != 1.0f);
      if (lane == 0) {
        rescaled_s[warp] = rescaled ? 1 : 0;
      }
    }
    __syncthreads();

    // Values: rows of the row group x the warp's dims, one fp16 sum over the
    // 16 keys of the tile per output.
    const int next = next_tile(c + splits);
    if (next < tiles) {
      load_tile(/*values=*/false, next);
    }
    {
      const float scale_lo = scale_s[row_group * 16 + g];
      const float scale_hi = scale_s[row_group * 16 + g + 8];
      const float factor_lo = factor_s[row_group * 16 + g];
      const float factor_hi = factor_s[row_group * 16 + g + 8];
      bool rescaled = false;
#pragma unroll
      for (int w = 0; w < G::kGroupWarps; ++w) {
        rescaled = rescaled || rescaled_s[row_group * G::kGroupWarps + w] != 0;
      }
      uint32_t a0, a1, a2, a3;
      LoadMatrixX4(a0, a1, a2, a3, probs_lane);
      constexpr int kBatch = 16;  // n-slices per batch of fp16 sums
#pragma unroll
      for (int n0 = 0; n0 < G::kNSlices; n0 += kBatch) {
        uint32_t sums[kBatch][2];
#pragma unroll
        for (int i = 0; i < kBatch; i += 2) {
          uint32_t b0, b1, b2, b3;
          LoadMatrixX4Trans(b0, b1, b2, b3, value_lane + (n0 + i) * 16);
          MmaHalfProduct(sums[i][0], sums[i][1], a0, a1, a2, a3, b0, b1);
          MmaHalfProduct(sums[i + 1][0], sums[i + 1][1], a0, a1, a2, a3, b2,
                         b3);
        }
        if (rescaled) {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const float2 lo = UnpackHalves(sums[i][0]);
            const float2 hi = UnpackHalves(sums[i][1]);
            float* a = acc[n0 + i];
            a[0] = fmaf(a[0], scale_lo, lo.x * factor_lo);
            a[1] = fmaf(a[1], scale_lo, lo.y * factor_lo);
            a[2] = fmaf(a[2], scale_hi, hi.x * factor_hi);
            a[3] = fmaf(a[3], scale_hi, hi.y * factor_hi);
          }
        } else {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const float2 lo = UnpackHalves(sums[i][0]);
            const float2 hi = UnpackHalves(sums[i][1]);
            float* a = acc[n0 + i];
            a[0] = fmaf(lo.x, factor_lo, a[0]);
            a[1] = fmaf(lo.y, factor_lo, a[1]);
            a[2] = fmaf(hi.x, factor_hi, a[2]);
            a[3] = fmaf(hi.y, factor_hi, a[3]);
          }
        }
      }
    }
    __syncthreads();
    c = next;
  }

  // acc[ns][{0, 1}] are row g, dims 8 * ns + 2t + {0, 1} of the warp's range.
  const int row_lo = row_group * 16 + g;
  const size_t column = static_cast<size_t>(part) * G::kWarpDims + 2 * t;
  if (splits == 1) {
    const float sum_lo = sum_s[row_lo];
    const float sum_hi = sum_s[row_lo + 8];
    const float inv_lo = sum_lo > 0.0f ? 1.0f / sum_lo : 0.0f;
    const float inv_hi = sum_hi > 0.0f ? 1.0f / sum_hi : 0.0f;
    T* out_head = out + static_cast<size_t>(head) * rows * kD;
    uint32_t* out_lo = reinterpret_cast<uint32_t*>(
        out_head + static_cast<size_t>(r0 + row_lo) * kD + column);
    uint32_t* out_hi = reinterpret_cast<uint32_t*>(
        out_head + static_cast<size_t>(r0 + row_lo + 8) * kD + column);
#pragma unroll
    for (int ns = 0; ns < G::kNSlices; ++ns) {
      out_lo[ns * 4] =
          PackOutput<T>(acc[ns][0] * inv_lo, acc[ns][1] * inv_lo);
      out_hi[ns * 4] =
          PackOutput<T>(acc[ns][2] * inv_hi, acc[ns][3] * inv_hi);
    }
    return;
  }
  const size_t partial = static_cast<size_t>(blockIdx.y) * splits + split;
  float* acc_lo = ws_acc + (partial * G::kRows + row_lo) * kD + column;
  float* acc_hi = ws_acc + (partial * G::kRows + row_lo + 8) * kD + column;
#pragma unroll
  for (int ns = 0; ns < G::kNSlices; ++ns) {
    *reinterpret_cast<float2*>(acc_lo + ns * 8) =
        make_float2(acc[ns][0], acc[ns][1]);
    *reinterpret_cast<float2*>(acc_hi + ns * 8) =
        make_float2(acc[ns][2], acc[ns][3]);
  }
  if (tid < G::kRows) {
    ws_m[partial * G::kRows + tid] = max_s[tid];
    ws_l[partial * G::kRows + tid] = sum_s[tid];
  }
}

// One block per query row, two dims per thread: merges the partial sums of
// the splits.
template <typename T, int kD>
__global__ void __launch_bounds__(kD / 2)
    CombineKernel(const float* __restrict__ ws_acc,
                  const float* __restrict__ ws_m,
                  const float* __restrict__ ws_l, int splits,
                  T* __restrict__ out) {
  constexpr int kCombineThreads = kD / 2;
  constexpr int kCombineWarps = kCombineThreads / 32;
  constexpr int kBlockRows = kBlockElements / kD;
  __shared__ float max_s[kCombineWarps];
  const int row = blockIdx.x;
  const int r = row % kBlockRows;
  const int tid = threadIdx.x;
  const size_t base = static_cast<size_t>(row / kBlockRows) * splits;
  float m = -INFINITY;
  for (int s = tid; s < splits; s += kCombineThreads) {
    m = fmaxf(m, ws_m[(base + s) * kBlockRows + r]);
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
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const size_t idx = (base + s) * kBlockRows + r;
    const float ms = ws_m[idx];
    const float w = ms > -INFINITY ? __expf(ms - m) : 0.0f;
    l = fmaf(w, ws_l[idx], l);
    const float2 partial =
        *reinterpret_cast<const float2*>(ws_acc + idx * kD + tid * 2);
    acc0 = fmaf(w, partial.x, acc0);
    acc1 = fmaf(w, partial.y, acc1);
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(row) * kD + tid * 2) =
      PackOutput<T>(acc0 * inv, acc1 * inv);
}

int BlockRows(int depth) { return kBlockElements / depth; }

int Tiles(const LiteRtNvidiaAttentionShape& shape) {
  return (shape.cache_len + shape.new_len + kTile - 1) / kTile;
}

int Words(const LiteRtNvidiaAttentionShape& shape) {
  return (Tiles(shape) + 31) / 32;
}

// The split count bounds the workspace, which is sized from the shape alone
// (the plan may be built on another device than the one that runs it).
int MaxSplits(const LiteRtNvidiaAttentionShape& shape) {
  const size_t part_bytes = static_cast<size_t>(shape.heads) * shape.rows *
                            (shape.depth + 2) * sizeof(float);
  const size_t fit = kMaxPartialBytes / part_bytes;
  const int tiles = Tiles(shape);
  const int splits =
      fit < 1 ? 1 : (fit > kMaxSplits ? kMaxSplits : static_cast<int>(fit));
  return splits < tiles ? splits : tiles;
}

size_t BitmapBytes(const LiteRtNvidiaAttentionShape& shape) {
  const size_t block_rows = BlockRows(shape.depth);
  const size_t bitmaps = shape.mask_rows + shape.mask_rows / block_rows;
  const size_t bytes = bitmaps * 2 * Words(shape) * sizeof(uint32_t);
  return (bytes + 255) / 256 * 256;
}

// One block fills a multiprocessor, so the blocks run in waves of
// `multiprocessors`. Costs in units of one tile of one block: a block takes
// its share of the tiles plus about 12 to start, and merging the partial sums
// of a row about depth / 256000 per split.
int ChooseSplits(const LiteRtNvidiaAttentionShape& shape, int multiprocessors) {
  const int blocks = shape.heads * (shape.rows / BlockRows(shape.depth));
  const double tiles = Tiles(shape);
  const double merge = static_cast<double>(shape.heads) * shape.rows *
                       shape.depth / 256000.0;
  const int max_splits = MaxSplits(shape);
  int best = 1;
  double best_cost = 0.0;
  for (int s = 1; s <= max_splits; ++s) {
    const int waves = (blocks * s + multiprocessors - 1) / multiprocessors;
    const double cost = waves * (tiles / s + 12.0) + (s > 1 ? merge * s : 0.0);
    if (s == 1 || cost < best_cost) {
      best_cost = cost;
      best = s;
    }
  }
  return best;
}

struct DeviceLimits {
  int multiprocessors = 0;
  int shared_memory_optin = 0;
  int major = 0;
};

cudaError_t GetDeviceLimits(DeviceLimits* limits) {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&limits->multiprocessors,
                                  cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&limits->major,
                                  cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) return status;
  return cudaDeviceGetAttribute(&limits->shared_memory_optin,
                                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                device);
}

template <typename T, int kD>
cudaError_t LaunchBlocks(const LiteRtNvidiaAttentionShape& shape, const T* q,
                         const Sources& src, const bool* mask,
                         const uint32_t* block_bits, float fill, int splits,
                         float* ws_acc, float* ws_m, float* ws_l, T* out,
                         cudaStream_t stream) {
  using G = Geometry<kD, IsBf16<T>::kValue>;
  static bool attribute_set = false;  // per instantiation
  if (!attribute_set) {
    const cudaError_t status = cudaFuncSetAttribute(
        TiledAttentionKernel<T, kD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(G::kBytes));
    if (status != cudaSuccess) return status;
    attribute_set = true;
  }
  const int blocks = shape.heads * (shape.rows / G::kRows);
  TiledAttentionKernel<T, kD>
      <<<dim3(splits, blocks), kThreads, G::kBytes, stream>>>(
          q, src, mask, block_bits, shape.mask_rows, shape.rows, fill, ws_acc,
          ws_m, ws_l, out);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess || splits == 1) return status;
  CombineKernel<T, kD><<<shape.heads * shape.rows, kD / 2, 0, stream>>>(
      ws_acc, ws_m, ws_l, splits, out);
  return cudaGetLastError();
}

template <typename T>
cudaError_t Launch(const LiteRtNvidiaAttentionShape& shape, const T* q,
                   const Sources& src, const bool* mask, float fill, T* out,
                   void* workspace, cudaStream_t stream) {
  DeviceLimits limits;
  cudaError_t status = GetDeviceLimits(&limits);
  if (status != cudaSuccess) return status;
  if (static_cast<size_t>(limits.shared_memory_optin) < kSharedBytes) {
    return cudaErrorInvalidConfiguration;
  }
  const int block_rows = BlockRows(shape.depth);
  const int splits = ChooseSplits(shape, limits.multiprocessors);
  const int seq = shape.cache_len + shape.new_len;
  const int words = Words(shape);
  uint32_t* row_bits = static_cast<uint32_t*>(workspace);
  uint32_t* block_bits =
      row_bits + static_cast<size_t>(shape.mask_rows) * 2 * words;
  float* ws_acc = reinterpret_cast<float*>(static_cast<char*>(workspace) +
                                           BitmapBytes(shape));
  const size_t parts = static_cast<size_t>(shape.heads) * shape.rows * splits;
  float* ws_m = ws_acc + parts * shape.depth;
  float* ws_l = ws_m + parts;
  if (mask != nullptr) {
    RowBitmapKernel<<<dim3((words + kWarps - 1) / kWarps, shape.mask_rows),
                      kThreads, 0, stream>>>(mask, seq, words, row_bits);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    BlockBitmapKernel<<<shape.mask_rows / block_rows, kThreads, 0, stream>>>(
        row_bits, block_rows, words, Tiles(shape), block_bits);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  } else {
    block_bits = nullptr;
  }
  if (shape.depth == 512) {
    return LaunchBlocks<T, 512>(shape, q, src, mask, block_bits, fill, splits,
                                ws_acc, ws_m, ws_l, out, stream);
  }
  return LaunchBlocks<T, 256>(shape, q, src, mask, block_bits, fill, splits,
                              ws_acc, ws_m, ws_l, out, stream);
}

}  // namespace

extern "C" bool LiteRtNvidiaTiledAttentionSupports(
    const LiteRtNvidiaAttentionShape* shape) {
  if (shape == nullptr || (shape->depth != 256 && shape->depth != 512) ||
      shape->heads <= 0 || shape->rows <= 0 || shape->mask_rows <= 0 ||
      shape->cache_len <= 0 || shape->new_len < 0) {
    return false;
  }
  const int block_rows = BlockRows(shape->depth);
  return shape->rows % block_rows == 0 && shape->mask_rows % block_rows == 0 &&
         shape->rows % shape->mask_rows == 0 &&
         (shape->new_len == 0 || shape->cache_len % kTile == 0) &&
         static_cast<int64_t>(shape->cache_len) + shape->new_len <=
             int64_t{kMaxBitmapWords} * 32 * kTile;
}

extern "C" bool LiteRtNvidiaTiledAttentionAvailable() {
  DeviceLimits limits;
  return GetDeviceLimits(&limits) == cudaSuccess && limits.major >= 8 &&
         static_cast<size_t>(limits.shared_memory_optin) >= kSharedBytes;
}

extern "C" size_t LiteRtNvidiaTiledAttentionWorkspaceBytes(
    const LiteRtNvidiaAttentionShape* shape) {
  if (!LiteRtNvidiaTiledAttentionSupports(shape)) {
    return 0;
  }
  const size_t parts =
      static_cast<size_t>(shape->heads) * shape->rows * MaxSplits(*shape);
  return BitmapBytes(*shape) +
         parts * (static_cast<size_t>(shape->depth) + 2) * sizeof(float);
}

extern "C" cudaError_t LiteRtNvidiaLaunchTiledAttention(
    const LiteRtNvidiaAttentionShape* shape, const void* q, bool q_bf16,
    const void* k, const void* v, const void* k_new, const void* v_new,
    const bool* mask, float fill, void* out, void* workspace,
    cudaStream_t stream) {
  if (!LiteRtNvidiaTiledAttentionSupports(shape) || q == nullptr ||
      k == nullptr || v == nullptr || out == nullptr || workspace == nullptr ||
      (shape->new_len > 0 && (k_new == nullptr || v_new == nullptr))) {
    return cudaErrorInvalidValue;
  }
  const Sources src = {static_cast<const __half*>(k),
                       static_cast<const __half*>(v),
                       static_cast<const __half*>(k_new),
                       static_cast<const __half*>(v_new),
                       shape->cache_len,
                       shape->new_len};
  if (q_bf16) {
    return Launch(*shape, static_cast<const __nv_bfloat16*>(q), src, mask,
                  fill, static_cast<__nv_bfloat16*>(out), workspace, stream);
  }
  return Launch(*shape, static_cast<const __half*>(q), src, mask, fill,
                static_cast<__half*>(out), workspace, stream);
}
