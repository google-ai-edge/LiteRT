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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_CUH_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_CUH_

// Device code of the blockwise INT4 GEMV: a few activation rows times a tile
// of weight rows whose INT4 values carry one FP16 scale per block of 32 input
// dims (TFLite blockwise quantization along the input dimension). Shared by
// the dense fully-connected launch and the mixture-of-experts launches of
// int4_block_gemv.cu.
//
// A thread owns whole input blocks (16 bytes of weights per row and 32
// activation values per activation row), so a block's scale is applied to
// its eight nibbles per word while they are converted and the products are
// accumulated in FP32 straight into the row accumulators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace litert::nvidia::int4_block {

constexpr int kBlock = 32;     // input dims per scale
constexpr int kTileRows = 8;   // weight rows per CTA tile (the largest)
constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;

// A tile of up to kTileRows consecutive weight rows: row i starts at
// weights + i * weight_stride bytes and its scales at scales + i *
// scale_stride halves.
struct WeightTile {
  const uint8_t* weights;
  size_t weight_stride;
  const __half* scales;
  size_t scale_stride;
  int valid_rows;
  // Compact scales instead of `scales`: a row's blocks' fractions (bytes,
  // scale_stride apart per row) of the row's maximum (scale_max, max_stride
  // halves apart per row).
  const uint8_t* fractions = nullptr;
  const __half* scale_max = nullptr;
  size_t max_stride = 0;
};

// The activation rows of a launch: row r is `rows[r]` (or r without a row
// index) of a [*, stride] matrix.
template <typename T>
struct ActivationRows {
  const T* base;
  size_t stride;
  const int32_t* rows;  // nullptr: the identity
  __device__ __forceinline__ const T* Row(int r) const {
    return base + static_cast<size_t>(rows == nullptr ? r : rows[r]) * stride;
  }
};

// Eight consecutive values of an activation row as four float2 pairs.
template <typename T>
__device__ __forceinline__ void LoadPairs(const T* values, float2 pairs[4]);

template <>
__device__ __forceinline__ void LoadPairs<__nv_bfloat16>(
    const __nv_bfloat16* values, float2 pairs[4]) {
  const uint4 bits = __ldg(reinterpret_cast<const uint4*>(values));
  const uint32_t words[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 pair;
    *reinterpret_cast<uint32_t*>(&pair) = words[i];
    pairs[i] = __bfloat1622float2(pair);
  }
}

template <>
__device__ __forceinline__ void LoadPairs<float>(const float* values,
                                                 float2 pairs[4]) {
  const float4 lo = __ldg(reinterpret_cast<const float4*>(values));
  const float4 hi = __ldg(reinterpret_cast<const float4*>(values) + 1);
  pairs[0] = make_float2(lo.x, lo.y);
  pairs[1] = make_float2(lo.z, lo.w);
  pairs[2] = make_float2(hi.x, hi.y);
  pairs[3] = make_float2(hi.z, hi.w);
}

__device__ __forceinline__ float Nibble(uint32_t word, int index) {
  // Two's complement sign extension of bits [4 index, 4 index + 4).
  const uint32_t bits = (word >> (4 * index)) & 0xfu;
  return static_cast<float>(static_cast<int>((bits ^ 8u) - 8u));
}

// With programmatic dependent launch, the point before which a kernel may
// not read what the kernel before it wrote: a no-op for a kernel launched
// without the dependency.
__device__ __forceinline__ void WaitForKernelBefore() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

// Accumulates, for every activation row r and tile row i, the dot products
// over the input blocks this thread owns (blockIdx-independent: block b for
// b = threadIdx.x, threadIdx.x + Threads, ...) into acc[r][i]. With
// `norm_scale` (an RMS norm fused into the product), activation k is
// multiplied by norm_scale[k] and the thread's sum of squares of the raw
// activations of each row goes to squares[r]; the caller divides by the
// row's RMS after the reduction. With kDependentLaunch, the thread's first
// weights are asked for before the wait on the kernel before: the weights
// are not its output, the activations are.
template <typename T, int Rows, int Threads, int TileRows = kTileRows,
          bool kCompact = false, bool kDependentLaunch = false>
__device__ __forceinline__ void AccumulateTile(const WeightTile& tile,
                                               const ActivationRows<T>& act,
                                               int input_size,
                                               float acc[Rows][TileRows],
                                               const float* norm_scale = nullptr,
                                               float* squares = nullptr) {
  const int blocks = input_size / kBlock;
  const T* rows[Rows];
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    rows[r] = act.Row(r);
  }
  // Compact scales (kCompact): the rows' maxima, loaded once; converted
  // where first used so the loads overlap the first weight loads.
  __half row_max[TileRows];
  if constexpr (kCompact) {
#pragma unroll
    for (int i = 0; i < TileRows; ++i) {
      row_max[i] = i < tile.valid_rows
                       ? __ldg(tile.scale_max + i * tile.max_stride)
                       : __float2half(0.0f);
    }
  }
  for (int b = threadIdx.x; b < blocks; b += Threads) {
    uint4 words[TileRows];
    float scales[TileRows];
#pragma unroll
    for (int i = 0; i < TileRows; ++i) {
      if (i < tile.valid_rows) {
        words[i] = __ldg(reinterpret_cast<const uint4*>(
            tile.weights + i * tile.weight_stride + b * (kBlock / 2)));
        if constexpr (kCompact) {
          scales[i] = __half2float(row_max[i]) * (1.0f / 255.0f) *
                      static_cast<float>(
                          __ldg(tile.fractions + i * tile.scale_stride + b));
        } else {
          scales[i] = __half2float(tile.scales[i * tile.scale_stride + b]);
        }
      } else {
        words[i] = make_uint4(0, 0, 0, 0);
        scales[i] = 0.0f;
      }
    }
    if constexpr (kDependentLaunch) {
      if (b == threadIdx.x) WaitForKernelBefore();
    }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      float2 pairs[Rows][4];
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        LoadPairs<T>(rows[r] + b * kBlock + w * 8, pairs[r]);
      }
      if (norm_scale != nullptr) {
        float2 scales2[4];
        LoadPairs<float>(norm_scale + b * kBlock + w * 8, scales2);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            squares[r] = fmaf(pairs[r][n].x, pairs[r][n].x, squares[r]);
            squares[r] = fmaf(pairs[r][n].y, pairs[r][n].y, squares[r]);
            pairs[r][n].x *= scales2[n].x;
            pairs[r][n].y *= scales2[n].y;
          }
        }
      }
#pragma unroll
      for (int i = 0; i < TileRows; ++i) {
        const uint32_t word = w == 0   ? words[i].x
                              : w == 1 ? words[i].y
                              : w == 2 ? words[i].z
                                       : words[i].w;
        const float scale = scales[i];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const float w0 = Nibble(word, 2 * n) * scale;
          const float w1 = Nibble(word, 2 * n + 1) * scale;
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            acc[r][i] = fmaf(pairs[r][n].x, w0, acc[r][i]);
            acc[r][i] = fmaf(pairs[r][n].y, w1, acc[r][i]);
          }
        }
      }
    }
  }
}

// Reduces acc over the CTA: afterwards thread 0 holds the tile sums in
// acc[r][i] (other threads: unspecified). `warp_sums` is shared memory of
// Rows * kTileRows * (Threads / 32) floats, unused by a single warp.
template <int Rows, int Threads, int TileRows = kTileRows>
__device__ __forceinline__ void ReduceTile(float acc[Rows][TileRows],
                                           float* warp_sums) {
  constexpr int kWarps = Threads / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
      for (int i = 0; i < TileRows; ++i) {
        acc[r][i] += __shfl_down_sync(kFullWarpMask, acc[r][i], offset);
      }
    }
  }
  if constexpr (kWarps > 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int i = 0; i < TileRows; ++i) {
          warp_sums[(r * TileRows + i) * kWarps + warp] = acc[r][i];
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int i = 0; i < TileRows; ++i) {
          float sum = 0.0f;
#pragma unroll
          for (int source = 0; source < kWarps; ++source) {
            sum += warp_sums[(r * TileRows + i) * kWarps + source];
          }
          acc[r][i] = sum;
        }
      }
    }
  }
}

// Sums `Rows` per-thread values over the CTA into thread 0's `values`.
// `scratch` is shared memory of Rows * (Threads / 32) floats.
template <int Rows, int Threads>
__device__ __forceinline__ void ReduceRows(float values[Rows], float* scratch) {
  constexpr int kWarps = Threads / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      values[r] += __shfl_down_sync(kFullWarpMask, values[r], offset);
    }
  }
  if constexpr (kWarps > 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < Rows; ++r) scratch[r * kWarps + warp] = values[r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        float sum = 0.0f;
#pragma unroll
        for (int source = 0; source < kWarps; ++source) {
          sum += scratch[r * kWarps + source];
        }
        values[r] = sum;
      }
    }
  }
}

// Gate activations of a gated feed-forward: act(gate) * up, with the tanh
// approximation (1) or the exact erf form (2) of GELU, or SiLU (3).
__device__ __forceinline__ float Gelu(float x, int gate) {
  if (gate == 3) {
    return x / (1.0f + __expf(-x));
  }
  if (gate == 1) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
  }
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// --- A verify step's rows on the tensor cores --------------------------------
//
// Two or more activation rows (a verify step's) times tiles of kMmaTile
// weight rows by mma.sync m16n8k16 with BF16 operands and FP32 accumulation:
// the weight rows are the A operand, the activation rows the B operand's
// kMmaRows columns (columns past the rows in use are zero). Lane (g, c) =
// (lane / 4, lane % 4) holds word c (elements 8c to 8c + 7) of a block of
// weight rows g and g + 8 and of activation row g, and a block's MMAs take
// each lane's elements in pairs (0, 4), (1, 5) | (2, 6), (3, 7): the order
// of a block's 32 products differs from the elements' but is the same for
// both operands. A nibble is an exact BF16 integer, and a block's sum is
// multiplied by its scale in FP32 after its MMAs, as in AccumulateTile.
// Activations that are FP32 (an expert's hidden row) or reach the product
// through an FP32 multiply (a fused norm's scale) are split into a BF16 high
// part and a BF16 low part, each with its own MMAs, so that they keep 16 of
// their 24 bits instead of 8. What differs from AccumulateTile's arithmetic
// is the order of the FP32 sums (inside the tensor core, and a block's sum
// before its scale) and those activations' remaining rounding.
constexpr int kMmaTile = 16;   // weight rows of an MMA tile
constexpr int kMmaRows = 8;    // activation rows of an MMA
constexpr int kMmaUnroll = 4;  // consecutive input blocks a warp loads at once

__device__ __forceinline__ void MmaBf16(float d[4], const uint32_t a[4],
                                        uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The nibbles of a weight word as BF16 pairs of exact integers, pair j
// holding nibbles j (low half) and j + 4. 0x4300 is BF16 128.0, whose
// mantissa's lowest bit is 1: the nibble plus 8 (its sign bit flipped) in
// the mantissa gives 136 + nibble, and subtracting 136 the nibble.
__device__ __forceinline__ void NibblePairs(uint32_t word, uint32_t pairs[4]) {
  const __nv_bfloat162 bias = __floats2bfloat162_rn(136.0f, 136.0f);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t bits = ((word >> (4 * j)) & 0x000f000fu) ^ 0x43084308u;
    const __nv_bfloat162 value =
        __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&bits), bias);
    pairs[j] = *reinterpret_cast<const uint32_t*>(&value);
  }
}

// FP32 values as BF16 pairs: the high parts (rounded to nearest) and the
// remainders' (lo).
__device__ __forceinline__ void SplitPair(float a, float b, uint32_t& hi,
                                          uint32_t& lo) {
  const __nv_bfloat162 high = __floats2bfloat162_rn(a, b);
  const float2 back = __bfloat1622float2(high);
  const __nv_bfloat162 low = __floats2bfloat162_rn(a - back.x, b - back.y);
  hi = *reinterpret_cast<const uint32_t*>(&high);
  lo = *reinterpret_cast<const uint32_t*>(&low);
}

// A lane's B operand for one block: elements 8c to 8c + 7 of its activation
// row as the pairs (0, 4), (1, 5), (2, 6), (3, 7), high parts in hi and,
// for FP32 values, low parts in lo.
struct ActivationFragment {
  uint32_t hi[4];
  uint32_t lo[4];
};

// The fragment from raw values `x` (eight, as four float pairs) times
// `scale` (or not).
__device__ __forceinline__ void SplitFragment(const float2 x[4],
                                              const float2* scale,
                                              ActivationFragment& f) {
  float v[8];
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    v[2 * n] = scale != nullptr ? x[n].x * scale[n].x : x[n].x;
    v[2 * n + 1] = scale != nullptr ? x[n].y * scale[n].y : x[n].y;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) SplitPair(v[j], v[j + 4], f.hi[j], f.lo[j]);
}

// Whether the activations of type T take low parts: FP32 rows, or BF16 rows
// with a fused norm.
template <typename T, bool kNorm>
constexpr bool kSplitActivations = !std::is_same_v<T, __nv_bfloat16> || kNorm;

// The block scale of a tile row (row < tile.valid_rows).
template <bool kCompact>
__device__ __forceinline__ float RowScale(const WeightTile& tile, int row,
                                          int block, float row_max) {
  if constexpr (kCompact) {
    return row_max * (1.0f / 255.0f) *
           static_cast<float>(
               __ldg(tile.fractions + row * tile.scale_stride + block));
  } else {
    return __half2float(tile.scales[row * tile.scale_stride + block]);
  }
}

// ldmatrix.x4: lanes 8j to 8j + 7 pass the addresses of the eight 16-byte
// rows of matrix j, and lane (g, c) receives word c of row g of each.
__device__ __forceinline__ void LoadMatrixX4(uint32_t r[4],
                                             const void* shared_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address));
}

// A warp's staging of a group's weights: for each tile and half h, eight
// rows (g + 8h) of kMmaUnroll 16-byte blocks, a row's blocks XOR-swizzled by
// its index so that the lanes' 16-byte stores and ldmatrix's row reads each
// fall in distinct banks.
template <int kTiles>
struct MmaStage {
  uint4 chunks[kTiles][2][8][kMmaUnroll];
};
static_assert(kMmaUnroll == 4, "the swizzle permutes four blocks of a row");

__device__ __forceinline__ int SwizzledBlock(int row, int block) {
  return block ^ ((row >> 1) & 3);
}

// Accumulates, for activation rows 0 to count - 1 of `act` and kTiles weight
// tiles of kMmaTile rows, the sums over the input blocks of the groups of
// kMmaUnroll consecutive blocks first, first + step, ...: lane (g, c) adds to
// acc[t] the values of tile t's rows g (acc[t][0], [1]) and g + 8 ([2], [3])
// for activation rows 2c ([0], [2]) and 2c + 1 ([1], [3]), the MMA
// accumulator's layout. A group's weights are read a whole block per lane
// (lane (g, c) block c of rows g and g + 8, as the one-row kernel reads them)
// and reach the MMA's layout through `stage` and ldmatrix; the next group's
// are in flight while a group multiplies. With kNorm the activations are
// multiplied by norm_scale on the way in and the lane's sum of squares of its
// raw values of row g goes to *squares (each of a row's values is one lane's
// of one block).
template <typename T, int kTiles, bool kCompact, bool kNorm>
__device__ __forceinline__ void AccumulateMma(const WeightTile (&tiles)[kTiles],
                                              const ActivationRows<T>& act,
                                              int count, int input_size,
                                              int first, int step,
                                              MmaStage<kTiles>& stage,
                                              float acc[kTiles][4],
                                              const float* norm_scale,
                                              float* squares) {
  constexpr bool kSplit = kSplitActivations<T, kNorm>;
  const int lane = threadIdx.x % kWarpSize;
  const int g = lane / 4;
  const int c = lane % 4;
  const int blocks = input_size / kBlock;
  const T* row = g < count ? act.Row(g) : nullptr;
  float row_max[kTiles][2] = {};
  if constexpr (kCompact) {
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (g + 8 * h < tiles[t].valid_rows) {
          row_max[t][h] = __half2float(
              __ldg(tiles[t].scale_max + (g + 8 * h) * tiles[t].max_stride));
        }
      }
    }
  }
  // Lane (g, c)'s block of a group starting at b0: its weights and scale.
  const auto load = [&](int b0, uint4 (&chunk)[kTiles][2],
                        float (&scale)[kTiles][2]) {
    const int b = b0 + c;
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = g + 8 * h;
        chunk[t][h] = make_uint4(0, 0, 0, 0);
        scale[t][h] = 0.0f;
        if (b < blocks && r < tiles[t].valid_rows) {
          chunk[t][h] = __ldg(reinterpret_cast<const uint4*>(
              tiles[t].weights + r * tiles[t].weight_stride + b * (kBlock / 2)));
          scale[t][h] = RowScale<kCompact>(tiles[t], r, b, row_max[t][h]);
        }
      }
    }
  };
  int b0 = first * kMmaUnroll;
  uint4 chunk[kTiles][2];
  float scale[kTiles][2];
  if (b0 < blocks) load(b0, chunk, scale);
  for (; b0 < blocks; b0 += step * kMmaUnroll) {
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        stage.chunks[t][h][g][SwizzledBlock(g, c)] = chunk[t][h];
      }
    }
    // The group's scales of rows g and g + 8, block by block, from the
    // quad's lanes.
    float scales[kMmaUnroll][kTiles][2];
#pragma unroll
    for (int u = 0; u < kMmaUnroll; ++u) {
#pragma unroll
      for (int t = 0; t < kTiles; ++t) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          scales[u][t][h] = __shfl_sync(kFullWarpMask, scale[t][h],
                                        (lane & ~3) | u);
        }
      }
    }
    const int next = b0 + step * kMmaUnroll;
    if (next < blocks) load(next, chunk, scale);
    __syncwarp();
    uint32_t words[kTiles][2][kMmaUnroll];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = lane % 8;
        LoadMatrixX4(words[t][h],
                     &stage.chunks[t][h][m][SwizzledBlock(m, lane / 8)]);
      }
    }
    __syncwarp();
#pragma unroll
    for (int u = 0; u < kMmaUnroll; ++u) {
      const int b = b0 + u;
      if (b >= blocks) break;
      ActivationFragment x = {};
      if (row != nullptr) {
        const T* values = row + b * kBlock + 8 * c;
        if constexpr (!kSplit) {
          const uint4 bits = __ldg(reinterpret_cast<const uint4*>(values));
          x.hi[0] = __byte_perm(bits.x, bits.z, 0x5410);
          x.hi[1] = __byte_perm(bits.x, bits.z, 0x7632);
          x.hi[2] = __byte_perm(bits.y, bits.w, 0x5410);
          x.hi[3] = __byte_perm(bits.y, bits.w, 0x7632);
        } else {
          float2 raw[4];
          LoadPairs<T>(values, raw);
          if constexpr (kNorm) {
            float2 norm[4];
            LoadPairs<float>(norm_scale + b * kBlock + 8 * c, norm);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
              *squares = fmaf(raw[n].x, raw[n].x, *squares);
              *squares = fmaf(raw[n].y, raw[n].y, *squares);
            }
            SplitFragment(raw, norm, x);
          } else {
            SplitFragment(raw, nullptr, x);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < kTiles; ++t) {
        uint32_t lo[4];
        uint32_t hi[4];
        NibblePairs(words[t][0][u], lo);
        NibblePairs(words[t][1][u], hi);
        const uint32_t a0[4] = {lo[0], hi[0], lo[1], hi[1]};
        const uint32_t a1[4] = {lo[2], hi[2], lo[3], hi[3]};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        MmaBf16(d, a0, x.hi[0], x.hi[1]);
        MmaBf16(d, a1, x.hi[2], x.hi[3]);
        if constexpr (kSplit) {
          MmaBf16(d, a0, x.lo[0], x.lo[1]);
          MmaBf16(d, a1, x.lo[2], x.lo[3]);
        }
        acc[t][0] = fmaf(d[0], scales[u][t][0], acc[t][0]);
        acc[t][1] = fmaf(d[1], scales[u][t][0], acc[t][1]);
        acc[t][2] = fmaf(d[2], scales[u][t][1], acc[t][2]);
        acc[t][3] = fmaf(d[3], scales[u][t][1], acc[t][3]);
      }
    }
  }
}

}  // namespace litert::nvidia::int4_block

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_CUH_
