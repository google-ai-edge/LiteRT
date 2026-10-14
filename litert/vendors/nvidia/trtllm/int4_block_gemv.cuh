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

#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"

#include <cstddef>
#include <cstdint>

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

// Accumulates, for every activation row r and tile row i, the dot products
// over the input blocks this thread owns (blockIdx-independent: block b for
// b = threadIdx.x, threadIdx.x + Threads, ...) into acc[r][i]. With
// `norm_scale` (an RMS norm fused into the product), activation k is
// multiplied by norm_scale[k] and the thread's sum of squares of the raw
// activations of each row goes to squares[r]; the caller divides by the
// row's RMS after the reduction.
template <typename T, int Rows, int Threads, int TileRows = kTileRows,
          bool kCompact = false>
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

// Gate activations of a gated feed-forward: gelu(gate) * up, with the tanh
// approximation (1) or the exact erf form (2) of GELU.
__device__ __forceinline__ float Gelu(float x, int gate) {
  if (gate == 1) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
  }
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

}  // namespace litert::nvidia::int4_block

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_CUH_
