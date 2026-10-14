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

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.cuh"
#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

namespace litert::nvidia::int4_block {
namespace {

// One CTA computes kTileRows output channels of every activation row (and
// with a gate, the gate and the up rows of those channels).
// With `norm_scale` the product includes an RMS norm of the activation
// row: y = (x * rsqrt(mean(x^2) + norm_eps) * norm_scale * norm_multiplier)
// . w.
// TileRows output channels per CTA: eight for wide projections, fewer for
// narrow ones so that enough CTAs are in flight to cover memory latency.
template <int Rows, int Threads, int Gate, int TileRows>
__global__ void __launch_bounds__(Threads)
    DenseKernel(const __nv_bfloat16* __restrict__ activation,
                const uint8_t* __restrict__ weights,
                const __half* __restrict__ scales,
                const float* __restrict__ norm_scale, float norm_eps,
                float norm_multiplier, __nv_bfloat16* __restrict__ output,
                int input_size, int output_size) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[Rows * TileRows * kWarps];
  const int n0 = blockIdx.x * TileRows;
  if (n0 >= output_size) {
    return;
  }
  const int blocks = input_size / kBlock;
  const size_t row_bytes = static_cast<size_t>(input_size) / 2;
  const int valid = min(TileRows, output_size - n0);
  const ActivationRows<__nv_bfloat16> act{activation,
                                          static_cast<size_t>(input_size),
                                          nullptr};
  float acc[Rows][TileRows] = {};
  float squares[Rows] = {};
  const WeightTile tile{weights + n0 * row_bytes, row_bytes,
                        scales + static_cast<size_t>(n0) * blocks,
                        static_cast<size_t>(blocks), valid};
  AccumulateTile<__nv_bfloat16, Rows, Threads, TileRows>(
      tile, act, input_size, acc, norm_scale, squares);
  ReduceTile<Rows, Threads, TileRows>(acc, warp_sums);
  float up[Rows][TileRows] = {};
  if constexpr (Gate != 0) {
    const size_t up_row = static_cast<size_t>(output_size) + n0;
    const WeightTile up_tile{weights + up_row * row_bytes, row_bytes,
                             scales + up_row * blocks,
                             static_cast<size_t>(blocks), valid};
    __syncthreads();  // warp_sums is reused
    // The up projection reads the same normalized activations; its sums of
    // squares are the gate's over again and are dropped.
    float up_squares[Rows] = {};
    AccumulateTile<__nv_bfloat16, Rows, Threads, TileRows>(
        up_tile, act, input_size, up, norm_scale, up_squares);
    ReduceTile<Rows, Threads, TileRows>(up, warp_sums);
  }
  float row_factor[Rows];
#pragma unroll
  for (int r = 0; r < Rows; ++r) row_factor[r] = 1.0f;
  if (norm_scale != nullptr) {
    if constexpr (kWarps > 1) __syncthreads();  // warp_sums is reused
    ReduceRows<Rows, Threads>(squares, warp_sums);
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      row_factor[r] =
          rsqrtf(squares[r] / static_cast<float>(input_size) + norm_eps) *
          norm_multiplier;
    }
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
      for (int i = 0; i < TileRows; ++i) {
        if (i < valid) {
          float value = acc[r][i] * row_factor[r];
          if constexpr (Gate != 0) {
            value = Gelu(value, Gate) * up[r][i] * row_factor[r];
          }
          output[static_cast<size_t>(r) * output_size + n0 + i] =
              __float2bfloat16_rn(value);
        }
      }
    }
  }
}

struct DenseNorm {
  const float* scale = nullptr;
  float eps = 0.0f;
  float multiplier = 1.0f;
};

// Enough CTAs to keep the device's memory system busy.
constexpr int kDenseMinCtas = 1024;

template <int Rows, int Threads, int Gate>
cudaError_t LaunchDense(const __nv_bfloat16* activation,
                        const uint8_t* weights, const __half* scales,
                        const DenseNorm& norm, __nv_bfloat16* output,
                        int input_size, int output_size, cudaStream_t stream) {
  const auto launch = [&](auto tile_rows) {
    constexpr int kRows = decltype(tile_rows)::value;
    const int grid = (output_size + kRows - 1) / kRows;
    DenseKernel<Rows, Threads, Gate, kRows><<<grid, Threads, 0, stream>>>(
        activation, weights, scales, norm.scale, norm.eps, norm.multiplier,
        output, input_size, output_size);
    return cudaPeekAtLastError();
  };
  if (output_size >= kDenseMinCtas * kTileRows) {
    return launch(std::integral_constant<int, kTileRows>());
  }
  if (output_size >= kDenseMinCtas * 4) {
    return launch(std::integral_constant<int, 4>());
  }
  return launch(std::integral_constant<int, 2>());
}

template <int Rows, int Threads>
cudaError_t LaunchDenseGate(const __nv_bfloat16* activation,
                            const uint8_t* weights, const __half* scales,
                            const DenseNorm& norm, __nv_bfloat16* output,
                            int input_size, int output_size, int gate,
                            cudaStream_t stream) {
  switch (gate) {
    case 0:
      return LaunchDense<Rows, Threads, 0>(activation, weights, scales, norm,
                                           output, input_size, output_size,
                                           stream);
    case 1:
      return LaunchDense<Rows, Threads, 1>(activation, weights, scales, norm,
                                           output, input_size, output_size,
                                           stream);
    case 2:
      return LaunchDense<Rows, Threads, 2>(activation, weights, scales, norm,
                                           output, input_size, output_size,
                                           stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <int Threads>
cudaError_t LaunchDenseRows(const __nv_bfloat16* activation,
                            const uint8_t* weights, const __half* scales,
                            const DenseNorm& norm, __nv_bfloat16* output,
                            int rows, int input_size, int output_size,
                            int gate, cudaStream_t stream) {
  switch (rows) {
#define LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(n)                                \
  case n:                                                                    \
    return LaunchDenseGate<n, Threads>(activation, weights, scales, norm,     \
                                       output, input_size, output_size, gate, \
                                       stream);
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(1)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(2)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(3)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(4)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(5)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(6)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(7)
    LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(8)
#undef LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE
    default:
      return cudaErrorInvalidValue;
  }
}
static_assert(kLiteRtNvidiaInt4BlockGemvMaxRows == 8,
              "LaunchDenseRows instantiates rows 1 to 8");

// --- Mixture of experts -----------------------------------------------------

constexpr int kMaxTopK = 32;
constexpr int kMaxExperts = 1024;
constexpr int kPairRows = kTileRows;  // routed (row, slot) pairs per chunk

// The weight tile of expert `expert`'s output rows [o0, o0 + kTileRows) of a
// projection whose rows interleave the experts.
__device__ __forceinline__ WeightTile ExpertTile(const uint8_t* weights,
                                                 const __half* scales,
                                                 int experts, int expert,
                                                 int o0, int out_rows,
                                                 int input_size) {
  const size_t row_bytes = static_cast<size_t>(input_size) / 2;
  const size_t blocks = static_cast<size_t>(input_size) / kBlock;
  const size_t first = static_cast<size_t>(o0) * experts + expert;
  return WeightTile{weights + first * row_bytes, row_bytes * experts,
                    scales + first * blocks, blocks * experts,
                    min(kTileRows, out_rows - o0)};
}

// Routing from the router's logits (BF16 [rows, experts], one launch of the
// normed GEMV or GEMM over the arena's router rows): one warp per row takes
// the softmax, the top_k largest by repeated argmax (ties to the lower
// index) and, with `renormalize`, divides their probabilities by their sum.
constexpr int kMaxRouteDim = 4096;

__global__ void MoeRouteKernel(const __nv_bfloat16* __restrict__ logits,
                               int rows, int experts, int top_k,
                               int renormalize,
                               int32_t* __restrict__ top_indices,
                               float* __restrict__ top_weights) {
  constexpr int kMaxPerLane = kMaxExperts / kWarpSize;  // 32
  const int row = blockIdx.x * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (row >= rows) return;
  const __nv_bfloat16* src = logits + static_cast<size_t>(row) * experts;
  float values[kMaxPerLane];
  float largest = -INFINITY;
#pragma unroll
  for (int i = 0; i < kMaxPerLane; ++i) {
    const int e = lane + i * kWarpSize;
    values[i] = e < experts ? __bfloat162float(src[e]) : -INFINITY;
    largest = fmaxf(largest, values[i]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    largest = fmaxf(largest, __shfl_xor_sync(kFullWarpMask, largest, offset));
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxPerLane; ++i) {
    values[i] = lane + i * kWarpSize < experts ? expf(values[i] - largest) : 0.0f;
    sum += values[i];
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(kFullWarpMask, sum, offset);
  }
  const float inv_sum = 1.0f / sum;
  float chosen_sum = 0.0f;
  for (int j = 0; j < top_k; ++j) {
    // The lane's best remaining expert, then the warp's (ties: lowest index).
    float best = -1.0f;
    int best_e = experts;
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i) {
      const int e = lane + i * kWarpSize;
      if (e < experts && (values[i] > best || (values[i] == best && e < best_e))) {
        best = values[i];
        best_e = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const float other = __shfl_xor_sync(kFullWarpMask, best, offset);
      const int other_e = __shfl_xor_sync(kFullWarpMask, best_e, offset);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    const float probability = best * inv_sum;
    chosen_sum += probability;
    if (lane == 0) {
      top_indices[row * top_k + j] = best_e;
      top_weights[row * top_k + j] = probability;
    }
    // Remove it from the lane that holds it.
    if (best_e % kWarpSize == lane) values[best_e / kWarpSize] = -1.0f;
  }
  if (renormalize && lane == 0) {
    const float inv = chosen_sum > 0.0f ? 1.0f / chosen_sum : 0.0f;
    for (int j = 0; j < top_k; ++j) top_weights[row * top_k + j] *= inv;
  }
}

// Decode: grid (hidden_dim / kTileRows, rows * top_k). hidden[pair][o] =
// gelu(x . gate_e) * (x . up_e) for the pair's row and expert. With
// `norm_scale` the row is RMS-normalized on the way in (the pre-expert norm).
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeHiddenKernel(const __nv_bfloat16* __restrict__ activation,
                    const int32_t* __restrict__ top_indices,
                    const float* __restrict__ norm_scale, float norm_eps,
                    LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                    float* __restrict__ hidden) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kTileRows * kWarps];
  const int pair = blockIdx.y;
  const int row = pair / shape.top_k;
  const int expert = top_indices[pair];
  const int o0 = blockIdx.x * kTileRows;
  if (expert < 0 || expert >= shape.experts) return;
  const ActivationRows<__nv_bfloat16> act{
      activation + static_cast<size_t>(row) * shape.model_dim,
      static_cast<size_t>(shape.model_dim), nullptr};
  float gate[1][kTileRows] = {};
  float up[1][kTileRows] = {};
  float squares[1] = {};
  float up_squares[1] = {};
  AccumulateTile<__nv_bfloat16, 1, Threads>(
      ExpertTile(arena.gate, static_cast<const __half*>(arena.gate_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim),
      act, shape.model_dim, gate, norm_scale, squares);
  ReduceTile<1, Threads>(gate, warp_sums);
  __syncthreads();
  AccumulateTile<__nv_bfloat16, 1, Threads>(
      ExpertTile(arena.up, static_cast<const __half*>(arena.up_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim),
      act, shape.model_dim, up, norm_scale, up_squares);
  ReduceTile<1, Threads>(up, warp_sums);
  float factor = 1.0f;
  if (norm_scale != nullptr) {
    if constexpr (kWarps > 1) __syncthreads();
    ReduceRows<1, Threads>(squares, warp_sums);
    factor = rsqrtf(squares[0] / static_cast<float>(shape.model_dim) + norm_eps);
  }
  if (threadIdx.x == 0) {
    float* out = hidden + static_cast<size_t>(pair) * shape.hidden_dim + o0;
    const int valid = min(kTileRows, shape.hidden_dim - o0);
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (i < valid) {
        out[i] = Gelu(gate[0][i] * factor, shape.gate) * up[0][i] * factor;
      }
    }
  }
}

// Decode: grid (model_dim / kTileRows, rows * top_k). partial[pair][o] =
// w_pair * (hidden[pair] . down_e[o]) for the pair's expert e, so every
// routed expert's tiles run in parallel; MoeCombineKernel sums the pairs.
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeDownKernel(const float* __restrict__ hidden,
                  const int32_t* __restrict__ top_indices,
                  const float* __restrict__ top_weights,
                  const float* __restrict__ per_expert_scale,
                  LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                  float* __restrict__ partial) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kTileRows * kWarps];
  const int pair = blockIdx.y;
  const int o0 = blockIdx.x * kTileRows;
  const int expert = top_indices[pair];
  float* dst = partial + static_cast<size_t>(pair) * shape.model_dim + o0;
  const int valid = min(kTileRows, shape.model_dim - o0);
  if (expert < 0 || expert >= shape.experts) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < valid; ++i) dst[i] = 0.0f;
    }
    return;
  }
  const ActivationRows<float> act{
      hidden + static_cast<size_t>(pair) * shape.hidden_dim,
      static_cast<size_t>(shape.hidden_dim), nullptr};
  float acc[1][kTileRows] = {};
  AccumulateTile<float, 1, Threads>(
      ExpertTile(arena.down, static_cast<const __half*>(arena.down_scales),
                 shape.experts, expert, o0, shape.model_dim, shape.hidden_dim),
      act, shape.hidden_dim, acc);
  ReduceTile<1, Threads>(acc, warp_sums);
  if (threadIdx.x == 0) {
    const float weight = top_weights[pair] * per_expert_scale[expert];
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (i < valid) dst[i] = weight * acc[0][i];
    }
  }
}

// Decode: out[r][o] = sum_j partial[r * top_k + j][o], as BF16.
__global__ void MoeCombineKernel(const float* __restrict__ partial, int top_k,
                                 int model_dim,
                                 __nv_bfloat16* __restrict__ output) {
  const int row = blockIdx.y;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= model_dim) return;
  const float* src = partial + static_cast<size_t>(row) * top_k * model_dim + o;
  float sum = 0.0f;
  for (int j = 0; j < top_k; ++j) {
    sum += src[static_cast<size_t>(j) * model_dim];
  }
  output[static_cast<size_t>(row) * model_dim + o] = __float2bfloat16_rn(sum);
}

// Prefill routing: the (row, slot) pairs grouped by expert.
//   counts[e]      pairs routed to e
//   offsets[e]     exclusive prefix sum of counts (offsets[experts] = total)
//   pairs[offsets[e] + i]  the i-th pair of e, as row * top_k + slot
__global__ void MoeCountKernel(const int32_t* __restrict__ top_indices,
                               int pairs, int experts,
                               int32_t* __restrict__ counts) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int expert = top_indices[pair];
  if (expert >= 0 && expert < experts) atomicAdd(counts + expert, 1);
}

__global__ void __launch_bounds__(kMaxExperts)
    MoeScanKernel(const int32_t* __restrict__ counts, int experts,
                  int32_t* __restrict__ offsets, int32_t* __restrict__ cursors) {
  __shared__ int32_t partial[kMaxExperts];
  const int e = threadIdx.x;
  partial[e] = e < experts ? counts[e] : 0;
  __syncthreads();
  // Hillis-Steele inclusive scan over kMaxExperts entries.
  for (int stride = 1; stride < kMaxExperts; stride *= 2) {
    const int32_t add = e >= stride ? partial[e - stride] : 0;
    __syncthreads();
    partial[e] += add;
    __syncthreads();
  }
  if (e < experts) {
    const int32_t exclusive = partial[e] - counts[e];
    offsets[e] = exclusive;
    cursors[e] = exclusive;
  }
  if (e == 0) offsets[experts] = partial[kMaxExperts - 1];
}

__global__ void MoeFillKernel(const int32_t* __restrict__ top_indices,
                              int pairs, int experts,
                              int32_t* __restrict__ cursors,
                              int32_t* __restrict__ grouped) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int expert = top_indices[pair];
  if (expert < 0 || expert >= experts) return;
  grouped[atomicAdd(cursors + expert, 1)] = pair;
}

// Prefill groups the routed pairs by expert into tiles of kGemmTile rows for
// the tensor-core GEMM (subbyte_gemm.h): expert e owns ceil(count_e /
// kGemmTile) tiles, its pairs' rows first and zero rows after. Tiles past
// the last expert's are unused. Device-side counts decide the tiling, so the
// launches cover the largest possible tiling (MaxPaddedRows) and the unused
// tiles exit at once.
constexpr int kGemmTile = 128;

__device__ __forceinline__ int TilesOf(int count) {
  return (count + kGemmTile - 1) / kGemmTile;
}

// tile_offsets[e] = tiles of the experts before e; tile_offsets[experts] =
// the tiles in use.
__global__ void __launch_bounds__(kMaxExperts)
    MoeTileScanKernel(const int32_t* __restrict__ counts, int experts,
                      int32_t* __restrict__ tile_offsets) {
  __shared__ int32_t partial[kMaxExperts];
  const int e = threadIdx.x;
  partial[e] = e < experts ? TilesOf(counts[e]) : 0;
  __syncthreads();
  for (int stride = 1; stride < kMaxExperts; stride *= 2) {
    const int32_t add = e >= stride ? partial[e - stride] : 0;
    __syncthreads();
    partial[e] += add;
    __syncthreads();
  }
  if (e < experts) tile_offsets[e] = partial[e] - TilesOf(counts[e]);
  if (e == 0) tile_offsets[experts] = partial[kMaxExperts - 1];
}

// grid (experts + 1): block e fills expert e's tiles; the last block marks
// the unused tiles.
//   tile_experts[t]   the expert of tile t, or -1
//   row_source[r]     the token of padded row r, -1 for a zero row, -2 unused
//   row_self[r]       r for a row in use, -2 otherwise (the hidden pre-pass)
//   pair_row[pair]    the padded row of a routed pair
__global__ void MoeTileFillKernel(const int32_t* __restrict__ counts,
                                  const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ grouped,
                                  const int32_t* __restrict__ tile_offsets,
                                  int experts, int top_k, int max_tiles,
                                  int32_t* __restrict__ tile_experts,
                                  int32_t* __restrict__ row_source,
                                  int32_t* __restrict__ row_self,
                                  int32_t* __restrict__ pair_row) {
  const int e = blockIdx.x;
  if (e == experts) {
    const int used = tile_offsets[experts];
    for (int t = used + threadIdx.x; t < max_tiles; t += blockDim.x) {
      tile_experts[t] = -1;
      for (int i = 0; i < kGemmTile; ++i) {
        row_source[t * kGemmTile + i] = -2;
        row_self[t * kGemmTile + i] = -2;
      }
    }
    return;
  }
  const int count = counts[e];
  const int first_tile = tile_offsets[e];
  const int tiles = TilesOf(count);
  for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
    tile_experts[first_tile + t] = e;
  }
  const int first_row = first_tile * kGemmTile;
  for (int i = threadIdx.x; i < tiles * kGemmTile; i += blockDim.x) {
    const int row = first_row + i;
    if (i < count) {
      const int pair = grouped[offsets[e] + i];
      row_source[row] = pair / top_k;
      pair_row[pair] = row;
    } else {
      row_source[row] = -1;
    }
    row_self[row] = row;
  }
}

// out[r][o] = sum_j top_weights[r][j] * per_expert_scale[e_j] *
// down[pair_row[r * top_k + j]][o], as BF16.
__global__ void MoeGatherCombineKernel(const __nv_bfloat16* __restrict__ down,
                                       const int32_t* __restrict__ pair_row,
                                       const int32_t* __restrict__ top_indices,
                                       const float* __restrict__ top_weights,
                                       const float* __restrict__ per_expert_scale,
                                       int top_k, int experts, int model_dim,
                                       __nv_bfloat16* __restrict__ output) {
  const int row = blockIdx.y;
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= model_dim) return;
  float sum = 0.0f;
  for (int j = 0; j < top_k; ++j) {
    const int pair = row * top_k + j;
    const int expert = top_indices[pair];
    if (expert < 0 || expert >= experts) continue;
    const float weight = top_weights[pair] * per_expert_scale[expert];
    sum = fmaf(weight,
               __bfloat162float(
                   down[static_cast<size_t>(pair_row[pair]) * model_dim + o]),
               sum);
  }
  output[static_cast<size_t>(row) * model_dim + o] = __float2bfloat16_rn(sum);
}

// The most padded rows a tiling can take: every pair plus up to kGemmTile - 1
// padding rows per expert, in whole tiles.
int MaxPaddedRows(const LiteRtNvidiaMoeShape& shape) {
  const int64_t pairs = static_cast<int64_t>(shape.rows) * shape.top_k;
  const int64_t tiles =
      (pairs + kGemmTile - 1) / kGemmTile + shape.experts;
  return static_cast<int>(tiles * kGemmTile);
}

size_t RoundUp256(size_t bytes) { return (bytes + 255) / 256 * 256; }

struct MoeWorkspace {
  // Routing done in the launch (LiteRtNvidiaLaunchMoeInt4BlockFused).
  size_t top_indices = 0;  // i32 [rows * top_k]
  size_t top_weights = 0;  // f32 [rows * top_k]
  size_t logits = 0;       // bf16 [rows, experts]
  size_t router_gemm = 0;  // the router GEMM's workspace (prefill)
  // Decode.
  size_t hidden = 0;   // f32 [rows * top_k, hidden_dim]
  size_t partial = 0;  // f32 [rows * top_k, model_dim]
  // Prefill: the routing tables, then the GEMM buffers over the padded rows
  // P = MaxPaddedRows. The down GEMM's output reuses the scaled activations.
  size_t counts = 0;        // i32 [experts]
  size_t offsets = 0;       // i32 [experts + 1]
  size_t cursors = 0;       // i32 [experts]
  size_t grouped = 0;       // i32 [rows * top_k]
  size_t tile_offsets = 0;  // i32 [experts + 1]
  size_t tile_experts = 0;  // i32 [P / 128]
  size_t row_source = 0;    // i32 [P]
  size_t row_self = 0;      // i32 [P]
  size_t pair_row = 0;      // i32 [rows * top_k]
  size_t row_scale = 0;     // f32 [P]
  size_t scaled_x = 0;      // f16 [P, model_dim]; later bf16 down [P, model_dim]
  size_t gemm_hidden = 0;   // bf16 [P, hidden_dim]
  size_t scaled_hidden = 0; // f16 [P, hidden_dim]
  size_t total = 0;
};

MoeWorkspace LayoutWorkspace(const LiteRtNvidiaMoeShape& shape) {
  MoeWorkspace w;
  const size_t pairs = static_cast<size_t>(shape.rows) * shape.top_k;
  auto take = [&w](size_t bytes) {
    const size_t at = w.total;
    w.total += RoundUp256(bytes);
    return at;
  };
  w.top_indices = take(pairs * sizeof(int32_t));
  w.top_weights = take(pairs * sizeof(float));
  w.logits = take(static_cast<size_t>(shape.rows) * shape.experts * sizeof(uint16_t));
  if (shape.rows > kLiteRtNvidiaInt4BlockGemvMaxRows) {
    const LiteRtNvidiaGemmShape router = {shape.rows, shape.model_dim,
                                          shape.experts,
                                          kLiteRtNvidiaGemmGateNone, 32};
    w.router_gemm = take(LiteRtNvidiaSubbyteGemmWorkspaceBytes(&router));
  }
  if (shape.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    w.hidden = take(pairs * shape.hidden_dim * sizeof(float));
    w.partial = take(pairs * shape.model_dim * sizeof(float));
    return w;
  }
  const size_t padded = static_cast<size_t>(MaxPaddedRows(shape));
  w.counts = take(static_cast<size_t>(shape.experts) * sizeof(int32_t));
  w.offsets = take(static_cast<size_t>(shape.experts + 1) * sizeof(int32_t));
  w.cursors = take(static_cast<size_t>(shape.experts) * sizeof(int32_t));
  w.grouped = take(pairs * sizeof(int32_t));
  w.tile_offsets = take(static_cast<size_t>(shape.experts + 1) * sizeof(int32_t));
  w.tile_experts = take(padded / kGemmTile * sizeof(int32_t));
  w.row_source = take(padded * sizeof(int32_t));
  w.row_self = take(padded * sizeof(int32_t));
  w.pair_row = take(pairs * sizeof(int32_t));
  w.row_scale = take(padded * sizeof(float));
  w.scaled_x = take(padded * shape.model_dim * sizeof(uint16_t));
  w.gemm_hidden = take(padded * shape.hidden_dim * sizeof(uint16_t));
  w.scaled_hidden = take(padded * shape.hidden_dim * sizeof(uint16_t));
  return w;
}

template <int Threads>
cudaError_t LaunchMoe(const LiteRtNvidiaMoeShape& shape,
                      const LiteRtNvidiaMoeArena& arena,
                      const __nv_bfloat16* activation,
                      const int32_t* top_indices, const float* top_weights,
                      const float* per_expert_scale, const float* norm_scale,
                      float norm_eps, char* workspace, __nv_bfloat16* output,
                      cudaStream_t stream) {
  const MoeWorkspace w = LayoutWorkspace(shape);
  float* hidden = reinterpret_cast<float*>(workspace + w.hidden);
  const int hidden_tiles = (shape.hidden_dim + kTileRows - 1) / kTileRows;
  const int model_tiles = (shape.model_dim + kTileRows - 1) / kTileRows;
  // The down projection's input is the short hidden dim.
  constexpr int kDownThreads = 32;
  if (shape.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    float* partial = reinterpret_cast<float*>(workspace + w.partial);
    MoeHiddenKernel<Threads>
        <<<dim3(hidden_tiles, shape.rows * shape.top_k), Threads, 0, stream>>>(
            activation, top_indices, norm_scale, norm_eps, arena, shape,
            hidden);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    MoeDownKernel<kDownThreads>
        <<<dim3(model_tiles, shape.rows * shape.top_k), kDownThreads, 0,
           stream>>>(hidden, top_indices, top_weights, per_expert_scale, arena,
                     shape, partial);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    constexpr int kCombineThreads = 256;
    MoeCombineKernel<<<dim3((shape.model_dim + kCombineThreads - 1) /
                                kCombineThreads,
                            shape.rows),
                       kCombineThreads, 0, stream>>>(partial, shape.top_k,
                                                     shape.model_dim, output);
    return cudaPeekAtLastError();
  }
  const int pairs = shape.rows * shape.top_k;
  const int padded = MaxPaddedRows(shape);
  const int max_tiles = padded / kGemmTile;
  int32_t* counts = reinterpret_cast<int32_t*>(workspace + w.counts);
  int32_t* offsets = reinterpret_cast<int32_t*>(workspace + w.offsets);
  int32_t* cursors = reinterpret_cast<int32_t*>(workspace + w.cursors);
  int32_t* grouped = reinterpret_cast<int32_t*>(workspace + w.grouped);
  int32_t* tile_offsets = reinterpret_cast<int32_t*>(workspace + w.tile_offsets);
  int32_t* tile_experts = reinterpret_cast<int32_t*>(workspace + w.tile_experts);
  int32_t* row_source = reinterpret_cast<int32_t*>(workspace + w.row_source);
  int32_t* row_self = reinterpret_cast<int32_t*>(workspace + w.row_self);
  int32_t* pair_row = reinterpret_cast<int32_t*>(workspace + w.pair_row);
  float* row_scale = reinterpret_cast<float*>(workspace + w.row_scale);
  void* scaled_x = workspace + w.scaled_x;
  auto* gemm_hidden = reinterpret_cast<__nv_bfloat16*>(workspace + w.gemm_hidden);
  void* scaled_hidden = workspace + w.scaled_hidden;
  auto* down = reinterpret_cast<__nv_bfloat16*>(scaled_x);  // after GEMM 1
  cudaError_t status =
      cudaMemsetAsync(counts, 0, shape.experts * sizeof(int32_t), stream);
  if (status != cudaSuccess) return status;
  constexpr int kRouteThreads = 256;
  const int route_blocks = (pairs + kRouteThreads - 1) / kRouteThreads;
  MoeCountKernel<<<route_blocks, kRouteThreads, 0, stream>>>(
      top_indices, pairs, shape.experts, counts);
  MoeScanKernel<<<1, kMaxExperts, 0, stream>>>(counts, shape.experts, offsets,
                                               cursors);
  MoeFillKernel<<<route_blocks, kRouteThreads, 0, stream>>>(
      top_indices, pairs, shape.experts, cursors, grouped);
  MoeTileScanKernel<<<1, kMaxExperts, 0, stream>>>(counts, shape.experts,
                                                   tile_offsets);
  MoeTileFillKernel<<<shape.experts + 1, kGemmTile, 0, stream>>>(
      counts, offsets, grouped, tile_offsets, shape.experts, shape.top_k,
      max_tiles, tile_experts, row_source, row_self, pair_row);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  // Gate and up as one gated GEMM over the gathered, scaled (and normed)
  // rows.
  status = LiteRtNvidiaLaunchGemmScaleRowsNormed(
      activation, row_source, padded, shape.model_dim, norm_scale, norm_eps,
      1.0f, row_scale, scaled_x, stream);
  if (status != cudaSuccess) return status;
  const uint64_t model_row_bytes = static_cast<uint64_t>(shape.model_dim) / 2;
  const uint64_t model_scales = static_cast<uint64_t>(shape.model_dim) / kBlock;
  const LiteRtNvidiaGroupedGemmWeights gate_up = {
      arena.gate,
      arena.up,
      arena.gate_scales,
      arena.up_scales,
      model_row_bytes * shape.experts,
      model_scales * shape.experts,
      model_row_bytes,
      model_scales,
      tile_experts};
  const LiteRtNvidiaGemmShape gate_shape = {padded, shape.model_dim,
                                            shape.hidden_dim, shape.gate, 32};
  status = LiteRtNvidiaLaunchBf16Int4GroupedGemm(
      &gate_shape, &gate_up, scaled_x, row_scale, gemm_hidden, stream);
  if (status != cudaSuccess) return status;
  // The down projection over the hidden rows (zero rows stay zero).
  status = LiteRtNvidiaLaunchGemmScaleRows(gemm_hidden, row_self, padded,
                                           shape.hidden_dim, row_scale,
                                           scaled_hidden, stream);
  if (status != cudaSuccess) return status;
  const uint64_t hidden_row_bytes = static_cast<uint64_t>(shape.hidden_dim) / 2;
  const uint64_t hidden_scales = static_cast<uint64_t>(shape.hidden_dim) / kBlock;
  const LiteRtNvidiaGroupedGemmWeights down_weights = {
      arena.down,
      nullptr,
      arena.down_scales,
      nullptr,
      hidden_row_bytes * shape.experts,
      hidden_scales * shape.experts,
      hidden_row_bytes,
      hidden_scales,
      tile_experts};
  const LiteRtNvidiaGemmShape down_shape = {
      padded, shape.hidden_dim, shape.model_dim, kLiteRtNvidiaGemmGateNone, 32};
  status = LiteRtNvidiaLaunchBf16Int4GroupedGemm(
      &down_shape, &down_weights, scaled_hidden, row_scale, down, stream);
  if (status != cudaSuccess) return status;
  constexpr int kCombineThreads = 256;
  MoeGatherCombineKernel<<<dim3((shape.model_dim + kCombineThreads - 1) /
                                    kCombineThreads,
                                shape.rows),
                           kCombineThreads, 0, stream>>>(
      down, pair_row, top_indices, top_weights, per_expert_scale, shape.top_k,
      shape.experts, shape.model_dim, output);
  return cudaPeekAtLastError();
}

}  // namespace
}  // namespace litert::nvidia::int4_block

extern "C" bool LiteRtNvidiaMoeSupports(const LiteRtNvidiaMoeShape* shape) {
  using namespace litert::nvidia::int4_block;
  if (shape == nullptr || shape->rows < 1 || shape->top_k < 1 ||
      shape->top_k > kMaxTopK || shape->experts < 1 ||
      shape->experts > kMaxExperts || shape->model_dim < kBlock ||
      shape->model_dim % kBlock != 0 || shape->hidden_dim < kBlock ||
      shape->hidden_dim % kBlock != 0 ||
      (shape->gate != 1 && shape->gate != 2) ||
      static_cast<int64_t>(shape->rows) * shape->top_k >
          std::numeric_limits<int32_t>::max() / 4) {
    return false;
  }
  if (shape->rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    return true;
  }
  // Prefill runs the tensor-core GEMM: its input sizes are multiples of 64
  // and its outputs even.
  const int padded = MaxPaddedRows(*shape);
  const LiteRtNvidiaGemmShape gate_shape = {padded, shape->model_dim,
                                            shape->hidden_dim, shape->gate, 32};
  const LiteRtNvidiaGemmShape down_shape = {
      padded, shape->hidden_dim, shape->model_dim, kLiteRtNvidiaGemmGateNone, 32};
  return LiteRtNvidiaSubbyteGemmSupports(&gate_shape) &&
         LiteRtNvidiaSubbyteGemmSupports(&down_shape) &&
         LiteRtNvidiaSubbyteGemmAvailable();
}

extern "C" size_t LiteRtNvidiaMoeWorkspaceBytes(
    const LiteRtNvidiaMoeShape* shape) {
  if (!LiteRtNvidiaMoeSupports(shape)) return 0;
  return litert::nvidia::int4_block::LayoutWorkspace(*shape).total;
}

namespace {

cudaError_t LaunchMoeChecked(const LiteRtNvidiaMoeShape* shape,
                             const LiteRtNvidiaMoeArena* arena,
                             const void* activation,
                             const int32_t* top_indices,
                             const float* top_weights,
                             const float* per_expert_scale,
                             const float* norm_scale, float norm_eps,
                             void* workspace, void* output,
                             cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!LiteRtNvidiaMoeSupports(shape) || arena == nullptr ||
      arena->gate == nullptr || arena->up == nullptr ||
      arena->down == nullptr || arena->gate_scales == nullptr ||
      arena->up_scales == nullptr || arena->down_scales == nullptr ||
      activation == nullptr || top_indices == nullptr ||
      top_weights == nullptr || per_expert_scale == nullptr ||
      workspace == nullptr || output == nullptr ||
      (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  if (shape->model_dim / kBlock <= kWarpSize) {
    return LaunchMoe<32>(*shape, *arena, typed_activation, top_indices,
                         top_weights, per_expert_scale, norm_scale, norm_eps,
                         static_cast<char*>(workspace), typed_output, stream);
  }
  return LaunchMoe<128>(*shape, *arena, typed_activation, top_indices,
                        top_weights, per_expert_scale, norm_scale, norm_eps,
                        static_cast<char*>(workspace), typed_output, stream);
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4Block(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const int32_t* top_indices,
    const float* top_weights, const float* per_expert_scale, void* workspace,
    void* output, cudaStream_t stream) {
  return LaunchMoeChecked(shape, arena, activation, top_indices, top_weights,
                          per_expert_scale, nullptr, 0.0f, workspace, output,
                          stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4BlockFused(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const float* per_expert_scale,
    const float* router_norm_scale, float router_multiplier,
    const float* expert_norm_scale, float norm_eps, int renormalize,
    void* workspace, void* output, cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!LiteRtNvidiaMoeSupports(shape) || arena == nullptr ||
      arena->router == nullptr || arena->router_scales == nullptr ||
      activation == nullptr || router_norm_scale == nullptr ||
      expert_norm_scale == nullptr || norm_eps < 0.0f || workspace == nullptr ||
      shape->model_dim > kMaxRouteDim || shape->experts % 2 != 0) {
    return cudaErrorInvalidValue;
  }
  const MoeWorkspace w = LayoutWorkspace(*shape);
  auto* bytes = static_cast<char*>(workspace);
  auto* top_indices = reinterpret_cast<int32_t*>(bytes + w.top_indices);
  auto* top_weights = reinterpret_cast<float*>(bytes + w.top_weights);
  auto* logits = reinterpret_cast<__nv_bfloat16*>(bytes + w.logits);
  // The router's logits: the normed GEMV over a few rows, the normed GEMM
  // over many (prefill rows are a multiple of 128).
  cudaError_t status;
  if (shape->rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    status = LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
        activation, arena->router, arena->router_scales, router_norm_scale,
        norm_eps, router_multiplier, logits, shape->rows, shape->model_dim,
        shape->experts, /*gate=*/0, stream);
  } else {
    const LiteRtNvidiaGemmShape router = {shape->rows, shape->model_dim,
                                          shape->experts,
                                          kLiteRtNvidiaGemmGateNone, 32};
    if (!LiteRtNvidiaSubbyteGemmSupports(&router)) {
      return cudaErrorInvalidValue;
    }
    status = LiteRtNvidiaLaunchBf16Int4GemmNormed(
        &router, activation, router_norm_scale, norm_eps, router_multiplier,
        arena->router, arena->router_scales, logits, bytes + w.router_gemm,
        stream);
  }
  if (status != cudaSuccess) return status;
  constexpr int kRouteWarps = 4;
  const int route_blocks = (shape->rows + kRouteWarps - 1) / kRouteWarps;
  MoeRouteKernel<<<route_blocks, kRouteWarps * kWarpSize, 0, stream>>>(
      logits, shape->rows, shape->experts, shape->top_k, renormalize,
      top_indices, top_weights);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  return LaunchMoeChecked(shape, arena, activation, top_indices, top_weights,
                          per_expert_scale, expert_norm_scale, norm_eps,
                          workspace, output, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream) {
  return LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
      activation, packed_weights, scales, nullptr, 0.0f, 1.0f, output, rows,
      input_size, output_size, gate, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    const float* norm_scale, float norm_eps, float norm_multiplier,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (activation == nullptr || packed_weights == nullptr ||
      scales == nullptr || output == nullptr || rows <= 0 ||
      rows > kLiteRtNvidiaInt4BlockGemvMaxRows || input_size <= 0 ||
      input_size % kBlock != 0 || output_size <= 0 || gate < 0 || gate > 2 ||
      (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __half*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  const DenseNorm norm = {norm_scale, norm_eps, norm_multiplier};
  // As many warps as the input blocks fill, up to four: 22 blocks (an
  // expert's down projection) one warp, 66 (the dense down projection)
  // three, 88 and more four.
  const int blocks = input_size / kBlock;
  if (blocks <= 32) {
    return LaunchDenseRows<32>(typed_activation, packed_weights, typed_scales,
                               norm, typed_output, rows, input_size,
                               output_size, gate, stream);
  }
  if (blocks <= 64) {
    return LaunchDenseRows<64>(typed_activation, packed_weights, typed_scales,
                               norm, typed_output, rows, input_size,
                               output_size, gate, stream);
  }
  if (blocks <= 96) {
    return LaunchDenseRows<96>(typed_activation, packed_weights, typed_scales,
                               norm, typed_output, rows, input_size,
                               output_size, gate, stream);
  }
  return LaunchDenseRows<128>(typed_activation, packed_weights, typed_scales,
                              norm, typed_output, rows, input_size,
                              output_size, gate, stream);
}
