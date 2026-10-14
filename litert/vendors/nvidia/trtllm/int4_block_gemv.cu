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

#include <cstddef>
#include <cstdint>
#include <limits>

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.cuh"
#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

namespace litert::nvidia::int4_block {
namespace {

// One CTA computes kTileRows output channels of every activation row (and
// with a gate, the gate and the up rows of those channels).
template <int Rows, int Threads, int Gate>
__global__ void __launch_bounds__(Threads)
    DenseKernel(const __nv_bfloat16* __restrict__ activation,
                const uint8_t* __restrict__ weights,
                const __half* __restrict__ scales,
                __nv_bfloat16* __restrict__ output, int input_size,
                int output_size) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[Rows * kTileRows * kWarps];
  const int n0 = blockIdx.x * kTileRows;
  if (n0 >= output_size) {
    return;
  }
  const int blocks = input_size / kBlock;
  const size_t row_bytes = static_cast<size_t>(input_size) / 2;
  const int valid = min(kTileRows, output_size - n0);
  const ActivationRows<__nv_bfloat16> act{activation,
                                          static_cast<size_t>(input_size),
                                          nullptr};
  float acc[Rows][kTileRows] = {};
  const WeightTile tile{weights + n0 * row_bytes, row_bytes,
                        scales + static_cast<size_t>(n0) * blocks,
                        static_cast<size_t>(blocks), valid};
  AccumulateTile<__nv_bfloat16, Rows, Threads>(tile, act, input_size, acc);
  ReduceTile<Rows, Threads>(acc, warp_sums);
  float up[Rows][kTileRows] = {};
  if constexpr (Gate != 0) {
    const size_t up_row = static_cast<size_t>(output_size) + n0;
    const WeightTile up_tile{weights + up_row * row_bytes, row_bytes,
                             scales + up_row * blocks,
                             static_cast<size_t>(blocks), valid};
    __syncthreads();  // warp_sums is reused
    AccumulateTile<__nv_bfloat16, Rows, Threads>(up_tile, act, input_size,
                                                 up);
    ReduceTile<Rows, Threads>(up, warp_sums);
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
      for (int i = 0; i < kTileRows; ++i) {
        if (i < valid) {
          float value = acc[r][i];
          if constexpr (Gate != 0) {
            value = Gelu(value, Gate) * up[r][i];
          }
          output[static_cast<size_t>(r) * output_size + n0 + i] =
              __float2bfloat16_rn(value);
        }
      }
    }
  }
}

template <int Rows, int Threads, int Gate>
cudaError_t LaunchDense(const __nv_bfloat16* activation,
                        const uint8_t* weights, const __half* scales,
                        __nv_bfloat16* output, int input_size,
                        int output_size, cudaStream_t stream) {
  const int grid = (output_size + kTileRows - 1) / kTileRows;
  DenseKernel<Rows, Threads, Gate><<<grid, Threads, 0, stream>>>(
      activation, weights, scales, output, input_size, output_size);
  return cudaPeekAtLastError();
}

template <int Rows, int Threads>
cudaError_t LaunchDenseGate(const __nv_bfloat16* activation,
                            const uint8_t* weights, const __half* scales,
                            __nv_bfloat16* output, int input_size,
                            int output_size, int gate, cudaStream_t stream) {
  switch (gate) {
    case 0:
      return LaunchDense<Rows, Threads, 0>(activation, weights, scales, output,
                                           input_size, output_size, stream);
    case 1:
      return LaunchDense<Rows, Threads, 1>(activation, weights, scales, output,
                                           input_size, output_size, stream);
    case 2:
      return LaunchDense<Rows, Threads, 2>(activation, weights, scales, output,
                                           input_size, output_size, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <int Threads>
cudaError_t LaunchDenseRows(const __nv_bfloat16* activation,
                            const uint8_t* weights, const __half* scales,
                            __nv_bfloat16* output, int rows, int input_size,
                            int output_size, int gate, cudaStream_t stream) {
  switch (rows) {
#define LITERT_NVIDIA_INT4_BLOCK_ROWS_CASE(n)                             \
  case n:                                                                 \
    return LaunchDenseGate<n, Threads>(activation, weights, scales, output, \
                                       input_size, output_size, gate,     \
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

// Decode: grid (hidden_dim / kTileRows, rows * top_k). hidden[pair][o] =
// gelu(x . gate_e) * (x . up_e) for the pair's row and expert.
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeHiddenKernel(const __nv_bfloat16* __restrict__ activation,
                    const int32_t* __restrict__ top_indices,
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
  AccumulateTile<__nv_bfloat16, 1, Threads>(
      ExpertTile(arena.gate, static_cast<const __half*>(arena.gate_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim),
      act, shape.model_dim, gate);
  ReduceTile<1, Threads>(gate, warp_sums);
  __syncthreads();
  AccumulateTile<__nv_bfloat16, 1, Threads>(
      ExpertTile(arena.up, static_cast<const __half*>(arena.up_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim),
      act, shape.model_dim, up);
  ReduceTile<1, Threads>(up, warp_sums);
  if (threadIdx.x == 0) {
    float* out = hidden + static_cast<size_t>(pair) * shape.hidden_dim + o0;
    const int valid = min(kTileRows, shape.hidden_dim - o0);
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (i < valid) out[i] = Gelu(gate[0][i], shape.gate) * up[0][i];
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
                      const float* per_expert_scale, char* workspace,
                      __nv_bfloat16* output, cudaStream_t stream) {
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
            activation, top_indices, arena, shape, hidden);
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
  // Gate and up as one gated GEMM over the gathered, scaled rows.
  status = LiteRtNvidiaLaunchGemmScaleRows(activation, row_source, padded,
                                           shape.model_dim, row_scale,
                                           scaled_x, stream);
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

extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4Block(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const int32_t* top_indices,
    const float* top_weights, const float* per_expert_scale, void* workspace,
    void* output, cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!LiteRtNvidiaMoeSupports(shape) || arena == nullptr ||
      arena->gate == nullptr || arena->up == nullptr ||
      arena->down == nullptr || arena->gate_scales == nullptr ||
      arena->up_scales == nullptr || arena->down_scales == nullptr ||
      activation == nullptr || top_indices == nullptr ||
      top_weights == nullptr || per_expert_scale == nullptr ||
      workspace == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  if (shape->model_dim / kBlock <= kWarpSize) {
    return LaunchMoe<32>(*shape, *arena, typed_activation, top_indices,
                         top_weights, per_expert_scale,
                         static_cast<char*>(workspace), typed_output, stream);
  }
  return LaunchMoe<128>(*shape, *arena, typed_activation, top_indices,
                        top_weights, per_expert_scale,
                        static_cast<char*>(workspace), typed_output, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (activation == nullptr || packed_weights == nullptr ||
      scales == nullptr || output == nullptr || rows <= 0 ||
      rows > kLiteRtNvidiaInt4BlockGemvMaxRows || input_size <= 0 ||
      input_size % kBlock != 0 || output_size <= 0 || gate < 0 || gate > 2) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __half*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  // Short inputs (the down projection of an expert: 22 blocks) keep a warp
  // busy; longer ones four.
  if (input_size / kBlock <= kWarpSize) {
    return LaunchDenseRows<32>(typed_activation, packed_weights, typed_scales,
                               typed_output, rows, input_size, output_size,
                               gate, stream);
  }
  return LaunchDenseRows<128>(typed_activation, packed_weights, typed_scales,
                              typed_output, rows, input_size, output_size,
                              gate, stream);
}
