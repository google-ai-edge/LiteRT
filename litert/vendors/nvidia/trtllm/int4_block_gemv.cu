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

// Decode: grid (model_dim / kTileRows, rows). out[r][o] = sum_j w_j *
// (hidden[r][j] . down_e_j[o]).
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeDownKernel(const float* __restrict__ hidden,
                  const int32_t* __restrict__ top_indices,
                  const float* __restrict__ top_weights,
                  const float* __restrict__ per_expert_scale,
                  LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                  __nv_bfloat16* __restrict__ output) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kTileRows * kWarps];
  const int row = blockIdx.y;
  const int o0 = blockIdx.x * kTileRows;
  float out[kTileRows] = {};
  for (int j = 0; j < shape.top_k; ++j) {
    const int pair = row * shape.top_k + j;
    const int expert = top_indices[pair];
    if (expert < 0 || expert >= shape.experts) continue;
    const ActivationRows<float> act{
        hidden + static_cast<size_t>(pair) * shape.hidden_dim,
        static_cast<size_t>(shape.hidden_dim), nullptr};
    float acc[1][kTileRows] = {};
    AccumulateTile<float, 1, Threads>(
        ExpertTile(arena.down, static_cast<const __half*>(arena.down_scales),
                   shape.experts, expert, o0, shape.model_dim,
                   shape.hidden_dim),
        act, shape.hidden_dim, acc);
    if constexpr (kWarps > 1) __syncthreads();
    ReduceTile<1, Threads>(acc, warp_sums);
    if (threadIdx.x == 0) {
      const float weight = top_weights[pair] * per_expert_scale[expert];
#pragma unroll
      for (int i = 0; i < kTileRows; ++i) {
        out[i] = fmaf(weight, acc[0][i], out[i]);
      }
    }
  }
  if (threadIdx.x == 0) {
    const int valid = min(kTileRows, shape.model_dim - o0);
    __nv_bfloat16* dst =
        output + static_cast<size_t>(row) * shape.model_dim + o0;
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (i < valid) dst[i] = __float2bfloat16_rn(out[i]);
    }
  }
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

// Prefill: grid (hidden_dim / kTileRows, experts). An expert's tile of
// gate and up rows against its routed rows, kPairRows at a time.
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeGroupedHiddenKernel(const __nv_bfloat16* __restrict__ activation,
                           const int32_t* __restrict__ offsets,
                           const int32_t* __restrict__ grouped,
                           LiteRtNvidiaMoeArena arena,
                           LiteRtNvidiaMoeShape shape,
                           float* __restrict__ hidden) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kPairRows * kTileRows * kWarps];
  __shared__ int32_t rows_s[kPairRows];
  __shared__ int32_t pairs_s[kPairRows];
  const int expert = blockIdx.y;
  const int o0 = blockIdx.x * kTileRows;
  const int begin = offsets[expert];
  const int count = offsets[expert + 1] - begin;
  if (count == 0) return;
  const WeightTile gate_tile =
      ExpertTile(arena.gate, static_cast<const __half*>(arena.gate_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim);
  const WeightTile up_tile =
      ExpertTile(arena.up, static_cast<const __half*>(arena.up_scales),
                 shape.experts, expert, o0, shape.hidden_dim, shape.model_dim);
  for (int chunk = 0; chunk < count; chunk += kPairRows) {
    const int valid = min(kPairRows, count - chunk);
    __syncthreads();  // rows_s and warp_sums of the previous chunk
    if (threadIdx.x < kPairRows) {
      const int pair = grouped[begin + chunk + min(threadIdx.x, valid - 1)];
      pairs_s[threadIdx.x] = pair;
      rows_s[threadIdx.x] = pair / shape.top_k;
    }
    __syncthreads();
    const ActivationRows<__nv_bfloat16> act{
        activation, static_cast<size_t>(shape.model_dim), rows_s};
    float gate[kPairRows][kTileRows] = {};
    AccumulateTile<__nv_bfloat16, kPairRows, Threads>(gate_tile, act,
                                                     shape.model_dim, gate);
    ReduceTile<kPairRows, Threads>(gate, warp_sums);
    __syncthreads();
    float up[kPairRows][kTileRows] = {};
    AccumulateTile<__nv_bfloat16, kPairRows, Threads>(up_tile, act,
                                                     shape.model_dim, up);
    ReduceTile<kPairRows, Threads>(up, warp_sums);
    if (threadIdx.x == 0) {
      const int out_rows = min(kTileRows, shape.hidden_dim - o0);
#pragma unroll
      for (int k = 0; k < kPairRows; ++k) {
        if (k < valid) {
          float* out =
              hidden + static_cast<size_t>(pairs_s[k]) * shape.hidden_dim + o0;
#pragma unroll
          for (int i = 0; i < kTileRows; ++i) {
            if (i < out_rows) out[i] = Gelu(gate[k][i], shape.gate) * up[k][i];
          }
        }
      }
    }
  }
}

// Prefill: grid (model_dim / kTileRows, experts). An expert's tile of down
// rows against the hidden rows of its pairs; the weighted results add into
// the FP32 accumulator of the pair's row.
template <int Threads>
__global__ void __launch_bounds__(Threads)
    MoeGroupedDownKernel(const float* __restrict__ hidden,
                         const int32_t* __restrict__ offsets,
                         const int32_t* __restrict__ grouped,
                         const float* __restrict__ top_weights,
                         const float* __restrict__ per_expert_scale,
                         LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                         float* __restrict__ accumulator) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kPairRows * kTileRows * kWarps];
  __shared__ int32_t pairs_s[kPairRows];
  const int expert = blockIdx.y;
  const int o0 = blockIdx.x * kTileRows;
  const int begin = offsets[expert];
  const int count = offsets[expert + 1] - begin;
  if (count == 0) return;
  const WeightTile tile =
      ExpertTile(arena.down, static_cast<const __half*>(arena.down_scales),
                 shape.experts, expert, o0, shape.model_dim, shape.hidden_dim);
  const float expert_scale = per_expert_scale[expert];
  for (int chunk = 0; chunk < count; chunk += kPairRows) {
    const int valid = min(kPairRows, count - chunk);
    __syncthreads();
    if (threadIdx.x < kPairRows) {
      pairs_s[threadIdx.x] =
          grouped[begin + chunk + min(threadIdx.x, valid - 1)];
    }
    __syncthreads();
    const ActivationRows<float> act{hidden, static_cast<size_t>(shape.hidden_dim),
                                    pairs_s};
    float acc[kPairRows][kTileRows] = {};
    AccumulateTile<float, kPairRows, Threads>(tile, act, shape.hidden_dim, acc);
    ReduceTile<kPairRows, Threads>(acc, warp_sums);
    if (threadIdx.x == 0) {
      const int out_rows = min(kTileRows, shape.model_dim - o0);
#pragma unroll
      for (int k = 0; k < kPairRows; ++k) {
        if (k < valid) {
          const int pair = pairs_s[k];
          const float weight = top_weights[pair] * expert_scale;
          float* out = accumulator +
                       static_cast<size_t>(pair / shape.top_k) * shape.model_dim +
                       o0;
#pragma unroll
          for (int i = 0; i < kTileRows; ++i) {
            if (i < out_rows) atomicAdd(out + i, weight * acc[k][i]);
          }
        }
      }
    }
  }
}

__global__ void MoeFinishKernel(const float* __restrict__ accumulator,
                                size_t count,
                                __nv_bfloat16* __restrict__ output) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = __float2bfloat16_rn(accumulator[i]);
}

size_t RoundUp256(size_t bytes) { return (bytes + 255) / 256 * 256; }

struct MoeWorkspace {
  size_t hidden = 0;       // f32 [rows * top_k, hidden_dim]
  size_t counts = 0;       // i32 [experts]
  size_t offsets = 0;      // i32 [experts + 1]
  size_t cursors = 0;      // i32 [experts]
  size_t grouped = 0;      // i32 [rows * top_k]
  size_t accumulator = 0;  // f32 [rows, model_dim]
  size_t total = 0;
};

MoeWorkspace LayoutWorkspace(const LiteRtNvidiaMoeShape& shape) {
  MoeWorkspace w;
  const size_t pairs = static_cast<size_t>(shape.rows) * shape.top_k;
  w.hidden = 0;
  w.total = RoundUp256(pairs * shape.hidden_dim * sizeof(float));
  if (shape.rows > kLiteRtNvidiaInt4BlockGemvMaxRows) {
    w.counts = w.total;
    w.total += RoundUp256(static_cast<size_t>(shape.experts) * sizeof(int32_t));
    w.offsets = w.total;
    w.total +=
        RoundUp256(static_cast<size_t>(shape.experts + 1) * sizeof(int32_t));
    w.cursors = w.total;
    w.total += RoundUp256(static_cast<size_t>(shape.experts) * sizeof(int32_t));
    w.grouped = w.total;
    w.total += RoundUp256(pairs * sizeof(int32_t));
    w.accumulator = w.total;
    w.total += RoundUp256(static_cast<size_t>(shape.rows) * shape.model_dim *
                          sizeof(float));
  }
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
    MoeHiddenKernel<Threads>
        <<<dim3(hidden_tiles, shape.rows * shape.top_k), Threads, 0, stream>>>(
            activation, top_indices, arena, shape, hidden);
    cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    MoeDownKernel<kDownThreads>
        <<<dim3(model_tiles, shape.rows), kDownThreads, 0, stream>>>(
            hidden, top_indices, top_weights, per_expert_scale, arena, shape,
            output);
    return cudaPeekAtLastError();
  }
  const int pairs = shape.rows * shape.top_k;
  int32_t* counts = reinterpret_cast<int32_t*>(workspace + w.counts);
  int32_t* offsets = reinterpret_cast<int32_t*>(workspace + w.offsets);
  int32_t* cursors = reinterpret_cast<int32_t*>(workspace + w.cursors);
  int32_t* grouped = reinterpret_cast<int32_t*>(workspace + w.grouped);
  float* accumulator = reinterpret_cast<float*>(workspace + w.accumulator);
  const size_t outputs = static_cast<size_t>(shape.rows) * shape.model_dim;
  cudaError_t status =
      cudaMemsetAsync(counts, 0, shape.experts * sizeof(int32_t), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(accumulator, 0, outputs * sizeof(float), stream);
  if (status != cudaSuccess) return status;
  constexpr int kRouteThreads = 256;
  const int route_blocks = (pairs + kRouteThreads - 1) / kRouteThreads;
  MoeCountKernel<<<route_blocks, kRouteThreads, 0, stream>>>(
      top_indices, pairs, shape.experts, counts);
  MoeScanKernel<<<1, kMaxExperts, 0, stream>>>(counts, shape.experts, offsets,
                                               cursors);
  MoeFillKernel<<<route_blocks, kRouteThreads, 0, stream>>>(
      top_indices, pairs, shape.experts, cursors, grouped);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  MoeGroupedHiddenKernel<Threads>
      <<<dim3(hidden_tiles, shape.experts), Threads, 0, stream>>>(
          activation, offsets, grouped, arena, shape, hidden);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  MoeGroupedDownKernel<kDownThreads>
      <<<dim3(model_tiles, shape.experts), kDownThreads, 0, stream>>>(
          hidden, offsets, grouped, top_weights, per_expert_scale, arena, shape,
          accumulator);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  const int finish_blocks = static_cast<int>((outputs + 255) / 256);
  MoeFinishKernel<<<finish_blocks, 256, 0, stream>>>(accumulator, outputs,
                                                     output);
  return cudaPeekAtLastError();
}

}  // namespace
}  // namespace litert::nvidia::int4_block

extern "C" bool LiteRtNvidiaMoeSupports(const LiteRtNvidiaMoeShape* shape) {
  using namespace litert::nvidia::int4_block;
  return shape != nullptr && shape->rows >= 1 && shape->top_k >= 1 &&
         shape->top_k <= kMaxTopK && shape->experts >= 1 &&
         shape->experts <= kMaxExperts && shape->model_dim >= kBlock &&
         shape->model_dim % kBlock == 0 && shape->hidden_dim >= kBlock &&
         shape->hidden_dim % kBlock == 0 &&
         (shape->gate == 1 || shape->gate == 2) &&
         static_cast<int64_t>(shape->rows) * shape->top_k <=
             std::numeric_limits<int32_t>::max() / 2;
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
