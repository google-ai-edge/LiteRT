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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
// Launched, when LITERT_NVIDIA_GEMV_DEPENDENT_LAUNCH asks for it, as a
// programmatic dependent of the kernel before it (LaunchDense): a CTA lets
// the next kernel launch as soon as it has started (trigger 0) or once its
// output is written (1), so the next kernel's blocks fill the multiprocessors
// this one's last wave leaves idle, and it asks for its first weights before
// it waits for the activations. The early trigger is for a stream run
// eagerly: in a captured CUDA graph the kernel after this one starts at the
// trigger and its own wait does not hold it until this one completes (the
// RTX Spark, CUDA 13.4), so a capture gets the late one. Off by default: in
// TensorRT-RTX's captured steps the launch measured no faster with the early
// trigger and 1.5% slower with the late one (Qwen 3.5 27B, 2026-11-04).
template <int Rows, int Threads, int Gate, int TileRows>
__global__ void __launch_bounds__(Threads)
    DenseKernel(const __nv_bfloat16* __restrict__ activation,
                const uint8_t* __restrict__ weights,
                const __half* __restrict__ scales,
                const float* __restrict__ norm_scale, float norm_eps,
                float norm_multiplier, __nv_bfloat16* __restrict__ output,
                int input_size, int output_size, int trigger) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[Rows * TileRows * kWarps];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (trigger == 0) cudaTriggerProgrammaticLaunchCompletion();
#endif
  const int n0 = blockIdx.x * TileRows;
  if (n0 >= output_size) {
    return;  // an exited block counts as triggered
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
  AccumulateTile<__nv_bfloat16, Rows, Threads, TileRows, /*kCompact=*/false,
                 /*kDependentLaunch=*/true>(tile, act, input_size, acc,
                                            norm_scale, squares);
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (trigger != 0) {
    // The block's output is in before the next kernel may launch.
    if (threadIdx.x == 0) __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

struct DenseNorm {
  const float* scale = nullptr;
  float eps = 0.0f;
  float multiplier = 1.0f;
};

// Enough CTAs to keep the device's memory system busy.
constexpr int kDenseMinCtas = 1024;

// LITERT_NVIDIA_GEMV_DEPENDENT_LAUNCH: unset or 0, the dense kernels are
// launched in plain stream order (the default: in TensorRT-RTX's captured
// steps the dependent launch measured no faster with the early trigger and
// 1.5% slower with the late one, Qwen 3.5 27B, 2026-11-04); 1 or `auto`,
// they are launched as programmatic dependents of the kernel before them,
// and a block lets the kernel after it launch as soon as it has started when
// the stream runs eagerly, or once its output is written when the stream is
// being captured into a CUDA graph (where the next kernel starts at the
// trigger and its wait does not hold it: Gemma 4 26B's decode went
// non-finite under TensorRT-RTX's whole-graph capture with the early
// trigger); `early` and `late` force one trigger (diagnostics).
enum class DependentLaunch { kOff, kAuto, kEarly, kLate };

DependentLaunch DependentLaunchMode() {
  static const DependentLaunch mode = [] {
    const char* value = std::getenv("LITERT_NVIDIA_GEMV_DEPENDENT_LAUNCH");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
      return DependentLaunch::kOff;
    }
    if (std::strcmp(value, "early") == 0) return DependentLaunch::kEarly;
    if (std::strcmp(value, "late") == 0) return DependentLaunch::kLate;
    return DependentLaunch::kAuto;
  }();
  return mode;
}

// Whether the stream is being captured into a CUDA graph (a stream joined to
// a capture through an event counts).
bool StreamCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
         status != cudaStreamCaptureStatusNone;
}

template <int Rows, int Threads, int Gate>
cudaError_t LaunchDense(const __nv_bfloat16* activation,
                        const uint8_t* weights, const __half* scales,
                        const DenseNorm& norm, __nv_bfloat16* output,
                        int input_size, int output_size, cudaStream_t stream) {
  const auto launch = [&](auto tile_rows) {
    constexpr int kRows = decltype(tile_rows)::value;
    const int grid = (output_size + kRows - 1) / kRows;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid);
    config.blockDim = dim3(Threads);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    const DependentLaunch mode = DependentLaunchMode();
    if (mode != DependentLaunch::kOff) {
      config.attrs = &attribute;
      config.numAttrs = 1;
    }
    const int trigger =
        mode == DependentLaunch::kLate ||
                (mode == DependentLaunch::kAuto && StreamCapturing(stream))
            ? 1
            : 0;
    return cudaLaunchKernelEx(&config, DenseKernel<Rows, Threads, Gate, kRows>,
                              activation, weights, scales, norm.scale, norm.eps,
                              norm.multiplier, output, input_size, output_size,
                              trigger);
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
    case 3:
      return LaunchDense<Rows, Threads, 3>(activation, weights, scales, norm,
                                           output, input_size, output_size,
                                           stream);
    default:
      return cudaErrorInvalidValue;
  }
}

// Rows 2 to kLiteRtNvidiaInt4BlockGemvMaxRows (a verify step) on the tensor
// cores (AccumulateMma): a CTA computes kMmaTile output channels of every
// row (with a gate, those of the gate and of the up projection), its
// kMmaWarps warps taking every kMmaWarps-th input block, and sums the warps'
// results in a fixed order. With a gated feed-forward or the fused norm the
// epilogue is DenseKernel's.
constexpr int kMmaWarps = 4;
constexpr int kMmaThreads = kMmaWarps * kWarpSize;
static_assert(kMmaThreads == kMmaTile * kMmaRows,
              "a thread of the epilogue per output of a tile");
static_assert(kLiteRtNvidiaInt4BlockGemvMaxRows <= kMmaRows,
              "a verify step's rows fit the MMA's columns");

// Stores a warp's accumulators of kTiles tiles (AccumulateMma's layout) to
// sums[warp][t][channel][row] and, with `norm`, the warp's sums of squares of
// its rows to squares_s[warp][row].
template <int kTiles>
__device__ __forceinline__ void StoreWarpSums(
    const float acc[kTiles][4], float squares, bool norm,
    float (*sums)[kTiles][kMmaTile][kMmaRows], float (*squares_s)[kMmaRows]) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int g = lane / 4;
  const int c = lane % 4;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    sums[warp][t][g][2 * c] = acc[t][0];
    sums[warp][t][g][2 * c + 1] = acc[t][1];
    sums[warp][t][g + 8][2 * c] = acc[t][2];
    sums[warp][t][g + 8][2 * c + 1] = acc[t][3];
  }
  if (norm) {
    squares += __shfl_xor_sync(kFullWarpMask, squares, 1);
    squares += __shfl_xor_sync(kFullWarpMask, squares, 2);
    if (c == 0) squares_s[warp][g] = squares;
  }
}

template <int Gate, bool kNorm>
__global__ void __launch_bounds__(kMmaThreads)
    DenseMmaKernel(const __nv_bfloat16* __restrict__ activation,
                   const uint8_t* __restrict__ weights,
                   const __half* __restrict__ scales,
                   const float* __restrict__ norm_scale, float norm_eps,
                   float norm_multiplier, __nv_bfloat16* __restrict__ output,
                   int rows, int input_size, int output_size) {
  constexpr int kTiles = Gate != 0 ? 2 : 1;
  __shared__ MmaStage<kTiles> stages[kMmaWarps];
  __shared__ float sums[kMmaWarps][kTiles][kMmaTile][kMmaRows];
  __shared__ float squares_s[kMmaWarps][kMmaRows];
  const int n0 = blockIdx.x * kMmaTile;
  const int blocks = input_size / kBlock;
  const size_t row_bytes = static_cast<size_t>(input_size) / 2;
  const int valid = min(kMmaTile, output_size - n0);
  WeightTile tiles[kTiles];
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    // The up projection's rows follow the gate's.
    const size_t first = static_cast<size_t>(t) * output_size + n0;
    tiles[t] = WeightTile{weights + first * row_bytes, row_bytes,
                          scales + first * blocks,
                          static_cast<size_t>(blocks), valid};
  }
  const ActivationRows<__nv_bfloat16> act{
      activation, static_cast<size_t>(input_size), nullptr};
  float acc[kTiles][4] = {};
  float squares = 0.0f;
  const int warp = threadIdx.x / kWarpSize;
  AccumulateMma<__nv_bfloat16, kTiles, /*kCompact=*/false, kNorm>(
      tiles, act, rows, input_size, warp, kMmaWarps, stages[warp], acc,
      norm_scale, &squares);
  StoreWarpSums<kTiles>(acc, squares, kNorm, sums, squares_s);
  __syncthreads();
  const int channel = threadIdx.x / kMmaRows;
  const int r = threadIdx.x % kMmaRows;
  if (r >= rows || channel >= valid) return;
  float value = 0.0f;
  float up = 0.0f;
  float row_squares = 0.0f;
#pragma unroll
  for (int w = 0; w < kMmaWarps; ++w) {
    value += sums[w][0][channel][r];
    if constexpr (Gate != 0) up += sums[w][1][channel][r];
    if constexpr (kNorm) row_squares += squares_s[w][r];
  }
  float row_factor = 1.0f;
  if constexpr (kNorm) {
    row_factor =
        rsqrtf(row_squares / static_cast<float>(input_size) + norm_eps) *
        norm_multiplier;
  }
  value *= row_factor;
  if constexpr (Gate != 0) {
    value = Gelu(value, Gate) * up * row_factor;
  }
  output[static_cast<size_t>(r) * output_size + n0 + channel] =
      __float2bfloat16_rn(value);
}

template <int Gate>
cudaError_t LaunchDenseMmaGate(const __nv_bfloat16* activation,
                               const uint8_t* weights, const __half* scales,
                               const DenseNorm& norm, __nv_bfloat16* output,
                               int rows, int input_size, int output_size,
                               cudaStream_t stream) {
  const int grid = (output_size + kMmaTile - 1) / kMmaTile;
  if (norm.scale != nullptr) {
    DenseMmaKernel<Gate, true><<<grid, kMmaThreads, 0, stream>>>(
        activation, weights, scales, norm.scale, norm.eps, norm.multiplier,
        output, rows, input_size, output_size);
  } else {
    DenseMmaKernel<Gate, false><<<grid, kMmaThreads, 0, stream>>>(
        activation, weights, scales, nullptr, 0.0f, 1.0f, output, rows,
        input_size, output_size);
  }
  return cudaPeekAtLastError();
}

cudaError_t LaunchDenseMma(const __nv_bfloat16* activation,
                           const uint8_t* weights, const __half* scales,
                           const DenseNorm& norm, __nv_bfloat16* output,
                           int rows, int input_size, int output_size, int gate,
                           cudaStream_t stream) {
  switch (gate) {
    case 0:
      return LaunchDenseMmaGate<0>(activation, weights, scales, norm, output,
                                   rows, input_size, output_size, stream);
    case 1:
      return LaunchDenseMmaGate<1>(activation, weights, scales, norm, output,
                                   rows, input_size, output_size, stream);
    case 2:
      return LaunchDenseMmaGate<2>(activation, weights, scales, norm, output,
                                   rows, input_size, output_size, stream);
    case 3:
      return LaunchDenseMmaGate<3>(activation, weights, scales, norm, output,
                                   rows, input_size, output_size, stream);
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

// The FP32 kernels (DenseKernel) at any row count: every row's sums are the
// ones the row would get alone. The router's logits take them in a verify
// step too, so that its rows choose their experts as decode would.
cudaError_t LaunchDenseFp32(const __nv_bfloat16* activation,
                            const uint8_t* weights, const __half* scales,
                            const DenseNorm& norm, __nv_bfloat16* output,
                            int rows, int input_size, int output_size,
                            int gate, cudaStream_t stream) {
  // As many warps as the input blocks fill, up to four: 22 blocks (an
  // expert's down projection) one warp, 66 (the dense down projection)
  // three, 88 and more four.
  const int blocks = input_size / kBlock;
  if (blocks <= 32) {
    return LaunchDenseRows<32>(activation, weights, scales, norm, output, rows,
                               input_size, output_size, gate, stream);
  }
  if (blocks <= 64) {
    return LaunchDenseRows<64>(activation, weights, scales, norm, output, rows,
                               input_size, output_size, gate, stream);
  }
  if (blocks <= 96) {
    return LaunchDenseRows<96>(activation, weights, scales, norm, output, rows,
                               input_size, output_size, gate, stream);
  }
  return LaunchDenseRows<128>(activation, weights, scales, norm, output, rows,
                              input_size, output_size, gate, stream);
}

// --- Mixture of experts -----------------------------------------------------

constexpr int kMaxTopK = 32;
constexpr int kMaxExperts = 1024;
constexpr int kPairRows = kTileRows;  // routed (row, slot) pairs per chunk

// The weight tile of expert `expert`'s output rows [o0, o0 + tile_rows) of a
// projection whose rows interleave the experts (the model's order) or follow
// each other expert by expert (`expert_major`).
__device__ __forceinline__ WeightTile ExpertTile(const uint8_t* weights,
                                                 const void* scales,
                                                 const void* scale_max,
                                                 int compact, int expert_major,
                                                 int experts, int expert,
                                                 int o0, int out_rows,
                                                 int input_size,
                                                 int tile_rows = kTileRows) {
  const size_t row_bytes = static_cast<size_t>(input_size) / 2;
  const size_t blocks = static_cast<size_t>(input_size) / kBlock;
  // The expert's first row of the tile, and the rows from one to the next.
  const size_t first = expert_major != 0
                           ? static_cast<size_t>(expert) * out_rows + o0
                           : static_cast<size_t>(o0) * experts + expert;
  const size_t step = expert_major != 0 ? 1 : experts;
  WeightTile tile{weights + first * row_bytes, row_bytes * step, nullptr,
                  blocks * step, min(tile_rows, out_rows - o0)};
  if (compact != 0) {
    tile.fractions = static_cast<const uint8_t*>(scales) + first * blocks;
    tile.scale_max = static_cast<const __half*>(scale_max) + first;
    tile.max_stride = step;
  } else {
    tile.scales = static_cast<const __half*>(scales) + first * blocks;
  }
  return tile;
}

// Routing from the router's logits (BF16 [rows, experts], one launch of the
// normed GEMV or GEMM over the arena's router rows): one warp per row takes
// the softmax, the top_k largest by repeated argmax (ties to the lower
// index) and, with `renormalize`, divides their probabilities by their sum.
constexpr int kMaxRouteDim = 4096;

// A row's routing from its logits (BF16 [experts]), by one warp: softmax,
// then the top_k largest by repeated argmax (ties to the lower index). Slot
// `slot`'s expert is returned after slot + 1 rounds; with `write` all top_k
// rounds run and the row's indices and weights (divided by their sum with
// `renormalize`) go to `indices` / `weights`. kPerLane logits per lane, 4
// for up to 128 experts (the per-lane scans are the serial work).
template <int kPerLane>
__device__ __forceinline__ int RouteRow(const __nv_bfloat16* __restrict__ src,
                                        int experts, int top_k, int slot,
                                        int renormalize, bool write, int lane,
                                        int32_t* __restrict__ indices,
                                        float* __restrict__ weights) {
  float values[kPerLane];
  float largest = -INFINITY;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
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
  for (int i = 0; i < kPerLane; ++i) {
    values[i] = lane + i * kWarpSize < experts ? expf(values[i] - largest) : 0.0f;
    sum += values[i];
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(kFullWarpMask, sum, offset);
  }
  const float inv_sum = 1.0f / sum;
  int picks[kMaxTopK];
  float probabilities[kMaxTopK];
  float chosen_sum = 0.0f;
  int chosen = experts;
  const int rounds = write ? top_k : slot + 1;
  for (int j = 0; j < rounds; ++j) {
    float best = -1.0f;
    int best_e = experts;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
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
    if (j == slot) chosen = best_e;
    if (j < kMaxTopK) {
      picks[j] = best_e;
      probabilities[j] = probability;
    }
    if (best_e % kWarpSize == lane) values[best_e / kWarpSize] = -1.0f;
  }
  if (write && lane == 0) {
    const float inv =
        renormalize ? (chosen_sum > 0.0f ? 1.0f / chosen_sum : 0.0f) : 1.0f;
    for (int j = 0; j < top_k; ++j) {
      indices[j] = picks[j];
      weights[j] = probabilities[j] * inv;
    }
  }
  return chosen;
}

// The routing as its own kernel (prefill: the grouped GEMMs need every
// row's experts before anything else), one warp per row.
template <int kPerLane>
__global__ void MoeRouteKernel(const __nv_bfloat16* __restrict__ logits,
                               int rows, int experts, int top_k,
                               int renormalize,
                               int32_t* __restrict__ top_indices,
                               float* __restrict__ top_weights) {
  const int row = blockIdx.x * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (row >= rows) return;
  RouteRow<kPerLane>(logits + static_cast<size_t>(row) * experts, experts,
                     top_k, /*slot=*/0, renormalize, /*write=*/true, lane,
                     top_indices + static_cast<size_t>(row) * top_k,
                     top_weights + static_cast<size_t>(row) * top_k);
}

// Decode: grid (hidden_dim / kTileRows, rows * top_k). hidden[pair][o] =
// gelu(x . gate_e) * (x . up_e) for the pair's row and expert. With
// `norm_scale` the row is RMS-normalized on the way in (the pre-expert norm).
template <int Threads, bool kCompact>
__global__ void __launch_bounds__(Threads)
    MoeHiddenKernel(const __nv_bfloat16* __restrict__ activation,
                    const int32_t* __restrict__ top_indices,
                    const __nv_bfloat16* __restrict__ logits, int renormalize,
                    int32_t* __restrict__ route_indices,
                    float* __restrict__ route_weights,
                    const float* __restrict__ norm_scale, float norm_eps,
                    LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                    float* __restrict__ hidden) {
  constexpr int kWarps = Threads / kWarpSize;
  __shared__ float warp_sums[kTileRows * kWarps];
  __shared__ int routed_expert;
  const int pair = blockIdx.y;
  const int row = pair / shape.top_k;
  const int slot = pair - row * shape.top_k;
  int expert;
  if (logits != nullptr) {
    // The routing folded in (decode): warp 0 takes this slot's expert from
    // the row's logits, and the first block of slot 0 writes the row's whole
    // top-k for the down kernel; the other warps wait.
    if (threadIdx.x < kWarpSize) {
      const bool write = blockIdx.x == 0 && slot == 0;
      const __nv_bfloat16* src =
          logits + static_cast<size_t>(row) * shape.experts;
      int32_t* indices = route_indices + static_cast<size_t>(row) * shape.top_k;
      float* weights = route_weights + static_cast<size_t>(row) * shape.top_k;
      const int chosen =
          shape.experts <= 4 * kWarpSize
              ? RouteRow<4>(src, shape.experts, shape.top_k, slot, renormalize,
                            write, threadIdx.x, indices, weights)
              : RouteRow<kMaxExperts / kWarpSize>(
                    src, shape.experts, shape.top_k, slot, renormalize, write,
                    threadIdx.x, indices, weights);
      if (threadIdx.x == 0) routed_expert = chosen;
    }
    __syncthreads();
    expert = routed_expert;
  } else {
    expert = top_indices[pair];
  }
  const int o0 = blockIdx.x * kTileRows;
  if (expert < 0 || expert >= shape.experts) return;
  const ActivationRows<__nv_bfloat16> act{
      activation + static_cast<size_t>(row) * shape.model_dim,
      static_cast<size_t>(shape.model_dim), nullptr};
  float gate[1][kTileRows] = {};
  float up[1][kTileRows] = {};
  float squares[1] = {};
  float up_squares[1] = {};
  AccumulateTile<__nv_bfloat16, 1, Threads, kTileRows, kCompact>(
      ExpertTile(arena.gate, arena.gate_scales, arena.gate_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 expert, o0, shape.hidden_dim, shape.model_dim),
      act, shape.model_dim, gate, norm_scale, squares);
  ReduceTile<1, Threads>(gate, warp_sums);
  __syncthreads();
  AccumulateTile<__nv_bfloat16, 1, Threads, kTileRows, kCompact>(
      ExpertTile(arena.up, arena.up_scales, arena.up_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 expert, o0, shape.hidden_dim, shape.model_dim),
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

// A few rows (a verify step: 2 to kLiteRtNvidiaInt4BlockGemvMaxRows) grouped
// by expert (LiteRtNvidiaMoeArena::group_rows): a group holds up to
// kGroupMaxRows routed (row, slot) pairs of one expert, in pair order, and
// the groups come in the order of their first pair, empty slots after them.
// The table takes the place of the router's logits in the workspace, which
// nothing reads once the rows are routed. The grouped kernels multiply on the
// tensor cores (a group's rows are the MMA's columns) and read compact scales
// only, the format the plugin loads.
constexpr int kGroupMaxRows = kLiteRtNvidiaInt4BlockGemvMaxRows;
static_assert(kGroupMaxRows <= kMmaRows, "a group's rows fit the MMA");
constexpr int kGroupMaxPairs = kGroupMaxRows * kMaxTopK;
constexpr int kNoExpert = 0xffff;

struct MoeGroup {
  uint16_t expert;  // kNoExpert: a pair routed to no expert
  uint8_t count;    // 0: an empty slot
  uint8_t pairs[kGroupMaxRows];
};

// Eight BF16 values (one 16-byte word) as four float pairs.
__device__ __forceinline__ void WordPairs(uint4 bits, float2 pairs[4]) {
  const uint32_t words[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pairs[i] = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&words[i]));
  }
}

// The grouped launch's prologue. Block 0, a thread per pair: with `logits`
// (the fused launches), warp r routes row r as MoeRouteKernel does into
// route_indices and route_weights, which top_indices then is, and the pairs
// of top_indices are grouped by expert into `groups`. With `norm_scale` (the
// expert norm), block 1 + r splits row r's values times the scale into hi and
// lo [rows, model_dim] (SplitPair) and puts its RMS factor in factors[r], the
// sum of squares taken in the order the grouped hidden kernel took it in
// every group before (AccumulateMma's per lane, then StoreWarpSums' and the
// epilogue's): the kernel then reads the products' operands and factors it
// computed. One launch where the routing, the grouping and the norm took
// three before, and the split done once instead of in every group.
__global__ void __launch_bounds__(kGroupMaxPairs)
    MoeRouteGroupKernel(const __nv_bfloat16* __restrict__ logits,
                        int renormalize, int32_t* route_indices,
                        float* route_weights, const int32_t* top_indices,
                        const __nv_bfloat16* __restrict__ activation,
                        const float* __restrict__ norm_scale, float norm_eps,
                        LiteRtNvidiaMoeShape shape, MoeGroup* groups,
                        __nv_bfloat16* __restrict__ hi,
                        __nv_bfloat16* __restrict__ lo,
                        float* __restrict__ factors) {
  if (blockIdx.x > 0) {
    // The norm comes with the fused launches only: model_dim <= kMaxRouteDim.
    __shared__ uint4 row_s[kMaxRouteDim / 8];
    __shared__ float squares_s[kMmaWarps][4];
    const int r = blockIdx.x - 1;
    const int words = shape.model_dim / 8;
    const uint4* row = reinterpret_cast<const uint4*>(activation) +
                       static_cast<size_t>(r) * words;
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
      row_s[i] = __ldg(row + i);
    }
    __syncthreads();
    if (threadIdx.x < kMmaWarps * 4) {
      // Lane c of warp w of the grouped kernel: elements 8c to 8c + 7 of the
      // blocks of its groups of kMmaUnroll, in order.
      const int w = threadIdx.x / 4;
      const int c = threadIdx.x % 4;
      const int blocks = shape.model_dim / kBlock;
      float squares = 0.0f;
      for (int b0 = w * kMmaUnroll; b0 < blocks;
           b0 += kMmaWarps * kMmaUnroll) {
#pragma unroll
        for (int u = 0; u < kMmaUnroll; ++u) {
          if (b0 + u >= blocks) break;
          float2 raw[4];
          WordPairs(row_s[(b0 + u) * (kBlock / 8) + c], raw);
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            squares = fmaf(raw[n].x, raw[n].x, squares);
            squares = fmaf(raw[n].y, raw[n].y, squares);
          }
        }
      }
      squares_s[w][c] = squares;
    }
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
      float2 raw[4];
      float2 scale[4];
      WordPairs(row_s[i], raw);
      LoadPairs<float>(norm_scale + 8 * i, scale);
      uint32_t high[4];
      uint32_t low[4];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        SplitPair(__fmul_rn(raw[n].x, scale[n].x),
                  __fmul_rn(raw[n].y, scale[n].y), high[n], low[n]);
      }
      const size_t at = static_cast<size_t>(r) * words + i;
      reinterpret_cast<uint4*>(hi)[at] =
          make_uint4(high[0], high[1], high[2], high[3]);
      reinterpret_cast<uint4*>(lo)[at] =
          make_uint4(low[0], low[1], low[2], low[3]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kMmaWarps; ++w) {
        total += (squares_s[w][0] + squares_s[w][1]) +
                 (squares_s[w][2] + squares_s[w][3]);
      }
      factors[r] =
          rsqrtf(total / static_cast<float>(shape.model_dim) + norm_eps);
    }
    return;
  }
  __shared__ int32_t expert_s[kGroupMaxPairs];
  __shared__ int32_t rank_s[kGroupMaxPairs];
  __shared__ int32_t leader_s[kGroupMaxPairs];
  __shared__ int32_t slot_s[kGroupMaxPairs];
  const int pairs = shape.rows * shape.top_k;
  const int experts = shape.experts;
  const int warp = threadIdx.x / kWarpSize;
  if (logits != nullptr && warp < shape.rows) {
    const __nv_bfloat16* src =
        logits + static_cast<size_t>(warp) * shape.experts;
    int32_t* indices =
        route_indices + static_cast<size_t>(warp) * shape.top_k;
    float* weights = route_weights + static_cast<size_t>(warp) * shape.top_k;
    if (shape.experts <= 4 * kWarpSize) {
      RouteRow<4>(src, shape.experts, shape.top_k, /*slot=*/0, renormalize,
                  /*write=*/true, threadIdx.x % kWarpSize, indices, weights);
    } else {
      RouteRow<kMaxExperts / kWarpSize>(
          src, shape.experts, shape.top_k, /*slot=*/0, renormalize,
          /*write=*/true, threadIdx.x % kWarpSize, indices, weights);
    }
  }
  __syncthreads();  // the routing's indices
  const int p = threadIdx.x;
  if (p < pairs) {
    const int expert = top_indices[p];
    expert_s[p] = expert >= 0 && expert < experts ? expert : kNoExpert;
  }
  __syncthreads();
  // The pair's rank among the pairs of its expert: every kGroupMaxRows of
  // them are a group (one row routes to an expert once, so the routing's own
  // groups never split). A pair without an expert is a group of its own.
  if (p < pairs) {
    int rank = 0;
    if (expert_s[p] != kNoExpert) {
      for (int q = 0; q < p; ++q) rank += expert_s[q] == expert_s[p] ? 1 : 0;
    }
    rank_s[p] = rank;
  }
  __syncthreads();
  if (p < pairs) {
    int leader = p;
    if (expert_s[p] != kNoExpert) {
      const int first = rank_s[p] - rank_s[p] % kGroupMaxRows;
      for (int q = 0; q < p; ++q) {
        if (expert_s[q] == expert_s[p] && rank_s[q] == first) {
          leader = q;
          break;
        }
      }
    }
    leader_s[p] = leader;
  }
  __syncthreads();
  if (p == 0) {
    int slot = 0;
    for (int q = 0; q < pairs; ++q) {
      if (leader_s[q] == q) slot_s[q] = slot++;
    }
    for (; slot < pairs; ++slot) groups[slot].count = 0;
  }
  __syncthreads();
  if (p < pairs && leader_s[p] == p) {
    MoeGroup group;
    group.expert = static_cast<uint16_t>(expert_s[p]);
    int count = 0;
    for (int q = p; q < pairs && count < kGroupMaxRows; ++q) {
      if (leader_s[q] == p) group.pairs[count++] = static_cast<uint8_t>(q);
    }
    for (int i = count; i < kGroupMaxRows; ++i) group.pairs[i] = 0;
    group.count = static_cast<uint8_t>(count);
    groups[slot_s[p]] = group;
  }
}

// A group's rows on the tensor cores (AccumulateMma): the expert's tile of
// kMmaTile rows is read once for all of them. Grid (hidden_dim / kMmaTile,
// rows * top_k): block (tile, group slot).
//   hidden[pair][o] = gelu(x_r . gate_e) * (x_r . up_e)
// With kNorm the rows are the prologue's split ones (norm_hi, norm_lo and
// factors, MoeRouteGroupKernel); the hidden values go out as the high and
// low parts the down projection multiplies (hidden_hi, hidden_lo).
template <bool kNorm>
__global__ void __launch_bounds__(kMmaThreads)
    MoeHiddenGroupedKernel(const __nv_bfloat16* __restrict__ activation,
                           const MoeGroup* __restrict__ groups,
                           const __nv_bfloat16* __restrict__ norm_hi,
                           const __nv_bfloat16* __restrict__ norm_lo,
                           const float* __restrict__ factors,
                           LiteRtNvidiaMoeArena arena,
                           LiteRtNvidiaMoeShape shape,
                           __nv_bfloat16* __restrict__ hidden_hi,
                           __nv_bfloat16* __restrict__ hidden_lo) {
  __shared__ MmaStage<2> stages[kMmaWarps];
  __shared__ float sums[kMmaWarps][2][kMmaTile][kMmaRows];
  __shared__ int32_t rows_s[kMmaRows];
  const MoeGroup group = groups[blockIdx.y];
  if (group.count == 0 || group.expert == kNoExpert) return;
  if (threadIdx.x < kMmaRows) {
    rows_s[threadIdx.x] =
        group.pairs[threadIdx.x < group.count ? threadIdx.x : 0] /
        shape.top_k;
  }
  __syncthreads();
  const int o0 = blockIdx.x * kMmaTile;
  const WeightTile tiles[2] = {
      ExpertTile(arena.gate, arena.gate_scales, arena.gate_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 group.expert, o0, shape.hidden_dim, shape.model_dim,
                 kMmaTile),
      ExpertTile(arena.up, arena.up_scales, arena.up_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 group.expert, o0, shape.hidden_dim, shape.model_dim,
                 kMmaTile)};
  float acc[2][4] = {};
  const int warp = threadIdx.x / kWarpSize;
  if constexpr (kNorm) {
    const ActivationRows<__nv_bfloat16> act{
        norm_hi, static_cast<size_t>(shape.model_dim), rows_s, norm_lo};
    AccumulateMma<__nv_bfloat16, 2, /*kCompact=*/true, /*kNorm=*/false,
                  /*kPreSplit=*/true>(tiles, act, group.count,
                                      shape.model_dim, warp, kMmaWarps,
                                      stages[warp], acc, nullptr, nullptr);
  } else {
    const ActivationRows<__nv_bfloat16> act{
        activation, static_cast<size_t>(shape.model_dim), rows_s};
    AccumulateMma<__nv_bfloat16, 2, /*kCompact=*/true, /*kNorm=*/false>(
        tiles, act, group.count, shape.model_dim, warp, kMmaWarps,
        stages[warp], acc, nullptr, nullptr);
  }
  StoreWarpSums<2>(acc, 0.0f, false, sums, nullptr);
  __syncthreads();
  const int channel = threadIdx.x / kMmaRows;
  const int r = threadIdx.x % kMmaRows;
  if (r >= group.count || channel >= tiles[0].valid_rows) return;
  float gate = 0.0f;
  float up = 0.0f;
#pragma unroll
  for (int w = 0; w < kMmaWarps; ++w) {
    gate += sums[w][0][channel][r];
    up += sums[w][1][channel][r];
  }
  const float factor = kNorm ? factors[rows_s[r]] : 1.0f;
  // Rounded before its low part is taken, as the FP32 value it was.
  const float value = __fmul_rn(Gelu(gate * factor, shape.gate) * up, factor);
  const __nv_bfloat16 high = __float2bfloat16_rn(value);
  const size_t at =
      static_cast<size_t>(group.pairs[r]) * shape.hidden_dim + o0 + channel;
  hidden_hi[at] = high;
  hidden_lo[at] = __float2bfloat16_rn(value - __bfloat162float(high));
}

// partial[pair][o] = w_pair * (hidden[pair] . down_e[o]) for the pairs of a
// group, on the tensor cores, from the hidden rows as the hidden kernel split
// them. Grid (model_dim / kMmaTile, rows * top_k): block (tile, group slot).
__global__ void __launch_bounds__(kMmaThreads)
    MoeDownGroupedKernel(const __nv_bfloat16* __restrict__ hidden_hi,
                         const __nv_bfloat16* __restrict__ hidden_lo,
                         const MoeGroup* __restrict__ groups,
                         const float* __restrict__ top_weights,
                         const float* __restrict__ per_expert_scale,
                         LiteRtNvidiaMoeArena arena, LiteRtNvidiaMoeShape shape,
                         float* __restrict__ partial) {
  __shared__ MmaStage<1> stages[kMmaWarps];
  __shared__ float sums[kMmaWarps][1][kMmaTile][kMmaRows];
  __shared__ int32_t pairs_s[kMmaRows];
  const MoeGroup group = groups[blockIdx.y];
  if (group.count == 0) return;
  const int o0 = blockIdx.x * kMmaTile;
  const int valid = min(kMmaTile, shape.model_dim - o0);
  if (group.expert == kNoExpert) {
    // The pair's partial is zero, as the pair's own block writes it.
    float* dst =
        partial + static_cast<size_t>(group.pairs[0]) * shape.model_dim + o0;
    if (threadIdx.x < valid) dst[threadIdx.x] = 0.0f;
    return;
  }
  if (threadIdx.x < kMmaRows) {
    pairs_s[threadIdx.x] =
        group.pairs[threadIdx.x < group.count ? threadIdx.x : 0];
  }
  __syncthreads();
  const WeightTile tiles[1] = {
      ExpertTile(arena.down, arena.down_scales, arena.down_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 group.expert, o0, shape.model_dim, shape.hidden_dim,
                 kMmaTile)};
  const ActivationRows<__nv_bfloat16> act{
      hidden_hi, static_cast<size_t>(shape.hidden_dim), pairs_s, hidden_lo};
  float acc[1][4] = {};
  const int warp = threadIdx.x / kWarpSize;
  AccumulateMma<__nv_bfloat16, 1, /*kCompact=*/true, /*kNorm=*/false,
                /*kPreSplit=*/true>(tiles, act, group.count,
                                    shape.hidden_dim, warp, kMmaWarps,
                                    stages[warp], acc, nullptr, nullptr);
  StoreWarpSums<1>(acc, 0.0f, false, sums, nullptr);
  __syncthreads();
  const int channel = threadIdx.x / kMmaRows;
  const int r = threadIdx.x % kMmaRows;
  if (r >= group.count || channel >= valid) return;
  float value = 0.0f;
#pragma unroll
  for (int w = 0; w < kMmaWarps; ++w) value += sums[w][0][channel][r];
  const int pair = pairs_s[r];
  partial[static_cast<size_t>(pair) * shape.model_dim + o0 + channel] =
      top_weights[pair] * per_expert_scale[group.expert] * value;
}

// Decode: grid (model_dim / kTileRows, rows * top_k). partial[pair][o] =
// w_pair * (hidden[pair] . down_e[o]) for the pair's expert e, so every
// routed expert's tiles run in parallel; MoeCombineKernel sums the pairs.
template <int Threads, bool kCompact>
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
  AccumulateTile<float, 1, Threads, kTileRows, kCompact>(
      ExpertTile(arena.down, arena.down_scales, arena.down_scale_max,
                 arena.compact_scales, arena.expert_major, shape.experts,
                 expert, o0, shape.model_dim, shape.hidden_dim),
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

// The layer block's epilogue, one block per row (model_dim <= kBlockMaxDim):
//   dense' = rms(dense) * dense_post, moe' = rms(moe) * moe_post,
//   out = (x + rms(dense' + moe') * sum_post) * multiplier,
// moe being the row's expert partials summed (decode) or its gathered down
// rows times their weights (prefill, kGather).
constexpr int kBlockThreads = 1024;
constexpr int kBlockMaxDim = 4096;

__device__ __forceinline__ float BlockSum(float value, float* scratch) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  __syncthreads();  // the previous sum's reads of scratch are done
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = lane < kBlockThreads / kWarpSize ? scratch[lane] : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    total += __shfl_xor_sync(kFullWarpMask, total, offset);
  }
  return total;
}

template <bool kGather>
__global__ void __launch_bounds__(kBlockThreads)
    MoeBlockCombineKernel(const __nv_bfloat16* __restrict__ x,
                          const float* __restrict__ partial,
                          const __nv_bfloat16* __restrict__ down,
                          const int32_t* __restrict__ pair_row,
                          const int32_t* __restrict__ top_indices,
                          const float* __restrict__ top_weights,
                          const float* __restrict__ per_expert_scale,
                          const __nv_bfloat16* __restrict__ dense,
                          const float* __restrict__ dense_post,
                          const float* __restrict__ moe_post,
                          const float* __restrict__ sum_post, float eps,
                          float multiplier, int top_k, int experts,
                          int model_dim, __nv_bfloat16* __restrict__ output) {
  constexpr int kPerThread = kBlockMaxDim / kBlockThreads;
  __shared__ float scratch[kWarpSize];
  const int row = blockIdx.x;
  const size_t base = static_cast<size_t>(row) * model_dim;
  float xv[kPerThread], dv[kPerThread], mv[kPerThread];
  float dense_sq = 0.0f;
  float moe_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int o = threadIdx.x + i * kBlockThreads;
    xv[i] = dv[i] = mv[i] = 0.0f;
    if (o >= model_dim) continue;
    xv[i] = __bfloat162float(x[base + o]);
    dv[i] = __bfloat162float(dense[base + o]);
    float m = 0.0f;
    for (int j = 0; j < top_k; ++j) {
      const int pair = row * top_k + j;
      if constexpr (kGather) {
        const int expert = top_indices[pair];
        if (expert < 0 || expert >= experts) continue;
        m = fmaf(top_weights[pair] * per_expert_scale[expert],
                 __bfloat162float(
                     down[static_cast<size_t>(pair_row[pair]) * model_dim + o]),
                 m);
      } else {
        m += partial[static_cast<size_t>(pair) * model_dim + o];
      }
    }
    mv[i] = m;
    dense_sq = fmaf(dv[i], dv[i], dense_sq);
    moe_sq = fmaf(mv[i], mv[i], moe_sq);
  }
  const float inv_dim = 1.0f / static_cast<float>(model_dim);
  const float dense_factor =
      rsqrtf(BlockSum(dense_sq, scratch) * inv_dim + eps);
  const float moe_factor = rsqrtf(BlockSum(moe_sq, scratch) * inv_dim + eps);
  float cv[kPerThread];
  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int o = threadIdx.x + i * kBlockThreads;
    cv[i] = 0.0f;
    if (o >= model_dim) continue;
    cv[i] = dv[i] * dense_factor * dense_post[o] +
            mv[i] * moe_factor * moe_post[o];
    sum_sq = fmaf(cv[i], cv[i], sum_sq);
  }
  const float sum_factor = rsqrtf(BlockSum(sum_sq, scratch) * inv_dim + eps);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int o = threadIdx.x + i * kBlockThreads;
    if (o >= model_dim) continue;
    output[base + o] = __float2bfloat16_rn(
        (xv[i] + cv[i] * sum_factor * sum_post[o]) * multiplier);
  }
}

// What LaunchMoe needs to end a layer block instead of a plain combine.
struct BlockEpilogue {
  const LiteRtNvidiaMoeBlockParams* params;
  const __nv_bfloat16* x;
  const __nv_bfloat16* dense_out;
  float eps;
};

template <bool kGather>
cudaError_t LaunchBlockEpilogue(const LiteRtNvidiaMoeShape& shape,
                                const BlockEpilogue& block,
                                const float* partial,
                                const __nv_bfloat16* down,
                                const int32_t* pair_row,
                                const int32_t* top_indices,
                                const float* top_weights,
                                const float* per_expert_scale,
                                __nv_bfloat16* output, cudaStream_t stream) {
  // The dense side has to be in before the epilogue reads it.
  cudaError_t status = cudaStreamWaitEvent(stream, block.params->join, 0);
  if (status != cudaSuccess) return status;
  MoeBlockCombineKernel<kGather><<<shape.rows, kBlockThreads, 0, stream>>>(
      block.x, partial, down, pair_row, top_indices, top_weights,
      per_expert_scale, block.dense_out, block.params->dense_post_scale,
      block.params->moe_post_scale, block.params->sum_post_scale, block.eps,
      block.params->residual_multiplier, shape.top_k, shape.experts,
      shape.model_dim, output);
  return cudaPeekAtLastError();
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

// The rows of some output channels of a projection from the model's order
// to their places in an expert-major one (LiteRtNvidiaLaunchMoeExpertRows),
// a thread per 16 bytes.
__global__ void MoeExpertRowsKernel(const uint4* __restrict__ model_rows,
                                    uint4* __restrict__ expert_rows,
                                    int first_channel, int experts,
                                    int expert_channels, int row_words,
                                    size_t words) {
  const size_t word =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word >= words) return;
  const size_t row = word / row_words;
  const size_t channel = first_channel + row / experts;
  const size_t expert = row % experts;
  expert_rows[(expert * expert_channels + channel) * row_words +
              word % row_words] = model_rows[word];
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
//                     (also the rows of a GEMM block with none of the
//                     expert's rows, which the GEMMs skip)
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
      row_self[row] = row;
    } else if (i / kLiteRtNvidiaGroupedGemmBlockRows *
                   kLiteRtNvidiaGroupedGemmBlockRows <
               count) {
      row_source[row] = -1;
      row_self[row] = row;
    } else {
      row_source[row] = -2;
      row_self[row] = -2;
    }
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
  // A layer block's dense feed-forward (dense_hidden > 0).
  size_t dense_hidden = 0;  // bf16 [rows, dense_hidden]
  size_t dense_out = 0;     // bf16 [rows, model_dim]
  size_t dense_gemm = 0;    // the dense GEMMs' workspace (prefill)
  size_t total = 0;
};

// The expert pieces first, in the same places with or without the dense
// feed-forward's, which follow.
MoeWorkspace LayoutWorkspace(const LiteRtNvidiaMoeShape& shape,
                             int dense_hidden, int dense_gate);

MoeWorkspace LayoutWorkspace(const LiteRtNvidiaMoeShape& shape) {
  return LayoutWorkspace(shape, 0, 0);
}

MoeWorkspace LayoutWorkspace(const LiteRtNvidiaMoeShape& shape,
                             int dense_hidden, int dense_gate) {
  MoeWorkspace w;
  const size_t pairs = static_cast<size_t>(shape.rows) * shape.top_k;
  auto take = [&w](size_t bytes) {
    const size_t at = w.total;
    w.total += RoundUp256(bytes);
    return at;
  };
  auto take_dense = [&] {
    if (dense_hidden <= 0) return;
    w.dense_hidden = take(static_cast<size_t>(shape.rows) * dense_hidden *
                          sizeof(uint16_t));
    w.dense_out = take(static_cast<size_t>(shape.rows) * shape.model_dim *
                       sizeof(uint16_t));
    if (shape.rows > kLiteRtNvidiaInt4BlockGemvMaxRows) {
      const LiteRtNvidiaGemmShape gate_shape = {
          shape.rows, shape.model_dim, dense_hidden, dense_gate, 32};
      const LiteRtNvidiaGemmShape down_shape = {
          shape.rows, dense_hidden, shape.model_dim,
          kLiteRtNvidiaGemmGateNone, 32};
      w.dense_gemm =
          take(std::max(LiteRtNvidiaSubbyteGemmWorkspaceBytes(&gate_shape),
                        LiteRtNvidiaSubbyteGemmWorkspaceBytes(&down_shape)));
    }
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
    take_dense();
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
  take_dense();
  return w;
}

// Whether the launch groups the rows by expert: the arena asks for it and
// holds compact scales, the rows are a verify step's, and the groups fit
// where the logits were.
bool GroupsRows(const LiteRtNvidiaMoeShape& shape,
                const LiteRtNvidiaMoeArena& arena) {
  return arena.group_rows != 0 && arena.compact_scales != 0 &&
         shape.rows >= 2 &&
         shape.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows &&
         static_cast<size_t>(shape.rows) * shape.top_k * sizeof(MoeGroup) <=
             RoundUp256(static_cast<size_t>(shape.rows) * shape.experts *
                        sizeof(uint16_t));
}

template <int Threads>
cudaError_t LaunchMoe(const LiteRtNvidiaMoeShape& shape,
                      const LiteRtNvidiaMoeArena& arena,
                      const __nv_bfloat16* activation,
                      const int32_t* top_indices, const float* top_weights,
                      const float* per_expert_scale, const float* norm_scale,
                      float norm_eps, char* workspace, __nv_bfloat16* output,
                      cudaStream_t stream,
                      const BlockEpilogue* block = nullptr,
                      const __nv_bfloat16* logits = nullptr,
                      int renormalize = 1) {
  const MoeWorkspace w = LayoutWorkspace(shape);
  float* hidden = reinterpret_cast<float*>(workspace + w.hidden);
  const int hidden_tiles = (shape.hidden_dim + kTileRows - 1) / kTileRows;
  const int model_tiles = (shape.model_dim + kTileRows - 1) / kTileRows;
  // The down projection's input is the short hidden dim.
  constexpr int kDownThreads = 32;
  if (shape.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    float* partial = reinterpret_cast<float*>(workspace + w.partial);
    const dim3 hidden_grid(hidden_tiles, shape.rows * shape.top_k);
    const dim3 down_grid(model_tiles, shape.rows * shape.top_k);
    auto* route_indices = reinterpret_cast<int32_t*>(workspace + w.top_indices);
    auto* route_weights = reinterpret_cast<float*>(workspace + w.top_weights);
    cudaError_t status;
    if (GroupsRows(shape, arena)) {
      // An expert's tile rows are read once for the rows that route to it.
      // The prologue routes the rows (fused launches: from the logits, which
      // the groups then replace) and splits the normed ones.
      auto* groups = reinterpret_cast<MoeGroup*>(workspace + w.logits);
      const int pairs = shape.rows * shape.top_k;
      auto* norm_hi = reinterpret_cast<__nv_bfloat16*>(partial);
      __nv_bfloat16* norm_lo =
          norm_hi + static_cast<size_t>(shape.rows) * shape.model_dim;
      auto* factors = reinterpret_cast<float*>(
          norm_lo + static_cast<size_t>(shape.rows) * shape.model_dim);
      auto* hidden_hi = reinterpret_cast<__nv_bfloat16*>(hidden);
      __nv_bfloat16* hidden_lo =
          hidden_hi + static_cast<size_t>(pairs) * shape.hidden_dim;
      const int prologue_blocks = norm_scale != nullptr ? 1 + shape.rows : 1;
      MoeRouteGroupKernel<<<prologue_blocks, kGroupMaxPairs, 0, stream>>>(
          logits, renormalize, route_indices, route_weights, top_indices,
          activation, norm_scale, norm_eps, shape, groups, norm_hi, norm_lo,
          factors);
      const dim3 hidden_mma_grid((shape.hidden_dim + kMmaTile - 1) / kMmaTile,
                                 pairs);
      const dim3 down_mma_grid((shape.model_dim + kMmaTile - 1) / kMmaTile,
                               pairs);
      if (norm_scale != nullptr) {
        MoeHiddenGroupedKernel<true>
            <<<hidden_mma_grid, kMmaThreads, 0, stream>>>(
                activation, groups, norm_hi, norm_lo, factors, arena, shape,
                hidden_hi, hidden_lo);
      } else {
        MoeHiddenGroupedKernel<false>
            <<<hidden_mma_grid, kMmaThreads, 0, stream>>>(
                activation, groups, nullptr, nullptr, nullptr, arena, shape,
                hidden_hi, hidden_lo);
      }
      MoeDownGroupedKernel<<<down_mma_grid, kMmaThreads, 0, stream>>>(
          hidden_hi, hidden_lo, groups, top_weights, per_expert_scale, arena,
          shape, partial);
      status = cudaPeekAtLastError();
      if (status != cudaSuccess) return status;
    } else {
      if (arena.compact_scales != 0) {
        MoeHiddenKernel<Threads, true><<<hidden_grid, Threads, 0, stream>>>(
            activation, top_indices, logits, renormalize, route_indices,
            route_weights, norm_scale, norm_eps, arena, shape, hidden);
      } else {
        MoeHiddenKernel<Threads, false><<<hidden_grid, Threads, 0, stream>>>(
            activation, top_indices, logits, renormalize, route_indices,
            route_weights, norm_scale, norm_eps, arena, shape, hidden);
      }
      status = cudaPeekAtLastError();
      if (status != cudaSuccess) return status;
      if (arena.compact_scales != 0) {
        MoeDownKernel<kDownThreads, true>
            <<<down_grid, kDownThreads, 0, stream>>>(
                hidden, top_indices, top_weights, per_expert_scale, arena,
                shape, partial);
      } else {
        MoeDownKernel<kDownThreads, false>
            <<<down_grid, kDownThreads, 0, stream>>>(
                hidden, top_indices, top_weights, per_expert_scale, arena,
                shape, partial);
      }
      status = cudaPeekAtLastError();
      if (status != cudaSuccess) return status;
    }
    if (block != nullptr) {
      return LaunchBlockEpilogue<false>(shape, *block, partial, nullptr,
                                        nullptr, top_indices, top_weights,
                                        per_expert_scale, output, stream);
    }
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
  // The rows from a channel of an expert to its next channel, and from an
  // expert to the next: the model's order interleaves the experts.
  const uint64_t channel_rows = arena.expert_major != 0 ? 1 : shape.experts;
  const uint64_t gate_expert_rows =
      arena.expert_major != 0 ? shape.hidden_dim : 1;
  const uint64_t down_expert_rows =
      arena.expert_major != 0 ? shape.model_dim : 1;
  const uint64_t model_row_bytes = static_cast<uint64_t>(shape.model_dim) / 2;
  const uint64_t model_scales = static_cast<uint64_t>(shape.model_dim) / kBlock;
  const LiteRtNvidiaGroupedGemmWeights gate_up = {
      arena.gate,
      arena.up,
      arena.gate_scales,
      arena.up_scales,
      model_row_bytes * channel_rows,
      model_scales * channel_rows,
      model_row_bytes * gate_expert_rows,
      model_scales * gate_expert_rows,
      tile_experts,
      arena.gate_scale_max,
      arena.up_scale_max,
      channel_rows,
      gate_expert_rows,
      arena.compact_scales,
      row_source,
      arena.host_memory};
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
      hidden_row_bytes * channel_rows,
      hidden_scales * channel_rows,
      hidden_row_bytes * down_expert_rows,
      hidden_scales * down_expert_rows,
      tile_experts,
      arena.down_scale_max,
      nullptr,
      channel_rows,
      down_expert_rows,
      arena.compact_scales,
      row_source,
      arena.host_memory};
  const LiteRtNvidiaGemmShape down_shape = {
      padded, shape.hidden_dim, shape.model_dim, kLiteRtNvidiaGemmGateNone, 32};
  status = LiteRtNvidiaLaunchBf16Int4GroupedGemm(
      &down_shape, &down_weights, scaled_hidden, row_scale, down, stream);
  if (status != cudaSuccess) return status;
  if (block != nullptr) {
    return LaunchBlockEpilogue<true>(shape, *block, nullptr, down, pair_row,
                                     top_indices, top_weights,
                                     per_expert_scale, output, stream);
  }
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

extern "C" void LiteRtNvidiaCompactScaleRow(const float* scales,
                                            int32_t blocks,
                                            uint8_t* fractions,
                                            uint16_t* max_half) {
  float largest = 0.0f;
  for (int32_t b = 0; b < blocks; ++b) {
    largest = fmaxf(largest, scales[b]);
  }
  // The maximum as stored (FP16), so the fractions are of the value the
  // kernels multiply by.
  const __half half = __float2half_rn(largest);
  const __half_raw raw = half;
  *max_half = raw.x;
  const float stored = __half2float(half);
  for (int32_t b = 0; b < blocks; ++b) {
    const float fraction =
        stored > 0.0f ? scales[b] / stored * 255.0f + 0.5f : 0.0f;
    const int value = static_cast<int>(fraction);
    fractions[b] = static_cast<uint8_t>(value < 0 ? 0 : value > 255 ? 255 : value);
  }
}

extern "C" cudaError_t LiteRtNvidiaLaunchMoeExpertRows(
    const void* model_rows, void* expert_rows, int32_t first_channel,
    int32_t channels, int32_t experts, int32_t expert_channels,
    int32_t row_bytes, cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (model_rows == nullptr || expert_rows == nullptr || first_channel < 0 ||
      channels < 1 || experts < 1 || channels > expert_channels ||
      first_channel > expert_channels - channels || row_bytes < 16 ||
      row_bytes % 16 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  const size_t words =
      static_cast<size_t>(channels) * experts * (row_bytes / 16);
  const size_t blocks = (words + kThreads - 1) / kThreads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return cudaErrorInvalidValue;
  }
  MoeExpertRowsKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      static_cast<const uint4*>(model_rows), static_cast<uint4*>(expert_rows),
      first_channel, experts, expert_channels, row_bytes / 16, words);
  return cudaPeekAtLastError();
}

extern "C" bool LiteRtNvidiaMoeSupports(const LiteRtNvidiaMoeShape* shape) {
  using namespace litert::nvidia::int4_block;
  if (shape == nullptr || shape->rows < 1 || shape->top_k < 1 ||
      shape->top_k > kMaxTopK || shape->experts < 1 ||
      shape->experts > kMaxExperts || shape->model_dim < kBlock ||
      shape->model_dim % kBlock != 0 || shape->hidden_dim < kBlock ||
      shape->hidden_dim % kBlock != 0 ||
      (shape->gate != 1 && shape->gate != 2 && shape->gate != 3) ||
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

extern "C" size_t LiteRtNvidiaMoeBlockWorkspaceBytes(
    const LiteRtNvidiaMoeShape* shape, int32_t dense_hidden,
    int32_t dense_gate) {
  using namespace litert::nvidia::int4_block;
  if (!LiteRtNvidiaMoeSupports(shape) || dense_hidden < kBlock ||
      dense_hidden % kBlock != 0 || (dense_gate != 1 && dense_gate != 2) ||
      shape->model_dim > kBlockMaxDim) {
    return 0;
  }
  if (shape->rows > kLiteRtNvidiaInt4BlockGemvMaxRows) {
    const LiteRtNvidiaGemmShape gate_shape = {
        shape->rows, shape->model_dim, dense_hidden, dense_gate, 32};
    const LiteRtNvidiaGemmShape down_shape = {
        shape->rows, dense_hidden, shape->model_dim,
        kLiteRtNvidiaGemmGateNone, 32};
    if (!LiteRtNvidiaSubbyteGemmSupports(&gate_shape) ||
        !LiteRtNvidiaSubbyteGemmSupports(&down_shape)) {
      return 0;
    }
  }
  return LayoutWorkspace(*shape, dense_hidden, dense_gate).total;
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
                             cudaStream_t stream,
                             const litert::nvidia::int4_block::BlockEpilogue*
                                 block = nullptr,
                             const __nv_bfloat16* logits = nullptr,
                             int renormalize = 1) {
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
                         static_cast<char*>(workspace), typed_output, stream,
                         block, logits, renormalize);
  }
  // Up to 96 input blocks (the 26B's 88): 96-thread hidden blocks. A thread
  // still takes one input block as with 128, and the fourth warp would only
  // add zeros after the others to sums that start at +0, so the bits are the
  // same. The kernel holds 128 registers a thread: five 96-thread blocks fit
  // a multiprocessor where four of 128 did, and the 26B's 704 blocks of a
  // decode step take 1.7 waves on the 5080 instead of 2.1 (expert launch
  // 39.0 to 37.2 us).
  if (shape->model_dim / kBlock <= 3 * kWarpSize) {
    return LaunchMoe<96>(*shape, *arena, typed_activation, top_indices,
                         top_weights, per_expert_scale, norm_scale, norm_eps,
                         static_cast<char*>(workspace), typed_output, stream,
                         block, logits, renormalize);
  }
  return LaunchMoe<128>(*shape, *arena, typed_activation, top_indices,
                        top_weights, per_expert_scale, norm_scale, norm_eps,
                        static_cast<char*>(workspace), typed_output, stream,
                        block, logits, renormalize);
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

namespace {

// The routing of the fused launches: the router's logits into the
// workspace, then (prefill) top_indices/top_weights at its front; in decode
// the hidden kernel routes from the logits and writes them itself.
cudaError_t LaunchFusedRouting(const LiteRtNvidiaMoeShape* shape,
                               const LiteRtNvidiaMoeArena* arena,
                               const void* activation,
                               const float* router_norm_scale,
                               float router_multiplier, float norm_eps,
                               int renormalize, void* workspace,
                               cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  const MoeWorkspace w = LayoutWorkspace(*shape);
  auto* bytes = static_cast<char*>(workspace);
  auto* top_indices = reinterpret_cast<int32_t*>(bytes + w.top_indices);
  auto* top_weights = reinterpret_cast<float*>(bytes + w.top_weights);
  auto* logits = reinterpret_cast<__nv_bfloat16*>(bytes + w.logits);
  // The router's logits: the normed GEMV over a few rows, the normed GEMM
  // over many (prefill rows are a multiple of 128).
  cudaError_t status;
  if (shape->rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    status = LaunchDenseFp32(
        static_cast<const __nv_bfloat16*>(activation), arena->router,
        static_cast<const __half*>(arena->router_scales),
        DenseNorm{router_norm_scale, norm_eps, router_multiplier}, logits,
        shape->rows, shape->model_dim, shape->experts, /*gate=*/0, stream);
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
  if (shape->rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    // Decode: the hidden kernel routes from the logits itself; rows grouped
    // by expert, the grouped launch's prologue.
    return cudaSuccess;
  }
  constexpr int kRouteWarps = 4;
  const int route_blocks = (shape->rows + kRouteWarps - 1) / kRouteWarps;
  if (shape->experts <= 4 * kWarpSize) {
    MoeRouteKernel<4><<<route_blocks, kRouteWarps * kWarpSize, 0, stream>>>(
        logits, shape->rows, shape->experts, shape->top_k, renormalize,
        top_indices, top_weights);
  } else {
    MoeRouteKernel<kMaxExperts / kWarpSize>
        <<<route_blocks, kRouteWarps * kWarpSize, 0, stream>>>(
            logits, shape->rows, shape->experts, shape->top_k, renormalize,
            top_indices, top_weights);
  }
  status = cudaPeekAtLastError();
  return cudaPeekAtLastError();
}

// The logits a few rows' launch routes from (the hidden kernel in decode,
// the prologue of rows grouped by expert); prefill rows are routed already.
const __nv_bfloat16* DecodeRoutingLogits(const LiteRtNvidiaMoeShape& shape,
                                         const char* logits) {
  using namespace litert::nvidia::int4_block;
  return shape.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows
             ? reinterpret_cast<const __nv_bfloat16*>(logits)
             : nullptr;
}

bool FusedArgumentsValid(const LiteRtNvidiaMoeShape* shape,
                         const LiteRtNvidiaMoeArena* arena,
                         const void* activation,
                         const float* router_norm_scale,
                         const float* expert_norm_scale, float norm_eps,
                         void* workspace) {
  using namespace litert::nvidia::int4_block;
  return LiteRtNvidiaMoeSupports(shape) && arena != nullptr &&
         arena->router != nullptr && arena->router_scales != nullptr &&
         activation != nullptr && router_norm_scale != nullptr &&
         expert_norm_scale != nullptr && norm_eps >= 0.0f &&
         workspace != nullptr && shape->model_dim <= kMaxRouteDim &&
         shape->experts % 2 == 0;
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4BlockFused(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const float* per_expert_scale,
    const float* router_norm_scale, float router_multiplier,
    const float* expert_norm_scale, float norm_eps, int renormalize,
    void* workspace, void* output, cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!FusedArgumentsValid(shape, arena, activation, router_norm_scale,
                           expert_norm_scale, norm_eps, workspace)) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t status = LaunchFusedRouting(
      shape, arena, activation, router_norm_scale, router_multiplier,
      norm_eps, renormalize, workspace, stream);
  if (status != cudaSuccess) return status;
  const MoeWorkspace w = LayoutWorkspace(*shape);
  auto* bytes = static_cast<char*>(workspace);
  return LaunchMoeChecked(
      shape, arena, activation,
      reinterpret_cast<const int32_t*>(bytes + w.top_indices),
      reinterpret_cast<const float*>(bytes + w.top_weights), per_expert_scale,
      expert_norm_scale, norm_eps, workspace, output, stream,
      /*block=*/nullptr, DecodeRoutingLogits(*shape, bytes + w.logits),
      renormalize);
}

extern "C" cudaError_t LiteRtNvidiaLaunchMoeLayerBlock(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const float* per_expert_scale,
    const float* router_norm_scale, float router_multiplier,
    const float* expert_norm_scale, float norm_eps, int renormalize,
    const LiteRtNvidiaMoeBlockParams* block, void* workspace, void* output,
    cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!FusedArgumentsValid(shape, arena, activation, router_norm_scale,
                           expert_norm_scale, norm_eps, workspace) ||
      block == nullptr || block->dense_norm_scale == nullptr ||
      block->dense_post_scale == nullptr || block->moe_post_scale == nullptr ||
      block->sum_post_scale == nullptr || block->side_stream == nullptr ||
      block->fork == nullptr || block->join == nullptr ||
      arena->dense_gate_up == nullptr ||
      arena->dense_gate_up_scales == nullptr ||
      arena->dense_down == nullptr || arena->dense_down_scales == nullptr ||
      LiteRtNvidiaMoeBlockWorkspaceBytes(shape, block->dense_hidden,
                                         block->dense_gate) == 0) {
    return cudaErrorInvalidValue;
  }
  const MoeWorkspace w =
      LayoutWorkspace(*shape, block->dense_hidden, block->dense_gate);
  auto* bytes = static_cast<char*>(workspace);
  auto* dense_hidden = reinterpret_cast<__nv_bfloat16*>(bytes + w.dense_hidden);
  auto* dense_out = reinterpret_cast<__nv_bfloat16*>(bytes + w.dense_out);
  // The dense feed-forward on the side stream, from the fork.
  cudaError_t status = cudaEventRecord(block->fork, stream);
  if (status != cudaSuccess) return status;
  status = cudaStreamWaitEvent(block->side_stream, block->fork, 0);
  if (status != cudaSuccess) return status;
  if (shape->rows <= kLiteRtNvidiaInt4BlockGemvMaxRows) {
    status = LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
        activation, arena->dense_gate_up, arena->dense_gate_up_scales,
        block->dense_norm_scale, norm_eps, 1.0f, dense_hidden, shape->rows,
        shape->model_dim, block->dense_hidden, block->dense_gate,
        block->side_stream);
    if (status == cudaSuccess) {
      status = LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
          dense_hidden, arena->dense_down, arena->dense_down_scales, dense_out,
          shape->rows, block->dense_hidden, shape->model_dim, 0,
          block->side_stream);
    }
  } else {
    const LiteRtNvidiaGemmShape gate_shape = {
        shape->rows, shape->model_dim, block->dense_hidden, block->dense_gate,
        32};
    const LiteRtNvidiaGemmShape down_shape = {
        shape->rows, block->dense_hidden, shape->model_dim,
        kLiteRtNvidiaGemmGateNone, 32};
    status = LiteRtNvidiaLaunchBf16Int4GemmNormed(
        &gate_shape, activation, block->dense_norm_scale, norm_eps, 1.0f,
        arena->dense_gate_up, arena->dense_gate_up_scales, dense_hidden,
        bytes + w.dense_gemm, block->side_stream);
    if (status == cudaSuccess) {
      status = LiteRtNvidiaLaunchBf16Int4Gemm(
          &down_shape, dense_hidden, arena->dense_down,
          arena->dense_down_scales, dense_out, bytes + w.dense_gemm,
          block->side_stream);
    }
  }
  if (status != cudaSuccess) return status;
  status = cudaEventRecord(block->join, block->side_stream);
  if (status != cudaSuccess) return status;
  // The routing and the experts here; the epilogue waits for the join.
  status = LaunchFusedRouting(shape, arena, activation, router_norm_scale,
                              router_multiplier, norm_eps, renormalize,
                              workspace, stream);
  if (status != cudaSuccess) return status;
  const BlockEpilogue epilogue{
      block, static_cast<const __nv_bfloat16*>(activation), dense_out,
      norm_eps};
  return LaunchMoeChecked(
      shape, arena, activation,
      reinterpret_cast<const int32_t*>(bytes + w.top_indices),
      reinterpret_cast<const float*>(bytes + w.top_weights), per_expert_scale,
      expert_norm_scale, norm_eps, workspace, output, stream, &epilogue,
      DecodeRoutingLogits(*shape, bytes + w.logits), renormalize);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream) {
  return LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
      activation, packed_weights, scales, nullptr, 0.0f, 1.0f, output, rows,
      input_size, output_size, gate, stream);
}

namespace {

bool ValidRowsLaunch(const void* activation, const uint8_t* packed_weights,
                     const void* scales, const float* norm_scale,
                     float norm_eps, const void* output, int rows,
                     int input_size, int output_size, int gate) {
  using namespace litert::nvidia::int4_block;
  return !(activation == nullptr || packed_weights == nullptr ||
           scales == nullptr || output == nullptr || rows <= 0 ||
           rows > kLiteRtNvidiaInt4BlockGemvMaxRows || input_size <= 0 ||
           input_size % kBlock != 0 || output_size <= 0 || gate < 0 ||
           gate > 3 || (norm_scale != nullptr && norm_eps < 0.0f));
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvSlotRowsNormed(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    const float* norm_scale, float norm_eps, float norm_multiplier,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream) {
  using namespace litert::nvidia::int4_block;
  if (!ValidRowsLaunch(activation, packed_weights, scales, norm_scale,
                       norm_eps, output, rows, input_size, output_size,
                       gate)) {
    return cudaErrorInvalidValue;
  }
  // DenseKernel keeps each row's accumulation in the order of a row alone.
  const DenseNorm norm = {norm_scale, norm_eps, norm_multiplier};
  return LaunchDenseFp32(static_cast<const __nv_bfloat16*>(activation),
                         packed_weights, static_cast<const __half*>(scales),
                         norm, static_cast<__nv_bfloat16*>(output), rows,
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
      input_size % kBlock != 0 || output_size <= 0 || gate < 0 || gate > 3 ||
      (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __half*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  const DenseNorm norm = {norm_scale, norm_eps, norm_multiplier};
  if (rows >= 2) {
    return LaunchDenseMma(typed_activation, packed_weights, typed_scales, norm,
                          typed_output, rows, input_size, output_size, gate,
                          stream);
  }
  return LaunchDenseFp32(typed_activation, packed_weights, typed_scales, norm,
                         typed_output, rows, input_size, output_size, gate,
                         stream);
}
