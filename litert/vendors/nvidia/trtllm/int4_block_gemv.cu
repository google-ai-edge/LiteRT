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

}  // namespace
}  // namespace litert::nvidia::int4_block

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
