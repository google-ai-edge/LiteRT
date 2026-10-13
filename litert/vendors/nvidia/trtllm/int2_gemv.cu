// Copyright 2026 Google LLC.
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Adapted from TensorRT Edge-LLM's Apache-2.0 CUDA INT4
// groupwise GEMV kernel:
// https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/cpp/kernels/int4GroupwiseGemmKernels/int4WoQGemvCuda.cu
//
// The Edge-LLM kernel is derived from MIT-HAN Lab's LLM-AWQ GEMV. This
// adaptation retains its CUDA C++ small-M design and activation reuse across
// output rows, while consuming LiteRT's raw signed INT2/INT4 row-major weights,
// BF16 activations, and per-channel BF16 scales.
//
// Copyright (c) 2023 MIT-HAN Lab
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda_bf16.h>

#include <cstdint>

#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kOutputsPerBlock = 8;
constexpr int kLongInputThreshold = 4096;

// One CTA computes eight output channels. Each thread loads a weight word
// once and reuses it across the Rows activation rows, and loads an activation
// word once and reuses it across all eight channels, following Edge-LLM's
// small-M GEMV mapping. Accumulation and the block reduction stay FP32 for
// BF16 accuracy. Rows is the batch of activation rows (1 for decode, the
// draft tokens plus one for a verify step).
template <int BitWidth, int Rows, int BlockSize>
__global__ void Bf16SubbytePerChannelGemvKernel(
    const __nv_bfloat16* __restrict__ activation,
    const uint8_t* __restrict__ packed_weights,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ output, int input_size, int output_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int first_output_channel = blockIdx.x * kOutputsPerBlock;
  if (first_output_channel >= output_size) {
    return;
  }

  constexpr int kWeightsPerWord = 32 / BitWidth;
  constexpr uint32_t kMask = (1u << BitWidth) - 1;
  constexpr uint32_t kSignBit = 1u << (BitWidth - 1);
  const int packed_words_per_row = input_size / kWeightsPerWord;
  float accumulators[Rows][kOutputsPerBlock] = {};

  for (int packed_word = threadIdx.x; packed_word < packed_words_per_row;
       packed_word += blockDim.x) {
    uint32_t weights[kOutputsPerBlock] = {};
#pragma unroll
    for (int row = 0; row < kOutputsPerBlock; ++row) {
      const int output_channel = first_output_channel + row;
      if (output_channel < output_size) {
        const auto* weight_words = reinterpret_cast<const uint32_t*>(
            packed_weights +
            static_cast<int64_t>(output_channel) * input_size * BitWidth / 8);
        weights[row] = weight_words[packed_word];
      }
    }
    const __nv_bfloat162* activation_pairs[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      activation_pairs[r] = reinterpret_cast<const __nv_bfloat162*>(
          activation + static_cast<int64_t>(r) * input_size +
          static_cast<int64_t>(packed_word) * kWeightsPerWord);
    }

#pragma unroll
    for (int pair_index = 0; pair_index < kWeightsPerWord / 2; ++pair_index) {
      float2 values[Rows];
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        values[r] = __bfloat1622float2(__ldg(activation_pairs[r] + pair_index));
      }
#pragma unroll
      for (int row = 0; row < kOutputsPerBlock; ++row) {
        const uint32_t bits0 = weights[row] & kMask;
        const float weight0 =
            static_cast<float>(static_cast<int>((bits0 ^ kSignBit) - kSignBit));
        weights[row] >>= BitWidth;
        const uint32_t bits1 = weights[row] & kMask;
        const float weight1 =
            static_cast<float>(static_cast<int>((bits1 ^ kSignBit) - kSignBit));
        weights[row] >>= BitWidth;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          accumulators[r][row] =
              fmaf(values[r].x, weight0, accumulators[r][row]);
          accumulators[r][row] =
              fmaf(values[r].y, weight1, accumulators[r][row]);
        }
      }
    }
  }

  constexpr int kWarpsPerBlock = BlockSize / kWarpSize;
  __shared__ float warp_sums[Rows][kOutputsPerBlock][kWarpsPerBlock];
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
      for (int row = 0; row < kOutputsPerBlock; ++row) {
        accumulators[r][row] +=
            __shfl_down_sync(kFullWarpMask, accumulators[r][row], offset);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
      for (int row = 0; row < kOutputsPerBlock; ++row) {
        warp_sums[r][row][warp] = accumulators[r][row];
      }
    }
  }
  __syncthreads();

  if (warp == 0 && lane < kOutputsPerBlock) {
    const int output_channel = first_output_channel + lane;
    if (output_channel < output_size) {
      const float scale = __bfloat162float(scales[output_channel]);
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        float accumulator = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < kWarpsPerBlock;
             ++source_warp) {
          accumulator += warp_sums[r][lane][source_warp];
        }
        output[static_cast<int64_t>(r) * output_size + output_channel] =
            __float2bfloat16_rn(accumulator * scale);
      }
    }
  }
}

template <int BitWidth, int Rows, int BlockSize>
cudaError_t Launch(const __nv_bfloat16* activation,
                   const uint8_t* packed_weights,
                   const __nv_bfloat16* scales, __nv_bfloat16* output,
                   int input_size, int output_size, cudaStream_t stream) {
  const int blocks = (output_size + kOutputsPerBlock - 1) / kOutputsPerBlock;
  Bf16SubbytePerChannelGemvKernel<BitWidth, Rows, BlockSize>
      <<<blocks, BlockSize, 0, stream>>>(
          activation, packed_weights, scales, output, input_size, output_size);
  return cudaPeekAtLastError();
}

template <int BitWidth, int BlockSize>
cudaError_t LaunchRows(const __nv_bfloat16* activation,
                       const uint8_t* packed_weights,
                       const __nv_bfloat16* scales, __nv_bfloat16* output,
                       int rows, int input_size, int output_size,
                       cudaStream_t stream) {
  switch (rows) {
#define LITERT_NVIDIA_GEMV_ROWS_CASE(n)                                  \
  case n:                                                                \
    return Launch<BitWidth, n, BlockSize>(activation, packed_weights,    \
                                          scales, output, input_size,    \
                                          output_size, stream);
    LITERT_NVIDIA_GEMV_ROWS_CASE(1)
    LITERT_NVIDIA_GEMV_ROWS_CASE(2)
    LITERT_NVIDIA_GEMV_ROWS_CASE(3)
    LITERT_NVIDIA_GEMV_ROWS_CASE(4)
    LITERT_NVIDIA_GEMV_ROWS_CASE(5)
    LITERT_NVIDIA_GEMV_ROWS_CASE(6)
    LITERT_NVIDIA_GEMV_ROWS_CASE(7)
    LITERT_NVIDIA_GEMV_ROWS_CASE(8)
#undef LITERT_NVIDIA_GEMV_ROWS_CASE
    default:
      return cudaErrorInvalidValue;
  }
}
static_assert(kLiteRtNvidiaSubbyteGemvMaxRows == 8,
              "LaunchRows instantiates rows 1 to 8");

bool InvalidArguments(const void* activation, const uint8_t* packed_weights,
                      const void* scales, void* output, int bit_width,
                      int rows, int input_size, int output_size) {
  return activation == nullptr || packed_weights == nullptr ||
         scales == nullptr || output == nullptr || input_size <= 0 ||
         input_size % 16 != 0 || output_size <= 0 || rows <= 0 ||
         rows > kLiteRtNvidiaSubbyteGemvMaxRows ||
         (bit_width != 2 && bit_width != 4);
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int input_size, int output_size, cudaStream_t stream) {
  return LiteRtNvidiaLaunchBf16SubbytePerChannelGemv(
      activation, packed_weights, scales, output, /*bit_width=*/2, input_size,
      output_size, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16SubbytePerChannelGemv(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int bit_width, int input_size, int output_size,
    cudaStream_t stream) {
  return LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
      activation, packed_weights, scales, output, bit_width, /*rows=*/1,
      input_size, output_size, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int bit_width, int rows, int input_size, int output_size,
    cudaStream_t stream) {
  // uint32_t weight loads require a 16-value row stride. Gemma4 E2B decode
  // fully-connected dimensions satisfy this alignment.
  if (InvalidArguments(activation, packed_weights, scales, output, bit_width,
                       rows, input_size, output_size)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __nv_bfloat16*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  if (input_size >= kLongInputThreshold) {
    return bit_width == 2
               ? LaunchRows<2, 256>(typed_activation, packed_weights,
                                    typed_scales, typed_output, rows,
                                    input_size, output_size, stream)
               : LaunchRows<4, 256>(typed_activation, packed_weights,
                                    typed_scales, typed_output, rows,
                                    input_size, output_size, stream);
  }
  return bit_width == 2
             ? LaunchRows<2, 128>(typed_activation, packed_weights,
                                  typed_scales, typed_output, rows, input_size,
                                  output_size, stream)
             : LaunchRows<4, 128>(typed_activation, packed_weights,
                                  typed_scales, typed_output, rows, input_size,
                                  output_size, stream);
}
