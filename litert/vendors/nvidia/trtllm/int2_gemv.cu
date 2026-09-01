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

#include <cuda_bf16.h>

#include <cstdint>

#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kOutputsPerBlock = 8;
constexpr int kLongInputThreshold = 4096;

// One CTA computes eight output channels. Each thread loads an activation word
// once and reuses it across all eight rows, following Edge-LLM's small-M GEMV
// mapping. Accumulation and the block reduction stay FP32 for BF16 accuracy.
template <int BitWidth, int BlockSize>
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
  float accumulators[kOutputsPerBlock] = {};

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
    const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(
        activation + static_cast<int64_t>(packed_word) * kWeightsPerWord);

#pragma unroll
    for (int pair_index = 0; pair_index < kWeightsPerWord / 2; ++pair_index) {
      const float2 values =
          __bfloat1622float2(__ldg(activation_pairs + pair_index));
#pragma unroll
      for (int row = 0; row < kOutputsPerBlock; ++row) {
        const uint32_t bits0 = weights[row] & kMask;
        const int weight0 = static_cast<int>((bits0 ^ kSignBit) - kSignBit);
        weights[row] >>= BitWidth;
        const uint32_t bits1 = weights[row] & kMask;
        const int weight1 = static_cast<int>((bits1 ^ kSignBit) - kSignBit);
        weights[row] >>= BitWidth;
        accumulators[row] =
            fmaf(values.x, static_cast<float>(weight0), accumulators[row]);
        accumulators[row] =
            fmaf(values.y, static_cast<float>(weight1), accumulators[row]);
      }
    }
  }

  constexpr int kWarpsPerBlock = BlockSize / kWarpSize;
  __shared__ float warp_sums[kOutputsPerBlock][kWarpsPerBlock];
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int row = 0; row < kOutputsPerBlock; ++row) {
      accumulators[row] +=
          __shfl_down_sync(kFullWarpMask, accumulators[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < kOutputsPerBlock; ++row) {
      warp_sums[row][warp] = accumulators[row];
    }
  }
  __syncthreads();

  if (warp == 0 && lane < kOutputsPerBlock) {
    float accumulator = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarpsPerBlock; ++source_warp) {
      accumulator += warp_sums[lane][source_warp];
    }
    const int output_channel = first_output_channel + lane;
    if (output_channel < output_size) {
      const float scale = __bfloat162float(scales[output_channel]);
      output[output_channel] = __float2bfloat16_rn(accumulator * scale);
    }
  }
}

template <int BitWidth, int BlockSize>
cudaError_t Launch(const __nv_bfloat16* activation,
                   const uint8_t* packed_weights,
                   const __nv_bfloat16* scales, __nv_bfloat16* output,
                   int input_size, int output_size, cudaStream_t stream) {
  const int blocks = (output_size + kOutputsPerBlock - 1) / kOutputsPerBlock;
  Bf16SubbytePerChannelGemvKernel<BitWidth, BlockSize>
      <<<blocks, BlockSize, 0, stream>>>(
          activation, packed_weights, scales, output, input_size, output_size);
  return cudaPeekAtLastError();
}

bool InvalidArguments(const void* activation, const uint8_t* packed_weights,
                      const void* scales, void* output, int bit_width,
                      int input_size, int output_size) {
  return activation == nullptr || packed_weights == nullptr ||
         scales == nullptr || output == nullptr || input_size <= 0 ||
         input_size % 16 != 0 || output_size <= 0 ||
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
  // uint32_t weight loads require a 16-value row stride. Gemma4 E2B decode
  // fully-connected dimensions satisfy this alignment.
  if (InvalidArguments(activation, packed_weights, scales, output, bit_width,
                       input_size, output_size)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __nv_bfloat16*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  if (input_size >= kLongInputThreshold) {
    return bit_width == 2
               ? Launch<2, 256>(typed_activation, packed_weights, typed_scales,
                                typed_output, input_size, output_size, stream)
               : Launch<4, 256>(typed_activation, packed_weights, typed_scales,
                                typed_output, input_size, output_size, stream);
  }
  return bit_width == 2
             ? Launch<2, 128>(typed_activation, packed_weights, typed_scales,
                              typed_output, input_size, output_size, stream)
             : Launch<4, 128>(typed_activation, packed_weights, typed_scales,
                              typed_output, input_size, output_size, stream);
}
