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

#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kLongInputThreshold = 4096;
constexpr int kLargeOutputThreshold = 65536;

// A lane reads one aligned uint32_t holding 16 weights from each channel per
// loop iteration, so every row load is a coalesced 128-byte transaction.
// RowsPerWarp=4 balances parallelism and activation reuse for short-K body up
// projections. Long-K body down projections use two rows per warp for enough
// CTAs, while the large vocabulary head uses eight rows to reuse activations.
template <int RowsPerWarp, int WarpsPerBlock>
__global__ void Bf16Int2PerChannelGemvKernel(
    const __nv_bfloat16* __restrict__ activation,
    const uint8_t* __restrict__ packed_weights,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ output, int input_size, int output_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int first_output_channel =
      blockIdx.x * WarpsPerBlock * RowsPerWarp + warp * RowsPerWarp;
  if (first_output_channel >= output_size) {
    return;
  }

  const int packed_words_per_row = input_size / 16;
  float accumulators[RowsPerWarp] = {};

  for (int packed_word = lane; packed_word < packed_words_per_row;
       packed_word += kWarpSize) {
    uint32_t weights[RowsPerWarp] = {};
#pragma unroll
    for (int row = 0; row < RowsPerWarp; ++row) {
      const int output_channel = first_output_channel + row;
      if (output_channel < output_size) {
        const auto* weight_words = reinterpret_cast<const uint32_t*>(
            packed_weights +
            static_cast<int64_t>(output_channel) * input_size / 4);
        weights[row] = weight_words[packed_word];
      }
    }
    const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(
        activation + static_cast<int64_t>(packed_word) * 16);

#pragma unroll
    for (int pair_index = 0; pair_index < 8; ++pair_index) {
      const float2 values =
          __bfloat1622float2(__ldg(activation_pairs + pair_index));
#pragma unroll
      for (int row = 0; row < RowsPerWarp; ++row) {
        const int weight0 = static_cast<int>(weights[row] & 1u) -
                            static_cast<int>(weights[row] & 2u);
        weights[row] >>= 2;
        const int weight1 = static_cast<int>(weights[row] & 1u) -
                            static_cast<int>(weights[row] & 2u);
        weights[row] >>= 2;
        accumulators[row] =
            fmaf(values.x, static_cast<float>(weight0), accumulators[row]);
        accumulators[row] =
            fmaf(values.y, static_cast<float>(weight1), accumulators[row]);
      }
    }
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int row = 0; row < RowsPerWarp; ++row) {
      accumulators[row] +=
          __shfl_down_sync(kFullWarpMask, accumulators[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < RowsPerWarp; ++row) {
      const int output_channel = first_output_channel + row;
      if (output_channel < output_size) {
        const float scale = __bfloat162float(scales[output_channel]);
        output[output_channel] =
            __float2bfloat16_rn(accumulators[row] * scale);
      }
    }
  }
}

template <int RowsPerWarp, int WarpsPerBlock>
cudaError_t LaunchVariant(const __nv_bfloat16* activation,
                          const uint8_t* packed_weights,
                          const __nv_bfloat16* scales,
                          __nv_bfloat16* output, int input_size,
                          int output_size, cudaStream_t stream) {
  constexpr int kOutputsPerBlock = RowsPerWarp * WarpsPerBlock;
  const int blocks =
      (output_size + kOutputsPerBlock - 1) / kOutputsPerBlock;
  Bf16Int2PerChannelGemvKernel<RowsPerWarp, WarpsPerBlock>
      <<<blocks, kWarpSize * WarpsPerBlock, 0, stream>>>(
          activation, packed_weights, scales, output, input_size, output_size);
  return cudaPeekAtLastError();
}

bool InvalidArguments(const void* activation, const uint8_t* packed_weights,
                      const void* scales, void* output, int input_size,
                      int output_size) {
  return activation == nullptr || packed_weights == nullptr ||
         scales == nullptr || output == nullptr || input_size <= 0 ||
         input_size % 16 != 0 || output_size <= 0;
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int input_size, int output_size, cudaStream_t stream) {
  // uint32_t weight loads require a 16-value row stride. Gemma4 E2B decode
  // fully-connected dimensions satisfy this alignment.
  if (InvalidArguments(activation, packed_weights, scales, output, input_size,
                       output_size)) {
    return cudaErrorInvalidValue;
  }
  const auto* typed_activation =
      static_cast<const __nv_bfloat16*>(activation);
  const auto* typed_scales = static_cast<const __nv_bfloat16*>(scales);
  auto* typed_output = static_cast<__nv_bfloat16*>(output);
  // Eight rows per warp amortizes activation loads for the very large
  // vocabulary head. Long-K body down projections need more CTAs to occupy
  // the GPU: Gemma4's K=12288, N=1536 shape produces only 48 CTAs with eight
  // rows, but 192 CTAs with two rows on an 84-SM RTX 5080. Short-K body layers
  // retain four rows per warp.
  if (output_size >= kLargeOutputThreshold) {
    return LaunchVariant<8, 4>(typed_activation, packed_weights, typed_scales,
                               typed_output, input_size, output_size, stream);
  }
  if (input_size >= kLongInputThreshold) {
    return LaunchVariant<2, 4>(typed_activation, packed_weights, typed_scales,
                               typed_output, input_size, output_size, stream);
  }
  return LaunchVariant<4, 4>(typed_activation, packed_weights, typed_scales,
                             typed_output, input_size, output_size, stream);
}
