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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_QKV_NORM_ROPE_CUH_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_QKV_NORM_ROPE_CUH_

// The megakernel's copy of odml.qkv_norm_rope's device code
// (trtllm/qkv_norm_rope.cu, whose kernel's arithmetic this is, operation for
// operation): one head of one token by one block, as a function of the
// head's row and of how the block's threads wait for each other
// (block_sync.cuh), so that a persistent kernel can run the same heads. The
// production kernel does not include this file.

#include <cuda_bf16.h>

#include <cstdint>

#include "litert/vendors/nvidia/experimental/megakernel/block_sync.cuh"

namespace litert::nvidia::megakernel::qkv_norm_rope {

constexpr int kThreads = 256;
constexpr int kMaxDepth = 1024;

// By a block of kThreads threads (they wait for each other by Sync): `row`
// [depth] RMS-normalized (when `normalize`, times `scale` when not null)
// and rotated by the half-rotation
// RoPE at *position (when `rotate`; the first rotary / 2 pairs (i, i +
// depth / 2) turn, the timescales spaced over all depth / 2 pairs) into
// `out` [depth]. `values` is shared memory of kMaxDepth floats, `partial`
// of kThreads / 32.
template <typename Sync = CtaSync>
__device__ __forceinline__ void NormRopeHead(
    const __nv_bfloat16* __restrict__ row, const float* __restrict__ scale,
    bool normalize, bool rotate, int depth, int rotary, float epsilon,
    float min_timescale, float max_timescale,
    const int32_t* __restrict__ position_ptr, __nv_bfloat16* __restrict__ out,
    float* values, float* partial) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < depth; i += kThreads) {
    const float x = __bfloat162float(row[i]);
    values[i] = x;
    sum += x * x;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
  Sync::Sync();
  if (threadIdx.x < 32) {
    float total = threadIdx.x < kThreads / 32 ? partial[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      total += __shfl_xor_sync(0xffffffffu, total, offset);
    }
    if (threadIdx.x == 0) partial[0] = total;
  }
  Sync::Sync();
  const float inv_rms =
      normalize ? rsqrtf(partial[0] / static_cast<float>(depth) + epsilon)
                : 1.0f;
  Sync::Sync();  // values complete, partial read
  if (normalize) {
    for (int i = threadIdx.x; i < depth; i += kThreads) {
      values[i] *= inv_rms * (scale != nullptr ? scale[i] : 1.0f);
    }
  }
  Sync::Sync();
  if (rotate) {
    // Proportional RoPE: the timescales are spaced over all depth / 2 pairs
    // (i, i + depth / 2) and only the first rotary / 2 pairs turn, as in the
    // composite's decomposition. Equal to the plain RoPE when rotary = depth.
    const int half = rotary / 2;
    const float position = static_cast<float>(*position_ptr);
    const float log_step =
        logf(max_timescale / min_timescale) / static_cast<float>(depth / 2);
    for (int i = threadIdx.x; i < half; i += kThreads) {
      const float inv_timescale =
          expf(-static_cast<float>(i) * log_step) / min_timescale;
      float sine, cosine;
      sincosf(position * inv_timescale, &sine, &cosine);
      const float x1 = values[i];
      const float x2 = values[i + depth / 2];
      values[i] = x1 * cosine - x2 * sine;
      values[i + depth / 2] = x2 * cosine + x1 * sine;
    }
    Sync::Sync();
  }
  for (int i = threadIdx.x; i < depth; i += kThreads) {
    out[i] = __float2bfloat16_rn(values[i]);
  }
}

}  // namespace litert::nvidia::megakernel::qkv_norm_rope

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_QKV_NORM_ROPE_CUH_
