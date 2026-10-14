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

#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"

#include "third_party/gpus/cuda/include/cuda_bf16.h"

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxDepth = 1024;

// grid (heads + 2 kv_heads, tokens): one CTA per head of one token.
__global__ void __launch_bounds__(kThreads)
    QkvNormRopeKernel(const __nv_bfloat16* __restrict__ qkv,
                      const int32_t* __restrict__ positions,
                      const float* __restrict__ q_scale,
                      const float* __restrict__ k_scale,
                      LiteRtNvidiaQkvNormRopeParams params,
                      __nv_bfloat16* __restrict__ q,
                      __nv_bfloat16* __restrict__ k,
                      __nv_bfloat16* __restrict__ v) {
  __shared__ float values[kMaxDepth];
  __shared__ float partial[kThreads / 32];
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int depth = params.depth;
  const int total_heads = params.heads + 2 * params.kv_heads;
  const __nv_bfloat16* row =
      qkv + (static_cast<size_t>(token) * total_heads + head) * depth;
  // Which projection, its scale and its output.
  const bool is_q = head < params.heads;
  const bool is_k = !is_q && head < params.heads + params.kv_heads;
  const float* scale = is_q ? q_scale : (is_k ? k_scale : nullptr);
  const bool normalize = is_q || is_k || params.has_v_norm != 0;
  const bool rotate = is_q || is_k;
  __nv_bfloat16* out;
  if (is_q) {
    out = q + (static_cast<size_t>(head) * params.tokens + token) * depth;
  } else if (is_k) {
    out = k + (static_cast<size_t>(head - params.heads) * params.tokens +
               token) *
                  depth;
  } else {
    out = v + (static_cast<size_t>(head - params.heads - params.kv_heads) *
                   params.tokens +
               token) *
                  depth;
  }

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
  __syncthreads();
  if (threadIdx.x < 32) {
    float total = threadIdx.x < kThreads / 32 ? partial[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      total += __shfl_xor_sync(0xffffffffu, total, offset);
    }
    if (threadIdx.x == 0) partial[0] = total;
  }
  __syncthreads();
  const float inv_rms =
      normalize ? rsqrtf(partial[0] / static_cast<float>(depth) + params.epsilon)
                : 1.0f;
  __syncthreads();  // values complete, partial read
  if (normalize) {
    for (int i = threadIdx.x; i < depth; i += kThreads) {
      values[i] *= inv_rms * (scale != nullptr ? scale[i] : 1.0f);
    }
  }
  __syncthreads();
  if (rotate) {
    const int half = params.rotary / 2;
    const float position = static_cast<float>(positions[token]);
    const float log_step =
        logf(params.max_timescale / params.min_timescale) /
        static_cast<float>(half);
    for (int i = threadIdx.x; i < half; i += kThreads) {
      const float inv_timescale =
          expf(-static_cast<float>(i) * log_step) / params.min_timescale;
      float sine, cosine;
      sincosf(position * inv_timescale, &sine, &cosine);
      const float x1 = values[i];
      const float x2 = values[i + depth / 2];
      values[i] = x1 * cosine - x2 * sine;
      values[i + depth / 2] = x2 * cosine + x1 * sine;
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < depth; i += kThreads) {
    out[i] = __float2bfloat16_rn(values[i]);
  }
}

}  // namespace

extern "C" bool LiteRtNvidiaQkvNormRopeSupports(
    const LiteRtNvidiaQkvNormRopeParams* params) {
  return params != nullptr && params->tokens >= 1 && params->heads >= 1 &&
         params->kv_heads >= 1 && params->depth >= 2 &&
         params->depth <= kMaxDepth && params->depth % 2 == 0 &&
         params->rotary >= 0 && params->rotary <= params->depth &&
         params->rotary % 2 == 0 && params->epsilon >= 0.0f &&
         params->min_timescale > 0.0f &&
         params->max_timescale >= params->min_timescale;
}

extern "C" cudaError_t LiteRtNvidiaLaunchQkvNormRope(
    const LiteRtNvidiaQkvNormRopeParams* params, const void* qkv,
    const int32_t* positions, const float* q_scale, const float* k_scale,
    void* q, void* k, void* v, cudaStream_t stream) {
  if (!LiteRtNvidiaQkvNormRopeSupports(params) || qkv == nullptr ||
      positions == nullptr || q_scale == nullptr || k_scale == nullptr ||
      q == nullptr || k == nullptr || v == nullptr) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(params->heads + 2 * params->kv_heads, params->tokens);
  QkvNormRopeKernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv), positions, q_scale, k_scale,
      *params, static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v));
  return cudaPeekAtLastError();
}
