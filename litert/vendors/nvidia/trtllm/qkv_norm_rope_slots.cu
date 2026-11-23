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

#include "litert/vendors/nvidia/trtllm/qkv_norm_rope_slots.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxDepth = 1024;

struct SlotTable {
  LiteRtNvidiaQkvNormRopeSlot slot[kLiteRtNvidiaQkvNormRopeMaxSlots];
};

// grid (heads + 2 kv_heads, tokens, slots): one CTA per head of one token of
// one slot. The block's work is QkvNormRopeKernel's (qkv_norm_rope.cu), kept
// here as a copy so that the single-sequence kernel's object code stays what
// it is: a change to that kernel's arithmetic must be made here too, and
// qkv_norm_rope_slots_test compares the two byte for byte.
__global__ void __launch_bounds__(kThreads)
    QkvNormRopeSlotsKernel(SlotTable table,
                           const float* __restrict__ q_scale,
                           const float* __restrict__ k_scale,
                           LiteRtNvidiaQkvNormRopeParams params) {
  __shared__ float values[kMaxDepth];
  __shared__ float partial[kThreads / 32];
  const LiteRtNvidiaQkvNormRopeSlot& slot = table.slot[blockIdx.z];
  const auto* qkv = static_cast<const __nv_bfloat16*>(slot.qkv);
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
    out = static_cast<__nv_bfloat16*>(slot.q) +
          (static_cast<size_t>(head) * params.tokens + token) * depth;
  } else if (is_k) {
    out = static_cast<__nv_bfloat16*>(slot.k) +
          (static_cast<size_t>(head - params.heads) * params.tokens + token) *
              depth;
  } else {
    out = static_cast<__nv_bfloat16*>(slot.v) +
          (static_cast<size_t>(head - params.heads - params.kv_heads) *
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
    // Proportional RoPE: the timescales are spaced over all depth / 2 pairs
    // (i, i + depth / 2) and only the first rotary / 2 pairs turn.
    const int half = params.rotary / 2;
    const float position = static_cast<float>(slot.positions[token]);
    const float log_step =
        logf(params.max_timescale / params.min_timescale) /
        static_cast<float>(depth / 2);
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

extern "C" cudaError_t LiteRtNvidiaLaunchQkvNormRopeSlots(
    const LiteRtNvidiaQkvNormRopeParams* params,
    const LiteRtNvidiaQkvNormRopeSlot* slots, int32_t num_slots,
    const float* q_scale, const float* k_scale, cudaStream_t stream) {
  if (!LiteRtNvidiaQkvNormRopeSupports(params) || slots == nullptr ||
      num_slots < 1 || num_slots > kLiteRtNvidiaQkvNormRopeMaxSlots ||
      q_scale == nullptr || k_scale == nullptr) {
    return cudaErrorInvalidValue;
  }
  SlotTable table = {};
  for (int32_t s = 0; s < num_slots; ++s) {
    if (slots[s].qkv == nullptr || slots[s].positions == nullptr ||
        slots[s].q == nullptr || slots[s].k == nullptr ||
        slots[s].v == nullptr) {
      return cudaErrorInvalidValue;
    }
    table.slot[s] = slots[s];
  }
  const dim3 grid(params->heads + 2 * params->kv_heads, params->tokens,
                  num_slots);
  QkvNormRopeSlotsKernel<<<grid, kThreads, 0, stream>>>(table, q_scale,
                                                        k_scale, *params);
  return cudaPeekAtLastError();
}
