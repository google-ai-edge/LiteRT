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

// The gated DeltaNet layer of gated_delta_net.h in four launches: the
// prologue (convolution, norms and gates, into the recurrent kernel's
// head-major layout), the convolution state, the recurrent update
// (gated_delta.h) and the gated norm. Everything between the projections is
// FP32; the projections are read and the output written once, in BF16.

#include "litert/vendors/nvidia/trtllm/gated_delta_net.h"

#include <cuda_bf16.h>

#include <cstdint>

#include "litert/vendors/nvidia/trtllm/gated_delta.h"

namespace {

constexpr int kDim = 128;  // head_dim: a block's threads, one per channel
constexpr int kMaxTaps = 8;
constexpr int kStateThreads = 256;

struct Workspace {
  size_t q, k, v, beta, g, o, total;
};

Workspace Layout(const LiteRtNvidiaGatedDeltaNetParams& p) {
  const auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  const size_t rows = static_cast<size_t>(p.tokens) * kDim * sizeof(float);
  Workspace w{};
  w.q = 0;
  w.k = align(w.q + p.key_heads * rows);
  w.v = align(w.k + p.key_heads * rows);
  w.beta = align(w.v + p.value_heads * rows);
  w.g = align(w.beta + static_cast<size_t>(p.value_heads) * p.tokens * 4);
  w.o = align(w.g + static_cast<size_t>(p.value_heads) * p.tokens * 4);
  w.total = align(w.o + p.value_heads * rows);
  return w;
}

__device__ __forceinline__ int ValidRows(const int32_t* valid_len, int tokens) {
  return valid_len != nullptr ? min(max(valid_len[0], 0), tokens) : tokens;
}

// The sum of `x` over the block's kDim threads (four warps).
__device__ __forceinline__ float BlockSum(float x, float* scratch) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  }
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = x;
  __syncthreads();
  return scratch[0] + scratch[1] + scratch[2] + scratch[3];
}

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }

// The constants of gated_delta_net.h, by name.
struct Constants {
  const float* decay;
  const float* q_scale;
  const float* k_scale;
  const float* norm_weight;
  const float* conv_weight;
  const float* qkv_bias;
  const float* z_bias;
  const float* a_bias;
  const float* b_bias;
};

__device__ __forceinline__ Constants SplitConstants(
    const float* constants, const LiteRtNvidiaGatedDeltaNetParams& p) {
  const int channels = (2 * p.key_heads + p.value_heads) * kDim;
  Constants c;
  c.decay = constants;
  c.q_scale = c.decay + p.value_heads;
  c.k_scale = c.q_scale + kDim;
  c.norm_weight = c.k_scale + kDim;
  c.conv_weight = c.norm_weight + kDim;
  c.qkv_bias = c.conv_weight + static_cast<size_t>(channels) * p.taps;
  c.z_bias = c.qkv_bias + channels;
  c.a_bias = c.z_bias + p.value_heads * kDim;
  c.b_bias = c.a_bias + p.value_heads;
  return c;
}

// Position `position` of xx = conv_state ++ the qkv rows (with their bias),
// for channel c.
__device__ __forceinline__ float ConvInput(const __nv_bfloat16* __restrict__ proj,
                                           const float* conv_state, const Constants& w,
                                           const LiteRtNvidiaGatedDeltaNetParams& p,
                                           int history, int c, int position) {
  return position < history
             ? conv_state[static_cast<size_t>(c) * history + position]
             : __bfloat162float(
                   proj[static_cast<size_t>(position - history) * p.proj_stride +
                        p.qkv_offset + c]) +
                   w.qkv_bias[c];
}

// grid (tokens, 2 * key_heads + value_heads): row t of one head's channels
// (the q heads, then the k heads, then the v heads), a thread per channel.
__global__ void __launch_bounds__(kDim)
    PrologueKernel(const __nv_bfloat16* __restrict__ proj, const float* conv_state,
                   const float* __restrict__ constants,
                   const int32_t* __restrict__ valid_len,
                   LiteRtNvidiaGatedDeltaNetParams p, float* __restrict__ q,
                   float* __restrict__ k, float* __restrict__ v,
                   float* __restrict__ beta, float* __restrict__ g) {
  __shared__ float scratch[kDim / 32];
  const int t = blockIdx.x;
  const int group = blockIdx.y;
  const int d = threadIdx.x;
  const int c = group * kDim + d;
  const bool live = t < ValidRows(valid_len, p.tokens);
  const int history = p.taps - 1;
  const Constants w = SplitConstants(constants, p);
  float y = 0.0f;
  if (live) {
    for (int j = 0; j < p.taps; ++j) {
      y = fmaf(w.conv_weight[static_cast<size_t>(c) * p.taps + j],
               ConvInput(proj, conv_state, w, p, history, c, t + j), y);
    }
    if (p.silu != 0) y = Silu(y);
  }
  if (group < 2 * p.key_heads) {
    const bool is_q = group < p.key_heads;
    const float squares = BlockSum(y * y, scratch);
    const float factor = rsqrtf(squares / kDim + (is_q ? p.q_eps : p.k_eps));
    const int head = is_q ? group : group - p.key_heads;
    (is_q ? q : k)[(static_cast<size_t>(head) * p.tokens + t) * kDim + d] =
        live ? y * factor * (is_q ? w.q_scale : w.k_scale)[d] : 0.0f;
    return;
  }
  const int head = group - 2 * p.key_heads;
  v[(static_cast<size_t>(head) * p.tokens + t) * kDim + d] = y;
  if (d == 0) {
    float rate = 0.0f;
    float gate = 0.0f;
    if (live) {
      const __nv_bfloat16* row = proj + static_cast<size_t>(t) * p.proj_stride;
      const float b = __bfloat162float(row[p.b_offset + head]) + w.b_bias[head];
      const float a = __bfloat162float(row[p.a_offset + head]) + w.a_bias[head];
      rate = 1.0f / (1.0f + expf(-b));
      gate = w.decay[head] * logf(1.0f + expf(fminf(a, 50.0f)));
    }
    beta[static_cast<size_t>(head) * p.tokens + t] = rate;
    g[static_cast<size_t>(head) * p.tokens + t] = gate;
  }
}

// A thread per channel: the last taps - 1 valid inputs, read before the state
// is overwritten (in-place safe; runs after PrologueKernel).
__global__ void __launch_bounds__(kStateThreads)
    ConvStateKernel(const __nv_bfloat16* __restrict__ proj, const float* conv_state,
                    const float* __restrict__ constants,
                    const int32_t* __restrict__ valid_len,
                    LiteRtNvidiaGatedDeltaNetParams p, int channels,
                    float* new_conv_state) {
  const int c = blockIdx.x * kStateThreads + threadIdx.x;
  if (c >= channels) return;
  const int history = p.taps - 1;
  const int valid = ValidRows(valid_len, p.tokens);
  const Constants w = SplitConstants(constants, p);
  float kept[kMaxTaps];
  for (int j = 0; j < history; ++j) {
    kept[j] = ConvInput(proj, conv_state, w, p, history, c, valid + j);
  }
  for (int j = 0; j < history; ++j) {
    new_conv_state[static_cast<size_t>(c) * history + j] = kept[j];
  }
}

// grid (tokens, value_heads): the gated norm of row t of a head.
__global__ void __launch_bounds__(kDim)
    EpilogueKernel(const float* __restrict__ o, const __nv_bfloat16* __restrict__ proj,
                   const float* __restrict__ constants,
                   const int32_t* __restrict__ valid_len,
                   LiteRtNvidiaGatedDeltaNetParams p,
                   __nv_bfloat16* __restrict__ out) {
  __shared__ float scratch[kDim / 32];
  const int t = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  const Constants w = SplitConstants(constants, p);
  const float x = o[(static_cast<size_t>(head) * p.tokens + t) * kDim + d];
  const float squares = BlockSum(x * x, scratch);
  float value = 0.0f;
  if (t < ValidRows(valid_len, p.tokens)) {
    const float z =
        __bfloat162float(
            proj[static_cast<size_t>(t) * p.proj_stride + p.z_offset + head * kDim + d]) +
        w.z_bias[head * kDim + d];
    value = x * rsqrtf(squares / kDim + p.out_eps) * w.norm_weight[d] * Silu(z);
  }
  out[(static_cast<size_t>(t) * p.value_heads + head) * kDim + d] =
      __float2bfloat16_rn(value);
}

}  // namespace

extern "C" size_t LiteRtNvidiaGatedDeltaNetConstantCount(
    const LiteRtNvidiaGatedDeltaNetParams* params) {
  const int channels = (2 * params->key_heads + params->value_heads) * params->head_dim;
  const size_t values = static_cast<size_t>(params->value_heads) * params->head_dim;
  return 3 * static_cast<size_t>(params->value_heads) + 3 * params->head_dim +
         static_cast<size_t>(channels) * (params->taps + 1) + values;
}

extern "C" bool LiteRtNvidiaGatedDeltaNetSupports(
    const LiteRtNvidiaGatedDeltaNetParams* p) {
  if (p == nullptr || p->tokens < 1 || p->key_heads < 1 || p->value_heads < 1 ||
      p->value_heads % p->key_heads != 0 || p->head_dim != kDim || p->taps < 2 ||
      p->taps > kMaxTaps || p->value_heads > 65535 ||
      2 * p->key_heads + p->value_heads > 65535 || !(p->q_eps >= 0.0f) ||
      !(p->k_eps >= 0.0f) || !(p->out_eps >= 0.0f)) {
    return false;
  }
  const int64_t channels = static_cast<int64_t>(2 * p->key_heads + p->value_heads) * kDim;
  const int64_t values = static_cast<int64_t>(p->value_heads) * kDim;
  const auto fits = [&](int64_t offset, int64_t width) {
    return offset >= 0 && offset + width <= p->proj_stride;
  };
  return fits(p->qkv_offset, channels) && fits(p->z_offset, values) &&
         fits(p->a_offset, p->value_heads) && fits(p->b_offset, p->value_heads);
}

extern "C" size_t LiteRtNvidiaGatedDeltaNetWorkspaceBytes(
    const LiteRtNvidiaGatedDeltaNetParams* params) {
  if (!LiteRtNvidiaGatedDeltaNetSupports(params)) return 0;
  return Layout(*params).total;
}

extern "C" cudaError_t LiteRtNvidiaLaunchGatedDeltaNet(
    const LiteRtNvidiaGatedDeltaNetParams* params, const void* proj,
    const float* conv_state, const float* state, const float* constants,
    const int32_t* valid_len, void* out, float* new_conv_state,
    float* new_state, void* workspace, cudaStream_t stream) {
  if (!LiteRtNvidiaGatedDeltaNetSupports(params) || proj == nullptr ||
      conv_state == nullptr || state == nullptr || constants == nullptr ||
      out == nullptr || new_conv_state == nullptr || new_state == nullptr ||
      workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  const LiteRtNvidiaGatedDeltaNetParams& p = *params;
  const Workspace w = Layout(p);
  char* base = static_cast<char*>(workspace);
  float* q = reinterpret_cast<float*>(base + w.q);
  float* k = reinterpret_cast<float*>(base + w.k);
  float* v = reinterpret_cast<float*>(base + w.v);
  float* beta = reinterpret_cast<float*>(base + w.beta);
  float* g = reinterpret_cast<float*>(base + w.g);
  float* o = reinterpret_cast<float*>(base + w.o);
  const auto* typed_proj = static_cast<const __nv_bfloat16*>(proj);
  const int channels = (2 * p.key_heads + p.value_heads) * kDim;
  PrologueKernel<<<dim3(p.tokens, 2 * p.key_heads + p.value_heads), kDim, 0, stream>>>(
      typed_proj, conv_state, constants, valid_len, p, q, k, v, beta, g);
  ConvStateKernel<<<(channels + kStateThreads - 1) / kStateThreads, kStateThreads, 0,
                    stream>>>(typed_proj, conv_state, constants, valid_len, p, channels,
                              new_conv_state);
  const LiteRtNvidiaGatedDeltaParams update = {p.value_heads, p.key_heads, p.tokens,
                                               kDim, kDim};
  cudaError_t status = LiteRtNvidiaLaunchGatedDeltaUpdate(
      &update, q, k, v, beta, g, state, valid_len, o, new_state, stream);
  if (status != cudaSuccess) return status;
  EpilogueKernel<<<dim3(p.tokens, p.value_heads), kDim, 0, stream>>>(
      o, typed_proj, constants, valid_len, p, static_cast<__nv_bfloat16*>(out));
  return cudaPeekAtLastError();
}
