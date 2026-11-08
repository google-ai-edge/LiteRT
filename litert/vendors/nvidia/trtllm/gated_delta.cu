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

#include "litert/vendors/nvidia/trtllm/gated_delta.h"

#include <cstdint>

namespace {

// The recurrent update. A block owns 32 value columns of one head (one per
// lane); its four warps split the key rows, so every thread keeps
// key_dim / 4 state values of its column in registers for all rows of the
// call. Per row: stage k_t and q_t, reduce S^T k over the warps, update,
// reduce S^T q over the warps.
constexpr int kColumns = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kColumns * kWarps;

template <int kKeyDim>
__global__ void __launch_bounds__(kThreads)
    GatedDeltaUpdateKernel(const float* __restrict__ q,
                           const float* __restrict__ k,
                           const float* __restrict__ v,
                           const float* __restrict__ beta,
                           const float* __restrict__ g, const float* state,
                           const int32_t* __restrict__ valid_len,
                           LiteRtNvidiaGatedDeltaParams p,
                           float* __restrict__ out, float* new_state) {
  constexpr int kRows = kKeyDim / kWarps;
  __shared__ float key[2][kKeyDim];
  __shared__ float query[2][kKeyDim];
  __shared__ float key_partial[kWarps][kColumns];
  __shared__ float out_partial[2][kWarps][kColumns];

  const int head = blockIdx.x;
  const int lane = threadIdx.x % kColumns;
  const int warp = threadIdx.x / kColumns;
  const int column = blockIdx.y * kColumns + lane;
  const int key_head = head / (p.heads / p.key_heads);
  const int tokens = p.tokens;
  const int value_dim = p.value_dim;
  int active = tokens;
  if (valid_len != nullptr) {
    active = min(max(valid_len[0], 0), tokens);
  }

  // This thread's part of the state: rows [warp * kRows, + kRows) of
  // `column`, read before anything is written (in-place safe).
  const size_t state_base =
      (static_cast<size_t>(head) * kKeyDim + warp * kRows) * value_dim +
      column;
  float s[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    s[r] = state[state_base + static_cast<size_t>(r) * value_dim];
  }

  const float* q_head = q + static_cast<size_t>(key_head) * tokens * kKeyDim;
  const float* k_head = k + static_cast<size_t>(key_head) * tokens * kKeyDim;
  const float* v_head = v + static_cast<size_t>(head) * tokens * value_dim;
  const float* beta_head = beta + static_cast<size_t>(head) * tokens;
  const float* g_head = g + static_cast<size_t>(head) * tokens;
  float* out_head = out + static_cast<size_t>(head) * tokens * value_dim;

  for (int t = 0; t < active; ++t) {
    const int buffer = t & 1;
    for (int i = threadIdx.x; i < kKeyDim; i += kThreads) {
      key[buffer][i] = k_head[static_cast<size_t>(t) * kKeyDim + i];
      query[buffer][i] = q_head[static_cast<size_t>(t) * kKeyDim + i];
    }
    const float g_t = g_head[t];
    const float beta_t = beta_head[t];
    const float value = v_head[static_cast<size_t>(t) * value_dim + column];
    __syncthreads();  // A: k_t, q_t staged

    const float decay = g_t <= 0.0f ? expf(g_t) : 1.0f;
    const float rate = beta_t > 0.0f ? beta_t : 0.0f;
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s[r] *= decay;
      kv += s[r] * key[buffer][warp * kRows + r];
    }
    key_partial[warp][lane] = kv;
    __syncthreads();  // B: S^T k partials complete

    float memory = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) memory += key_partial[w][lane];
    const float delta = (value - memory) * rate;
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s[r] += key[buffer][warp * kRows + r] * delta;
      o += s[r] * query[buffer][warp * kRows + r];
    }
    out_partial[buffer][warp][lane] = o;
    __syncthreads();  // C: S^T q partials complete; key_partial consumed
    if (warp == 0) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) total += out_partial[buffer][w][lane];
      out_head[static_cast<size_t>(t) * value_dim + column] = total;
    }
  }
  for (int t = active + warp; t < tokens; t += kWarps) {
    out_head[static_cast<size_t>(t) * value_dim + column] = 0.0f;
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    new_state[state_base + static_cast<size_t>(r) * value_dim] = s[r];
  }
}

constexpr int kConvThreads = 256;
constexpr int kMaxTaps = 8;

__device__ __forceinline__ float ConvWindow(const float* __restrict__ x,
                                            const float* state, int channels,
                                            int history, int c, int position) {
  // Position `position` of xx = state ++ x along time, for channel c.
  return position < history
             ? state[static_cast<size_t>(c) * history + position]
             : x[static_cast<size_t>(position - history) * channels + c];
}

// grid (ceil(channels / 256), tokens): y_t of 256 channels.
__global__ void __launch_bounds__(kConvThreads)
    ShortConvOutputKernel(const float* __restrict__ x, const float* state,
                          const float* __restrict__ weight,
                          LiteRtNvidiaShortConvParams p,
                          float* __restrict__ y) {
  const int c = blockIdx.x * kConvThreads + threadIdx.x;
  const int t = blockIdx.y;
  if (c >= p.channels) return;
  const int history = p.taps - 1;
  float acc = 0.0f;
  for (int j = 0; j < p.taps; ++j) {
    acc += weight[static_cast<size_t>(c) * p.taps + j] *
           ConvWindow(x, state, p.channels, history, c, t + j);
  }
  if (p.silu != 0) acc = acc / (1.0f + expf(-acc));
  y[static_cast<size_t>(t) * p.channels + c] = acc;
}

// One thread per channel: the last taps - 1 valid inputs, read before the
// state is overwritten (in-place safe; runs after ShortConvOutputKernel).
__global__ void __launch_bounds__(kConvThreads)
    ShortConvStateKernel(const float* __restrict__ x, const float* state,
                         const int32_t* __restrict__ valid_len,
                         LiteRtNvidiaShortConvParams p, float* new_state) {
  const int c = blockIdx.x * kConvThreads + threadIdx.x;
  if (c >= p.channels) return;
  const int history = p.taps - 1;
  int valid = p.tokens;
  if (valid_len != nullptr) valid = min(max(valid_len[0], 0), p.tokens);
  float kept[kMaxTaps];
  for (int j = 0; j < history; ++j) {
    kept[j] = ConvWindow(x, state, p.channels, history, c, valid + j);
  }
  for (int j = 0; j < history; ++j) {
    new_state[static_cast<size_t>(c) * history + j] = kept[j];
  }
}

}  // namespace

extern "C" bool LiteRtNvidiaGatedDeltaSupports(
    const LiteRtNvidiaGatedDeltaParams* params) {
  return params != nullptr && params->heads >= 1 && params->key_heads >= 1 &&
         params->heads % params->key_heads == 0 && params->tokens >= 1 &&
         (params->key_dim == 64 || params->key_dim == 128 ||
          params->key_dim == 256) &&
         params->value_dim >= kColumns && params->value_dim % kColumns == 0;
}

extern "C" cudaError_t LiteRtNvidiaLaunchGatedDeltaUpdate(
    const LiteRtNvidiaGatedDeltaParams* params, const float* q, const float* k,
    const float* v, const float* beta, const float* g, const float* state,
    const int32_t* valid_len, float* out, float* new_state,
    cudaStream_t stream) {
  if (!LiteRtNvidiaGatedDeltaSupports(params) || q == nullptr ||
      k == nullptr || v == nullptr || beta == nullptr || g == nullptr ||
      state == nullptr || out == nullptr || new_state == nullptr) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(params->heads, params->value_dim / kColumns);
  switch (params->key_dim) {
    case 64:
      GatedDeltaUpdateKernel<64><<<grid, kThreads, 0, stream>>>(
          q, k, v, beta, g, state, valid_len, *params, out, new_state);
      break;
    case 128:
      GatedDeltaUpdateKernel<128><<<grid, kThreads, 0, stream>>>(
          q, k, v, beta, g, state, valid_len, *params, out, new_state);
      break;
    default:
      GatedDeltaUpdateKernel<256><<<grid, kThreads, 0, stream>>>(
          q, k, v, beta, g, state, valid_len, *params, out, new_state);
      break;
  }
  return cudaPeekAtLastError();
}

extern "C" bool LiteRtNvidiaShortConvSupports(
    const LiteRtNvidiaShortConvParams* params) {
  return params != nullptr && params->channels >= 1 && params->tokens >= 1 &&
         params->tokens <= 65535 && params->taps >= 2 &&
         params->taps <= kMaxTaps;
}

extern "C" cudaError_t LiteRtNvidiaLaunchShortConv(
    const LiteRtNvidiaShortConvParams* params, const float* x,
    const float* state, const float* weight, const int32_t* valid_len,
    float* y, float* new_state, cudaStream_t stream) {
  if (!LiteRtNvidiaShortConvSupports(params) || x == nullptr ||
      state == nullptr || weight == nullptr || y == nullptr ||
      new_state == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (params->channels + kConvThreads - 1) / kConvThreads;
  ShortConvOutputKernel<<<dim3(blocks, params->tokens), kConvThreads, 0,
                          stream>>>(x, state, weight, *params, y);
  ShortConvStateKernel<<<blocks, kConvThreads, 0, stream>>>(
      x, state, valid_len, *params, new_state);
  return cudaPeekAtLastError();
}
