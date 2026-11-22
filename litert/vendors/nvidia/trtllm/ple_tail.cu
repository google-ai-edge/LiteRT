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


#include "litert/vendors/nvidia/trtllm/ple_tail.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kChunk = 8;  // BF16 values per 16-byte load
constexpr int kMaxRows = 8;
constexpr int kMaxStreams = 8;
constexpr int kMaxTaps = 16;
// A thread sums at most this many chunks of a stream: streams of up to
// kThreads * kMaxChunks * kChunk values.
constexpr int kMaxChunks = 4;
constexpr int kSums = 4;  // key^2, x^2, key * x (weighted), value^2

__device__ __forceinline__ void Unpack(const uint4& raw, float (&out)[kChunk]) {
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int i = 0; i < kChunk / 2; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ float Bf16(const __nv_bfloat16* at) {
  return __uint_as_float(static_cast<uint32_t>(
                             __ldg(reinterpret_cast<const unsigned short*>(at)))
                         << 16);
}

// grid (ceil(hidden / kThreads), streams): a CTA owns kThreads channels of
// one stream, a thread a channel. Every CTA sums its whole stream for the
// norms and the gate (a few CTAs a stream read it again: kilobytes), then
// each thread runs its channel's gated value, norm and convolution over the
// rows.
template <int kRows>
__global__ void __launch_bounds__(kThreads)
    PleTailKernel(LiteRtNvidiaPleTailParams p,
                  const __nv_bfloat16* __restrict__ key,
                  const __nv_bfloat16* __restrict__ value,
                  const __nv_bfloat16* __restrict__ x, const float* state,
                  const float* __restrict__ constants,
                  __nv_bfloat16* __restrict__ out, float* new_state) {
  __shared__ float scratch[kWarps * kSums];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int stream = blockIdx.y;
  const int h = blockIdx.x * kThreads + threadIdx.x;
  const int width = p.streams * p.hidden;
  const int channel = stream * p.hidden + h;
  const bool owns = h < p.hidden;
  const int history = p.taps - 1;
  const float* key_weight = constants + static_cast<size_t>(stream) * p.hidden;
  const float* query_weight = key_weight + width;
  const float* conv_weight = constants + 2 * static_cast<size_t>(width);
  const float* taps = constants + 3 * static_cast<size_t>(width) +
                      static_cast<size_t>(channel) * p.taps;
  // The channel's own values, asked for before the stream's sums.
  float window[kMaxTaps - 1 + kRows];
  float weights[kMaxTaps];
  float values[kRows], residual[kRows];
  float norm_weight = 0.0f;
  if (owns) {
    norm_weight = __ldg(conv_weight + channel);
#pragma unroll
    for (int j = 0; j < kMaxTaps; ++j) {
      weights[j] = j < p.taps ? __ldg(taps + j) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < kMaxTaps - 1; ++j) {
      window[j] =
          j < history ? state[static_cast<size_t>(channel) * history + j] : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      values[r] = 0.0f;
      residual[r] = 0.0f;
      if (r < p.rows) {
        values[r] = Bf16(value + static_cast<size_t>(r) * p.hidden + h);
        residual[r] = Bf16(x + static_cast<size_t>(r) * width + channel);
      }
    }
  }
  const int chunks = p.hidden / kChunk;
  const float inverse = 1.0f / static_cast<float>(p.hidden);
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r >= p.rows) break;
    const __nv_bfloat16* key_row =
        key + static_cast<size_t>(r) * width + stream * p.hidden;
    const __nv_bfloat16* x_row =
        x + static_cast<size_t>(r) * width + stream * p.hidden;
    const __nv_bfloat16* value_row = value + static_cast<size_t>(r) * p.hidden;
    float sums[kSums] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < kMaxChunks; ++k) {
      const int c = threadIdx.x + k * kThreads;
      if (c < chunks) {
        float kv[kChunk], xv[kChunk], vv[kChunk];
        Unpack(__ldg(reinterpret_cast<const uint4*>(key_row) + c), kv);
        Unpack(__ldg(reinterpret_cast<const uint4*>(x_row) + c), xv);
        Unpack(__ldg(reinterpret_cast<const uint4*>(value_row) + c), vv);
        const float4* kw = reinterpret_cast<const float4*>(key_weight) + 2 * c;
        const float4* qw =
            reinterpret_cast<const float4*>(query_weight) + 2 * c;
        const float4 kw0 = __ldg(kw), kw1 = __ldg(kw + 1);
        const float4 qw0 = __ldg(qw), qw1 = __ldg(qw + 1);
        const float kws[kChunk] = {kw0.x, kw0.y, kw0.z, kw0.w,
                                   kw1.x, kw1.y, kw1.z, kw1.w};
        const float qws[kChunk] = {qw0.x, qw0.y, qw0.z, qw0.w,
                                   qw1.x, qw1.y, qw1.z, qw1.w};
#pragma unroll
        for (int i = 0; i < kChunk; ++i) {
          sums[0] = fmaf(kv[i], kv[i], sums[0]);
          sums[1] = fmaf(xv[i], xv[i], sums[1]);
          sums[2] = fmaf(kv[i] * kws[i], xv[i] * qws[i], sums[2]);
          sums[3] = fmaf(vv[i], vv[i], sums[3]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kSums; ++i) {
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        sums[i] += __shfl_xor_sync(kFullWarp, sums[i], offset);
      }
      if (lane == 0) scratch[warp * kSums + i] = sums[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kSums; ++i) {
      sums[i] = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) sums[i] += scratch[w * kSums + i];
    }
    // The scratch is free for the next row.
    __syncthreads();
    const float g = p.dot_scale * sums[2] *
                    rsqrtf(sums[0] * inverse + p.eps_key) *
                    rsqrtf(sums[1] * inverse + p.eps_query);
    const float sign = g > 0.0f ? 1.0f : (g < 0.0f ? -1.0f : 0.0f);
    const float root = sign * sqrtf(fmaxf(fabsf(g), p.floor));
    const float gate = 1.0f / (1.0f + __expf(-root));
    const float gated = gate * values[r];
    window[kMaxTaps - 1 + r] =
        gated * rsqrtf(gate * gate * sums[3] * inverse + p.eps_conv) *
        norm_weight;
    residual[r] += gated;
  }
  if (!owns) return;
  // The window holds the state at [0, history) and the rows' inputs at
  // kMaxTaps - 1 on: row r's taps read positions r .. r + history of
  // state ++ inputs.
  const auto at = [&](int position) {
    float picked = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxTaps - 1 + kRows; ++i) {
      const int place = position < history ? position
                                           : kMaxTaps - 1 + position - history;
      picked = i == place ? window[i] : picked;
    }
    return picked;
  };
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r >= p.rows) break;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxTaps; ++j) {
      if (j < p.taps) acc = fmaf(weights[j], at(r + j), acc);
    }
    if (p.silu != 0) acc = acc / (1.0f + __expf(-acc));
    out[static_cast<size_t>(r) * width + channel] =
        __float2bfloat16_rn(residual[r] + acc);
  }
  // The last taps - 1 inputs: read above, before the state is written (it
  // may be the same memory).
#pragma unroll
  for (int j = 0; j < kMaxTaps - 1; ++j) {
    if (j < history) {
      new_state[static_cast<size_t>(channel) * history + j] = at(p.rows + j);
    }
  }
}

}  // namespace

extern "C" bool LiteRtNvidiaPleTailSupports(
    const LiteRtNvidiaPleTailParams* p) {
  return p != nullptr && p->rows >= 1 && p->rows <= kMaxRows &&
         p->streams >= 1 && p->streams <= kMaxStreams && p->hidden >= kChunk &&
         p->hidden % kChunk == 0 &&
         p->hidden <= kThreads * kMaxChunks * kChunk && p->taps >= 2 &&
         p->taps <= kMaxTaps && p->eps_key >= 0.0f && p->eps_query >= 0.0f &&
         p->eps_conv >= 0.0f && p->floor >= 0.0f &&
         p->dot_scale == p->dot_scale;
}

extern "C" size_t LiteRtNvidiaPleTailConstantCount(
    const LiteRtNvidiaPleTailParams* p) {
  if (!LiteRtNvidiaPleTailSupports(p)) return 0;
  return static_cast<size_t>(p->streams) * p->hidden * (3 + p->taps);
}

extern "C" cudaError_t LiteRtNvidiaLaunchPleTail(
    const LiteRtNvidiaPleTailParams* p, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* x, const float* state,
    const float* constants, __nv_bfloat16* out, float* new_state,
    cudaStream_t stream) {
  if (!LiteRtNvidiaPleTailSupports(p) || key == nullptr || value == nullptr ||
      x == nullptr || state == nullptr || constants == nullptr ||
      out == nullptr || new_state == nullptr ||
      // The sums read 16 bytes at a time.
      reinterpret_cast<uintptr_t>(key) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(value) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(constants) % 16 != 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((p->hidden + kThreads - 1) / kThreads, p->streams);
  if (p->rows == 1) {
    PleTailKernel<1><<<grid, kThreads, 0, stream>>>(*p, key, value, x, state,
                                                    constants, out, new_state);
  } else {
    PleTailKernel<kMaxRows><<<grid, kThreads, 0, stream>>>(
        *p, key, value, x, state, constants, out, new_state);
  }
  return cudaPeekAtLastError();
}
