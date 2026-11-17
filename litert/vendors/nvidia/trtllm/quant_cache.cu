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

#include "litert/vendors/nvidia/trtllm/quant_cache.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr size_t kMaxBlocks = 4096;

__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// TFLite's QUANTIZE with a zero point of 0: x / scale rounded half away from
// zero, saturated to [lo, hi].
__device__ __forceinline__ int Quantize(float x, float scale, float lo,
                                        float hi) {
  return static_cast<int>(fminf(fmaxf(roundf(x / scale), lo), hi));
}

// Element (h, t, d) of a [heads, tokens, depth] tensor, or of a [tokens,
// heads, depth] one.
__device__ __forceinline__ size_t RowIndex(bool token_major, int heads,
                                           int tokens, int depth, size_t h,
                                           size_t t, size_t d) {
  return (token_major ? t * heads + h : h * tokens + t) * depth + d;
}

// The kernels below give a thread the items FirstItem(), + ItemStride(), ...
int Blocks(size_t items) {
  const size_t blocks = (items + kThreads - 1) / kThreads;
  return static_cast<int>(blocks < 1 ? 1
                                     : blocks > kMaxBlocks ? kMaxBlocks
                                                           : blocks);
}

__device__ __forceinline__ size_t FirstItem() {
  return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t ItemStride() {
  return static_cast<size_t>(gridDim.x) * blockDim.x;
}

template <typename T>
__global__ void QuantizeQueryKernel(const T* __restrict__ q, size_t count,
                                    float scale, __half* __restrict__ out) {
  for (size_t i = FirstItem(); i < count; i += ItemStride()) {
    out[i] = __float2half(
        static_cast<float>(Quantize(ToFloat(q[i]), scale, -128.0f, 127.0f)) *
        scale);
  }
}

template <typename T>
__global__ void QuantizeRowsKernel(const T* __restrict__ k,
                                   const T* __restrict__ v, bool v_token_major,
                                   int heads, int tokens, int depth,
                                   float k_scale, float v_scale,
                                   int8_t* __restrict__ k_out,
                                   int8_t* __restrict__ v_out) {
  const size_t count = static_cast<size_t>(heads) * tokens * depth;
  for (size_t i = FirstItem(); i < 2 * count; i += ItemStride()) {
    if (i < count) {
      k_out[i] = static_cast<int8_t>(
          Quantize(ToFloat(k[i]), k_scale, -128.0f, 127.0f));
      continue;
    }
    const size_t e = i - count;
    const size_t row = e / depth;
    const size_t from = RowIndex(v_token_major, heads, tokens, depth,
                                 row / tokens, row % tokens, e % depth);
    v_out[e] = static_cast<int8_t>(
        Quantize(ToFloat(v[from]), v_scale, -128.0f, 127.0f));
  }
}

// A thread per byte the update writes: the bytes of the tokens' key rows,
// then those of every value row. The INT4 values of a value row are cache_len
// nibbles that need not start or end on a byte, so a thread rebuilds its byte
// from the values the cache keeps and the new ones.
template <typename T, int kBits>
__global__ void QuantCacheWriteKernel(int8_t* __restrict__ k_cache,
                                      int8_t* __restrict__ v_cache,
                                      const T* __restrict__ k,
                                      const T* __restrict__ v,
                                      bool v_token_major,
                                      const int32_t* __restrict__ k_position,
                                      const int32_t* __restrict__ v_position,
                                      float k_scale, float v_scale, int heads,
                                      int cache_len, int depth, int tokens) {
  constexpr float kLo = kBits == 8 ? -128.0f : -8.0f;
  constexpr float kHi = kBits == 8 ? 127.0f : 7.0f;
  const int last = cache_len - tokens;
  const int k_pos = min(max(k_position[0], 0), last);
  const int v_pos = min(max(v_position[0], 0), last);
  const size_t row_bytes = static_cast<size_t>(depth) * kBits / 8;
  const size_t key_items = static_cast<size_t>(heads) * tokens * row_bytes;
  const size_t row_items = kBits == 8 ? tokens : tokens / 2 + 1;
  const size_t value_rows = static_cast<size_t>(heads) * depth;
  const size_t items = key_items + value_rows * row_items;
  for (size_t i = FirstItem(); i < items; i += ItemStride()) {
    if (i < key_items) {
      const size_t row = i / row_bytes;  // head * tokens + token
      const size_t to =
          ((row / tokens) * cache_len + k_pos + row % tokens) * row_bytes +
          i % row_bytes;
      if constexpr (kBits == 8) {
        k_cache[to] =
            static_cast<int8_t>(Quantize(ToFloat(k[i]), k_scale, kLo, kHi));
      } else {
        const size_t from = row * depth + 2 * (i % row_bytes);
        const int lo = Quantize(ToFloat(k[from]), k_scale, kLo, kHi);
        const int hi = Quantize(ToFloat(k[from + 1]), k_scale, kLo, kHi);
        k_cache[to] = static_cast<int8_t>((lo & 0xf) | (hi << 4));
      }
      continue;
    }
    const size_t e = i - key_items;
    const size_t value_row = e / row_items;  // head * depth + dim
    const size_t first = value_row * cache_len + v_pos;
    if constexpr (kBits == 8) {
      const size_t t = e % row_items;
      v_cache[first + t] = static_cast<int8_t>(
          Quantize(ToFloat(v[RowIndex(v_token_major, heads, tokens, depth,
                                      value_row / depth, t,
                                      value_row % depth)]),
                   v_scale, kLo, kHi));
    } else {
      const size_t byte = (first >> 1) + e % row_items;
      if (byte > ((first + tokens - 1) >> 1)) {
        continue;
      }
      const int old = v_cache[byte];
      int nibbles[2] = {old & 0xf, (old >> 4) & 0xf};
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        // The row and the position of this value; a byte two rows share is
        // rebuilt the same way by both.
        const size_t at = 2 * byte + n;
        const size_t at_row = at / cache_len;
        const int position = static_cast<int>(at - at_row * cache_len);
        if (at_row < value_rows && position >= v_pos &&
            position < v_pos + tokens) {
          nibbles[n] =
              Quantize(ToFloat(v[RowIndex(v_token_major, heads, tokens, depth,
                                          at_row / depth, position - v_pos,
                                          at_row % depth)]),
                       v_scale, kLo, kHi) &
              0xf;
        }
      }
      v_cache[byte] = static_cast<int8_t>(nibbles[0] | (nibbles[1] << 4));
    }
  }
}

// The token of the chunk that a slot of the ring takes, negative when the
// slot keeps its entry: the decomposition's n - 1 - floor_mod(end - 1 - slot,
// cache_len).
__device__ __forceinline__ int SlotToken(int slot, int end, int n,
                                         int cache_len) {
  int back = (end - 1 - slot) % cache_len;
  if (back < 0) {
    back += cache_len;
  }
  return n - 1 - back;
}

// A thread per 16 bytes of a key row, then per byte of the value columns.
__global__ void QuantRingUpdateKernel(int8_t* __restrict__ k_cache,
                                      int8_t* __restrict__ v_cache,
                                      const int8_t* __restrict__ k_new,
                                      const int8_t* __restrict__ v_new,
                                      const int32_t* __restrict__ params,
                                      int heads, int cache_len, int depth,
                                      int tokens) {
  const int start = params[0];
  const int n = min(max(params[1] - start, 0), tokens);
  const int end = start + n;
  const size_t row_vectors = depth / 16;
  const size_t key_items = static_cast<size_t>(heads) * cache_len * row_vectors;
  const size_t value_items = static_cast<size_t>(heads) * depth * cache_len;
  for (size_t i = FirstItem(); i < key_items + value_items; i += ItemStride()) {
    if (i < key_items) {
      const size_t row = i / row_vectors;  // head * cache_len + slot
      const int token = SlotToken(static_cast<int>(row % cache_len), end, n,
                                  cache_len);
      if (token >= 0) {
        reinterpret_cast<int4*>(k_cache)[i] = reinterpret_cast<const int4*>(
            k_new)[((row / cache_len) * tokens + token) * row_vectors +
                   i % row_vectors];
      }
      continue;
    }
    const size_t e = i - key_items;
    const size_t value_row = e / cache_len;  // head * depth + dim
    const int token =
        SlotToken(static_cast<int>(e % cache_len), end, n, cache_len);
    if (token >= 0) {
      v_cache[e] = v_new[((value_row / depth) * tokens + token) * depth +
                         value_row % depth];
    }
  }
}

__global__ void HalfToBf16Kernel(const __half* __restrict__ in, size_t count,
                                 __nv_bfloat16* __restrict__ out) {
  for (size_t i = FirstItem(); i < count; i += ItemStride()) {
    out[i] = __float2bfloat16(__half2float(in[i]));
  }
}

bool ValidScale(float scale) { return std::isfinite(scale) && scale > 0.0f; }

template <typename T>
cudaError_t LaunchWrite(int8_t* k_cache, int8_t* v_cache, const T* k,
                        const T* v, bool v_token_major,
                        const int32_t* k_position, const int32_t* v_position,
                        int32_t cache_bits, float k_scale, float v_scale,
                        int32_t heads, int32_t cache_len, int32_t depth,
                        int32_t tokens, cudaStream_t stream) {
  const size_t row_bytes = static_cast<size_t>(depth) * cache_bits / 8;
  const size_t items =
      static_cast<size_t>(heads) * tokens * row_bytes +
      static_cast<size_t>(heads) * depth *
          (cache_bits == 8 ? tokens : tokens / 2 + 1);
  if (cache_bits == 8) {
    QuantCacheWriteKernel<T, 8><<<Blocks(items), kThreads, 0, stream>>>(
        k_cache, v_cache, k, v, v_token_major, k_position, v_position, k_scale,
        v_scale, heads, cache_len, depth, tokens);
  } else {
    QuantCacheWriteKernel<T, 4><<<Blocks(items), kThreads, 0, stream>>>(
        k_cache, v_cache, k, v, v_token_major, k_position, v_position, k_scale,
        v_scale, heads, cache_len, depth, tokens);
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchQuantizeQuery(const void* q,
                                                       bool q_bf16,
                                                       size_t count,
                                                       float scale, void* out,
                                                       cudaStream_t stream) {
  if (q == nullptr || out == nullptr || count == 0 || !ValidScale(scale)) {
    return cudaErrorInvalidValue;
  }
  if (q_bf16) {
    QuantizeQueryKernel<__nv_bfloat16>
        <<<Blocks(count), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q), count, scale,
            static_cast<__half*>(out));
  } else {
    QuantizeQueryKernel<__half><<<Blocks(count), kThreads, 0, stream>>>(
        static_cast<const __half*>(q), count, scale, static_cast<__half*>(out));
  }
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantizeRows(
    const void* k, const void* v, bool bf16, bool v_token_major, int32_t heads,
    int32_t tokens, int32_t depth, float k_scale, float v_scale, void* k_out,
    void* v_out, cudaStream_t stream) {
  if (k == nullptr || v == nullptr || k_out == nullptr || v_out == nullptr ||
      heads <= 0 || tokens <= 0 || depth <= 0 || !ValidScale(k_scale) ||
      !ValidScale(v_scale)) {
    return cudaErrorInvalidValue;
  }
  const size_t items = 2 * static_cast<size_t>(heads) * tokens * depth;
  if (bf16) {
    QuantizeRowsKernel<__nv_bfloat16><<<Blocks(items), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), v_token_major, heads, tokens,
        depth, k_scale, v_scale, static_cast<int8_t*>(k_out),
        static_cast<int8_t*>(v_out));
  } else {
    QuantizeRowsKernel<__half><<<Blocks(items), kThreads, 0, stream>>>(
        static_cast<const __half*>(k), static_cast<const __half*>(v),
        v_token_major, heads, tokens, depth, k_scale, v_scale,
        static_cast<int8_t*>(k_out), static_cast<int8_t*>(v_out));
  }
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantCacheWrite(
    void* k_cache, void* v_cache, const void* k, const void* v, bool bf16,
    bool v_token_major, const int32_t* k_position, const int32_t* v_position,
    int32_t cache_bits, float k_scale, float v_scale, int32_t heads,
    int32_t cache_len, int32_t depth, int32_t tokens, cudaStream_t stream) {
  if (k_cache == nullptr || v_cache == nullptr || k == nullptr ||
      v == nullptr || k_position == nullptr || v_position == nullptr ||
      (cache_bits != 8 && cache_bits != 4) || heads <= 0 || cache_len <= 0 ||
      depth <= 0 || depth % 2 != 0 || tokens <= 0 || tokens > cache_len ||
      !ValidScale(k_scale) || !ValidScale(v_scale)) {
    return cudaErrorInvalidValue;
  }
  if (bf16) {
    return LaunchWrite(static_cast<int8_t*>(k_cache),
                       static_cast<int8_t*>(v_cache),
                       static_cast<const __nv_bfloat16*>(k),
                       static_cast<const __nv_bfloat16*>(v), v_token_major,
                       k_position, v_position, cache_bits, k_scale, v_scale,
                       heads, cache_len, depth, tokens, stream);
  }
  return LaunchWrite(static_cast<int8_t*>(k_cache),
                     static_cast<int8_t*>(v_cache),
                     static_cast<const __half*>(k),
                     static_cast<const __half*>(v), v_token_major, k_position,
                     v_position, cache_bits, k_scale, v_scale, heads,
                     cache_len, depth, tokens, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantRingUpdate(
    void* k_cache, void* v_cache, const void* k_new, const void* v_new,
    const int32_t* params, int32_t heads, int32_t cache_len, int32_t depth,
    int32_t tokens, cudaStream_t stream) {
  if (k_cache == nullptr || v_cache == nullptr || k_new == nullptr ||
      v_new == nullptr || params == nullptr || heads <= 0 || cache_len <= 0 ||
      depth <= 0 || depth % 16 != 0 || tokens <= 0) {
    return cudaErrorInvalidValue;
  }
  const size_t items = static_cast<size_t>(heads) * cache_len * (depth / 16) +
                       static_cast<size_t>(heads) * depth * cache_len;
  QuantRingUpdateKernel<<<Blocks(items), kThreads, 0, stream>>>(
      static_cast<int8_t*>(k_cache), static_cast<int8_t*>(v_cache),
      static_cast<const int8_t*>(k_new), static_cast<const int8_t*>(v_new),
      params, heads, cache_len, depth, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchHalfToBf16(const void* in,
                                                    size_t count, void* out,
                                                    cudaStream_t stream) {
  if (in == nullptr || out == nullptr || count == 0) {
    return cudaErrorInvalidValue;
  }
  HalfToBf16Kernel<<<Blocks(count), kThreads, 0, stream>>>(
      static_cast<const __half*>(in), count,
      static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}
