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

#include "litert/vendors/nvidia/trtllm/cache_write_slots.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

struct WriteTable {
  LiteRtNvidiaCacheWrite write[kLiteRtNvidiaCacheWriteMaxWrites];
};

template <typename T>
__device__ __forceinline__ float LoadFloat(const T* p, int64_t i);
template <>
__device__ __forceinline__ float LoadFloat(const __half* p, int64_t i) {
  return __half2float(p[i]);
}
template <>
__device__ __forceinline__ float LoadFloat(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
template <>
__device__ __forceinline__ float LoadFloat(const float* p, int64_t i) {
  return p[i];
}

// The FP16 bits of element i: an FP16 source keeps its bits (a NaN's too).
template <typename T>
__device__ __forceinline__ __half LoadHalf(const T* p, int64_t i) {
  return __float2half(LoadFloat(p, i));
}
template <>
__device__ __forceinline__ __half LoadHalf(const __half* p, int64_t i) {
  return p[i];
}

// Int8CacheWriteKernel's quantization (compiler/int8_cache_write_kernel.cu),
// kept here as a copy so that kernel's object code stays what it is; the
// plugin's test compares the two byte for byte.
__device__ __forceinline__ int8_t Quantize(float x, float scale,
                                           int32_t zero_point) {
  const float lo = -128.0f * scale;
  const float hi = 127.0f * scale;
  x = fminf(fmaxf(x, lo), hi);
  int32_t q = __float2int_rn(x / scale) + zero_point;
  q = q < -128 ? -128 : q > 127 ? 127 : q;
  return static_cast<int8_t>(q);
}

// grid (blocks, writes): one thread per written element of one write. Key
// rows: the depth index runs fastest, so reads and writes are contiguous.
// Columns: the token index runs fastest, so the writes along the cache's
// last axis are.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    CacheWritesKernel(WriteTable table, int32_t heads, int32_t cache_len,
                      int32_t depth, int32_t tokens) {
  const LiteRtNvidiaCacheWrite& write = table.write[blockIdx.y];
  const T* update = static_cast<const T*>(write.update);
  int32_t pos = write.position[0];
  const int32_t last = cache_len - tokens;
  pos = pos < 0 ? 0 : pos > last ? last : pos;
  const bool columns = write.columns != 0;
  const int64_t elements = static_cast<int64_t>(heads) * tokens * depth;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int32_t h, t, d;
    if (columns) {
      t = static_cast<int32_t>(i % tokens);
      const int64_t rest = i / tokens;
      d = static_cast<int32_t>(rest % depth);
      h = static_cast<int32_t>(rest / depth);
    } else {
      d = static_cast<int32_t>(i % depth);
      const int64_t rest = i / depth;
      t = static_cast<int32_t>(rest % tokens);
      h = static_cast<int32_t>(rest / tokens);
    }
    const int64_t from = (static_cast<int64_t>(h) * tokens + t) * depth + d;
    const int64_t to =
        columns ? (static_cast<int64_t>(h) * depth + d) * cache_len + pos + t
                : (static_cast<int64_t>(h) * cache_len + pos + t) * depth + d;
    if (write.int8 != 0) {
      static_cast<int8_t*>(write.cache)[to] =
          Quantize(LoadFloat(update, from), write.scale, write.zero_point);
    } else {
      static_cast<__half*>(write.cache)[to] = LoadHalf(update, from);
    }
  }
}

template <typename T>
cudaError_t Launch(const WriteTable& table, int32_t num_writes, int32_t heads,
                   int32_t cache_len, int32_t depth, int32_t tokens,
                   cudaStream_t stream) {
  const int64_t elements = static_cast<int64_t>(heads) * tokens * depth;
  int64_t blocks = (elements + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  CacheWritesKernel<T><<<dim3(static_cast<unsigned>(blocks), num_writes),
                         kThreads, 0, stream>>>(table, heads, cache_len, depth,
                                                tokens);
  return cudaPeekAtLastError();
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchCacheWrites(
    const LiteRtNvidiaCacheWrite* writes, int32_t num_writes, int32_t source,
    int32_t heads, int32_t cache_len, int32_t depth, int32_t tokens,
    cudaStream_t stream) {
  if (writes == nullptr || num_writes < 1 ||
      num_writes > kLiteRtNvidiaCacheWriteMaxWrites || heads <= 0 ||
      cache_len <= 0 || depth <= 0 || tokens <= 0 || tokens > cache_len) {
    return cudaErrorInvalidValue;
  }
  WriteTable table = {};
  for (int32_t w = 0; w < num_writes; ++w) {
    if (writes[w].cache == nullptr || writes[w].update == nullptr ||
        writes[w].position == nullptr ||
        (writes[w].int8 != 0 && !(writes[w].scale > 0.0f))) {
      return cudaErrorInvalidValue;
    }
    table.write[w] = writes[w];
  }
  switch (source) {
    case kLiteRtNvidiaCacheWriteHalf:
      return Launch<__half>(table, num_writes, heads, cache_len, depth, tokens,
                            stream);
    case kLiteRtNvidiaCacheWriteBf16:
      return Launch<__nv_bfloat16>(table, num_writes, heads, cache_len, depth,
                                   tokens, stream);
    case kLiteRtNvidiaCacheWriteFloat:
      return Launch<float>(table, num_writes, heads, cache_len, depth, tokens,
                           stream);
    default:
      return cudaErrorInvalidValue;
  }
}
