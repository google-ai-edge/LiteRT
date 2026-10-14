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

#include "litert/vendors/nvidia/compiler/int8_cache_write_kernel.h"

#include <cstdint>

#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"

namespace litert::nvidia {
namespace {

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

__device__ __forceinline__ int8_t Quantize(float x, float scale,
                                           int32_t zero_point) {
  const float lo = -128.0f * scale;
  const float hi = 127.0f * scale;
  x = fminf(fmaxf(x, lo), hi);
  int32_t q = __float2int_rn(x / scale) + zero_point;
  q = q < -128 ? -128 : q > 127 ? 127 : q;
  return static_cast<int8_t>(q);
}

// One thread per written element. Key rows: the depth index runs fastest,
// so reads and writes are contiguous. Value columns: the token index runs
// fastest, so the writes along the cache's last axis are contiguous.
template <typename T, bool kColumns>
__global__ void Int8CacheWriteKernel(int8_t* __restrict__ cache,
                                     const T* __restrict__ update,
                                     const int32_t* __restrict__ position,
                                     int32_t heads, int32_t cache_len,
                                     int32_t depth, int32_t tokens,
                                     float scale, int32_t zero_point) {
  int32_t pos = position[0];
  const int32_t last = cache_len - tokens;
  pos = pos < 0 ? 0 : pos > last ? last : pos;
  const int64_t elements =
      static_cast<int64_t>(heads) * tokens * depth;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int32_t h, t, d;
    if (kColumns) {
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
    const float x =
        LoadFloat(update, (static_cast<int64_t>(h) * tokens + t) * depth + d);
    const int8_t q = Quantize(x, scale, zero_point);
    const int64_t index =
        kColumns ? (static_cast<int64_t>(h) * depth + d) * cache_len + pos + t
                 : (static_cast<int64_t>(h) * cache_len + pos + t) * depth + d;
    cache[index] = q;
  }
}

template <typename T>
cudaError_t Launch(int8_t* cache, const T* update, const int32_t* position,
                   int32_t heads, int32_t cache_len, int32_t depth,
                   int32_t tokens, bool value_columns, float scale,
                   int32_t zero_point, cudaStream_t stream) {
  const int64_t elements = static_cast<int64_t>(heads) * tokens * depth;
  const int threads = 256;
  int64_t blocks = (elements + threads - 1) / threads;
  if (blocks > 4096) blocks = 4096;
  if (value_columns) {
    Int8CacheWriteKernel<T, true><<<static_cast<int>(blocks), threads, 0,
                                    stream>>>(cache, update, position, heads,
                                              cache_len, depth, tokens, scale,
                                              zero_point);
  } else {
    Int8CacheWriteKernel<T, false><<<static_cast<int>(blocks), threads, 0,
                                     stream>>>(cache, update, position, heads,
                                               cache_len, depth, tokens, scale,
                                               zero_point);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchInt8CacheWrite(void* cache_out, const void* cache_in,
                                 size_t cache_bytes, const void* update,
                                 Int8CacheWriteSource source,
                                 const int32_t* position, int32_t heads,
                                 int32_t cache_len, int32_t depth,
                                 int32_t tokens, bool value_columns,
                                 float scale, int32_t zero_point,
                                 cudaStream_t stream) {
  if (cache_out == nullptr || cache_in == nullptr || update == nullptr ||
      position == nullptr || heads <= 0 || cache_len <= 0 || depth <= 0 ||
      tokens <= 0 || tokens > cache_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (cache_out != cache_in) {
    const cudaError_t copied = cudaMemcpyAsync(
        cache_out, cache_in, cache_bytes, cudaMemcpyDeviceToDevice, stream);
    if (copied != cudaSuccess) return copied;
  }
  auto* cache = static_cast<int8_t*>(cache_out);
  switch (source) {
    case Int8CacheWriteSource::kHalf:
      return Launch(cache, static_cast<const __half*>(update), position, heads,
                    cache_len, depth, tokens, value_columns, scale, zero_point,
                    stream);
    case Int8CacheWriteSource::kBf16:
      return Launch(cache, static_cast<const __nv_bfloat16*>(update), position,
                    heads, cache_len, depth, tokens, value_columns, scale,
                    zero_point, stream);
    case Int8CacheWriteSource::kFloat:
      return Launch(cache, static_cast<const float*>(update), position, heads,
                    cache_len, depth, tokens, value_columns, scale, zero_point,
                    stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace litert::nvidia
