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

#include "litert/vendors/nvidia/experimental/megakernel/megakernel_steps.h"

#include <cstdint>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

namespace {

constexpr int kThreads = 256;
constexpr int kNormThreads = 1024;

__device__ __forceinline__ float Bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__global__ void ToBf16Kernel(const float* in, __nv_bfloat16* out,
                             int32_t count) {
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    out[i] = __float2bfloat16_rn(in[i]);
  }
}

__global__ void RepeatKeysKernel(const uint16_t* qkv, uint16_t* out,
                                 int32_t q_width, int32_t k_width) {
  const int32_t total = q_width + 2 * k_width;
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    out[i] = i < q_width + k_width ? qkv[i] : qkv[i - k_width];
  }
}

__device__ __forceinline__ int32_t CacheRow(int32_t position, int32_t len,
                                            int32_t ring) {
  if (ring != 0) return ((position % len) + len) % len;
  return position < 0 ? 0 : (position >= len ? len - 1 : position);
}

// grid (heads, 2): the key (0) or value (1) row of one head.
__global__ void WriteCacheRowsKernel(const __nv_bfloat16* k,
                                     const __nv_bfloat16* v, __half* k_cache,
                                     __half* v_cache, const int32_t* position,
                                     int32_t len, int32_t depth, int32_t ring) {
  const int32_t row = CacheRow(*position, len, ring);
  const int64_t head = blockIdx.x;
  const __nv_bfloat16* source = (blockIdx.y == 0 ? k : v) + head * depth;
  __half* target = (blockIdx.y == 0 ? k_cache : v_cache) +
                   (head * len + row) * static_cast<int64_t>(depth);
  for (int32_t d = threadIdx.x; d < depth; d += blockDim.x) {
    target[d] = __float2half_rn(__bfloat162float(source[d]));
  }
}

// The sum of `value` over the block, in every thread.
__device__ float BlockSum(float value) {
  __shared__ float partial[kNormThreads / 32];
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = value;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < blockDim.x / 32; ++w) total += partial[w];
  __syncthreads();
  return total;
}

// The graph's rms_norm of `x` [dim] as its layers compute it, every one a
// BF16 tensor: r = 1 / sqrt(mean(x * x) + eps), with eps and the scale cast
// to BF16, then x * r * scale. Returns r.
__device__ float NormReciprocal(const __nv_bfloat16* x, float eps,
                                int32_t dim) {
  float sum = 0.0f;
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float value = __bfloat162float(x[i]);
    sum += Bf16(value * value);
  }
  const float mean = Bf16(BlockSum(sum) / static_cast<float>(dim));
  return Bf16(1.0f / Bf16(sqrtf(Bf16(mean + Bf16(eps)))));
}

// The norm's element i, BF16.
__device__ float NormAt(const __nv_bfloat16* x, float reciprocal,
                        const float* scale, int32_t i) {
  return Bf16(Bf16(__bfloat162float(x[i]) * reciprocal) * Bf16(scale[i]));
}

__global__ void NormResidualKernel(const __nv_bfloat16* o,
                                   const __nv_bfloat16* x, const float* scale,
                                   float eps, __nv_bfloat16* h, int32_t dim) {
  const float reciprocal = NormReciprocal(o, eps, dim);
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    h[i] = __float2bfloat16_rn(__bfloat162float(x[i]) +
                               NormAt(o, reciprocal, scale, i));
  }
}

__global__ void NormToFloatKernel(const __nv_bfloat16* x, const float* scale,
                                  float eps, float* out, int32_t dim) {
  const float reciprocal = NormReciprocal(x, eps, dim);
  for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = NormAt(x, reciprocal, scale, i);
  }
}

// The graph's scalars are BF16 constants too.
__global__ void SoftCapKernel(const __nv_bfloat16* z, float scale_in,
                              float scale_out, float* out, int32_t count) {
  const float in = Bf16(scale_in);
  const float cap = Bf16(scale_out);
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    const float scaled = Bf16(__bfloat162float(z[i]) * in);
    out[i] = Bf16(Bf16(tanhf(scaled)) * cap);
  }
}

// One block per layer, cache (key, value) and head: the head's row.
__global__ void RowsKernel(const LiteRtNvidiaMegakernelRowsParams params) {
  const LiteRtNvidiaMegakernelRowsLayer& layer = params.layer[blockIdx.x];
  if (blockIdx.z >= static_cast<unsigned>(layer.heads)) return;
  const int32_t row = CacheRow(*params.position, layer.len, layer.ring);
  if (blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    params.rows[blockIdx.x] = row;
  }
  const __half* cache = static_cast<const __half*>(
      blockIdx.y == 0 ? layer.k_cache : layer.v_cache);
  __half* out =
      static_cast<__half*>(blockIdx.y == 0 ? layer.k_row : layer.v_row);
  const int64_t head = blockIdx.z;
  const __half* source =
      cache + (head * layer.len + row) * static_cast<int64_t>(layer.depth);
  for (int32_t d = threadIdx.x; d < layer.depth; d += blockDim.x) {
    out[head * layer.depth + d] = source[d];
  }
}

int Blocks(int32_t count) {
  const int blocks = (count + kThreads - 1) / kThreads;
  return blocks < 1 ? 1 : (blocks > 1024 ? 1024 : blocks);
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaMegakernelToBf16(const float* in, void* out,
                                                    int32_t count,
                                                    cudaStream_t stream) {
  if (in == nullptr || out == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  ToBf16Kernel<<<Blocks(count), kThreads, 0, stream>>>(
      in, static_cast<__nv_bfloat16*>(out), count);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelRepeatKeys(const void* qkv,
                                                        void* out,
                                                        int32_t q_width,
                                                        int32_t k_width,
                                                        cudaStream_t stream) {
  if (qkv == nullptr || out == nullptr || q_width <= 0 || k_width <= 0) {
    return cudaErrorInvalidValue;
  }
  RepeatKeysKernel<<<Blocks(q_width + 2 * k_width), kThreads, 0, stream>>>(
      static_cast<const uint16_t*>(qkv), static_cast<uint16_t*>(out), q_width,
      k_width);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelWriteCacheRows(
    const void* k, const void* v, void* k_cache, void* v_cache,
    const int32_t* position, int32_t heads, int32_t len, int32_t depth,
    int32_t ring, cudaStream_t stream) {
  if (k == nullptr || v == nullptr || k_cache == nullptr ||
      v_cache == nullptr || position == nullptr || heads <= 0 || len <= 0 ||
      depth <= 0) {
    return cudaErrorInvalidValue;
  }
  WriteCacheRowsKernel<<<dim3(heads, 2), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<__half*>(k_cache),
      static_cast<__half*>(v_cache), position, len, depth, ring);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelNormResidual(
    const void* o, const void* x, const float* scale, float eps, void* h,
    int32_t dim, cudaStream_t stream) {
  if (o == nullptr || x == nullptr || scale == nullptr || h == nullptr ||
      dim <= 0) {
    return cudaErrorInvalidValue;
  }
  NormResidualKernel<<<1, kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(o),
      static_cast<const __nv_bfloat16*>(x), scale, eps,
      static_cast<__nv_bfloat16*>(h), dim);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelNormToFloat(const void* x,
                                                         const float* scale,
                                                         float eps, float* out,
                                                         int32_t dim,
                                                         cudaStream_t stream) {
  if (x == nullptr || scale == nullptr || out == nullptr || dim <= 0) {
    return cudaErrorInvalidValue;
  }
  NormToFloatKernel<<<1, kNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), scale, eps, out, dim);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelSoftCap(const void* z,
                                                     float scale_in,
                                                     float scale_out,
                                                     float* out, int32_t count,
                                                     cudaStream_t stream) {
  if (z == nullptr || out == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  SoftCapKernel<<<Blocks(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(z), scale_in, scale_out, out, count);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchMegakernelRows(
    const LiteRtNvidiaMegakernelRowsParams* params, cudaStream_t stream) {
  if (params == nullptr || params->layers <= 0 ||
      params->layers > kLiteRtNvidiaMegakernelRowsMaxLayers ||
      params->position == nullptr || params->rows == nullptr) {
    return cudaErrorInvalidValue;
  }
  int32_t heads = 0;
  for (int32_t l = 0; l < params->layers; ++l) {
    const auto& layer = params->layer[l];
    if (layer.heads <= 0 || layer.len <= 0 || layer.depth <= 0 ||
        layer.k_cache == nullptr || layer.v_cache == nullptr ||
        layer.k_row == nullptr || layer.v_row == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (layer.heads > heads) heads = layer.heads;
  }
  RowsKernel<<<dim3(params->layers, 2, heads), 128, 0, stream>>>(*params);
  return cudaGetLastError();
}
