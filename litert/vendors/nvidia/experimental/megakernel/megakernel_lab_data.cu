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

#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lab_data.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 4096;

// A 64-bit mix of the element's index and the fill's seed (splitmix64).
__device__ __forceinline__ uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Uniform in [0, 1) from 24 bits of the mix.
__device__ __forceinline__ float Unit(uint64_t bits) {
  return static_cast<float>(bits >> 40) * (1.0f / 16777216.0f);
}

__global__ void FillBytesKernel(uint8_t* dst, size_t bytes, uint64_t seed,
                                int lo, int span) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
       i < bytes; i += static_cast<size_t>(gridDim.x) * kThreads) {
    const uint64_t bits = Mix(seed ^ (i * 0x632be59bd9b4e019ull));
    dst[i] = static_cast<uint8_t>(span >= 256 ? bits & 0xff
                                              : lo + (bits >> 33) % span);
  }
}

__global__ void FillHalfKernel(__half* dst, size_t count, uint64_t seed,
                               float lo, float hi) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * kThreads) {
    const float u = Unit(Mix(seed ^ (i * 0x632be59bd9b4e019ull)));
    dst[i] = __float2half_rn(lo + (hi - lo) * u);
  }
}

__global__ void FillFloatKernel(float* dst, size_t count, uint64_t seed,
                                float lo, float hi) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * kThreads) {
    const float u = Unit(Mix(seed ^ (i * 0x632be59bd9b4e019ull)));
    dst[i] = lo + (hi - lo) * u;
  }
}

int Grid(size_t count) {
  const size_t blocks = (count + kThreads - 1) / kThreads;
  return blocks < kBlocks ? static_cast<int>(blocks < 1 ? 1 : blocks) : kBlocks;
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaMegakernelLabFillBytes(void* dst,
                                                          size_t bytes,
                                                          uint64_t seed, int lo,
                                                          int hi,
                                                          cudaStream_t stream) {
  if (dst == nullptr || lo < 0 || hi > 255 || hi < lo) {
    return cudaErrorInvalidValue;
  }
  FillBytesKernel<<<Grid(bytes), kThreads, 0, stream>>>(
      static_cast<uint8_t*>(dst), bytes, seed, lo, hi - lo + 1);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelLabFillHalf(void* dst,
                                                         size_t count,
                                                         uint64_t seed,
                                                         float lo, float hi,
                                                         cudaStream_t stream) {
  if (dst == nullptr) return cudaErrorInvalidValue;
  FillHalfKernel<<<Grid(count), kThreads, 0, stream>>>(
      static_cast<__half*>(dst), count, seed, lo, hi);
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaMegakernelLabFillFloat(float* dst,
                                                          size_t count,
                                                          uint64_t seed,
                                                          float lo, float hi,
                                                          cudaStream_t stream) {
  if (dst == nullptr) return cudaErrorInvalidValue;
  FillFloatKernel<<<Grid(count), kThreads, 0, stream>>>(dst, count, seed, lo,
                                                        hi);
  return cudaGetLastError();
}
