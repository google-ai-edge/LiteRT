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

#include "litert/vendors/nvidia/dispatch/greedy_sampler_kernel.h"

#include <algorithm>
#include <climits>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxArgMaxBlocks = 1024;

struct ArgMaxPair {
  float value;
  int32_t index;
};

__device__ ArgMaxPair BetterArgMax(ArgMaxPair lhs, ArgMaxPair rhs) {
  // Preserve the CPU sampler's deterministic NaN behavior.
  if (lhs.index == 0 && isnan(lhs.value)) {
    return lhs;
  }
  if (rhs.index == 0 && isnan(rhs.value)) {
    return rhs;
  }
  if (isnan(lhs.value)) {
    return rhs;
  }
  if (isnan(rhs.value)) {
    return lhs;
  }
  if (rhs.value > lhs.value ||
      (rhs.value == lhs.value && rhs.index < lhs.index)) {
    return rhs;
  }
  return lhs;
}

__global__ void F32ArgMaxPartialsKernel(const float* input, size_t count,
                                        ArgMaxPair* partials) {
  __shared__ ArgMaxPair shared[kThreads];
  ArgMaxPair best{-INFINITY, INT_MAX};
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    best = BetterArgMax(best, {input[index], static_cast<int32_t>(index)});
  }
  shared[threadIdx.x] = best;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          BetterArgMax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void F32ArgMaxFinalKernel(const ArgMaxPair* partials,
                                     int num_partials,
                                     int32_t* device_result) {
  __shared__ ArgMaxPair shared[kThreads];
  ArgMaxPair best{-INFINITY, INT_MAX};
  for (int index = threadIdx.x; index < num_partials; index += blockDim.x) {
    best = BetterArgMax(best, partials[index]);
  }
  shared[threadIdx.x] = best;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          BetterArgMax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *device_result = shared[0].index;
  }
}

int ArgMaxBlockCount(size_t count) {
  return static_cast<int>(std::min((count + kThreads - 1) / kThreads,
                                   static_cast<size_t>(kMaxArgMaxBlocks)));
}

}  // namespace

extern "C" size_t LiteRtNvidiaF32ArgMaxWorkspaceBytes(size_t count) {
  if (count == 0 || count > static_cast<size_t>(INT_MAX)) {
    return 0;
  }
  return static_cast<size_t>(ArgMaxBlockCount(count)) * sizeof(ArgMaxPair);
}

extern "C" cudaError_t LiteRtNvidiaLaunchF32ArgMax(
    const float* input, size_t count, void* workspace, size_t workspace_bytes,
    int32_t* device_result, cudaStream_t stream) {
  const size_t required = LiteRtNvidiaF32ArgMaxWorkspaceBytes(count);
  if (input == nullptr || workspace == nullptr || device_result == nullptr ||
      required == 0 || workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }
  const int blocks = ArgMaxBlockCount(count);
  auto* partials = static_cast<ArgMaxPair*>(workspace);
  F32ArgMaxPartialsKernel<<<blocks, kThreads, 0, stream>>>(input, count,
                                                           partials);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  F32ArgMaxFinalKernel<<<1, kThreads, 0, stream>>>(partials, blocks,
                                                   device_result);
  return cudaPeekAtLastError();
}
