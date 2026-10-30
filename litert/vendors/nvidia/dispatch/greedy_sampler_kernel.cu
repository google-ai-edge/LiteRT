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

#include "cuda_fp16.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxArgMaxBlocks = 1024;

struct ArgMaxPair {
  float value;
  int32_t index;
};

__device__ ArgMaxPair BetterArgMax(ArgMaxPair lhs, ArgMaxPair rhs) {
  // Preserve the original single-row reduction's deterministic NaN behavior.
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

__device__ float ReadLogit(float value) { return value; }
__device__ float ReadLogit(__half value) { return __half2float(value); }

// Row blockIdx.y of `input` ([rows, count]) reduces into its own
// gridDim.x partials.
template <typename T>
__global__ void ArgMaxPartialsKernel(const T* input, size_t count,
                                     ArgMaxPair* partials) {
  input += static_cast<size_t>(blockIdx.y) * count;
  partials += static_cast<size_t>(blockIdx.y) * gridDim.x;
  __shared__ ArgMaxPair shared[kThreads];
  ArgMaxPair best{-INFINITY, INT_MAX};
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    best = BetterArgMax(best,
                         {ReadLogit(input[index]), static_cast<int32_t>(index)});
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

// Block blockIdx.x merges row blockIdx.x's partials into device_result[row].
__global__ void F32ArgMaxFinalKernel(const ArgMaxPair* partials,
                                     int num_partials,
                                     int32_t* device_result) {
  partials += static_cast<size_t>(blockIdx.x) * num_partials;
  device_result += blockIdx.x;
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

extern "C" size_t LiteRtNvidiaArgMaxRowsWorkspaceBytes(size_t rows,
                                                        size_t count) {
  if (rows == 0 || rows > static_cast<size_t>(INT_MAX) || count == 0 ||
      count > static_cast<size_t>(INT_MAX)) {
    return 0;
  }
  return rows * static_cast<size_t>(ArgMaxBlockCount(count)) *
         sizeof(ArgMaxPair);
}

extern "C" size_t LiteRtNvidiaF32ArgMaxWorkspaceBytes(size_t count) {
  return LiteRtNvidiaArgMaxRowsWorkspaceBytes(1, count);
}

namespace {

// `rows` rows of `count` logits, one launch pair: the partials kernel's grid is
// (blocks, rows), the final kernel's (rows).
template <typename T>
cudaError_t LaunchArgMax(const T* input, size_t rows, size_t count,
                         void* workspace, size_t workspace_bytes,
                         int32_t* device_result, cudaStream_t stream) {
  const size_t required = LiteRtNvidiaArgMaxRowsWorkspaceBytes(rows, count);
  if (input == nullptr || workspace == nullptr || device_result == nullptr ||
      required == 0 || workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }
  const int blocks = ArgMaxBlockCount(count);
  auto* partials = static_cast<ArgMaxPair*>(workspace);
  ArgMaxPartialsKernel<<<dim3(blocks, static_cast<unsigned>(rows)), kThreads,
                         0, stream>>>(input, count, partials);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  F32ArgMaxFinalKernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      partials, blocks, device_result);
  return cudaPeekAtLastError();
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchF32ArgMax(
    const float* input, size_t count, void* workspace, size_t workspace_bytes,
    int32_t* device_result, cudaStream_t stream) {
  return LaunchArgMax(input, 1, count, workspace, workspace_bytes,
                      device_result, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchF16ArgMax(
    const uint16_t* input, size_t count, void* workspace, size_t workspace_bytes,
    int32_t* device_result, cudaStream_t stream) {
  return LaunchArgMax(reinterpret_cast<const __half*>(input), 1, count,
                      workspace, workspace_bytes, device_result, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchF32ArgMaxRows(
    const float* input, size_t rows, size_t count, void* workspace,
    size_t workspace_bytes, int32_t* device_result, cudaStream_t stream) {
  return LaunchArgMax(input, rows, count, workspace, workspace_bytes,
                      device_result, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchF16ArgMaxRows(
    const uint16_t* input, size_t rows, size_t count, void* workspace,
    size_t workspace_bytes, int32_t* device_result, cudaStream_t stream) {
  return LaunchArgMax(reinterpret_cast<const __half*>(input), rows, count,
                      workspace, workspace_bytes, device_result, stream);
}
