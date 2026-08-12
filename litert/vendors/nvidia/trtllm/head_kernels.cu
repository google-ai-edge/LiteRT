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

#include "litert/vendors/nvidia/trtllm/head_kernels.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

__global__ void F32ToBf16Kernel(const float* input, __nv_bfloat16* output,
                                size_t count) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
}

__global__ void Bf16SoftCapToF32Kernel(const __nv_bfloat16* input,
                                       float* output, size_t count,
                                       float inverse_soft_cap, float soft_cap) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const float logit = __bfloat162float(input[index]);
    output[index] = tanhf(logit * inverse_soft_cap) * soft_cap;
  }
}

int BlockCount(size_t count) {
  return static_cast<int>(std::min((count + kThreads - 1) / kThreads,
                                   static_cast<size_t>(kMaxBlocks)));
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchF32ToBf16(const float* input,
                                                   void* output, size_t count,
                                                   cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  if (input == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  F32ToBf16Kernel<<<BlockCount(count), kThreads, 0, stream>>>(
      input, static_cast<__nv_bfloat16*>(output), count);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16SoftCapToF32(
    const void* input, float* output, size_t count, float soft_cap,
    cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  if (input == nullptr || output == nullptr || !std::isfinite(soft_cap) ||
      soft_cap <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  Bf16SoftCapToF32Kernel<<<BlockCount(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), output, count,
      1.0f / soft_cap, soft_cap);
  return cudaPeekAtLastError();
}
