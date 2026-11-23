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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_COMMON_CUH_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_COMMON_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"

// Device helpers shared by the QSA kernels (qsa_indexer.cu and
// qsa_sparse_attention.cu).
namespace litert::nvidia::qsa {

constexpr int kBlock = 256;

__device__ __forceinline__ float LoadAsFloat(const void* data, int32_t type,
                                             int64_t index) {
  if (type == kLiteRtNvidiaQsaFloat16) {
    return __half2float(static_cast<const __half*>(data)[index]);
  }
  if (type == kLiteRtNvidiaQsaBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
  }
  return static_cast<const float*>(data)[index];
}

// Eight consecutive values of `type` from data[index] (index a multiple of 8
// and the data 16-byte aligned) as FP32, in one or two 16-byte loads.
__device__ __forceinline__ void LoadEightAsFloat(const void* data,
                                                 int32_t type, int64_t index,
                                                 float (&out)[8]) {
  if (type == kLiteRtNvidiaQsaFloat32) {
    const float4* at =
        reinterpret_cast<const float4*>(static_cast<const float*>(data) + index);
    const float4 a = at[0];
    const float4 b = at[1];
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
    out[4] = b.x;
    out[5] = b.y;
    out[6] = b.z;
    out[7] = b.w;
    return;
  }
  const uint4 raw = *reinterpret_cast<const uint4*>(
      static_cast<const uint16_t*>(data) + index);
  const uint16_t* bits = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = type == kLiteRtNvidiaQsaFloat16
                 ? __half2float(__ushort_as_half(bits[i]))
                 : __bfloat162float(__ushort_as_bfloat16(bits[i]));
  }
}

__device__ __forceinline__ void StoreFromFloat(float value, float* out) {
  *out = value;
}
__device__ __forceinline__ void StoreFromFloat(float value, __half* out) {
  *out = __float2half(value);
}
__device__ __forceinline__ void StoreFromFloat(float value,
                                               __nv_bfloat16* out) {
  *out = __float2bfloat16(value);
}

__device__ __forceinline__ float ToFloat(float value) { return value; }
__device__ __forceinline__ float ToFloat(__half value) {
  return __half2float(value);
}
__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float WarpMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Sums over the CTA; every thread gets the total, summed in warp order (the
// same for every launch). `scratch` holds a float per warp.
__device__ __forceinline__ float BlockReduceSum(float value, float* scratch) {
  value = WarpSum(value);
  __syncthreads();
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < static_cast<int>(blockDim.x / 32); ++i) {
    total += scratch[i];
  }
  return total;
}

__device__ __forceinline__ float BlockReduceMax(float value, float* scratch) {
  value = WarpMax(value);
  __syncthreads();
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  float total = scratch[0];
  for (int i = 1; i < static_cast<int>(blockDim.x / 32); ++i) {
    total = fmaxf(total, scratch[i]);
  }
  return total;
}

}  // namespace litert::nvidia::qsa

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QSA_COMMON_CUH_
