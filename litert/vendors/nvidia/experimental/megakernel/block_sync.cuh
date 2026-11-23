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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_BLOCK_SYNC_CUH_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_BLOCK_SYNC_CUH_

// How the threads that run a block's work of the megakernel's attention and
// norm-rope device code (decode_attention.cuh, qkv_norm_rope.cuh) wait for
// each other: a kernel's whole block, or the group of a persistent kernel's
// CTA that runs the work beside warps doing something else (the megakernel's
// weight stream).

namespace litert::nvidia::megakernel {

// The whole CTA: what a kernel of one block per item uses.
struct CtaSync {
  __device__ __forceinline__ static void Sync() { __syncthreads(); }
  __device__ __forceinline__ static int SyncOr(int predicate) {
    return __syncthreads_or(predicate);
  }
};

// Named barrier kId of the first kThreads threads (whole warps) of the CTA.
template <int kId, int kThreads>
struct NamedSync {
  __device__ __forceinline__ static void Sync() {
    asm volatile("bar.sync %0, %1;\n" ::"n"(kId), "n"(kThreads) : "memory");
  }
  __device__ __forceinline__ static int SyncOr(int predicate) {
    int result;
    asm volatile(
        "{\n"
        ".reg .pred p, q;\n"
        "setp.ne.s32 p, %1, 0;\n"
        "bar.red.or.pred q, %2, %3, p;\n"
        "selp.s32 %0, 1, 0, q;\n"
        "}\n"
        : "=r"(result)
        : "r"(predicate), "n"(kId), "n"(kThreads)
        : "memory");
    return result;
  }
};

}  // namespace litert::nvidia::megakernel

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_BLOCK_SYNC_CUH_
