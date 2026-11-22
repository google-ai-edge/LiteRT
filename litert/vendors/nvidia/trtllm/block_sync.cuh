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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_BLOCK_SYNC_CUH_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_BLOCK_SYNC_CUH_

// How the threads that run a block's work of the slot kernels' device code
// (decode_attention_blocks.cuh) wait for each other. The kernels here give
// every item a block of its own, whose threads are the whole CTA; the device
// code takes the way of waiting as a parameter so that a kernel whose CTAs do
// other work beside it can pass a barrier of its own threads.

namespace litert::nvidia {

// The whole CTA: what a kernel of one block per item uses.
struct CtaSync {
  __device__ __forceinline__ static void Sync() { __syncthreads(); }
  __device__ __forceinline__ static int SyncOr(int predicate) {
    return __syncthreads_or(predicate);
  }
};

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_BLOCK_SYNC_CUH_
