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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_SLOTS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_SLOTS_H_

#include <cstdint>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"

// LiteRtNvidiaLaunchQkvNormRope (qkv_norm_rope.h) for several sequence slots
// in one launch (a batched decode step: one sequence per slot). A block of
// that launch works on one head of one token alone, so a slot's blocks here
// do what they do there, on the slot's own buffers and positions: each slot's
// q, k and v are that launch's bit for bit.

// The most slots of one launch.
constexpr int kLiteRtNvidiaQkvNormRopeMaxSlots = 8;

// One slot's buffers (device pointers), as the single-sequence launch takes
// them: qkv BF16 [tokens, (heads + 2 kv_heads) * depth], the tokens'
// positions, and the outputs q [heads, tokens, depth], k and v [kv_heads,
// tokens, depth].
struct LiteRtNvidiaQkvNormRopeSlot {
  const void* qkv;
  const int32_t* positions;
  void* q;
  void* k;
  void* v;
};

// `slots` points to `num_slots` (1 to kLiteRtNvidiaQkvNormRopeMaxSlots)
// slots of the shapes of `params` (params->tokens tokens each); the table is
// read during the call. The scales are every slot's.
extern "C" cudaError_t LiteRtNvidiaLaunchQkvNormRopeSlots(
    const LiteRtNvidiaQkvNormRopeParams* params,
    const LiteRtNvidiaQkvNormRopeSlot* slots, int32_t num_slots,
    const float* q_scale, const float* k_scale, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_SLOTS_H_
