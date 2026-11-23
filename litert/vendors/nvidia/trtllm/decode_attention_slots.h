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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_SLOTS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_SLOTS_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"

// The decode attention of decode_attention.h over FP16 or INT8 caches for
// several sequence slots in one launch (a batched decode step: one sequence
// per slot, each with its own caches). A slot's rows are the rows the
// single-sequence launch takes for that sequence, and its result is that
// launch's bit for bit: the same blocks run the same device code on the
// slot's own buffers, only side by side with the other slots' blocks, in two
// launches (the chunks of the keys, then the combine) however many slots.
// Nothing is shared between slots, so the bytes read are the sum of the
// slots' own: per slot and head, the keys and values of the chunks some row
// sees.

// The most slots of one launch.
constexpr int kLiteRtNvidiaDecodeAttentionMaxSlots = 8;

// One slot's buffers (device pointers): q and out [heads, rows, depth] in
// the launch's type, the slot's own caches and mask as
// LiteRtNvidiaLaunchDecodeAttention (FP16 caches) or
// LiteRtNvidiaLaunchDecodeAttentionInt8Window (INT8 caches) take them, and
// with a window the slot's own window start (a device int32).
struct LiteRtNvidiaDecodeAttentionSlot {
  const void* q;
  const void* k;
  const void* v;
  const bool* mask;             // [mask_rows, cache_len], or null
  const int32_t* window_start;  // null without a window
  void* out;
};

// The workspace of a launch over `slots` slots: `slots` times the
// single-sequence launch's (of `seq` keys, the window with one).
extern "C" size_t LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
    int32_t slots, int32_t heads, int32_t rows, int32_t seq, int32_t depth);

// `slots` points to `num_slots` (1 to kLiteRtNvidiaDecodeAttentionMaxSlots)
// slots of the same shapes; the table is read during the call. `int8` null:
// FP16 caches [heads, cache_len, depth], and `window` is cache_len. Otherwise
// INT8 caches with these parameters, of which `window` keys from each slot's
// window_start[0] are attended (window_start null in every slot: all of
// them, and `window` is cache_len). The other arguments are those of the
// single-sequence launches, with their limits.
extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionSlots(
    const LiteRtNvidiaDecodeAttentionSlot* slots, int32_t num_slots,
    bool q_bf16, const LiteRtNvidiaInt8CacheParams* int8, int32_t mask_rows,
    int32_t heads, int32_t rows, int32_t cache_len, int32_t window,
    int32_t depth, float fill, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_SLOTS_H_
