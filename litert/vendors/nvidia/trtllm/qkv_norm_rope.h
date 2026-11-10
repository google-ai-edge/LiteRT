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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// Gemma 4's odml.qkv_norm_rope composite in one launch: the fused qkv
// projection [tokens, (heads + 2 kv_heads) * depth] is split into query, key
// and value heads; every head is RMS-normalized over its depth (queries by
// q_scale, keys by k_scale, values without a scale when has_v_norm, else
// not at all); queries and keys are rotated by the token's position with
// the half-rotation RoPE over the first `rotary` dims (pairs (i, i + rotary
// / 2); inverse timescale of pair i = exp(-i * ln(max_timescale /
// min_timescale) / (rotary / 2)) / min_timescale, the composite's constants):
//   q [1, heads, tokens, depth], k and v [1, kv_heads, tokens, depth].
// BF16 in and out, FP32 inside. positions holds `tokens` int32.
struct LiteRtNvidiaQkvNormRopeParams {
  int32_t tokens;
  int32_t heads;
  int32_t kv_heads;
  int32_t depth;   // a multiple of 2, at most 1024
  int32_t rotary;  // dims rotated: depth * proportion, a multiple of 2
  int32_t has_v_norm;
  float epsilon;
  float min_timescale;
  float max_timescale;
};

extern "C" bool LiteRtNvidiaQkvNormRopeSupports(
    const LiteRtNvidiaQkvNormRopeParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchQkvNormRope(
    const LiteRtNvidiaQkvNormRopeParams* params, const void* qkv,
    const int32_t* positions, const float* q_scale, const float* k_scale,
    void* q, void* k, void* v, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QKV_NORM_ROPE_H_
