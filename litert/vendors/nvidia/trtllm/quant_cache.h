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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_CACHE_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// What surrounds attention over quantized KV caches (the caches of
// LiteRtNvidiaQuantAttentionParams in decode_attention.h: keys as rows
// [heads, cache_len, depth], values as columns [heads, depth, cache_len],
// INT8 or INT4 two per byte with the first in the low nibble): the QUANTIZE
// of the query and of the keys and values a step adds, and the cache updates
// of odml.cache_update and odml.fused_sdpa_cache_update. Quantization is the
// graphs' QUANTIZE with a zero point of 0 as the CPU backend computes it: to
// INT8, XNNPACK's conversion of a FLOAT16 tensor (x times the reciprocal of
// the scale rounded to FLOAT16, rounded half to even, saturated to [-128,
// 127]); to INT4, TFLite's kernel (x / scale rounded half away from zero,
// saturated to [-8, 7]). Float tensors are FP16, or BF16 where a `bf16`
// argument says so.

// out[i] (FP16) = the INT8 quantization of q[i], dequantized as XNNPACK does
// (times `scale` rounded to FLOAT16): the query a tensor-core kernel
// multiplies.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantizeQuery(const void* q,
                                                       bool q_bf16,
                                                       size_t count,
                                                       float scale, void* out,
                                                       cudaStream_t stream);

// The INT8 quantization of the keys and values of `tokens` tokens as rows
// k_out and v_out [heads, tokens, depth]. k is [heads, tokens, depth]; v is
// the same, or [tokens, heads, depth] with `v_token_major`.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantizeRows(
    const void* k, const void* v, bool bf16, bool v_token_major, int32_t heads,
    int32_t tokens, int32_t depth, float k_scale, float v_scale, void* k_out,
    void* v_out, cudaStream_t stream);

// odml.cache_update: quantizes the keys and values of `tokens` tokens (k and
// v as LiteRtNvidiaLaunchQuantizeRows takes them) to `cache_bits` bits and
// writes them into the caches, in place, at the positions *k_position and
// *v_position (device int32), each clamped to [0, cache_len - tokens] as
// DYNAMIC_UPDATE_SLICE clamps its start.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantCacheWrite(
    void* k_cache, void* v_cache, const void* k, const void* v, bool bf16,
    bool v_token_major, const int32_t* k_position, const int32_t* v_position,
    int32_t cache_bits, float k_scale, float v_scale, int32_t heads,
    int32_t cache_len, int32_t depth, int32_t tokens, cudaStream_t stream);

// The cache update of odml.fused_sdpa_cache_update over INT8 ring caches,
// in place: params (device int32) holds the position of the chunk's first
// token and the position after its last one. n = clamp(params[1] -
// params[0], 0, tokens) tokens are valid, and the token at position p goes to
// slot p % cache_len, the newest of those that share a slot. k_new and v_new
// are the tokens' INT8 rows [heads, tokens, depth].
extern "C" cudaError_t LiteRtNvidiaLaunchQuantRingUpdate(
    void* k_cache, void* v_cache, const void* k_new, const void* v_new,
    const int32_t* params, int32_t heads, int32_t cache_len, int32_t depth,
    int32_t tokens, cudaStream_t stream);

// out[i] (BF16) = in[i] (FP16).
extern "C" cudaError_t LiteRtNvidiaLaunchHalfToBf16(const void* in,
                                                    size_t count, void* out,
                                                    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_CACHE_H_
