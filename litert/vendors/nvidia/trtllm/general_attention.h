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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Attention and cache updates over KV caches in every storage form the
// composites define (odml.sdpa_transposed, odml.fused_sdpa_cache_update and
// odml.cache_update), for the forms the specialized kernels
// (decode_attention.h, tiled_attention.h, quant_cache.h) do not take: float
// or INT8 queries, bool or additive float masks, a score scale and a soft
// cap, any head depth up to 512, and caches of INT8, packed INT4 or UINT8
// (zero point 128) values with per-tensor scales, or of float values.
//
// The caches are the model's tensors as they are: keys as rows [heads,
// cache_len, depth], values as columns [heads, depth, cache_len]. INT4 holds
// two values per byte along the last axis, the first in the low nibble, as
// LiteRT packs it. A quantized value q stands for q * scale (UINT8: (byte -
// 128) * scale).
//
// Quantization follows TFLite's QUANTIZE with the CPU backend's rounding: to
// INT8 and UINT8 the value times the reciprocal of the scale rounded half to
// even (XNNPACK's conversion), to INT4 the quotient rounded half away from
// zero (TFLite's kernel), saturated to the type's range.

enum LiteRtNvidiaGeneralType : int32_t {
  kLiteRtNvidiaGeneralFloat32 = 0,
  kLiteRtNvidiaGeneralFloat16 = 1,
  kLiteRtNvidiaGeneralBFloat16 = 2,
  kLiteRtNvidiaGeneralInt8 = 3,
  kLiteRtNvidiaGeneralInt4 = 4,
  kLiteRtNvidiaGeneralUint8 = 5,
};

enum LiteRtNvidiaGeneralMask : int32_t {
  kLiteRtNvidiaGeneralMaskBool = 0,   // true sees the key
  kLiteRtNvidiaGeneralMaskFloat = 1,  // fp32, added to the score
};

// A float tensor a graph may have quantized: its element type (float32,
// float16 or bfloat16) and, when the graph rounds it to INT8 before the
// composite reads it, that scale (0 when it is read as it is).
struct LiteRtNvidiaGeneralFloat {
  int32_t type;
  float int8_scale;
};

// The two caches of a layer: their element type and, when quantized, the
// scales of the keys and the values.
struct LiteRtNvidiaGeneralCaches {
  int32_t type;
  float k_scale;
  float v_scale;
};

// Attention of `rows` query rows per KV head:
//   score[r][j] = softcap(scale * q[r] . k[j]) (+ mask[r % mask_rows][j] for
//                 a float mask)
//   out[r] = softmax_j(score[r]) . v
// over the cache_len keys of the caches followed by new_len keys of the step
// (odml.fused_sdpa_cache_update's chunk; 0 otherwise), where softcap(x) =
// c * tanh(x / c) when softcap c > 0. A bool mask leaves out the keys it
// hides, as the composites' fill (far below every score) does; a row whose
// bool mask hides every key averages all of them with equal weights, as that
// fill does too. The mask is [mask_rows, cache_len + new_len].
//
// q is [kv_heads, rows, depth] (the query heads of a KV head are its rows,
// head-major) and out the same in q's float type. new_k is [kv_heads,
// new_len, depth] and new_v the same, or [new_len, kv_heads, depth] with
// new_v_token_major; over quantized caches the new keys and values are
// attended as the caches will hold them (rounded to their grid).
struct LiteRtNvidiaGeneralAttentionParams {
  int32_t kv_heads;
  int32_t rows;
  int32_t depth;      // a multiple of 8, at most 512
  int32_t cache_len;  // INT4 caches: even
  int32_t new_len;
  int32_t mask_rows;
  int32_t mask_type;  // LiteRtNvidiaGeneralMask
  LiteRtNvidiaGeneralFloat q;
  LiteRtNvidiaGeneralCaches caches;
  LiteRtNvidiaGeneralFloat new_kv;
  int32_t new_v_token_major;
  float scale;
  float softcap;
};

extern "C" bool LiteRtNvidiaGeneralAttentionSupports(
    const LiteRtNvidiaGeneralAttentionParams* params);

extern "C" size_t LiteRtNvidiaGeneralAttentionWorkspaceBytes(
    const LiteRtNvidiaGeneralAttentionParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchGeneralAttention(
    const LiteRtNvidiaGeneralAttentionParams* params, const void* q,
    const void* k_cache, const void* v_cache, const void* new_k,
    const void* new_v, const void* mask, void* out, void* workspace,
    cudaStream_t stream);

// Writes `tokens` rows of keys and values into the caches, in place, as
// their type holds them (rounded to the grid of quantized caches). src_k is
// [heads, tokens, depth] and src_v the same, or [tokens, heads, depth] with
// src_v_token_major. Where they go:
//  - kLiteRtNvidiaGeneralWriteSlice (odml.cache_update): token t to position
//    start + t, start = *k_start (keys) or *v_start (values), each clamped to
//    [0, cache_len - tokens] as DYNAMIC_UPDATE_SLICE clamps;
//  - kLiteRtNvidiaGeneralWriteRing (odml.fused_sdpa_cache_update): with
//    p0 = ring[0], p1 = ring[1], the n = clamp(p1 - p0, 0, tokens) valid
//    tokens; token t >= max(0, n - cache_len) goes to slot (p0 + t) %
//    cache_len, none when p0 < 0.
enum LiteRtNvidiaGeneralWrite : int32_t {
  kLiteRtNvidiaGeneralWriteSlice = 0,
  kLiteRtNvidiaGeneralWriteRing = 1,
};

struct LiteRtNvidiaGeneralWriteParams {
  int32_t heads;
  int32_t cache_len;
  int32_t depth;  // INT4 caches: even
  int32_t tokens;
  int32_t mode;  // LiteRtNvidiaGeneralWrite
  LiteRtNvidiaGeneralCaches caches;
  LiteRtNvidiaGeneralFloat src;
  int32_t src_v_token_major;
};

extern "C" bool LiteRtNvidiaGeneralCacheWriteSupports(
    const LiteRtNvidiaGeneralWriteParams* params);

// k_start and v_start (slice) or ring (ring) are device int32.
extern "C" cudaError_t LiteRtNvidiaLaunchGeneralCacheWrite(
    const LiteRtNvidiaGeneralWriteParams* params, void* k_cache, void* v_cache,
    const void* src_k, const void* src_v, const int32_t* k_start,
    const int32_t* v_start, const int32_t* ring, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GENERAL_ATTENTION_H_
