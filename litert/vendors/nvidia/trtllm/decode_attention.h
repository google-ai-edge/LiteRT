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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Decode attention over a full KV cache: for every head h and query row r,
//   scores[j] = q[h][r] . k[h][j]        (fp32; scores where mask is false
//                                          are replaced by `fill`)
//   out[h][r] = softmax_j(scores) . v[h]  (fp32 accumulation).
// q and out are [heads, rows, depth] in FP16 (`q_bf16` false) or BF16; k and
// v are [heads, seq, depth] FP16; mask is [mask_rows, seq] bool with
// mask_rows 1 (shared by all rows) or `rows`, or null for no masking. Split
// over the sequence into chunks whose partial results are combined by a
// second launch through `workspace`, which needs
// LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, seq, depth) bytes.
// Supports rows <= 16 and depth in {128, 256, 512}.
extern "C" size_t LiteRtNvidiaDecodeAttentionWorkspaceBytes(int32_t heads,
                                                            int32_t rows,
                                                            int32_t seq,
                                                            int32_t depth);

// Per-tensor quantization of INT8 key and value caches (Gemma 4 E4B):
// real = (quantized - zero_point) * scale. value_columns says the value cache
// is stored [1, H, D, S] (the model's layout) rather than rows [1, H, S, D].
struct LiteRtNvidiaInt8CacheParams {
  float k_scale;
  int32_t k_zero_point;
  float v_scale;
  int32_t v_zero_point;
  bool value_columns;
};

// Whether the INT8-cache kernel is built for this head depth.
extern "C" bool LiteRtNvidiaDecodeAttentionInt8Supports(int32_t depth);

// Decode attention over INT8 caches: q [1, H, rows, D] (FP16 or BF16),
// k [1, H, S, D] INT8, v [1, H, S, D] or [1, H, D, S] INT8, the same mask,
// workspace (LiteRtNvidiaDecodeAttentionWorkspaceBytes) and output as the
// FP16 entry below.
extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionInt8(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaInt8CacheParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t seq, int32_t depth,
    float fill, void* out, void* workspace, cudaStream_t stream);

// INT8 caches read through a window: the caches and the mask hold cache_len
// keys per head, and `window` keys starting at window_start[0] (a device
// int32; a multiple of 16 keeps the value loads vectorized) are attended.
// The workspace is sized for `window` keys.
extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionInt8Window(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaInt8CacheParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t window, const int32_t* window_start, int32_t depth, float fill,
    void* out, void* workspace, cudaStream_t stream);

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttention(
    const void* q, bool q_bf16, const void* k, const void* v, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t seq, int32_t depth,
    float fill, void* out, void* workspace, cudaStream_t stream);

// The quantization of an attention block whose query, keys and values are
// all quantized (odml.sdpa_transposed and odml.fused_sdpa_cache_update over
// quantized tensors): an INT8 query with q_scale and caches that hold INT8
// values, or INT4 values two per byte (the first in the low nibble, as LiteRT
// packs them), with k_scale and v_scale. Every zero point is 0. The keys are
// rows [heads, cache_len, depth], the values columns [heads, depth,
// cache_len], both as the model stores them.
struct LiteRtNvidiaQuantAttentionParams {
  float q_scale;
  float k_scale;
  float v_scale;
  int32_t cache_bits;  // 8 or 4
};

// Depth 256, at most 16 rows, and caches of 8 or 4 bits.
extern "C" bool LiteRtNvidiaQuantDecodeAttentionSupports(int32_t rows,
                                                         int32_t depth,
                                                         int32_t cache_bits);

// Decode attention over quantized caches: q [heads, rows, depth] (FP16 or
// BF16) is the query before its QUANTIZE (or its dequantized value). The
// kernels compute the decomposition of odml.sdpa_transposed tensor by tensor
// with the operations of LiteRT's CPU backend (XNNPACK on x86 with AVX2, FMA3
// and F16C), so that an FP16 result has the CPU's bits: the query quantized
// to INT8 as XNNPACK converts a FLOAT16 tensor (times the reciprocal of the
// scale rounded to FLOAT16, half to even, saturated) and dequantized to
// FLOAT16 with the scale rounded to FLOAT16; a score one fused multiply-add
// per depth element with the key dequantized to fp32, then cast to FLOAT16;
// scores where the mask is false become `fill`; XNNPACK's softmax of a
// FLOAT16 tensor, which gives no weight at all to a key 9.7 below the
// largest score and rounds the others to FLOAT16; and
//   out[h][r] = weights . values[h]
// one fused multiply-add per key in the cache's order, cast to q's type
// (trtllm/quant_attention_reference.h is the same on the host).
// mask is [mask_rows, cache_len] bool, mask_rows a divisor of
// `rows`: row r reads mask row r % mask_rows, the order of mask_rows tokens
// under rows / mask_rows query heads. Chunks of keys no row sees are not
// read. With a `fill` of -inf the keys a mask hides stay out of the softmax
// and a row that sees no key gets zero; with a finite fill such a row has no
// defined result. `workspace` needs
// LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, cache_len, depth)
// bytes. Two launches: the scores of the chunks of the keys, then the softmax
// and the values of every row.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantDecodeAttention(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaQuantAttentionParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t depth, float fill, void* out, void* workspace, cudaStream_t stream);

// One token's key and value rows of a decode step before their QUANTIZE,
// [heads, depth] each in the queries' type, and where they go: the key
// cache's row and the value cache's column, read on the device and clamped
// to the cache.
struct LiteRtNvidiaQuantCacheToken {
  const void* k;
  const void* v;
  const int32_t* k_position;
  const int32_t* v_position;
};

// What LiteRtNvidiaQuantDecodeAttentionSupports takes, with INT4 value rows
// that start on a byte (an even cache_len).
extern "C" bool LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
    int32_t rows, int32_t depth, int32_t cache_bits, int32_t cache_len);

// The device memory a caller keeps for its launches of
// LiteRtNvidiaLaunchQuantDecodeAttentionUpdate with these heads and rows:
// zero before the first launch, and left zero by every launch.
extern "C" size_t LiteRtNvidiaQuantDecodeAttentionStateBytes(int32_t heads,
                                                             int32_t rows);

// LiteRtNvidiaLaunchQuantCacheWrite (quant_cache.h) of `token` followed by
// LiteRtNvidiaLaunchQuantDecodeAttention over the updated caches, with the
// same results bit for bit, in the attention's launches alone: every block
// writes the token's quantized rows into the caches before it reads them. A
// null token writes nothing. `state` is no longer read: the one launch it
// counted the blocks of folded the softmax in chunk by chunk, which is not
// the decomposition's arithmetic.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
    const void* q, bool q_bf16, void* k, void* v,
    const LiteRtNvidiaQuantCacheToken* token,
    const LiteRtNvidiaQuantAttentionParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t depth, float fill, void* out, void* workspace, void* state,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_H_
