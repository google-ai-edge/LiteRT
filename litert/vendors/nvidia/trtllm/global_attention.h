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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GLOBAL_ATTENTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GLOBAL_ATTENTION_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Tensor-core attention over a single-head KV cache (Gemma 4 global layers),
// for decode (16 query rows) and prefill (16 rows per prompt token):
//   scores[r][j] = q[r] . k[j]              (scores where the mask is false
//                                            become `fill`)
//   out[r]       = softmax_j(scores[r]) . v
// q and out are [rows, depth] in FP16 (`q_bf16` false) or BF16, k and v are
// [seq, depth] FP16, and mask is [mask_rows, seq] bool. Query row r uses mask
// row r % mask_rows, which is the row order of `mask_rows` tokens times
// rows / mask_rows query heads flattened head-major.
//
// Only the keys some query row can see are read: a first launch summarizes
// the mask, and the attention blocks skip what no row of theirs sees. Keys no
// row can see must therefore not matter to the result, which holds when
// exp(fill - max score) underflows (any customary mask fill does); rows that
// see no key at all still get the softmax over the whole sequence. The softmax
// runs online, so no [rows, seq] tensor is materialized.
//
// Decode shapes run on a kernel that reads every key and value once, straight
// into tensor-core operands that accumulate in FP32. 128 rows or more in
// multiples of 64, with mask_rows a multiple of 64, run on the kernels of
// tiled_attention.h where the device supports them.
//
// Supports depth 512 and rows a multiple of 16 that mask_rows divides.
// `workspace` needs LiteRtNvidiaGlobalAttentionWorkspaceBytes() bytes, which
// depends only on the shapes. Requires compute capability 8.0 or newer.
extern "C" bool LiteRtNvidiaGlobalAttentionSupports(int32_t rows,
                                                    int32_t mask_rows,
                                                    int32_t depth);

extern "C" size_t LiteRtNvidiaGlobalAttentionWorkspaceBytes(int32_t rows,
                                                            int32_t mask_rows,
                                                            int32_t seq,
                                                            int32_t depth);

extern "C" cudaError_t LiteRtNvidiaLaunchGlobalAttention(
    const void* q, bool q_bf16, const void* k, const void* v, const bool* mask,
    int32_t mask_rows, int32_t rows, int32_t seq, int32_t depth, float fill,
    void* out, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GLOBAL_ATTENTION_H_
