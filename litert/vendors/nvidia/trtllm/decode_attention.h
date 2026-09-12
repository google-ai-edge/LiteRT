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

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttention(
    const void* q, bool q_bf16, const void* k, const void* v,
    const bool* mask, int32_t mask_rows, int32_t heads, int32_t rows,
    int32_t seq, int32_t depth, float fill, void* out, void* workspace,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_DECODE_ATTENTION_H_
