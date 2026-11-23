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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HYPER_CONNECTION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HYPER_CONNECTION_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "cuda_runtime_api.h"

// Qwen 4's hyper-connection mix in one launch. For each of `rows` rows of
// streams x [rows, streams * hidden] (BF16):
//   normed = x_s * rsqrt(mean(x_s^2) + eps) * norm_weight      (per stream s,
//            norm_weight [streams * hidden] FP32: the grouped (1 + w)),
//            rounded to BF16
//   t      = silu(down_scale * (w_down . normed))   w_down [low_rank,
//            streams * hidden], t [low_rank]
//   u      = sigmoid(w_up . t)                      w_up [streams * hidden,
//            low_rank]
//   mixed  = mean over s of (u_s * normed_s)        [rows, hidden] BF16
// and normed itself [rows, streams * hidden] BF16 (the combine reads it).
// The weights are BF16 rows or, with `int4`, INT4 in blocks of 32 (the CUDA
// subbyte GEMV's layout: two's complement nibbles, low first, [rows][columns
// / 2] bytes; a half scale a block, [rows][columns / 32]); their rows start
// on 16 bytes.
//
// FP32 sums in an order the shapes alone decide (the same bits at every grid
// size and on every device). With `state` (LiteRtNvidiaHcMixStateBytes()
// bytes that are zero between launches, one per stream of launches) on a
// device with cooperative launches it is one launch, whose CTAs meet at a
// grid barrier between the projections; without, two launches of the same
// kernel, with the same bits.
struct LiteRtNvidiaHcMixParams {
  int32_t rows;      // 1 to 8
  int32_t streams;   // 1, 2, 4 or 8
  int32_t hidden;    // a multiple of 8 (of 32 with int4)
  int32_t low_rank;  // a multiple of 8 (of 32 with int4)
  float down_scale;
  float eps;
  int32_t int4;
};

extern "C" bool LiteRtNvidiaHcMixSupports(const LiteRtNvidiaHcMixParams* p);

extern "C" size_t LiteRtNvidiaHcMixWorkspaceBytes(
    const LiteRtNvidiaHcMixParams* p);

extern "C" size_t LiteRtNvidiaHcMixStateBytes();

extern "C" cudaError_t LiteRtNvidiaLaunchHcMix(
    const LiteRtNvidiaHcMixParams* p, const __nv_bfloat16* x,
    const float* norm_weight, const void* w_down, const __half* down_scales,
    const void* w_up, const __half* up_scales, __nv_bfloat16* mixed,
    __nv_bfloat16* normed, void* state, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_HYPER_CONNECTION_H_
