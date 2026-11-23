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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_PLE_TAIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_PLE_TAIL_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include "cuda_runtime_api.h"

// Qwen 4's per-layer embedding (PLE) past its key and value projections, in
// one launch. For each of `rows` rows, with x the hc streams [rows, streams *
// hidden], key the key projection's result of the same shape and value the
// value projection's [rows, hidden] (all BF16), per stream s:
//   kn_s   = key_s * rsqrt(mean(key_s^2) + eps_key) * key_weight_s
//   qn_s   = x_s * rsqrt(mean(x_s^2) + eps_query) * query_weight_s
//   g_s    = dot_scale * sum(kn_s * qn_s)
//   gate_s = sigmoid(sign(g_s) * sqrt(max(|g_s|, floor)))
//   gv_s   = gate_s * value
//   c_s    = gv_s * rsqrt(mean(gv_s^2) + eps_conv) * conv_weight_s
//   y      = act(the causal convolution of c over the rows with `taps` taps
//            per channel: taps [streams * hidden, taps], oldest first, over
//            state ++ c), act silu when `silu`
//   out    = x + gv + y                  [rows, streams * hidden] BF16
//   state' = the last taps - 1 inputs of each channel (state FP32 [streams *
//            hidden, taps - 1]; it may be `state` itself)
// `constants` (FP32): key_weight, query_weight and conv_weight [streams *
// hidden] each, then the taps. FP32 arithmetic, sums in a fixed order.
struct LiteRtNvidiaPleTailParams {
  int32_t rows;     // 1 to 8
  int32_t streams;  // 1 to 8
  int32_t hidden;   // a multiple of 8
  int32_t taps;     // 2 to 16
  float eps_key;
  float eps_query;
  float eps_conv;
  float dot_scale;
  float floor;
  int32_t silu;
};

extern "C" bool LiteRtNvidiaPleTailSupports(
    const LiteRtNvidiaPleTailParams* params);

// The FP32 values of `constants`.
extern "C" size_t LiteRtNvidiaPleTailConstantCount(
    const LiteRtNvidiaPleTailParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchPleTail(
    const LiteRtNvidiaPleTailParams* params, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* x, const float* state,
    const float* constants, __nv_bfloat16* out, float* new_state,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_PLE_TAIL_H_
