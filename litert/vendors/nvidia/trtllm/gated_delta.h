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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// Qwen 3.5's linear-attention layers (Gated DeltaNet), FP32 in and out.
//
// gated_delta_update (litert/experimental/custom_ops/gated_delta_net):
//   q, k [1, key_heads, tokens, key_dim], v [1, heads, tokens, value_dim],
//   beta, g [1, heads, tokens], state [1, heads, key_dim, value_dim] and an
//   optional int32 valid length -> out [1, heads, tokens, value_dim] and
//   state'. Value head h reads key head h / (heads / key_heads); for every
//   row t below the valid length (all rows without one):
//     S = exp(g_t) S;  S += k_t ((v_t - S^T k_t) beta_t)^T;  out_t = S^T q_t
//   with the right-padding sentinels of the reference: g_t > 0 leaves S
//   undecayed and beta_t <= 0 adds nothing. Rows at or after the valid
//   length output zero and leave the state alone.
// state and state' may be one buffer: a block owns a set of value columns,
// reads them before writing them, and no other block touches them.
struct LiteRtNvidiaGatedDeltaParams {
  int32_t heads;  // value heads, a multiple of key_heads
  int32_t key_heads;
  int32_t tokens;
  int32_t key_dim;    // 64, 128 or 256
  int32_t value_dim;  // a multiple of 32
};

extern "C" bool LiteRtNvidiaGatedDeltaSupports(
    const LiteRtNvidiaGatedDeltaParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchGatedDeltaUpdate(
    const LiteRtNvidiaGatedDeltaParams* params, const float* q, const float* k,
    const float* v, const float* beta, const float* g, const float* state,
    const int32_t* valid_len, float* out, float* new_state,
    cudaStream_t stream);

// odml.short_conv_step: the causal depthwise convolution in front of a
// Gated DeltaNet layer. x [1, tokens, channels], state [1, channels,
// taps - 1], weight [channels, 1, taps] and an optional int32 valid length
// -> y [1, tokens, channels], state'. Over the window xx = state ++ x along
// time, y_t = act(sum_j weight[c, j] xx[t + j]) with act SiLU (or identity),
// and state' = xx[valid .. valid + taps - 2] (valid = tokens without the
// input): the last taps - 1 valid inputs. state and state' may be one buffer
// (the outputs are written by a first launch, the state by a second).
struct LiteRtNvidiaShortConvParams {
  int32_t channels;
  int32_t tokens;
  int32_t taps;  // 2 to 8
  int32_t silu;
};

extern "C" bool LiteRtNvidiaShortConvSupports(
    const LiteRtNvidiaShortConvParams* params);

extern "C" cudaError_t LiteRtNvidiaLaunchShortConv(
    const LiteRtNvidiaShortConvParams* params, const float* x,
    const float* state, const float* weight, const int32_t* valid_len, float* y,
    float* new_state, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_H_
