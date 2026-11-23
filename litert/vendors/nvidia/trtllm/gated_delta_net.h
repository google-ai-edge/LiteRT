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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_NET_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_NET_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Qwen 3.5's gated DeltaNet layer from the output of its in-projections to the
// input of its output projection, over the rows of a prefill chunk or a decode
// step. Row t of `proj` (BF16, proj_stride elements) holds the projections at
// their offsets, before their biases: qkv (C = (2 * key_heads + value_heads) *
// head_dim channels), z (value_heads * head_dim), a and b (value_heads each);
// below, each is that plus its bias. With D = head_dim:
//   y      = silu(sum_j conv_weight[c][j] * xx[t + j][c]), xx = conv_state
//            (taps - 1 rows) followed by the qkv rows
//   q, k   = rms_norm(y over a key head's D channels) * q_scale | k_scale
//            (epsilon q_eps | k_eps); v = y over a value head's channels
//   beta   = sigmoid(b), g = decay * log(1 + exp(min(a, 50)))  per value head
//   o      = the gated delta rule (gated_delta.h) from `state`
//   out    = rms_norm(o over a head's D values) * norm_weight (out_eps)
//            * silu(z)                                   BF16 [t][value
//            head][D]
// Rows from valid_len on (the padding of a prefill chunk) take no part, as in
// the model where they are masked to zero: the states advance over the valid
// rows only, and those rows' outputs are zero.
struct LiteRtNvidiaGatedDeltaNetParams {
  int32_t tokens;
  int32_t key_heads;
  int32_t value_heads;  // a multiple of key_heads
  int32_t head_dim;     // 128
  int32_t taps;         // 2..8
  int32_t silu;         // the convolution's SiLU
  int32_t proj_stride;  // elements per row of proj
  int32_t qkv_offset;
  int32_t z_offset;
  int32_t a_offset;
  int32_t b_offset;
  float q_eps;
  float k_eps;
  float out_eps;
};

// The FP32 constants of a layer, in this order: decay [value_heads], q_scale,
// k_scale and norm_weight [head_dim] each, conv_weight [C][taps], then the
// projections' biases qkv [C], z [value_heads * head_dim], a and b
// [value_heads] (zeros for a projection without one).
extern "C" size_t LiteRtNvidiaGatedDeltaNetConstantCount(
    const LiteRtNvidiaGatedDeltaNetParams* params);

extern "C" bool LiteRtNvidiaGatedDeltaNetSupports(
    const LiteRtNvidiaGatedDeltaNetParams* params);

extern "C" size_t LiteRtNvidiaGatedDeltaNetWorkspaceBytes(
    const LiteRtNvidiaGatedDeltaNetParams* params);

// conv_state FP32 [C][taps - 1], state FP32 [value_heads][D][D] (rows of a
// head's key dims); valid_len (nullptr: every row) a device INT32. The new
// states may be the old ones (an in-place update).
extern "C" cudaError_t LiteRtNvidiaLaunchGatedDeltaNet(
    const LiteRtNvidiaGatedDeltaNetParams* params, const void* proj,
    const float* conv_state, const float* state, const float* constants,
    const int32_t* valid_len, void* out, float* new_conv_state,
    float* new_state, void* workspace, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_GATED_DELTA_NET_H_
