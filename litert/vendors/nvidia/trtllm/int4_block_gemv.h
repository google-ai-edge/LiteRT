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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// GEMV over TFLite blockwise-quantized INT4 weights (Gemma 4 26B-A4B): raw
// row-major signed INT4 values, two per byte and the even element in the low
// nibble, with one FP16 scale per block of 32 input dims:
//   y[r][n] = sum_b scales[n][b] * sum_{k in block b} activation[r][k] *
//   w[n][k]
// Activations and outputs are BF16; products and sums are FP32. The weights
// are [output_size, input_size / 2] bytes, the scales [output_size,
// input_size / 32] halves, both the layout of the model file.
//
// `rows` activation rows, 1 for decode and up to
// kLiteRtNvidiaInt4BlockGemvMaxRows for a verify step, share every weight
// read. input_size must be a multiple of 32.
//
// `gate` selects a gated feed-forward: the weights and scales then hold
// 2 * output_size rows, the gate projection followed by the up projection,
// and output[r][n] = gelu(y_gate[r][n]) * y_up[r][n] with the tanh
// approximation (1) or the exact erf form (2) of GELU; 0 is a plain GEMV.
constexpr int kLiteRtNvidiaInt4BlockGemvMaxRows = 8;

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    void* output, int rows, int input_size, int output_size, int gate,
    cudaStream_t stream);

// Mixture-of-experts feed-forward over blockwise INT4 experts (the "moe"
// custom op of Gemma 4 26B-A4B): for every activation row r and each of its
// top_k routed experts e = top_indices[r][j],
//   hidden = gelu(x_r . gate_e) * (x_r . up_e)          [hidden_dim]
//   out_r += top_weights[r][j] * per_expert_scale[e] * (hidden . down_e)
// The expert weights keep the model's layout, which interleaves the experts
// within the output dimension: row o of expert e of the gate and up
// projections is row o * experts + e of a [hidden_dim * experts,
// model_dim] INT4 matrix, and of the down projection row o * experts + e of
// [model_dim * experts, hidden_dim]; the scales are FP16 per block of 32
// input dims in the same row order. Activations and outputs are BF16.
struct LiteRtNvidiaMoeArena {
  const uint8_t* gate;
  const uint8_t* up;
  const uint8_t* down;
  const void* gate_scales;
  const void* up_scales;
  const void* down_scales;
};

struct LiteRtNvidiaMoeShape {
  int32_t rows;
  int32_t top_k;
  int32_t experts;
  int32_t model_dim;   // a multiple of 32
  int32_t hidden_dim;  // a multiple of 32
  int32_t gate;        // 1: tanh GELU, 2: erf GELU
};

// Whether the launch takes the shape: rows >= 1, top_k in [1, 32], experts
// in [1, 1024], the dims multiples of 32.
extern "C" bool LiteRtNvidiaMoeSupports(const LiteRtNvidiaMoeShape* shape);

// The scratch bytes a launch of the shape needs.
extern "C" size_t LiteRtNvidiaMoeWorkspaceBytes(
    const LiteRtNvidiaMoeShape* shape);

// A few rows (decode, a verify step: rows <= kLiteRtNvidiaInt4BlockGemvMaxRows)
// run one GEMV per routed expert; more rows (prefill) are grouped by expert
// so that an expert's weights are read once per tile of tokens.
extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4Block(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const int32_t* top_indices,
    const float* top_weights, const float* per_expert_scale, void* workspace,
    void* output, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_H_
