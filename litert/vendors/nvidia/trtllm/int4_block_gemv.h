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

// The same product with an RMS norm of the activation rows fused in front
// of it (Gemma's rms_norm composite feeding a projection): each row x is
// replaced by x * rsqrt(mean(x^2) + norm_eps) * norm_scale[k] *
// norm_multiplier, with `norm_scale` an FP32 vector of input_size values.
// A null norm_scale is the plain product.
extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
    const void* activation, const uint8_t* packed_weights, const void* scales,
    const float* norm_scale, float norm_eps, float norm_multiplier,
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
  // The router's INT4 blockwise rows [experts, model_dim] and FP16 scales
  // (the fused launch only; null otherwise).
  const uint8_t* router;
  const void* router_scales;
  // The layer's dense feed-forward (the layer-block launch only; null
  // otherwise): its gate and up INT4 rows stacked [2 * dense_hidden,
  // model_dim] with FP16 scales, and its down rows [model_dim, dense_hidden]
  // with theirs.
  const uint8_t* dense_gate_up;
  const void* dense_gate_up_scales;
  const uint8_t* dense_down;
  const void* dense_down_scales;
  // Compact expert scales (compact_scales != 0): gate_scales, up_scales and
  // down_scales then hold one byte per block, the block's scale as a
  // fraction (in 255ths) of its row's maximum, and these hold the rows'
  // maxima as FP16, one per row in the piece's row order.
  const void* gate_scale_max;
  const void* up_scale_max;
  const void* down_scale_max;
  int32_t compact_scales;
};

// A row of block scales into the compact format: the row's maximum as FP16
// bits, then each block's scale as round(255 * scale / maximum) (scales are
// non-negative; a negative one quantizes to 0).
extern "C" void LiteRtNvidiaCompactScaleRow(const float* scales, int32_t blocks,
                                            uint8_t* fractions,
                                            uint16_t* max_half);

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

// The whole mixture-of-experts block of a layer from the residual stream
// `activation` [rows, model_dim]: the router's RMS norm (router_norm_scale,
// times router_multiplier), the router's logits from the arena's router
// rows, a softmax, the top_k experts (renormalized to a sum of one with
// `renormalize`), and the experts over the row normalized with
// expert_norm_scale; both norms share norm_eps. rows <= model_dim <= 4096.
extern "C" cudaError_t LiteRtNvidiaLaunchMoeInt4BlockFused(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const float* per_expert_scale,
    const float* router_norm_scale, float router_multiplier,
    const float* expert_norm_scale, float norm_eps, int renormalize,
    void* workspace, void* output, cudaStream_t stream);

// The rest of a layer's feed-forward block around the experts.
struct LiteRtNvidiaMoeBlockParams {
  int32_t dense_hidden;           // the dense feed-forward's hidden size
  int32_t dense_gate;             // its GELU: 1 tanh approximation, 2 erf
  float residual_multiplier;      // scales the block's output
  const float* dense_norm_scale;  // [model_dim] the dense input norm
  const float* dense_post_scale;  // [model_dim] the dense output norm
  const float* moe_post_scale;    // [model_dim] the experts' output norm
  const float* sum_post_scale;    // [model_dim] the norm of their sum
  cudaStream_t side_stream;       // the dense feed-forward runs here
  cudaEvent_t fork;               // the events that fork and join it
  cudaEvent_t join;
};

// The scratch bytes LiteRtNvidiaLaunchMoeLayerBlock needs.
extern "C" size_t LiteRtNvidiaMoeBlockWorkspaceBytes(
    const LiteRtNvidiaMoeShape* shape, int32_t dense_hidden,
    int32_t dense_gate);

// LiteRtNvidiaLaunchMoeInt4BlockFused and the rest of the layer's
// feed-forward block, Gemma 4's:
//   dense = down(gelu(gate(n_d(x))) * up(n_d(x)))      (the arena's dense rows)
//   output = (x + n_s(n_dp(dense) + n_mp(moe))) * residual_multiplier
// where every n is an RMS norm (norm_eps) scaled by its vector. The dense
// feed-forward runs on block->side_stream, forked from `stream` through
// block->fork and joined back through block->join before the epilogue.
extern "C" cudaError_t LiteRtNvidiaLaunchMoeLayerBlock(
    const LiteRtNvidiaMoeShape* shape, const LiteRtNvidiaMoeArena* arena,
    const void* activation, const float* per_expert_scale,
    const float* router_norm_scale, float router_multiplier,
    const float* expert_norm_scale, float norm_eps, int renormalize,
    const LiteRtNvidiaMoeBlockParams* block, void* workspace, void* output,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_INT4_BLOCK_GEMV_H_
