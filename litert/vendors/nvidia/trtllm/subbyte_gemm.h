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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

// Tensor-core GEMM of many BF16 activation rows (prefill) with signed INT4
// weights (two two's-complement values per byte, least-significant first)
// and BF16 per-channel scales:
//   y[m][n] = scales[n] * sum_k activation[m][k] * weights[n][k]
// The output is BF16.
//
// A block multiplies 128 activation rows by 128 weight rows. It reads the
// activations, which a first launch scales per row by a power of two into
// [-1, 1] and converts to FP16, and converts the weights to FP16 while it
// multiplies. The weights are the raw TFLite row-major bytes [channels,
// input_size], the layout the decode GEMV (int2_gemv.h) reads too, so prefill
// and decode engines can share one copy of them: a block stages 64 input dims
// of its 128 weight rows at a time. Products accumulate in FP16 over 128 input
// dims at a time, which cannot overflow (activations of at most 1 times
// weights of at most 8), and in FP32 beyond; the error this adds stays below
// the rounding of a BF16 result.
enum LiteRtNvidiaGemmGate : int32_t {
  // output[m][n] = y[m][n] for output_size channels.
  kLiteRtNvidiaGemmGateNone = 0,
  // weights and scales hold 2 * output_size channels, the gate projection
  // followed by the up projection of a feed-forward block:
  // output[m][n] = gelu(y[m][n]) * y[m][output_size + n], with the tanh
  // approximation of GELU or the exact one.
  kLiteRtNvidiaGemmGateGeluTanh = 1,
  kLiteRtNvidiaGemmGateGeluErf = 2,
};

struct LiteRtNvidiaGemmShape {
  int32_t rows;
  int32_t input_size;
  int32_t output_size;
  int32_t gate;  // LiteRtNvidiaGemmGate
  // 0: one BF16 scale per channel ([channels]). 32: TFLite blockwise
  // quantization, one FP16 scale per block of 32 input dims ([channels,
  // input_size / 32]), applied to the weights as they are converted.
  int32_t scale_block;
};

// rows a multiple of 128, input_size a multiple of 64, an even output_size,
// scale_block 0 or 32.
extern "C" bool LiteRtNvidiaSubbyteGemmSupports(
    const LiteRtNvidiaGemmShape* shape);

// Whether the current device can run the kernel (compute capability 8.0 or
// newer with about 81 KB of shared memory per block).
extern "C" bool LiteRtNvidiaSubbyteGemmAvailable();

extern "C" size_t LiteRtNvidiaSubbyteGemmWorkspaceBytes(
    const LiteRtNvidiaGemmShape* shape);

// The bytes of the row-major weights the launch reads: [channels,
// input_size / 2], with a gate the gate projection followed by the up
// projection (2 * output_size channels).
extern "C" size_t LiteRtNvidiaSubbyteGemmWeightBytes(
    const LiteRtNvidiaGemmShape* shape);

// The blocks of a launch, one per 128 activation rows and column tile.
extern "C" int32_t LiteRtNvidiaSubbyteGemmBlocks(
    const LiteRtNvidiaGemmShape* shape);

// Whether the launch keeps the current device busy: it has at least three
// blocks per two multiprocessors. Part of the device idles during smaller
// products.
extern "C" bool LiteRtNvidiaSubbyteGemmFillsDevice(
    const LiteRtNvidiaGemmShape* shape);

// A grouped GEMM over experts whose rows interleave within a projection
// (Gemma 4 26B-A4B's experts: row o of expert e is row o * experts + e):
// row tile t of the activations belongs to expert tile_experts[t], and the
// weights of channel c of expert e start at weights + e * expert_stride +
// c * channel_stride bytes (their scales at scales + e *
// expert_scale_stride + c * scale_channel_stride halves). With a gate the
// up projection has its own base pointers instead of following the gate
// channels. The activations are already in the GEMM's tile layout
// (LiteRtNvidiaLaunchGemmScaleRows); the padded rows of a tile past its
// expert's last row produce rows the caller ignores.
struct LiteRtNvidiaGroupedGemmWeights {
  const uint8_t* gate;
  const uint8_t* up;  // gate mode only
  const void* gate_scales;
  const void* up_scales;          // gate mode only
  uint64_t channel_stride;        // bytes
  uint64_t scale_channel_stride;  // halves
  uint64_t expert_stride;         // bytes
  uint64_t expert_scale_stride;   // halves
  const int32_t* tile_experts;    // [rows / 128]
  // Compact scales (compact_scales != 0): gate_scales/up_scales then hold a
  // byte per block, the fraction (255ths) of the channel's maximum, and
  // these the maxima as FP16, max_expert_stride and max_channel_stride
  // halves apart per expert and channel; the strides above count bytes.
  const void* gate_scale_max;
  const void* up_scale_max;
  uint64_t max_channel_stride;
  uint64_t max_expert_stride;
  int32_t compact_scales;
};

// The activation pre-pass of the GEMM on its own: `rows` rows of
// `input_size` BF16 values, row r taken from source row row_indices[r]
// (-1 gives a zero row, an index below -1 leaves the row untouched, nullptr
// gives the identity), scaled
// per row into the tile layout `scaled` (FP16, rows * input_size) with the
// scale factors in `row_scale` (rows floats). rows a multiple of 128.
extern "C" cudaError_t LiteRtNvidiaLaunchGemmScaleRows(
    const void* activation, const int32_t* row_indices, int32_t rows,
    int32_t input_size, float* row_scale, void* scaled, cudaStream_t stream);

// The pre-pass with an RMS norm of each gathered row (see
// LiteRtNvidiaLaunchBf16Int4GemmNormed).
extern "C" cudaError_t LiteRtNvidiaLaunchGemmScaleRowsNormed(
    const void* activation, const int32_t* row_indices, int32_t rows,
    int32_t input_size, const float* norm_scale, float norm_eps,
    float norm_multiplier, float* row_scale, void* scaled, cudaStream_t stream);

// The GEMM over pre-scaled activations (LiteRtNvidiaLaunchGemmScaleRows) and
// grouped weights. `shape.rows` is the padded row count, a multiple of 128.
extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4GroupedGemm(
    const LiteRtNvidiaGemmShape* shape,
    const LiteRtNvidiaGroupedGemmWeights* weights, const void* scaled,
    const float* row_scale, void* output, cudaStream_t stream);

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4Gemm(
    const LiteRtNvidiaGemmShape* shape, const void* activation,
    const uint8_t* weights, const void* scales, void* output, void* workspace,
    cudaStream_t stream);

// The same with an RMS norm of the activation rows fused into the pre-pass:
// row x becomes x * rsqrt(mean(x^2) + norm_eps) * norm_scale[k] *
// norm_multiplier (norm_scale an FP32 vector of input_size values; null is
// the plain product).
extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4GemmNormed(
    const LiteRtNvidiaGemmShape* shape, const void* activation,
    const float* norm_scale, float norm_eps, float norm_multiplier,
    const uint8_t* weights, const void* scales, void* output, void* workspace,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SUBBYTE_GEMM_H_
