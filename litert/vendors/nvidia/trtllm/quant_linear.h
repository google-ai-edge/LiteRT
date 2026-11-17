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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_H_

#include <cstdint>

#include "driver_types.h"

// Linear algebra on INT8 activations with INT4 or centered INT2 weights, for
// models whose every product is statically quantized: a QUANTIZE rounds the
// activation to INT8, the product accumulates integers and its INT8 result is
// rounded and saturated again. Those rounding points are part of the trained
// function, so the kernels reproduce LiteRT's CPU path bit for bit where it is
// integer or defined in floats of one width:
//   QUANTIZE (XNNPACK f16-qs8-vcvt)  q = sat8(rne(x * fl16(1 / scale)))
//   DEQUANTIZE (XNNPACK qs8-f16-vcvt)  x = fl16(q * fl16(scale))
//   FULLY_CONNECTED, INT4 per-channel weights (XNNPACK qs8-qc4w, fp32
//     requantization)  q = sat8(rne(float(acc) * multiplier[row])),
//     multiplier = input scale * weight scale[row] / output scale in FP32
//   FULLY_CONNECTED and BATCH_MATMUL, blockwise centered INT2 weights (the
//     TFLite kernels' EvalBlockwiseFilter and EvalInt8BlockwiseRhs)
//     q = sat8(round(double(weight scale[row]) * (dot + sum(x) / 2) *
//                    (double(input scale) / double(output scale))))
// with rne round-to-nearest-even, round half away from zero, fl16 rounding
// to FLOAT16 and sat8 saturation to [-128, 127]. Where the model's graph
// holds a FLOAT16 tensor between float ops, the kernels round to FLOAT16
// there too. The outputs of an activation row depend on that row alone.

// The element type of a kernel input.
enum LiteRtNvidiaQuantType : int32_t {
  kLiteRtNvidiaQuantInt8 = 0,
  kLiteRtNvidiaQuantFp16 = 1,
  kLiteRtNvidiaQuantBf16 = 2,
  kLiteRtNvidiaQuantFp32 = 3,
};

enum LiteRtNvidiaQuantLayout : int32_t {
  // FULLY_CONNECTED weights [rows, columns], row-major.
  kLiteRtNvidiaQuantLayoutRows = 0,
  // BATCH_MATMUL weights [segments, columns, rows / segments]: block b
  // multiplies run b of the activation row.
  kLiteRtNvidiaQuantLayoutBlocks = 1,
};

enum LiteRtNvidiaQuantRequant : int32_t {
  // INT4 weights: sat8(rne(float(acc) * multipliers[row])).
  kLiteRtNvidiaQuantRequantFp32 = 0,
  // Centered INT2 weights: sat8(round(double(multipliers[row]) * acc *
  // ratio)), multipliers the weight scales.
  kLiteRtNvidiaQuantRequantDouble = 1,
  // Centered INT2 weights, the vocabulary head (TFLite's dynamic range
  // contract): FP32 logits fl16(float(double(multipliers[row]) * acc *
  // double(row_scale[m]))), then cap_out * tanh(logit * cap_in) when cap_in
  // is not 0.
  kLiteRtNvidiaQuantRequantLogits = 2,
};

// One product y = requantize(W . x) per activation row.
struct LiteRtNvidiaQuantLinearParams {
  int32_t layout;     // LiteRtNvidiaQuantLayout
  int32_t bit_width;  // 4: signed INT4; 2: centered INT2, value q + 0.5
  int32_t rows;       // outputs of an activation row
  int32_t columns;    // inputs of one output
  // The activation row holds `segments` runs of `columns` values. Rows
  // layout: output r reads run r / (rows / segments), so 1 gives every output
  // the same input (the q, k and v projections) and 2 the halves of a split
  // input. Blocks layout: the blocks.
  int32_t segments;
  // Output order: the output with natural index (a, b, c) in [swap0, swap1,
  // rows / (swap0 * swap1)] is written at (b, a, c), the regrouping between the
  // stages of a factorized MLP. 1, 1 keeps the natural order.
  int32_t swap0;
  int32_t swap1;
  // Rows layout only: a float activation (LiteRtNvidiaQuantType) is quantized
  // on the way in, q = sat8(rne(fl16(x) * quantize_scale)).
  int32_t activation;
  int32_t requant;  // LiteRtNvidiaQuantRequant
  int32_t lut;      // 1: q = lut[q + 128] follows the requantization
  float quantize_scale;
  float cap_in;
  float cap_out;
  double ratio;  // kLiteRtNvidiaQuantRequantDouble: input scale / output scale
};

extern "C" bool LiteRtNvidiaQuantLinearSupports(
    const LiteRtNvidiaQuantLinearParams* params);

// `m` activation rows [m, segments * columns] (INT8, or the float type of
// params->activation) times the packed weights as the model file holds them
// (INT4 two per byte, INT2 four per byte, least significant first; several
// row-major matrices of one width concatenated are one matrix), FP32
// multipliers [rows], an INT8 table of 256 entries when params->lut, FP32
// row scales [m] for the logits. Output [m, rows]: INT8, or FP32 logits.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantLinear(
    const LiteRtNvidiaQuantLinearParams* params, int32_t m,
    const void* activation, const uint8_t* weights, const float* multipliers,
    const int8_t* lut, const float* row_scale, void* output,
    cudaStream_t stream);

enum LiteRtNvidiaQuantGlueQuantize : int32_t {
  kLiteRtNvidiaQuantGlueNone = 0,
  // q = sat8(rne(z * quantize_scale)), quantize_scale = fl16(1 / scale).
  kLiteRtNvidiaQuantGlueStatic = 1,
  // TFLite's dynamic range quantization of a row: scale = max|z| / 127,
  // q = clamp(round(z * (127 / max|z|)), -127, 127); an all-zero row has
  // scale 1.
  kLiteRtNvidiaQuantGlueDynamic = 2,
};

// The float steps between two products of a block, per row of `width`
// values, each rounded where the model's graph holds a FLOAT16 tensor:
//   x' = fl16(fl16(x * c_x) + fl16(rms_norm(fl16(y * branch_scale);
//                                           branch_weights) * c_y))
//        with `branch` (the residual update behind a projection whose INT8
//        result is y), else x' = fl16(x)
//   z = fl16(rms_norm(x'; norm_weights)) with `norm`, else z = x'
//   q = quantize(z)
// rms_norm(v; w) = (v * rsqrt(mean(v^2) + eps)) * w in FP32.
struct LiteRtNvidiaQuantGlueParams {
  int32_t width;   // at most 4096
  int32_t x_type;  // LiteRtNvidiaQuantType of x, a float type
  int32_t branch;
  int32_t norm;
  int32_t quantize;    // LiteRtNvidiaQuantGlueQuantize
  int32_t z_type;      // LiteRtNvidiaQuantType of z_out: FP16 or FP32
  float branch_scale;  // fl16(scale of y)
  float branch_eps;
  float c_x;
  float c_y;
  float norm_eps;
  float quantize_scale;
};

extern "C" bool LiteRtNvidiaQuantGlueSupports(
    const LiteRtNvidiaQuantGlueParams* params);

// x [m, width] float, y [m, width] INT8 and branch_weights [width] FP32 with
// `branch`, norm_weights [width] FP32 with `norm`. Outputs, each optional
// (null): x_out [m, width] FP16 = x', z_out [m, width], q_out [m, width]
// INT8, row_scale_out [m] FP32 (the dynamic quantization's scales).
extern "C" cudaError_t LiteRtNvidiaLaunchQuantGlue(
    const LiteRtNvidiaQuantGlueParams* params, int32_t m, const void* x,
    const int8_t* y, const float* branch_weights, const float* norm_weights,
    void* x_out, void* z_out, int8_t* q_out, float* row_scale_out,
    cudaStream_t stream);

// The glue in front of a product and the product in one launch: per
// activation row, LiteRtNvidiaLaunchQuantGlue with `glue` (static
// quantization) and then LiteRtNvidiaLaunchQuantLinear with `linear` (rows
// layout, INT8 activations) on its q, with the results of the two launches,
// bit for bit. The CTAs of a row each run its glue, so the launch suits few
// rows (decode); `linear`'s rows may hold up to 256 weight words.
extern "C" bool LiteRtNvidiaQuantGlueLinearSupports(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantLinearParams* linear);

// The inputs of both launches (q stays on the chip); outputs x_out and z_out
// as LiteRtNvidiaLaunchQuantGlue's (optional), the product's INT8 [m, rows].
extern "C" cudaError_t LiteRtNvidiaLaunchQuantGlueLinear(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantLinearParams* linear, int32_t m, const void* x,
    const int8_t* y, const float* branch_weights, const float* norm_weights,
    const uint8_t* weights, const float* multipliers, const int8_t* lut,
    void* x_out, void* z_out, void* output, cudaStream_t stream);

// The steps between the q, k and v projections and the attention: every head
// of the INT8 projection results [tokens, (q_heads + k_heads + v_heads) *
// depth] (queries, keys, values) is dequantized (fl16(q * scale)),
// RMS-normalized over its depth with its weights and rounded to FLOAT16;
// queries and keys are rotated by the token's position with the
// half-rotation RoPE of odml.rope in FLOAT16 steps (cos and sin rounded to
// FLOAT16, every product and sum too), angle = float(position) *
// inverse_timescales[i] for the pair (i, i + depth / 2), an inverse
// timescale of 0 leaving its pair as it is (a proportion below 1); queries
// are then multiplied by q_multiplier. A head count of 0 leaves a projection
// out: a block that reads another block's cache has queries only, a block
// that only writes its cache keys and values.
struct LiteRtNvidiaQuantQkvRopeParams {
  int32_t tokens;
  int32_t q_heads;
  int32_t k_heads;
  int32_t v_heads;
  int32_t depth;  // a multiple of 2, at most 1024
  float q_scale;  // fl16 of the INT8 scales
  float k_scale;
  float v_scale;
  float epsilon;
  float q_multiplier;
};

extern "C" bool LiteRtNvidiaQuantQkvRopeSupports(
    const LiteRtNvidiaQuantQkvRopeParams* params);

// positions [tokens] INT32; q_norm, k_norm, v_norm [depth] FP32;
// inverse_timescales [depth / 2] FP32. FP16 outputs: q [q_heads, tokens,
// depth], k [k_heads, tokens, depth], v [tokens, v_heads, depth]. The norm
// weights and the output of a projection that is left out may be null.
extern "C" cudaError_t LiteRtNvidiaLaunchQuantQkvRope(
    const LiteRtNvidiaQuantQkvRopeParams* params, const int8_t* qkv,
    const int32_t* positions, const float* q_norm, const float* k_norm,
    const float* v_norm, const float* inverse_timescales, void* q, void* k,
    void* v, cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_QUANT_LINEAR_H_
