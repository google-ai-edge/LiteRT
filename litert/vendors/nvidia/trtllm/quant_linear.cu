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

#include "litert/vendors/nvidia/trtllm/quant_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;
// Outputs of one CTA of the rows layout: a warp reads the words of eight
// weight rows and reuses the activation bytes it prepared across them.
constexpr int kRowsPerBlock = 8;
constexpr int kGlueThreads = 256;
constexpr int kGlueMaxWidth = 4096;
// The kernels that load every weight word of their rows first
// (QuantRowsPrefetchKernel, QuantGlueRowsKernel) take rows of up to
// kPrefetchWords words per lane: 2,048 INT4 or 4,096 INT2 inputs.
constexpr int kPrefetchWords = 8;
// Outputs of a warp of QuantGlueRowsKernel when it only quantizes a float
// activation row: measured on the RTX 5080 against 2 and 8.
constexpr int kQuantizedRows = 4;
// Outputs of a warp of QuantGlueRowsKernel: rows of more than two words per
// lane, and the others. Measured on the RTX 5080 against 4 and 32.
constexpr int kFusedRowsLong = 8;
constexpr int kFusedRowsShort = 16;
constexpr int kMaxDepth = 1024;

__device__ __forceinline__ float RoundToFp16(float value) {
  return __half2float(__float2half_rn(value));
}

// Element `index` of a float input as the model's FLOAT16 tensor holds it.
__device__ __forceinline__ float LoadFp16Value(const void* data, int type,
                                               int64_t index) {
  switch (type) {
    case kLiteRtNvidiaQuantFp16:
      return __half2float(static_cast<const __half*>(data)[index]);
    case kLiteRtNvidiaQuantBf16:
      return RoundToFp16(
          __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]));
    default:
      return RoundToFp16(static_cast<const float*>(data)[index]);
  }
}

__device__ __forceinline__ int Saturate8(int value) {
  return max(-128, min(127, value));
}

// XNNPACK's f16-qs8-vcvt: the scale is the FLOAT16 reciprocal.
__device__ __forceinline__ int QuantizeFp16(float value, float scale) {
  return Saturate8(__float2int_rn(value * scale));
}

// Where output `row` is written (LiteRtNvidiaQuantLinearParams::swap0).
__device__ __forceinline__ int OutputIndex(
    const LiteRtNvidiaQuantLinearParams& p, int row) {
  if (p.swap0 == 1) return row;
  const int inner = p.rows / (p.swap0 * p.swap1);
  const int a = row / (p.swap1 * inner);
  const int b = row / inner % p.swap1;
  return (b * p.swap0 + a) * inner + row % inner;
}

// Requantizes and stores output `row` of activation row `m`. `exact` is the
// integer the reference accumulates: sum(x * q) for INT4 weights, twice
// sum(x * (q + 0.5)) for centered INT2.
__device__ __forceinline__ void StoreOutput(
    const LiteRtNvidiaQuantLinearParams& p, int row, int m, int exact,
    const float* multipliers, const int8_t* lut, const float* row_scale,
    void* output) {
  int q;
  if (p.requant == kLiteRtNvidiaQuantRequantFp32) {
    q = Saturate8(__float2int_rn(static_cast<float>(exact) * multipliers[row]));
  } else {
    double value = static_cast<double>(multipliers[row]) *
                   (0.5 * static_cast<double>(exact));
    if (p.requant == kLiteRtNvidiaQuantRequantLogits) {
      value *= static_cast<double>(row_scale[m]);
      float logit = RoundToFp16(static_cast<float>(value));
      if (p.cap_in != 0.0f) {
        logit = tanhf(logit * p.cap_in) * p.cap_out;
      }
      static_cast<float*>(output)[static_cast<int64_t>(m) * p.rows + row] =
          logit;
      return;
    }
    value = round(value * p.ratio);
    q = static_cast<int>(fmin(fmax(value, -128.0), 127.0));
  }
  if (p.lut != 0) {
    q = lut[q + 128];
  }
  static_cast<int8_t*>(
      output)[static_cast<int64_t>(m) * p.rows + OutputIndex(p, row)] =
      static_cast<int8_t>(q);
}

// Rows layout. grid (rows / kRowsPerBlock, m), one warp per CTA: lane l takes
// the 32-bit weight words l, l + 32, ... of the eight rows. The activations
// of a word are held as byte vectors that line up with the weight fields
// (INT4: the even and the odd fields; INT2: the fields 0..3 modulo 4), so a
// word costs one dp4a per vector and row. Weight fields are read with their
// sign bit flipped (non-negative bytes) and the offset is taken out through
// the activation sum.
template <int Bits>
__global__ void __launch_bounds__(kWarp)
    QuantRowsKernel(LiteRtNvidiaQuantLinearParams p,
                    const void* __restrict__ activation,
                    const uint8_t* __restrict__ weights,
                    const float* __restrict__ multipliers,
                    const int8_t* __restrict__ lut,
                    const float* __restrict__ row_scale,
                    void* __restrict__ output) {
  constexpr int kFields = 32 / Bits;
  constexpr int kVectors = Bits == 4 ? 2 : 4;
  constexpr uint32_t kSignBits = Bits == 4 ? 0x88888888u : 0xAAAAAAAAu;
  constexpr uint32_t kFieldMask = Bits == 4 ? 0x0F0F0F0Fu : 0x03030303u;
  const int lane = threadIdx.x;
  const int first_row = blockIdx.x * kRowsPerBlock;
  const int m = blockIdx.y;
  const int words = p.columns / kFields;
  const int segment = first_row / (p.rows / p.segments);
  const int64_t base =
      (static_cast<int64_t>(m) * p.segments + segment) * p.columns;
  const int64_t row_bytes = static_cast<int64_t>(p.columns) * Bits / 8;
  int accumulators[kRowsPerBlock] = {};
  int sum = 0;
  for (int word = lane; word < words; word += kWarp) {
    const int64_t first = base + static_cast<int64_t>(word) * kFields;
    uint32_t vectors[kVectors];
    if (p.activation == kLiteRtNvidiaQuantInt8) {
      const int* bytes = reinterpret_cast<const int*>(
          static_cast<const int8_t*>(activation) + first);
      if constexpr (Bits == 4) {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        vectors[0] = (a0 & 0xFFu) | ((a0 >> 8) & 0xFF00u) |
                     ((a1 << 16) & 0xFF0000u) | ((a1 << 8) & 0xFF000000u);
        vectors[1] = ((a0 >> 8) & 0xFFu) | ((a0 >> 16) & 0xFF00u) |
                     ((a1 << 8) & 0xFF0000u) | (a1 & 0xFF000000u);
      } else {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        const uint32_t a2 = bytes[2];
        const uint32_t a3 = bytes[3];
#pragma unroll
        for (int v = 0; v < kVectors; ++v) {
          vectors[v] = ((a0 >> (8 * v)) & 0xFFu) |
                       (((a1 >> (8 * v)) & 0xFFu) << 8) |
                       (((a2 >> (8 * v)) & 0xFFu) << 16) |
                       (((a3 >> (8 * v)) & 0xFFu) << 24);
        }
      }
    } else {
#pragma unroll
      for (int v = 0; v < kVectors; ++v) {
        vectors[v] = 0;
      }
#pragma unroll
      for (int field = 0; field < kFields; ++field) {
        const int q =
            QuantizeFp16(LoadFp16Value(activation, p.activation, first + field),
                         p.quantize_scale);
        vectors[field % kVectors] |= static_cast<uint32_t>(q & 0xFF)
                                     << (8 * (field / kVectors));
      }
    }
#pragma unroll
    for (int v = 0; v < kVectors; ++v) {
      sum = __dp4a(static_cast<int>(vectors[v]), 0x01010101, sum);
    }
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
      const auto* row_words = reinterpret_cast<const uint32_t*>(
          weights + (first_row + row) * row_bytes);
      const uint32_t bits = row_words[word] ^ kSignBits;
#pragma unroll
      for (int v = 0; v < kVectors; ++v) {
        accumulators[row] =
            __dp4a(static_cast<int>(vectors[v]),
                   static_cast<int>((bits >> (Bits * v)) & kFieldMask),
                   accumulators[row]);
      }
    }
  }
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
      accumulators[row] +=
          __shfl_xor_sync(kFullMask, accumulators[row], offset);
    }
    sum += __shfl_xor_sync(kFullMask, sum, offset);
  }
  if (lane < kRowsPerBlock) {
    int total = 0;
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
      if (row == lane) total = accumulators[row];
    }
    // Fields are read as q + 8 (INT4) and q + 2 (INT2, value q + 0.5).
    const int exact = Bits == 4 ? total - 8 * sum : 2 * total - 3 * sum;
    StoreOutput(p, first_row + lane, m, exact, multipliers, lut, row_scale,
                output);
  }
}

// The rows layout with every weight word of the CTA's rows loaded first:
// QuantRowsKernel's arithmetic for rows of at most 32 * Words words, whose
// loads are then all in flight at once instead of one round trip per word
// group.
template <int Bits, int Words>
__global__ void __launch_bounds__(kWarp)
    QuantRowsPrefetchKernel(LiteRtNvidiaQuantLinearParams p,
                            const void* __restrict__ activation,
                            const uint8_t* __restrict__ weights,
                            const float* __restrict__ multipliers,
                            const int8_t* __restrict__ lut,
                            const float* __restrict__ row_scale,
                            void* __restrict__ output) {
  constexpr int kFields = 32 / Bits;
  constexpr int kVectors = Bits == 4 ? 2 : 4;
  constexpr uint32_t kSignBits = Bits == 4 ? 0x88888888u : 0xAAAAAAAAu;
  constexpr uint32_t kFieldMask = Bits == 4 ? 0x0F0F0F0Fu : 0x03030303u;
  const int lane = threadIdx.x;
  const int first_row = blockIdx.x * kRowsPerBlock;
  const int m = blockIdx.y;
  const int words = p.columns / kFields;
  const int segment = first_row / (p.rows / p.segments);
  const int64_t base =
      (static_cast<int64_t>(m) * p.segments + segment) * p.columns;
  const int64_t row_bytes = static_cast<int64_t>(p.columns) * Bits / 8;
  uint32_t bits[kRowsPerBlock][Words];
#pragma unroll
  for (int row = 0; row < kRowsPerBlock; ++row) {
    const auto* row_words = reinterpret_cast<const uint32_t*>(
        weights + (first_row + row) * row_bytes);
#pragma unroll
    for (int i = 0; i < Words; ++i) {
      const int word = lane + i * kWarp;
      bits[row][i] = word < words ? row_words[word] ^ kSignBits : 0u;
    }
  }
  int accumulators[kRowsPerBlock] = {};
  int sum = 0;
#pragma unroll
  for (int i = 0; i < Words; ++i) {
    const int word = lane + i * kWarp;
    if (word >= words) continue;
    const int64_t first = base + static_cast<int64_t>(word) * kFields;
    uint32_t vectors[kVectors];
    if (p.activation == kLiteRtNvidiaQuantInt8) {
      const int* bytes = reinterpret_cast<const int*>(
          static_cast<const int8_t*>(activation) + first);
      if constexpr (Bits == 4) {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        vectors[0] = (a0 & 0xFFu) | ((a0 >> 8) & 0xFF00u) |
                     ((a1 << 16) & 0xFF0000u) | ((a1 << 8) & 0xFF000000u);
        vectors[1] = ((a0 >> 8) & 0xFFu) | ((a0 >> 16) & 0xFF00u) |
                     ((a1 << 8) & 0xFF0000u) | (a1 & 0xFF000000u);
      } else {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        const uint32_t a2 = bytes[2];
        const uint32_t a3 = bytes[3];
#pragma unroll
        for (int v = 0; v < kVectors; ++v) {
          vectors[v] = ((a0 >> (8 * v)) & 0xFFu) |
                       (((a1 >> (8 * v)) & 0xFFu) << 8) |
                       (((a2 >> (8 * v)) & 0xFFu) << 16) |
                       (((a3 >> (8 * v)) & 0xFFu) << 24);
        }
      }
    } else {
#pragma unroll
      for (int v = 0; v < kVectors; ++v) {
        vectors[v] = 0;
      }
#pragma unroll
      for (int field = 0; field < kFields; ++field) {
        const int q =
            QuantizeFp16(LoadFp16Value(activation, p.activation, first + field),
                         p.quantize_scale);
        vectors[field % kVectors] |= static_cast<uint32_t>(q & 0xFF)
                                     << (8 * (field / kVectors));
      }
    }
#pragma unroll
    for (int v = 0; v < kVectors; ++v) {
      sum = __dp4a(static_cast<int>(vectors[v]), 0x01010101, sum);
    }
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
#pragma unroll
      for (int v = 0; v < kVectors; ++v) {
        accumulators[row] =
            __dp4a(static_cast<int>(vectors[v]),
                   static_cast<int>((bits[row][i] >> (Bits * v)) & kFieldMask),
                   accumulators[row]);
      }
    }
  }
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
      accumulators[row] +=
          __shfl_xor_sync(kFullMask, accumulators[row], offset);
    }
    sum += __shfl_xor_sync(kFullMask, sum, offset);
  }
  if (lane < kRowsPerBlock) {
    int total = 0;
#pragma unroll
    for (int row = 0; row < kRowsPerBlock; ++row) {
      if (row == lane) total = accumulators[row];
    }
    const int exact = Bits == 4 ? total - 8 * sum : 2 * total - 3 * sum;
    StoreOutput(p, first_row + lane, m, exact, multipliers, lut, row_scale,
                output);
  }
}

// Blocks layout (centered INT2). grid (segments * groups, m), one warp per
// CTA: its lanes are `slices` runs of the block's input rows times 32 /
// slices weight words of 16 outputs each, so a lane multiplies one run of
// activations with one word column and the runs are added across the warp.
__global__ void __launch_bounds__(kWarp)
    QuantBlocksKernel(LiteRtNvidiaQuantLinearParams p, int slices,
                      const int8_t* __restrict__ activation,
                      const uint8_t* __restrict__ weights,
                      const float* __restrict__ multipliers,
                      const int8_t* __restrict__ lut,
                      int8_t* __restrict__ output) {
  constexpr int kOutputsPerWord = 16;
  const int lane = threadIdx.x;
  const int outputs = p.rows / p.segments;
  const int words = outputs / kOutputsPerWord;
  const int words_per_warp = kWarp / slices;
  const int groups = (words + words_per_warp - 1) / words_per_warp;
  const int block = blockIdx.x / groups;
  const int word = blockIdx.x % groups * words_per_warp + lane % words_per_warp;
  const int slice = lane / words_per_warp;
  const int m = blockIdx.y;
  const int run = p.columns / slices;
  const int64_t first_input =
      static_cast<int64_t>(block) * p.columns + slice * run;
  int accumulators[kOutputsPerWord] = {};
  int sum = 0;
  if (word < words) {
    const int8_t* inputs = activation +
                           static_cast<int64_t>(m) * p.segments * p.columns +
                           first_input;
    const auto* column =
        reinterpret_cast<const uint32_t*>(weights) + first_input * words + word;
    for (int k = 0; k < run; ++k) {
      const int x = inputs[k];
      const uint32_t bits =
          column[static_cast<int64_t>(k) * words] ^ 0xAAAAAAAAu;
      sum += x;
#pragma unroll
      for (int i = 0; i < kOutputsPerWord; ++i) {
        accumulators[i] += x * static_cast<int>((bits >> (2 * i)) & 3u);
      }
    }
  }
  for (int offset = words_per_warp; offset < kWarp; offset *= 2) {
#pragma unroll
    for (int i = 0; i < kOutputsPerWord; ++i) {
      accumulators[i] += __shfl_xor_sync(kFullMask, accumulators[i], offset);
    }
    sum += __shfl_xor_sync(kFullMask, sum, offset);
  }
  if (word >= words) return;
    // Every lane of a word holds its totals: the slices share the outputs.
#pragma unroll
  for (int i = 0; i < kOutputsPerWord; ++i) {
    if (i % slices == slice) {
      StoreOutput(p, block * outputs + word * kOutputsPerWord + i, m,
                  2 * accumulators[i] - 3 * sum, multipliers, lut, nullptr,
                  output);
    }
  }
}

// The sum of `value` over the CTA's threads; `partial` has a slot per warp.
__device__ __forceinline__ float BlockSum(float value, float* partial) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  if (threadIdx.x % kWarp == 0) partial[threadIdx.x / kWarp] = value;
  __syncthreads();
  float total = 0.0f;
  for (int warp = 0; warp < blockDim.x / kWarp; ++warp) {
    total += partial[warp];
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ float BlockMax(float value, float* partial) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
  }
  if (threadIdx.x % kWarp == 0) partial[threadIdx.x / kWarp] = value;
  __syncthreads();
  float total = 0.0f;
  for (int warp = 0; warp < blockDim.x / kWarp; ++warp) {
    total = fmaxf(total, partial[warp]);
  }
  __syncthreads();
  return total;
}

// sum + value^2 as a product and a sum, never fused: the host reference of
// the tests adds the squares of an RMS norm the same way.
__device__ __forceinline__ float AddSquare(float sum, float value) {
  return __fadd_rn(sum, __fmul_rn(value, value));
}

// rsqrt(mean + eps) as the decomposition's ADD and RSQRT give it.
__device__ __forceinline__ float NormFactor(float sum_squares, int count,
                                            float epsilon) {
  return 1.0f / sqrtf(sum_squares / static_cast<float>(count) + epsilon);
}

// The glue of activation row `row` by a CTA of kGlueThreads threads: thread t
// holds the elements t, t + 256, ... Element i of q goes to q_row[i]; x_out,
// z_out and row_scale_out are written where not null.
__device__ __forceinline__ void GlueRow(
    const LiteRtNvidiaQuantGlueParams& p, int row, const void* __restrict__ x,
    const int8_t* __restrict__ y, const float* __restrict__ branch_weights,
    const float* __restrict__ norm_weights, __half* __restrict__ x_out,
    void* __restrict__ z_out, int8_t* __restrict__ q_row,
    float* __restrict__ row_scale_out) {
  constexpr int kPerThread = kGlueMaxWidth / kGlueThreads;
  __shared__ float partial[kGlueThreads / kWarp];
  const int64_t base = static_cast<int64_t>(row) * p.width;
  float values[kPerThread];
  int count = 0;
  for (int i = threadIdx.x; i < p.width; i += kGlueThreads) {
    values[count++] = LoadFp16Value(x, p.x_type, base + i);
  }
  if (p.branch != 0) {
    float branch[kPerThread];
    float squares = 0.0f;
    for (int j = 0; j < count; ++j) {
      // XNNPACK's qs8-f16-vcvt: the scale is the FLOAT16 one.
      branch[j] = RoundToFp16(
          static_cast<float>(y[base + threadIdx.x + j * kGlueThreads]) *
          p.branch_scale);
      squares = AddSquare(squares, branch[j]);
    }
    const float factor =
        NormFactor(BlockSum(squares, partial), p.width, p.branch_eps);
    for (int j = 0; j < count; ++j) {
      const int i = threadIdx.x + j * kGlueThreads;
      const float scaled = branch[j] * factor;
      const float normed = RoundToFp16(scaled * branch_weights[i]);
      values[j] = RoundToFp16(RoundToFp16(values[j] * p.c_x) +
                              RoundToFp16(normed * p.c_y));
      if (x_out != nullptr) x_out[base + i] = __float2half_rn(values[j]);
    }
  } else if (x_out != nullptr) {
    for (int j = 0; j < count; ++j) {
      x_out[base + threadIdx.x + j * kGlueThreads] = __float2half_rn(values[j]);
    }
  }
  if (p.norm != 0) {
    float squares = 0.0f;
    for (int j = 0; j < count; ++j) {
      squares = AddSquare(squares, values[j]);
    }
    const float factor =
        NormFactor(BlockSum(squares, partial), p.width, p.norm_eps);
    for (int j = 0; j < count; ++j) {
      const float scaled = values[j] * factor;
      values[j] =
          RoundToFp16(scaled * norm_weights[threadIdx.x + j * kGlueThreads]);
    }
  }
  float inverse = p.quantize_scale;
  if (p.quantize == kLiteRtNvidiaQuantGlueDynamic) {
    float largest = 0.0f;
    for (int j = 0; j < count; ++j) {
      largest = fmaxf(largest, fabsf(values[j]));
    }
    largest = BlockMax(largest, partial);
    inverse = largest == 0.0f ? 0.0f : 127.0f / largest;
    if (threadIdx.x == 0 && row_scale_out != nullptr) {
      row_scale_out[row] = largest == 0.0f ? 1.0f : largest / 127.0f;
    }
  }
  for (int j = 0; j < count; ++j) {
    const int column = threadIdx.x + j * kGlueThreads;
    const int64_t i = base + column;
    if (z_out != nullptr) {
      if (p.z_type == kLiteRtNvidiaQuantFp32) {
        static_cast<float*>(z_out)[i] = values[j];
      } else {
        static_cast<__half*>(z_out)[i] = __float2half_rn(values[j]);
      }
    }
    if (q_row == nullptr) continue;
    if (p.quantize == kLiteRtNvidiaQuantGlueDynamic) {
      q_row[column] = static_cast<int8_t>(
          fminf(fmaxf(roundf(values[j] * inverse), -127.0f), 127.0f));
    } else {
      q_row[column] = static_cast<int8_t>(QuantizeFp16(values[j], inverse));
    }
  }
}

// grid (m): one CTA per row.
__global__ void __launch_bounds__(kGlueThreads)
    QuantGlueKernel(LiteRtNvidiaQuantGlueParams p, const void* __restrict__ x,
                    const int8_t* __restrict__ y,
                    const float* __restrict__ branch_weights,
                    const float* __restrict__ norm_weights,
                    __half* __restrict__ x_out, void* __restrict__ z_out,
                    int8_t* __restrict__ q_out,
                    float* __restrict__ row_scale_out) {
  GlueRow(p, blockIdx.x, x, y, branch_weights, norm_weights, x_out, z_out,
          q_out != nullptr
              ? q_out + static_cast<int64_t>(blockIdx.x) * p.width
              : nullptr,
          row_scale_out);
}

// The glue in front of a product and the product in one launch
// (LiteRtNvidiaLaunchQuantGlueLinear). grid (ceil(rows / (8 * Rows)), m) of
// kGlueThreads threads: every CTA runs the glue of its activation row
// (GlueRow, the arithmetic of the glue kernel) into shared memory, the first
// CTA of a row also writing x' and z, and warp w then takes the Rows outputs
// from (8 * blockIdx.x + w) * Rows as QuantRowsKernel computes them: one
// integer dot product per output, the same requantization. The weight words
// (at most Words per lane and row) are loaded before the glue, so their loads
// overlap it. With QuantizeOnly the CTA only quantizes the float activation
// row as the rows kernels do (a product's own QUANTIZE, LiteRtNvidiaLaunch-
// QuantLinear), once for all its warps instead of once per warp.
template <int Bits, int Rows, int Words, bool QuantizeOnly = false>
__global__ void __launch_bounds__(kGlueThreads) QuantGlueRowsKernel(
    LiteRtNvidiaQuantGlueParams g, LiteRtNvidiaQuantLinearParams p,
    const void* __restrict__ x, const int8_t* __restrict__ y,
    const float* __restrict__ branch_weights,
    const float* __restrict__ norm_weights, const uint8_t* __restrict__ weights,
    const float* __restrict__ multipliers, const int8_t* __restrict__ lut,
    __half* __restrict__ x_out, void* __restrict__ z_out,
    void* __restrict__ output) {
  constexpr int kRows = Rows;
  constexpr int kFields = 32 / Bits;
  constexpr int kVectors = Bits == 4 ? 2 : 4;
  constexpr uint32_t kSignBits = Bits == 4 ? 0x88888888u : 0xAAAAAAAAu;
  constexpr uint32_t kFieldMask = Bits == 4 ? 0x0F0F0F0Fu : 0x03030303u;
  __shared__ __align__(16) int8_t q[kGlueMaxWidth];
  const int lane = threadIdx.x % kWarp;
  const int first_row =
      (blockIdx.x * (kGlueThreads / kWarp) + threadIdx.x / kWarp) * kRows;
  const int m = blockIdx.y;
  const int words = p.columns / kFields;
  const int64_t row_bytes = static_cast<int64_t>(p.columns) * Bits / 8;
  const bool active = first_row < p.rows;
  uint32_t bits[kRows][Words];
#pragma unroll
  for (int row = 0; row < kRows; ++row) {
    const auto* row_words = reinterpret_cast<const uint32_t*>(
        weights + (active ? first_row + row : 0) * row_bytes);
#pragma unroll
    for (int i = 0; i < Words; ++i) {
      const int word = lane + i * kWarp;
      bits[row][i] = active && word < words ? row_words[word] ^ kSignBits : 0u;
    }
  }
  if constexpr (QuantizeOnly) {
    const int64_t base = static_cast<int64_t>(m) * g.width;
    for (int i = threadIdx.x; i < g.width; i += kGlueThreads) {
      q[i] = static_cast<int8_t>(
          QuantizeFp16(LoadFp16Value(x, g.x_type, base + i), g.quantize_scale));
    }
  } else {
    const bool first = blockIdx.x == 0;
    GlueRow(g, m, x, y, branch_weights, norm_weights, first ? x_out : nullptr,
            first ? z_out : nullptr, q, nullptr);
  }
  __syncthreads();
  if (!active) return;
  const int segment = first_row / (p.rows / p.segments);
  const int8_t* activation = q + segment * p.columns;
  int accumulators[kRows] = {};
  int sum = 0;
#pragma unroll
  for (int i = 0; i < Words; ++i) {
    const int word = lane + i * kWarp;
    if (word < words) {
      const int* bytes =
          reinterpret_cast<const int*>(activation + word * kFields);
      uint32_t vectors[kVectors];
      if constexpr (Bits == 4) {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        vectors[0] = (a0 & 0xFFu) | ((a0 >> 8) & 0xFF00u) |
                     ((a1 << 16) & 0xFF0000u) | ((a1 << 8) & 0xFF000000u);
        vectors[1] = ((a0 >> 8) & 0xFFu) | ((a0 >> 16) & 0xFF00u) |
                     ((a1 << 8) & 0xFF0000u) | (a1 & 0xFF000000u);
      } else {
        const uint32_t a0 = bytes[0];
        const uint32_t a1 = bytes[1];
        const uint32_t a2 = bytes[2];
        const uint32_t a3 = bytes[3];
#pragma unroll
        for (int v = 0; v < kVectors; ++v) {
          vectors[v] = ((a0 >> (8 * v)) & 0xFFu) |
                       (((a1 >> (8 * v)) & 0xFFu) << 8) |
                       (((a2 >> (8 * v)) & 0xFFu) << 16) |
                       (((a3 >> (8 * v)) & 0xFFu) << 24);
        }
      }
#pragma unroll
      for (int v = 0; v < kVectors; ++v) {
        sum = __dp4a(static_cast<int>(vectors[v]), 0x01010101, sum);
      }
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
#pragma unroll
        for (int v = 0; v < kVectors; ++v) {
          accumulators[row] = __dp4a(
              static_cast<int>(vectors[v]),
              static_cast<int>((bits[row][i] >> (Bits * v)) & kFieldMask),
              accumulators[row]);
        }
      }
    }
  }
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      accumulators[row] +=
          __shfl_xor_sync(kFullMask, accumulators[row], offset);
    }
    sum += __shfl_xor_sync(kFullMask, sum, offset);
  }
  if (lane < kRows) {
    int total = 0;
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
      if (row == lane) total = accumulators[row];
    }
    // Fields are read as q + 8 (INT4) and q + 2 (INT2, value q + 0.5).
    const int exact = Bits == 4 ? total - 8 * sum : 2 * total - 3 * sum;
    StoreOutput(p, first_row + lane, m, exact, multipliers, lut, nullptr,
                output);
  }
}

// The factorized MLP in one launch (LiteRtNvidiaLaunchQuantFactorizedMlp): a
// co-resident grid of a CTA per multiprocessor. Each CTA owns an equal share of
// the rows of every stage. Its weight rows of a stage are one bulk copy into
// shared memory (buffer A holds stages 0 and 2, buffer B stages 1 and 3) that
// nobody waits for until the stage runs: the copy of stage s + 1 is started
// when stage s begins, after that stage's activations and multipliers were
// requested, since a request waits behind the bulk copies that the
// multiprocessor started before it. Between the stages a grid barrier; the
// activations, which other CTAs wrote, come from L2.
constexpr int kFactorizedMlpStages = 4;
constexpr int kFactorizedMlpThreads = 256;
// Of a stage, a thread loads at most this many 16-byte runs of activations
// and requantizes at most this many rows.
constexpr int kFactorizedMlpChunks = 2;
constexpr int kFactorizedMlpRowsPerThread = 4;
// The 16-byte runs of a glue's inputs that a thread loads: up to 28 KB.
constexpr int kFactorizedMlpGlueChunks = 7;
constexpr int kFactorizedMlpTable = 256;
static_assert(kFactorizedMlpThreads == kFactorizedMlpTable,
              "a thread per table entry");

struct FactorizedMlpPointers {
  // The activation, then the results of stages 0 to 2.
  const int8_t* in[kFactorizedMlpStages];
  // The results of stages 0 to 2, then the output.
  int8_t* out[kFactorizedMlpStages];
  const uint8_t* weights[kFactorizedMlpStages];
  const float* multipliers[kFactorizedMlpStages];
  const int8_t* lut;
  // [0] arrivals at the grid barriers, [1] CTAs that are done.
  unsigned* state;
};

// The dynamic shared memory of a CTA: weight buffers A and B, the
// activation runs of a stage, the exact sums of its rows.
struct FactorizedMlpLayout {
  int32_t a_bytes;
  int32_t b_bytes;
  int32_t act_bytes;
};

// The glue in front of the MLP (LiteRtNvidiaLaunchQuantGlueFactorizedMlp),
// whose q is the MLP's activation.
struct FactorizedMlpGlue {
  LiteRtNvidiaQuantGlueParams params;
  const void* x;
  const int8_t* y;
  const float* branch_weights;
  const float* norm_weights;
  __half* x_out;
  void* z_out;
};

// The rows of a stage are shared out in steps of this many, so that every
// CTA's rows start on 16 bytes of the weights (rows of row_bytes bytes, a
// multiple of 4).
__host__ __device__ inline int FactorizedMlpStep(int row_bytes) {
  return row_bytes % 16 == 0 ? 1 : (row_bytes % 8 == 0 ? 2 : 4);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

constexpr int kFactorizedMlpWarps = kFactorizedMlpThreads / kWarp;

struct FactorizedMlpShare {
  int first;
  int last;
};

// The rows [first, last) of stage `p` that CTA `cta` of `ctas` owns.
__device__ __forceinline__ FactorizedMlpShare
FactorizedMlpRows(const LiteRtNvidiaQuantLinearParams& p, int cta, int ctas) {
  const int step = FactorizedMlpStep(p.columns / 4);
  const int steps = p.rows / step;
  return {steps * cta / ctas * step, steps * (cta + 1) / ctas * step};
}

struct FactorizedMlpPending {
  int4 chunks[kFactorizedMlpChunks];
  float multipliers[kFactorizedMlpRowsPerThread];
  int first_segment;
};

__device__ __forceinline__ uint32_t SharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void InitCopyBarrier(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
                   SharedAddress(barrier))
               : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// `bytes` from global `source` into shared `target`: the next phase of
// `barrier` completes when they are there. The buffer's earlier readers are
// done (a __syncthreads() in between).
__device__ __forceinline__ void BulkCopy(void* target, const void* source,
                                         int bytes, uint64_t* barrier) {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          SharedAddress(barrier)),
      "r"(bytes)
      : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n" ::"r"(SharedAddress(target)),
      "l"(source), "r"(bytes), "r"(SharedAddress(barrier))
      : "memory");
}

__device__ __forceinline__ void WaitCopy(uint64_t* barrier, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT;\n"
      "}\n" ::"r"(SharedAddress(barrier)),
      "r"(phase)
      : "memory");
}

// Every CTA's results of the stages so far are written before any CTA goes
// on: the arrivals reach `target`.
__device__ __forceinline__ void FactorizedMlpGridBarrier(unsigned* arrivals,
                                                         unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(arrivals)
                 : "memory");
    unsigned seen = 0;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n"
                   : "=r"(seen)
                   : "l"(arrivals)
                   : "memory");
    } while (seen < target);
  }
  __syncthreads();
}

// Requests stage `p`'s activation runs that this CTA's rows read (none when
// `in` is null: the glue in front makes the whole row) and the rows'
// multipliers.
__device__ __forceinline__ FactorizedMlpPending LoadFactorizedMlpStage(
    const LiteRtNvidiaQuantLinearParams& p, FactorizedMlpShare share,
    const int8_t* in, const float* multipliers) {
  FactorizedMlpPending pending = {};
  if (share.first == share.last) return pending;
  const int segment_rows = p.rows / p.segments;
  pending.first_segment = in != nullptr ? share.first / segment_rows : 0;
  const int chunks = in == nullptr ? 0
                                   : ((share.last - 1) / segment_rows -
                                      pending.first_segment + 1) *
                                         p.columns / 16;
  const int4* source = reinterpret_cast<const int4*>(
      in + static_cast<int64_t>(pending.first_segment) * p.columns);
#pragma unroll
  for (int k = 0; k < kFactorizedMlpChunks; ++k) {
    const int i = threadIdx.x + k * kFactorizedMlpThreads;
    if (i < chunks) pending.chunks[k] = __ldcg(source + i);
  }
#pragma unroll
  for (int k = 0; k < kFactorizedMlpRowsPerThread; ++k) {
    const int row = share.first + threadIdx.x + k * kFactorizedMlpThreads;
    if (row < share.last) pending.multipliers[k] = __ldg(multipliers + row);
  }
  return pending;
}

// The glue's inputs x, y, the branch norm's and the norm's weights (those it
// has) one after another in 16-byte runs: run i of input a is run
// starts[a] + i of the whole.
struct GlueRuns {
  const int4* sources[4];
  int starts[5];
};

__device__ __forceinline__ GlueRuns
FactorizedMlpGlueRuns(const FactorizedMlpGlue& g) {
  const int width = g.params.width;
  const int bytes[4] = {
      width * (g.params.x_type == kLiteRtNvidiaQuantFp32 ? 4 : 2),
      g.params.branch != 0 ? width : 0, g.params.branch != 0 ? 4 * width : 0,
      g.params.norm != 0 ? 4 * width : 0};
  GlueRuns runs;
  runs.sources[0] = static_cast<const int4*>(g.x);
  runs.sources[1] = reinterpret_cast<const int4*>(g.y);
  runs.sources[2] = reinterpret_cast<const int4*>(g.branch_weights);
  runs.sources[3] = reinterpret_cast<const int4*>(g.norm_weights);
  runs.starts[0] = 0;
  for (int a = 0; a < 4; ++a) {
    runs.starts[a + 1] = runs.starts[a] + bytes[a] / 16;
  }
  return runs;
}

struct GlueStaging {
  int4 chunks[kFactorizedMlpGlueChunks];
};

// Requests the glue's inputs, all at once (its own loads would make three
// round trips, behind the weight copies).
__device__ __forceinline__ GlueStaging
LoadFactorizedMlpGlue(const GlueRuns& runs) {
  GlueStaging staging;
#pragma unroll
  for (int k = 0; k < kFactorizedMlpGlueChunks; ++k) {
    const int i = threadIdx.x + k * kFactorizedMlpThreads;
    int a = 0;
    while (a < 3 && i >= runs.starts[a + 1]) ++a;
    if (i < runs.starts[4]) {
      staging.chunks[k] = __ldcg(runs.sources[a] + (i - runs.starts[a]));
    }
  }
  return staging;
}

__device__ __forceinline__ void StoreFactorizedMlpGlue(
    const GlueRuns& runs, const GlueStaging& staging, uint8_t* target) {
#pragma unroll
  for (int k = 0; k < kFactorizedMlpGlueChunks; ++k) {
    const int i = threadIdx.x + k * kFactorizedMlpThreads;
    if (i < runs.starts[4]) {
      reinterpret_cast<int4*>(target)[i] = staging.chunks[k];
    }
  }
}

__device__ __forceinline__ void StoreFactorizedMlpActivations(
    const LiteRtNvidiaQuantLinearParams& p, FactorizedMlpShare share,
    const FactorizedMlpPending& pending, int8_t* act) {
  if (share.first == share.last) return;
  const int segment_rows = p.rows / p.segments;
  const int chunks = ((share.last - 1) / segment_rows -
                      pending.first_segment + 1) *
                     p.columns / 16;
#pragma unroll
  for (int k = 0; k < kFactorizedMlpChunks; ++k) {
    const int i = threadIdx.x + k * kFactorizedMlpThreads;
    if (i < chunks) reinterpret_cast<int4*>(act)[i] = pending.chunks[k];
  }
}

// The exact sums of this CTA's rows of a stage into totals[row - first]:
// QuantRowsKernel's integers on units of kRowsPerBlock rows that start at
// multiples of kRowsPerBlock, Lanes lanes a unit (a warp runs 32 / Lanes
// units at once), rows outside the share skipped.
template <int Lanes>
__device__ __forceinline__ void FactorizedMlpSums(
    const LiteRtNvidiaQuantLinearParams& p, FactorizedMlpShare share,
    const uint32_t* __restrict__ weights, const int8_t* __restrict__ act,
    int first_segment, int* __restrict__ totals) {
  constexpr int kUnits = kWarp / Lanes;
  const int words = p.columns / 16;
  const int segment_rows = p.rows / p.segments;
  const int lane = threadIdx.x % Lanes;
  int unit = share.first / kRowsPerBlock +
             threadIdx.x / kWarp * kUnits + threadIdx.x % kWarp / Lanes;
  // The shuffles take every lane of the warp: a warp ends together.
  while (__any_sync(kFullMask, unit * kRowsPerBlock < share.last)) {
    const int unit_first = unit * kRowsPerBlock;
    const bool live = unit_first < share.last;
    const int8_t* run =
        act + (live ? unit_first / segment_rows - first_segment : 0) *
                  p.columns;
    int accumulators[kRowsPerBlock] = {};
    int sum = 0;
    for (int word = lane; live && word < words; word += Lanes) {
      const int4 bytes = *reinterpret_cast<const int4*>(run + 16 * word);
      const uint32_t a0 = bytes.x;
      const uint32_t a1 = bytes.y;
      const uint32_t a2 = bytes.z;
      const uint32_t a3 = bytes.w;
      uint32_t vectors[4];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        vectors[v] = ((a0 >> (8 * v)) & 0xFFu) |
                     (((a1 >> (8 * v)) & 0xFFu) << 8) |
                     (((a2 >> (8 * v)) & 0xFFu) << 16) |
                     (((a3 >> (8 * v)) & 0xFFu) << 24);
      }
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        sum = __dp4a(static_cast<int>(vectors[v]), 0x01010101, sum);
      }
#pragma unroll
      for (int row = 0; row < kRowsPerBlock; ++row) {
        const int r = unit_first + row;
        const uint32_t bits =
            r >= share.first && r < share.last
                ? weights[(r - share.first) * words + word] ^ 0xAAAAAAAAu
                : 0u;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          accumulators[row] =
              __dp4a(static_cast<int>(vectors[v]),
                     static_cast<int>((bits >> (2 * v)) & 0x03030303u),
                     accumulators[row]);
        }
      }
    }
#pragma unroll
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int row = 0; row < kRowsPerBlock; ++row) {
        accumulators[row] +=
            __shfl_xor_sync(kFullMask, accumulators[row], offset);
      }
      sum += __shfl_xor_sync(kFullMask, sum, offset);
    }
    if (live && lane < kRowsPerBlock) {
      int total = 0;
#pragma unroll
      for (int row = 0; row < kRowsPerBlock; ++row) {
        if (row == lane) total = accumulators[row];
      }
      const int r = unit_first + lane;
      if (r >= share.first && r < share.last) {
        totals[r - share.first] = 2 * total - 3 * sum;
      }
    }
    unit += kFactorizedMlpWarps * kUnits;
  }
}

// StoreOutput's requantization for kLiteRtNvidiaQuantRequantDouble.
__device__ __forceinline__ int RequantizeDouble(
    const LiteRtNvidiaQuantLinearParams& p, float multiplier, int exact) {
  double value = static_cast<double>(multiplier) *
                 (0.5 * static_cast<double>(exact));
  value = round(value * p.ratio);
  return static_cast<int>(fmin(fmax(value, -128.0), 127.0));
}

// Requantizes and stores this CTA's rows of a stage, a row per thread.
__device__ __forceinline__ void StoreFactorizedMlpRows(
    const LiteRtNvidiaQuantLinearParams& p, FactorizedMlpShare share,
    const FactorizedMlpPending& pending, const int* totals, const int8_t* table,
    int8_t* out) {
#pragma unroll
  for (int k = 0; k < kFactorizedMlpRowsPerThread; ++k) {
    const int row = share.first + threadIdx.x + k * kFactorizedMlpThreads;
    if (row >= share.last) break;
    int q = RequantizeDouble(p, pending.multipliers[k],
                             totals[row - share.first]);
    if (p.lut != 0) q = table[q + 128];
    out[OutputIndex(p, row)] = static_cast<int8_t>(q);
  }
}

#endif  // __CUDA_ARCH__ >= 900

template <bool Glue>
__global__ void __launch_bounds__(kFactorizedMlpThreads, 1)
    QuantFactorizedMlpKernel(LiteRtNvidiaQuantFactorizedMlpParams p,
                             FactorizedMlpPointers ptr,
                             FactorizedMlpLayout layout,
                             FactorizedMlpGlue glue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) uint8_t shared[];
  __shared__ uint64_t loaded[2];
  __shared__ int8_t table[kFactorizedMlpTable];
  uint8_t* buffers[2] = {shared, shared + layout.a_bytes};
  auto* act = reinterpret_cast<int8_t*>(shared + layout.a_bytes +
                                        layout.b_bytes);
  auto* totals = reinterpret_cast<int*>(shared + layout.a_bytes +
                                        layout.b_bytes + layout.act_bytes);
  FactorizedMlpShare shares[kFactorizedMlpStages];
#pragma unroll
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    shares[s] = FactorizedMlpRows(p.stages[s], blockIdx.x, gridDim.x);
  }
  // Thread 0 runs the grid barriers, warp 1 the copies.
  const bool copier = threadIdx.x == kWarp;
  const auto copy = [&](int s) {
    if (copier && shares[s].first != shares[s].last) {
      const int row_bytes = p.stages[s].columns / 4;
      BulkCopy(buffers[s % 2],
               ptr.weights[s] +
                   static_cast<int64_t>(shares[s].first) * row_bytes,
               (shares[s].last - shares[s].first) * row_bytes, &loaded[s % 2]);
    }
  };
  uint32_t phases[2] = {0, 0};
  if (ptr.lut != nullptr) table[threadIdx.x] = ptr.lut[threadIdx.x];
  if (copier) {
    InitCopyBarrier(&loaded[0]);
    InitCopyBarrier(&loaded[1]);
  }
#pragma unroll
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    const auto& stage = p.stages[s];
    const FactorizedMlpShare share = shares[s];
    const bool glued = Glue && s == 0;
    const FactorizedMlpPending pending = LoadFactorizedMlpStage(
        stage, share, glued ? nullptr : ptr.in[s], ptr.multipliers[s]);
    const GlueRuns runs = glued ? FactorizedMlpGlueRuns(glue) : GlueRuns{};
    const GlueStaging staging =
        glued ? LoadFactorizedMlpGlue(runs) : GlueStaging{};
    __syncthreads();
    // Buffer (s + 1) % 2 held stage s - 1, done before the grid barrier;
    // stage 1's copy waits until stage 0's weights are in.
    if (s == 0 || s + 1 < kFactorizedMlpStages) copy(s == 0 ? 0 : s + 1);
    if (glued) {
      // Every CTA runs the glue of the row into its activations from copies
      // of its inputs in buffer B (free until stage 1's copy), the first one
      // writes x' and z.
      StoreFactorizedMlpGlue(runs, staging, buffers[1]);
      __syncthreads();
      const uint8_t* inputs = buffers[1];
      const bool first = blockIdx.x == 0;
      GlueRow(glue.params, 0, inputs,
              reinterpret_cast<const int8_t*>(inputs + 16 * runs.starts[1]),
              reinterpret_cast<const float*>(inputs + 16 * runs.starts[2]),
              reinterpret_cast<const float*>(inputs + 16 * runs.starts[3]),
              first ? glue.x_out : nullptr, first ? glue.z_out : nullptr, act,
              nullptr);
    } else {
      StoreFactorizedMlpActivations(stage, share, pending, act);
    }
    __syncthreads();
    if (share.first != share.last) {
      WaitCopy(&loaded[s % 2], phases[s % 2]);
      phases[s % 2] ^= 1;
    }
    if (s == 0) copy(1);
    if (share.first != share.last) {
      const auto* weights = reinterpret_cast<const uint32_t*>(buffers[s % 2]);
      if (stage.columns / 16 <= kWarp / 2) {
        FactorizedMlpSums<kWarp / 2>(stage, share, weights, act,
                                     pending.first_segment, totals);
      } else {
        FactorizedMlpSums<kWarp>(stage, share, weights, act,
                                 pending.first_segment, totals);
      }
    }
    __syncthreads();
    StoreFactorizedMlpRows(stage, share, pending, totals, table, ptr.out[s]);
    if (s + 1 < kFactorizedMlpStages) {
      FactorizedMlpGridBarrier(ptr.state, (s + 1) * gridDim.x);
    }
  }
  // The last CTA leaves the state zero for the next launch: every CTA is
  // past the last grid barrier.
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(ptr.state + 1, 1u) == gridDim.x - 1) {
    ptr.state[0] = 0;
    ptr.state[1] = 0;
  }
#else
  __trap();
#endif
}

// grid (q_heads + k_heads + v_heads, tokens): one CTA per head of one token.
__global__ void __launch_bounds__(kGlueThreads) QuantQkvRopeKernel(
    LiteRtNvidiaQuantQkvRopeParams p, const int8_t* __restrict__ qkv,
    const int32_t* __restrict__ positions, const float* __restrict__ q_norm,
    const float* __restrict__ k_norm, const float* __restrict__ v_norm,
    const float* __restrict__ inverse_timescales, __half* __restrict__ q,
    __half* __restrict__ k, __half* __restrict__ v) {
  __shared__ float values[kMaxDepth];
  __shared__ float partial[kGlueThreads / kWarp];
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int depth = p.depth;
  const int total_heads = p.q_heads + p.k_heads + p.v_heads;
  const int8_t* row =
      qkv + (static_cast<int64_t>(token) * total_heads + head) * depth;
  const bool is_q = head < p.q_heads;
  const bool is_k = !is_q && head < p.q_heads + p.k_heads;
  const float scale = is_q ? p.q_scale : (is_k ? p.k_scale : p.v_scale);
  const float* weights = is_q ? q_norm : (is_k ? k_norm : v_norm);
  float squares = 0.0f;
  for (int i = threadIdx.x; i < depth; i += kGlueThreads) {
    const float value = RoundToFp16(static_cast<float>(row[i]) * scale);
    values[i] = value;
    squares = AddSquare(squares, value);
  }
  const float factor = NormFactor(BlockSum(squares, partial), depth, p.epsilon);
  for (int i = threadIdx.x; i < depth; i += kGlueThreads) {
    const float scaled = values[i] * factor;
    values[i] = RoundToFp16(scaled * weights[i]);
  }
  __syncthreads();
  if (!is_q && !is_k) {
    __half* out = v + (static_cast<int64_t>(token) * p.v_heads +
                       (head - p.q_heads - p.k_heads)) *
                          depth;
    for (int i = threadIdx.x; i < depth; i += kGlueThreads) {
      out[i] = __float2half_rn(values[i]);
    }
    return;
  }
  __half* out =
      is_q ? q + (static_cast<int64_t>(head) * p.tokens + token) * depth
           : k + (static_cast<int64_t>(head - p.q_heads) * p.tokens + token) *
                     depth;
  const int half = depth / 2;
  const float position = static_cast<float>(positions[token]);
  for (int i = threadIdx.x; i < half; i += kGlueThreads) {
    float sine, cosine;
    sincosf(position * inverse_timescales[i], &sine, &cosine);
    sine = RoundToFp16(sine);
    cosine = RoundToFp16(cosine);
    const float x1 = values[i];
    const float x2 = values[i + half];
    float first =
        RoundToFp16(RoundToFp16(x1 * cosine) - RoundToFp16(x2 * sine));
    float second =
        RoundToFp16(RoundToFp16(x2 * cosine) + RoundToFp16(x1 * sine));
    if (is_q) {
      first = RoundToFp16(first * p.q_multiplier);
      second = RoundToFp16(second * p.q_multiplier);
    }
    out[i] = __float2half_rn(first);
    out[i + half] = __float2half_rn(second);
  }
}

bool IsFloatType(int32_t type) {
  return type == kLiteRtNvidiaQuantFp16 || type == kLiteRtNvidiaQuantBf16 ||
         type == kLiteRtNvidiaQuantFp32;
}

// The input runs of a block the lanes of a warp split: a power of two, at
// most 32, that leaves each lane about 32 rows.
int BlockSlices(int columns) {
  int slices = 1;
  while (slices < kWarp && columns % (2 * slices) == 0 &&
         columns / (2 * slices) >= 32) {
    slices *= 2;
  }
  return slices;
}

}  // namespace

extern "C" bool LiteRtNvidiaQuantLinearSupports(
    const LiteRtNvidiaQuantLinearParams* p) {
  if (p == nullptr || p->rows <= 0 || p->columns <= 0 || p->segments <= 0 ||
      p->rows % p->segments != 0 || p->swap0 <= 0 || p->swap1 <= 0 ||
      p->rows % (p->swap0 * p->swap1) != 0 || (p->lut != 0 && p->lut != 1)) {
    return false;
  }
  // INT4 weights requantize in FP32, centered INT2 weights in doubles.
  if (p->bit_width == 4 ? p->requant != kLiteRtNvidiaQuantRequantFp32
                        : (p->bit_width != 2 ||
                           (p->requant != kLiteRtNvidiaQuantRequantDouble &&
                            p->requant != kLiteRtNvidiaQuantRequantLogits))) {
    return false;
  }
  if (p->requant == kLiteRtNvidiaQuantRequantLogits &&
      (p->lut != 0 || p->swap0 != 1 || p->swap1 != 1)) {
    return false;
  }
  if (p->requant == kLiteRtNvidiaQuantRequantDouble && !(p->ratio > 0.0)) {
    return false;
  }
  if (p->layout == kLiteRtNvidiaQuantLayoutBlocks) {
    return p->bit_width == 2 && p->activation == kLiteRtNvidiaQuantInt8 &&
           p->requant == kLiteRtNvidiaQuantRequantDouble &&
           p->rows / p->segments % 16 == 0;
  }
  if (p->layout != kLiteRtNvidiaQuantLayoutRows ||
      p->columns % (32 / p->bit_width) != 0 ||
      p->rows / p->segments % kRowsPerBlock != 0) {
    return false;
  }
  return p->activation == kLiteRtNvidiaQuantInt8 ||
         (IsFloatType(p->activation) && p->quantize_scale > 0.0f);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantLinear(
    const LiteRtNvidiaQuantLinearParams* p, int32_t m, const void* activation,
    const uint8_t* weights, const float* multipliers, const int8_t* lut,
    const float* row_scale, void* output, cudaStream_t stream) {
  if (!LiteRtNvidiaQuantLinearSupports(p) || m <= 0 || activation == nullptr ||
      weights == nullptr || multipliers == nullptr || output == nullptr ||
      (p->lut != 0 && lut == nullptr) ||
      (p->requant == kLiteRtNvidiaQuantRequantLogits && row_scale == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (p->layout == kLiteRtNvidiaQuantLayoutBlocks) {
    const int slices = BlockSlices(p->columns);
    const int words = p->rows / p->segments / 16;
    const int words_per_warp = kWarp / slices;
    const int groups = (words + words_per_warp - 1) / words_per_warp;
    QuantBlocksKernel<<<dim3(p->segments * groups, m), kWarp, 0, stream>>>(
        *p, slices, static_cast<const int8_t*>(activation), weights,
        multipliers, lut, static_cast<int8_t*>(output));
    return cudaPeekAtLastError();
  }
  const dim3 grid(p->rows / kRowsPerBlock, m);
  const int lane_words = (p->columns / (32 / p->bit_width) + kWarp - 1) / kWarp;
  // Rows of up to 256 words: every weight load of a CTA in flight at once, but
  // for rows of two words per lane (the factorized MLP's up-1 and down-1),
  // which take longer so on the RTX 5080 in the decode graph.
  if (lane_words == 1) {
    if (p->bit_width == 4) {
      QuantRowsPrefetchKernel<4, 2><<<grid, kWarp, 0, stream>>>(
          *p, activation, weights, multipliers, lut, row_scale, output);
    } else {
      QuantRowsPrefetchKernel<2, 2><<<grid, kWarp, 0, stream>>>(
          *p, activation, weights, multipliers, lut, row_scale, output);
    }
    return cudaPeekAtLastError();
  }
  // A decode step's float activation row that every row reads whole: it is
  // quantized once per CTA, not once per warp.
  if (m == 1 && IsFloatType(p->activation) && p->segments == 1 &&
      lane_words > 2 && lane_words <= kPrefetchWords &&
      p->requant != kLiteRtNvidiaQuantRequantLogits) {
    LiteRtNvidiaQuantGlueParams quantize = {};
    quantize.width = p->columns;
    quantize.x_type = p->activation;
    quantize.quantize = kLiteRtNvidiaQuantGlueStatic;
    quantize.quantize_scale = p->quantize_scale;
    const int rows_per_cta = kGlueThreads / kWarp * kQuantizedRows;
    const dim3 ctas((p->rows + rows_per_cta - 1) / rows_per_cta, 1);
    if (p->bit_width == 4) {
      QuantGlueRowsKernel<4, kQuantizedRows, kPrefetchWords, true>
          <<<ctas, kGlueThreads, 0, stream>>>(
              quantize, *p, activation, nullptr, nullptr, nullptr, weights,
              multipliers, lut, nullptr, nullptr, output);
    } else {
      QuantGlueRowsKernel<2, kQuantizedRows, kPrefetchWords, true>
          <<<ctas, kGlueThreads, 0, stream>>>(
              quantize, *p, activation, nullptr, nullptr, nullptr, weights,
              multipliers, lut, nullptr, nullptr, output);
    }
    return cudaPeekAtLastError();
  }
  if (lane_words > 2 && lane_words <= kPrefetchWords) {
    if (p->bit_width == 4) {
      QuantRowsPrefetchKernel<4, kPrefetchWords><<<grid, kWarp, 0, stream>>>(
          *p, activation, weights, multipliers, lut, row_scale, output);
    } else {
      QuantRowsPrefetchKernel<2, kPrefetchWords><<<grid, kWarp, 0, stream>>>(
          *p, activation, weights, multipliers, lut, row_scale, output);
    }
    return cudaPeekAtLastError();
  }
  if (p->bit_width == 4) {
    QuantRowsKernel<4><<<grid, kWarp, 0, stream>>>(
        *p, activation, weights, multipliers, lut, row_scale, output);
  } else {
    QuantRowsKernel<2><<<grid, kWarp, 0, stream>>>(
        *p, activation, weights, multipliers, lut, row_scale, output);
  }
  return cudaPeekAtLastError();
}

extern "C" bool LiteRtNvidiaQuantGlueSupports(
    const LiteRtNvidiaQuantGlueParams* p) {
  return p != nullptr && p->width > 0 && p->width <= kGlueMaxWidth &&
         IsFloatType(p->x_type) &&
         (p->z_type == kLiteRtNvidiaQuantFp16 ||
          p->z_type == kLiteRtNvidiaQuantFp32) &&
         p->branch_eps >= 0.0f && p->norm_eps >= 0.0f &&
         (p->quantize == kLiteRtNvidiaQuantGlueNone ||
          p->quantize == kLiteRtNvidiaQuantGlueDynamic ||
          (p->quantize == kLiteRtNvidiaQuantGlueStatic &&
           p->quantize_scale > 0.0f)) &&
         (p->branch == 0 || p->branch_scale > 0.0f);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantGlue(
    const LiteRtNvidiaQuantGlueParams* p, int32_t m, const void* x,
    const int8_t* y, const float* branch_weights, const float* norm_weights,
    void* x_out, void* z_out, int8_t* q_out, float* row_scale_out,
    cudaStream_t stream) {
  if (!LiteRtNvidiaQuantGlueSupports(p) || m <= 0 || x == nullptr ||
      (p->branch != 0 && (y == nullptr || branch_weights == nullptr)) ||
      (p->norm != 0 && norm_weights == nullptr) ||
      (p->quantize != kLiteRtNvidiaQuantGlueNone && q_out == nullptr)) {
    return cudaErrorInvalidValue;
  }
  QuantGlueKernel<<<m, kGlueThreads, 0, stream>>>(
      *p, x, y, branch_weights, norm_weights, static_cast<__half*>(x_out),
      z_out, q_out, row_scale_out);
  return cudaPeekAtLastError();
}

extern "C" bool LiteRtNvidiaQuantGlueLinearSupports(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantLinearParams* linear) {
  return LiteRtNvidiaQuantGlueSupports(glue) &&
         glue->quantize == kLiteRtNvidiaQuantGlueStatic &&
         LiteRtNvidiaQuantLinearSupports(linear) &&
         linear->layout == kLiteRtNvidiaQuantLayoutRows &&
         linear->activation == kLiteRtNvidiaQuantInt8 &&
         linear->requant != kLiteRtNvidiaQuantRequantLogits &&
         static_cast<int64_t>(linear->segments) * linear->columns ==
             glue->width &&
         linear->columns / (32 / linear->bit_width) <= kPrefetchWords * kWarp &&
         linear->rows / linear->segments % kFusedRowsShort == 0;
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantGlueLinear(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantLinearParams* linear, int32_t m, const void* x,
    const int8_t* y, const float* branch_weights, const float* norm_weights,
    const uint8_t* weights, const float* multipliers, const int8_t* lut,
    void* x_out, void* z_out, void* output, cudaStream_t stream) {
  if (!LiteRtNvidiaQuantGlueLinearSupports(glue, linear) || m <= 0 ||
      x == nullptr ||
      (glue->branch != 0 && (y == nullptr || branch_weights == nullptr)) ||
      (glue->norm != 0 && norm_weights == nullptr) || weights == nullptr ||
      multipliers == nullptr || output == nullptr ||
      (linear->lut != 0 && lut == nullptr)) {
    return cudaErrorInvalidValue;
  }
  // Rows of up to two words per lane take kFusedRowsShort outputs a warp.
  const bool short_rows =
      linear->columns / (32 / linear->bit_width) <= 2 * kWarp;
  const int rows_per_cta = kGlueThreads / kWarp *
                           (short_rows ? kFusedRowsShort : kFusedRowsLong);
  const dim3 grid((linear->rows + rows_per_cta - 1) / rows_per_cta, m);
  auto* half_out = static_cast<__half*>(x_out);
  if (linear->bit_width == 4 && short_rows) {
    QuantGlueRowsKernel<4, kFusedRowsShort, 2>
        <<<grid, kGlueThreads, 0, stream>>>(
            *glue, *linear, x, y, branch_weights, norm_weights, weights,
            multipliers, lut, half_out, z_out, output);
  } else if (linear->bit_width == 4) {
    QuantGlueRowsKernel<4, kFusedRowsLong, kPrefetchWords>
        <<<grid, kGlueThreads, 0, stream>>>(
            *glue, *linear, x, y, branch_weights, norm_weights, weights,
            multipliers, lut, half_out, z_out, output);
  } else if (short_rows) {
    QuantGlueRowsKernel<2, kFusedRowsShort, 2>
        <<<grid, kGlueThreads, 0, stream>>>(
            *glue, *linear, x, y, branch_weights, norm_weights, weights,
            multipliers, lut, half_out, z_out, output);
  } else {
    QuantGlueRowsKernel<2, kFusedRowsLong, kPrefetchWords>
        <<<grid, kGlueThreads, 0, stream>>>(
            *glue, *linear, x, y, branch_weights, norm_weights, weights,
            multipliers, lut, half_out, z_out, output);
  }
  return cudaPeekAtLastError();
}

namespace {

int64_t Align16(int64_t bytes) { return (bytes + 15) / 16 * 16; }

bool Aligned16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

// The grid and the shared memory of the one launch on the current device:
// ctas 0 where it cannot run. With `cooperative` the device schedules the
// grid only when all of it can be resident (another grid's CTAs, a second
// context's MLP say, could otherwise hold multiprocessors that its barriers
// wait for).
struct FactorizedMlpPlan {
  int ctas = 0;
  FactorizedMlpLayout layout = {};
  int shared_bytes = 0;
  bool cooperative = false;
};

template <bool Glue>
FactorizedMlpPlan PlanFactorizedMlp(
    const LiteRtNvidiaQuantFactorizedMlpParams& p,
    const LiteRtNvidiaQuantGlueParams* glue) {
  int device = 0;
  int ctas = 0;
  int major = 0;
  int optin = 0;
  int cooperative = 0;
  cudaFuncAttributes attributes = {};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&ctas, cudaDevAttrMultiProcessorCount, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) != cudaSuccess ||
      cudaFuncGetAttributes(&attributes, QuantFactorizedMlpKernel<Glue>) !=
          cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return {};
  }
  // The bulk copies and the barriers' memory order need code for compute
  // capability 9.0 or later, not the PTX of an older one.
  if (major < 9 || attributes.ptxVersion < 90 || ctas < 1) return {};
  FactorizedMlpPlan plan;
  int64_t most_rows = 0;
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    const auto& stage = p.stages[s];
    const int step = FactorizedMlpStep(stage.columns / 4);
    const int64_t rows = static_cast<int64_t>(step) *
                         ((stage.rows / step + ctas - 1) / ctas);
    const int64_t segment_rows = stage.rows / stage.segments;
    // A share spans at most this many runs of the activation.
    const int64_t runs = std::min<int64_t>(
        stage.segments, (rows + segment_rows - 1) / segment_rows + 1);
    const int64_t bytes = rows * stage.columns / 4;
    int32_t& buffer = s % 2 == 0 ? plan.layout.a_bytes : plan.layout.b_bytes;
    buffer = static_cast<int32_t>(std::max<int64_t>(buffer, bytes));
    plan.layout.act_bytes = static_cast<int32_t>(
        std::max<int64_t>(plan.layout.act_bytes, runs * stage.columns));
    most_rows = std::max(most_rows, rows);
  }
  // The glue's q is the whole row; buffer B holds its inputs first.
  int64_t glue_bytes = 0;
  if (glue != nullptr) {
    const int64_t width = glue->width;
    glue_bytes = width * (glue->x_type == kLiteRtNvidiaQuantFp32 ? 4 : 2) +
                 (glue->branch != 0 ? 5 * width : 0) +
                 (glue->norm != 0 ? 4 * width : 0);
    plan.layout.act_bytes = std::max<int32_t>(
        plan.layout.act_bytes, static_cast<int32_t>(Align16(width)));
    plan.layout.b_bytes = std::max<int32_t>(plan.layout.b_bytes,
                                            static_cast<int32_t>(glue_bytes));
  }
  if (most_rows > kFactorizedMlpRowsPerThread * kFactorizedMlpThreads ||
      glue_bytes > kFactorizedMlpGlueChunks * kFactorizedMlpThreads * 16 ||
      plan.layout.act_bytes >
          kFactorizedMlpChunks * kFactorizedMlpThreads * 16) {
    return {};
  }
  const int64_t shared = static_cast<int64_t>(plan.layout.a_bytes) +
                         plan.layout.b_bytes + plan.layout.act_bytes +
                         most_rows * static_cast<int64_t>(sizeof(int));
  int resident = 0;
  if (shared + static_cast<int64_t>(attributes.sharedSizeBytes) > optin ||
      cudaFuncSetAttribute(QuantFactorizedMlpKernel<Glue>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared)) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &resident, QuantFactorizedMlpKernel<Glue>, kFactorizedMlpThreads,
          static_cast<size_t>(shared)) != cudaSuccess ||
      resident < 1) {
    static_cast<void>(cudaGetLastError());
    return {};
  }
  plan.ctas = ctas;
  plan.shared_bytes = static_cast<int>(shared);
  plan.cooperative = cooperative != 0;
  return plan;
}

}  // namespace

extern "C" bool LiteRtNvidiaQuantFactorizedMlpSupports(
    const LiteRtNvidiaQuantFactorizedMlpParams* p) {
  if (p == nullptr) return false;
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    const auto& stage = p->stages[s];
    if (!LiteRtNvidiaQuantLinearSupports(&stage) ||
        stage.layout != kLiteRtNvidiaQuantLayoutRows || stage.bit_width != 2 ||
        stage.activation != kLiteRtNvidiaQuantInt8 ||
        stage.requant != kLiteRtNvidiaQuantRequantDouble ||
        (s > 0 && static_cast<int64_t>(stage.segments) * stage.columns !=
                      p->stages[s - 1].rows)) {
      return false;
    }
  }
  return true;
}

extern "C" size_t LiteRtNvidiaQuantFactorizedMlpStateBytes() {
  return 2 * sizeof(unsigned);
}

extern "C" size_t LiteRtNvidiaQuantFactorizedMlpWorkspaceBytes(
    const LiteRtNvidiaQuantFactorizedMlpParams* p) {
  if (!LiteRtNvidiaQuantFactorizedMlpSupports(p)) return 0;
  // The three results between the stages, then the q of a glue in front
  // for the launches without the one launch.
  size_t bytes = static_cast<size_t>(Align16(
      static_cast<int64_t>(p->stages[0].segments) * p->stages[0].columns));
  for (int s = 0; s + 1 < kFactorizedMlpStages; ++s) {
    bytes += static_cast<size_t>(Align16(p->stages[s].rows));
  }
  return bytes;
}

namespace {

// The one launch (with the glue in front when `glue` is not null), else the
// glue's launch and the four products.
cudaError_t LaunchFactorizedMlp(const LiteRtNvidiaQuantFactorizedMlpParams& p,
                                const FactorizedMlpGlue* glue,
                                const int8_t* activation,
                                const uint8_t* const weights[4],
                                const float* const multipliers[4],
                                const int8_t* lut, void* state, void* workspace,
                                int8_t* output, cudaStream_t stream) {
  FactorizedMlpPointers pointers = {};
  pointers.in[0] = activation;
  pointers.out[kFactorizedMlpStages - 1] = output;
  auto* intermediate = static_cast<int8_t*>(workspace);
  // The q of the glue for the launches without the one launch.
  int8_t* q = intermediate;
  intermediate += Align16(static_cast<int64_t>(p.stages[0].segments) *
                          p.stages[0].columns);
  bool aligned = (glue != nullptr || Aligned16(activation)) &&
                 Aligned16(workspace);
  for (int s = 0; s < kFactorizedMlpStages; ++s) {
    if (weights[s] == nullptr || multipliers[s] == nullptr ||
        (p.stages[s].lut != 0 && lut == nullptr)) {
      return cudaErrorInvalidValue;
    }
    pointers.weights[s] = weights[s];
    pointers.multipliers[s] = multipliers[s];
    aligned = aligned && Aligned16(weights[s]);
    if (s + 1 < kFactorizedMlpStages) {
      pointers.out[s] = intermediate;
      pointers.in[s + 1] = intermediate;
      intermediate += Align16(p.stages[s].rows);
    }
  }
  pointers.lut = lut;
  pointers.state = static_cast<unsigned*>(state);
  if (glue != nullptr) {
    aligned = aligned && Aligned16(glue->x) && Aligned16(glue->y) &&
              Aligned16(glue->branch_weights) && Aligned16(glue->norm_weights);
  }
  const FactorizedMlpPlan plan =
      state == nullptr || !aligned ? FactorizedMlpPlan{}
      : glue != nullptr            ? PlanFactorizedMlp<true>(p, &glue->params)
                                   : PlanFactorizedMlp<false>(p, nullptr);
  if (plan.ctas == 0) {
    if (glue != nullptr) {
      const cudaError_t status = LiteRtNvidiaLaunchQuantGlue(
          &glue->params, 1, glue->x, glue->y, glue->branch_weights,
          glue->norm_weights, glue->x_out, glue->z_out, q, nullptr, stream);
      if (status != cudaSuccess) return status;
      pointers.in[0] = q;
    }
    for (int s = 0; s < kFactorizedMlpStages; ++s) {
      const cudaError_t status = LiteRtNvidiaLaunchQuantLinear(
          &p.stages[s], 1, pointers.in[s], weights[s], multipliers[s], lut,
          /*row_scale=*/nullptr, pointers.out[s], stream);
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(plan.ctas);
  config.blockDim = dim3(kFactorizedMlpThreads);
  config.dynamicSmemBytes = static_cast<size_t>(plan.shared_bytes);
  config.stream = stream;
  cudaLaunchAttribute cooperative = {};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  config.attrs = &cooperative;
  config.numAttrs = plan.cooperative ? 1 : 0;
  const FactorizedMlpGlue none = {};
  return glue != nullptr
             ? cudaLaunchKernelEx(&config, QuantFactorizedMlpKernel<true>, p,
                                  pointers, plan.layout, *glue)
             : cudaLaunchKernelEx(&config, QuantFactorizedMlpKernel<false>, p,
                                  pointers, plan.layout, none);
}

}  // namespace

extern "C" cudaError_t LiteRtNvidiaLaunchQuantFactorizedMlp(
    const LiteRtNvidiaQuantFactorizedMlpParams* p, const int8_t* activation,
    const uint8_t* const weights[4], const float* const multipliers[4],
    const int8_t* lut, void* state, void* workspace, int8_t* output,
    cudaStream_t stream) {
  if (!LiteRtNvidiaQuantFactorizedMlpSupports(p) || activation == nullptr ||
      weights == nullptr || multipliers == nullptr || workspace == nullptr ||
      output == nullptr) {
    return cudaErrorInvalidValue;
  }
  return LaunchFactorizedMlp(*p, nullptr, activation, weights, multipliers, lut,
                             state, workspace, output, stream);
}

extern "C" bool LiteRtNvidiaQuantGlueFactorizedMlpSupports(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantFactorizedMlpParams* params) {
  return LiteRtNvidiaQuantGlueSupports(glue) &&
         glue->quantize == kLiteRtNvidiaQuantGlueStatic &&
         LiteRtNvidiaQuantFactorizedMlpSupports(params) &&
         static_cast<int64_t>(params->stages[0].segments) *
                 params->stages[0].columns ==
             glue->width;
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantGlueFactorizedMlp(
    const LiteRtNvidiaQuantGlueParams* glue,
    const LiteRtNvidiaQuantFactorizedMlpParams* p, const void* x,
    const int8_t* y, const float* branch_weights, const float* norm_weights,
    const uint8_t* const weights[4], const float* const multipliers[4],
    const int8_t* lut, void* state, void* workspace, void* x_out, void* z_out,
    int8_t* output, cudaStream_t stream) {
  if (!LiteRtNvidiaQuantGlueFactorizedMlpSupports(glue, p) || x == nullptr ||
      (glue->branch != 0 && (y == nullptr || branch_weights == nullptr)) ||
      (glue->norm != 0 && norm_weights == nullptr) || weights == nullptr ||
      multipliers == nullptr || workspace == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const FactorizedMlpGlue in_front = {
      *glue, x, y, branch_weights, norm_weights, static_cast<__half*>(x_out),
      z_out};
  return LaunchFactorizedMlp(*p, &in_front, nullptr, weights, multipliers, lut,
                             state, workspace, output, stream);
}

extern "C" bool LiteRtNvidiaQuantQkvRopeSupports(
    const LiteRtNvidiaQuantQkvRopeParams* p) {
  return p != nullptr && p->tokens >= 1 && p->q_heads >= 0 && p->k_heads >= 0 &&
         p->v_heads >= 0 && p->q_heads + p->k_heads + p->v_heads >= 1 &&
         p->depth >= 2 && p->depth <= kMaxDepth && p->depth % 2 == 0 &&
         p->epsilon >= 0.0f && (p->q_heads == 0 || p->q_scale > 0.0f) &&
         (p->k_heads == 0 || p->k_scale > 0.0f) &&
         (p->v_heads == 0 || p->v_scale > 0.0f);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantQkvRope(
    const LiteRtNvidiaQuantQkvRopeParams* p, const int8_t* qkv,
    const int32_t* positions, const float* q_norm, const float* k_norm,
    const float* v_norm, const float* inverse_timescales, void* q, void* k,
    void* v, cudaStream_t stream) {
  if (!LiteRtNvidiaQuantQkvRopeSupports(p) || qkv == nullptr ||
      positions == nullptr || inverse_timescales == nullptr ||
      (p->q_heads != 0 && (q_norm == nullptr || q == nullptr)) ||
      (p->k_heads != 0 && (k_norm == nullptr || k == nullptr)) ||
      (p->v_heads != 0 && (v_norm == nullptr || v == nullptr))) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(p->q_heads + p->k_heads + p->v_heads, p->tokens);
  QuantQkvRopeKernel<<<grid, kGlueThreads, 0, stream>>>(
      *p, qkv, positions, q_norm, k_norm, v_norm, inverse_timescales,
      static_cast<__half*>(q), static_cast<__half*>(k),
      static_cast<__half*>(v));
  return cudaPeekAtLastError();
}
