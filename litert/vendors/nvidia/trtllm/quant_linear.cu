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
// overlap it.
template <int Bits, int Rows, int Words>
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
  const bool first = blockIdx.x == 0;
  GlueRow(g, m, x, y, branch_weights, norm_weights, first ? x_out : nullptr,
          first ? z_out : nullptr, q, nullptr);
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
