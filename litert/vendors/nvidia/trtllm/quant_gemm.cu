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

#include "litert/vendors/nvidia/trtllm/quant_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "cuda_runtime.h"

namespace {

constexpr int kThreads = 256;
// A block's activation rows and weight rows.
constexpr int kTile = 128;
// The inputs a block stages at a time, and those of one MMA.
constexpr int kChunk = 128;
constexpr int kSlice = 32;
constexpr int kSlices = kChunk / kSlice;
// A staged activation row: its chunk and a pad that spreads the eight rows
// of an ldmatrix over the banks of the shared memory.
constexpr int kActivationPitch = kChunk + 16;
constexpr int kActivationStage = kTile * kActivationPitch;
constexpr int kLutBytes = 256;
// The inputs up to which a sum of 16 * q * x or 32 * (2 * q + 1) * x stays
// below 2^31 with room to spare.
constexpr int kMaxColumns = 65536;
constexpr int kQuantizeThreads = 256;
constexpr int kQuantizeMaxBlocks = 16384;

// The bytes of a weight row's chunk as the model file packs them, the pitch
// of the staged rows and a stage of them.
template <int Bits>
constexpr int kPackedChunk = kChunk * Bits / 8;
template <int Bits>
constexpr int kPackedPitch = kPackedChunk<Bits> + 16;
template <int Bits>
constexpr int kPackedStage = kTile * kPackedPitch<Bits>;
// Two stages (the chunk a block multiplies and the next one) and the table.
template <int Bits>
constexpr int kSharedBytes =
    2 * (kActivationStage + kPackedStage<Bits>) + kLutBytes;

__device__ __forceinline__ void MmaS8(int32_t& d0, int32_t& d1, int32_t& d2,
                                      int32_t& d3, uint32_t a0, uint32_t a1,
                                      uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void LoadMatrixX4(uint32_t& r0, uint32_t& r1,
                                             uint32_t& r2, uint32_t& r3,
                                             const void* shared_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(address));
}

__device__ __forceinline__ void CopyAsync16(void* shared_pointer,
                                            const void* global_pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(address), "l"(global_pointer)
               : "memory");
}

__device__ __forceinline__ void CommitAsync() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void WaitAsyncAll() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// Four INT4 weights (a 16-bit word, the first in its low bits) as four INT8
// bytes that hold 16 times their values: a field in the top of its byte has
// its sign bit there.
__device__ __forceinline__ uint32_t WidenInt4(uint32_t fields) {
  return ((fields & 0x000Fu) << 4) | ((fields & 0x00F0u) << 8) |
         ((fields & 0x0F00u) << 12) | ((fields & 0xF000u) << 16);
}

// Four centered INT2 weights (a byte, the first in its low bits) as four
// INT8 bytes that hold 32 * (2 * q + 1), q the field's two's-complement
// value: the field in the top two bits of its byte is 64 * q, and the bit
// below them adds the 32.
__device__ __forceinline__ uint32_t WidenInt2(uint32_t fields) {
  return ((fields & 0x03u) << 6) | ((fields & 0x0Cu) << 12) |
         ((fields & 0x30u) << 18) | ((fields & 0xC0u) << 24) | 0x20202020u;
}

__device__ __forceinline__ float RoundToFp16(float value) {
  return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ int Saturate8(int value) {
  return max(-128, min(127, value));
}

// Where output `row` is written (LiteRtNvidiaQuantLinearParams::swap0), as
// quant_linear.cu's OutputIndex places it.
__device__ __forceinline__ int OutputIndex(
    const LiteRtNvidiaQuantLinearParams& p, int row) {
  if (p.swap0 == 1) return row;
  const int inner = p.rows / (p.swap0 * p.swap1);
  const int a = row / (p.swap1 * inner);
  const int b = row / inner % p.swap1;
  return (b * p.swap0 + a) * inner + row % inner;
}

// How far from a tie (a magnitude of n + 0.5) the FP32 estimate of a
// centered INT2 result must be for its rounding to be the doubles': the
// estimate's four roundings put it within 129 * 4 * 2^-24 = 3.1e-5 of the
// doubles' product below a magnitude of 129, an eighth of this.
constexpr float kTieMargin = 2.5e-4f;

// A centered INT2 result as quant_linear.cu's StoreOutput requantizes it, in
// doubles, operation for operation. Out of line: only the warps that hold a
// result near a tie make the call, and the epilogue of the others stays free
// of the doubles, which a consumer GPU computes slowly.
__device__ __noinline__ int RequantizeInDoubles(int exact, float multiplier,
                                                double ratio) {
  double value =
      static_cast<double>(multiplier) * (0.5 * static_cast<double>(exact));
  value = round(value * ratio);
  return static_cast<int>(fmin(fmax(value, -128.0), 127.0));
}

// The INT8 result of an output, as quant_linear.cu's StoreOutput requantizes
// it. `exact` is the integer the row kernels accumulate: sum(x * q) for INT4
// weights, twice sum(x * (q + 0.5)) for centered INT2. INT4 results are
// StoreOutput's FP32 operations. Centered INT2 results are its doubles
// rounded half away from zero, which a consumer GPU computes slowly: an FP32
// estimate decides every result that is not within kTieMargin of a tie, and
// only those few take the doubles. `ratio` is p.ratio in FP32.
__device__ __forceinline__ int Requantize(
    const LiteRtNvidiaQuantLinearParams& p, int exact, float multiplier,
    float ratio, const int8_t* lut) {
  int q;
  if (p.requant == kLiteRtNvidiaQuantRequantFp32) {
    q = Saturate8(__float2int_rn(static_cast<float>(exact) * multiplier));
  } else {
    const float estimate =
        multiplier * (0.5f * static_cast<float>(exact)) * ratio;
    const float magnitude = fabsf(estimate);
    // Past 129 the doubles' product is past 128.5 too: saturated.
    const int rounded =
        static_cast<int>(fminf(magnitude, 129.0f) + 0.5f);
    q = Saturate8(estimate < 0.0f ? -rounded : rounded);
    if (magnitude < 129.0f &&
        fabsf(magnitude - floorf(magnitude) - 0.5f) < kTieMargin) {
      q = RequantizeInDoubles(exact, multiplier, p.ratio);
    }
  }
  if (p.lut != 0) {
    q = lut[q + 128];
  }
  return q;
}

// A float activation as the row kernels quantize it on their way in:
// q = sat8(rne(fl16(x) * scale)), over every element of the rows.
__global__ void __launch_bounds__(kQuantizeThreads)
    QuantizeActivationKernel(const void* __restrict__ activation, int type,
                             float scale, int64_t count,
                             int8_t* __restrict__ quantized) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kQuantizeThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kQuantizeThreads +
                   threadIdx.x;
       i < count; i += stride) {
    float value;
    switch (type) {
      case kLiteRtNvidiaQuantFp16:
        value = __half2float(static_cast<const __half*>(activation)[i]);
        break;
      case kLiteRtNvidiaQuantBf16:
        value = RoundToFp16(__bfloat162float(
            static_cast<const __nv_bfloat16*>(activation)[i]));
        break;
      default:
        value = RoundToFp16(static_cast<const float*>(activation)[i]);
        break;
    }
    quantized[i] = static_cast<int8_t>(Saturate8(__float2int_rn(value * scale)));
  }
}

// grid (rows / 128, m / 128): a block multiplies 128 activation rows by 128
// weight rows (outputs). A warp owns 64 of the activation rows and 32 of the
// outputs, as four 16-row operands times four 8-output operands of an MMA over
// 32 inputs. The inputs go through shared memory 128 at a time, two chunks in
// flight: the next chunk's copies run while a chunk multiplies. The results
// leave through shared memory too when `whole_rows` (the outputs of a block are
// 128 consecutive bytes of an activation row's results that start on 16 bytes):
// a thread's 64 results are single bytes of 16 rows, which as stores of their
// own took a fifth of the time of a launch over 256 inputs (the factorized
// MLP's up-2 at 1,024 rows on the 5080); the block stores rows of 16-byte
// vectors instead.
template <int Bits>
__global__ void __launch_bounds__(kThreads, 1)
    QuantGemmKernel(LiteRtNvidiaQuantLinearParams p, bool whole_rows,
                    const int8_t* __restrict__ activation,
                    const uint8_t* __restrict__ weights,
                    const float* __restrict__ multipliers,
                    const int8_t* __restrict__ lut,
                    int8_t* __restrict__ output) {
  constexpr int kPackedSlice = kSlice * Bits / 8;
  extern __shared__ __align__(16) unsigned char shared[];
  unsigned char* activation_s = shared;
  unsigned char* packed_s = shared + 2 * kActivationStage;
  int8_t* lut_s = reinterpret_cast<int8_t*>(
      shared + 2 * (kActivationStage + kPackedStage<Bits>));
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n0 = blockIdx.x * kTile;
  const int m0 = blockIdx.y * kTile;
  const int chunks = p.columns / kChunk;
  // The outputs of a block read one run of the activation rows.
  const int segment = n0 / (p.rows / p.segments);
  const int64_t stride = static_cast<int64_t>(p.segments) * p.columns;
  const int8_t* activation_tile =
      activation + m0 * stride + static_cast<int64_t>(segment) * p.columns;
  const int64_t row_bytes = static_cast<int64_t>(p.columns) * Bits / 8;
  const uint8_t* weight_tile = weights + n0 * row_bytes;

  // A chunk's copies: 128 activation rows of 128 bytes are 1,024 vectors of
  // 16 bytes, four per thread; the packed weight rows are 512 (INT4) or 256
  // (INT2) vectors.
  const auto copy_stage = [&](int chunk, int stage) {
    unsigned char* a = activation_s + stage * kActivationStage;
    const int a_vector = (tid % 8) * 16;
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      const int row = tid / 8 + quarter * 32;
      CopyAsync16(a + row * kActivationPitch + a_vector,
                  activation_tile + row * stride + chunk * kChunk + a_vector);
    }
    unsigned char* w = packed_s + stage * kPackedStage<Bits>;
    if constexpr (Bits == 4) {
      const int w_vector = (tid % 4) * 16;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = tid / 4 + half * 64;
        CopyAsync16(w + row * kPackedPitch<Bits> + w_vector,
                    weight_tile + row * row_bytes +
                        chunk * kPackedChunk<Bits> + w_vector);
      }
    } else {
      const int w_vector = (tid % 2) * 16;
      const int row = tid / 2;
      CopyAsync16(w + row * kPackedPitch<Bits> + w_vector,
                  weight_tile + row * row_bytes + chunk * kPackedChunk<Bits> +
                      w_vector);
    }
  };

  if (p.lut != 0) {
    lut_s[tid] = lut[tid];
  }
  copy_stage(0, 0);
  CommitAsync();
  WaitAsyncAll();
  __syncthreads();

  const int warp_row = (warp / 4) * 64;
  const int warp_col = (warp % 4) * 32;
  // The row an ldmatrix lane names: lanes 0-7 the rows 0-7 of a 16-row
  // operand at inputs 0-15 of the slice, 8-15 its rows 8-15, 16-31 the same
  // rows at inputs 16-31.
  const int a_row = warp_row + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_input = (lane >> 4) * 16;
  int32_t sums[4][4][4];
#pragma unroll
  for (int rg = 0; rg < 4; ++rg) {
#pragma unroll
    for (int cg = 0; cg < 4; ++cg) {
      sums[rg][cg][0] = sums[rg][cg][1] = sums[rg][cg][2] = sums[rg][cg][3] =
          0;
    }
  }
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int stage = chunk & 1;
    if (chunk + 1 < chunks) {
      copy_stage(chunk + 1, stage ^ 1);
    }
    CommitAsync();
    const unsigned char* a_stage = activation_s + stage * kActivationStage;
    const unsigned char* p_stage = packed_s + stage * kPackedStage<Bits>;
#pragma unroll
    for (int slice = 0; slice < kSlices; ++slice) {
      uint32_t a[4][4];
      uint32_t b[4][2];
#pragma unroll
      for (int rg = 0; rg < 4; ++rg) {
        LoadMatrixX4(a[rg][0], a[rg][1], a[rg][2], a[rg][3],
                     a_stage + (a_row + rg * 16) * kActivationPitch +
                         slice * kSlice + a_input);
      }
      // The thread's output g of each 8-output operand: its inputs 4t..4t+3
      // and 16+4t..16+4t+3 of the slice.
#pragma unroll
      for (int cg = 0; cg < 4; ++cg) {
        const unsigned char* row =
            p_stage + (warp_col + cg * 8 + g) * kPackedPitch<Bits> +
            slice * kPackedSlice;
        if constexpr (Bits == 4) {
          b[cg][0] =
              WidenInt4(*reinterpret_cast<const uint16_t*>(row + 2 * t));
          b[cg][1] =
              WidenInt4(*reinterpret_cast<const uint16_t*>(row + 8 + 2 * t));
        } else {
          b[cg][0] = WidenInt2(row[t]);
          b[cg][1] = WidenInt2(row[4 + t]);
        }
      }
#pragma unroll
      for (int rg = 0; rg < 4; ++rg) {
#pragma unroll
        for (int cg = 0; cg < 4; ++cg) {
          MmaS8(sums[rg][cg][0], sums[rg][cg][1], sums[rg][cg][2],
                sums[rg][cg][3], a[rg][0], a[rg][1], a[rg][2], a[rg][3],
                b[cg][0], b[cg][1]);
        }
      }
    }
    // The next chunk is whole, and this one's stage is free, for every warp.
    WaitAsyncAll();
    __syncthreads();
  }

  // A thread holds, of each operand pair, the outputs 2t and 2t + 1 for the
  // activation rows g and g + 8: the sums are 16 (INT4) or 32 (INT2) times
  // the row kernels' integers.
  constexpr int kShift = Bits == 4 ? 4 : 5;
  const float ratio = static_cast<float>(p.ratio);
  // The block's results [128 activation rows][128 outputs], on the pitch of
  // a staged activation row, over a stage no chunk uses any more. A thread
  // stores the two outputs it holds of an activation row at once.
  unsigned char* result_s = shared;
#pragma unroll
  for (int cg = 0; cg < 4; ++cg) {
    const int local = warp_col + cg * 8 + 2 * t;
    const float multiplier[2] = {multipliers[n0 + local],
                                 multipliers[n0 + local + 1]};
    const int column[2] = {whole_rows ? 0 : OutputIndex(p, n0 + local),
                           whole_rows ? 0 : OutputIndex(p, n0 + local + 1)};
#pragma unroll
    for (int rg = 0; rg < 4; ++rg) {
#pragma unroll
      for (int high = 0; high < 2; ++high) {
        const int row = warp_row + rg * 16 + g + high * 8;
        int q[2];
#pragma unroll
        for (int odd = 0; odd < 2; ++odd) {
          q[odd] = Requantize(p, sums[rg][cg][high * 2 + odd] >> kShift,
                              multiplier[odd], ratio, lut_s);
        }
        if (whole_rows) {
          *reinterpret_cast<uint16_t*>(result_s + row * kActivationPitch +
                                       local) =
              static_cast<uint16_t>((q[0] & 0xFF) | ((q[1] & 0xFF) << 8));
        } else {
          int8_t* results = output + static_cast<int64_t>(m0 + row) * p.rows;
          results[column[0]] = static_cast<int8_t>(q[0]);
          results[column[1]] = static_cast<int8_t>(q[1]);
        }
      }
    }
  }
  if (!whole_rows) {
    return;
  }
  __syncthreads();
  // 128 rows of 128 bytes are 1,024 vectors of 16 bytes, four per thread.
  int8_t* tile =
      output + static_cast<int64_t>(m0) * p.rows + OutputIndex(p, n0);
  const int vector = (tid % 8) * 16;
#pragma unroll
  for (int quarter = 0; quarter < 4; ++quarter) {
    const int row = tid / 8 + quarter * 32;
    *reinterpret_cast<int4*>(tile + static_cast<int64_t>(row) * p.rows +
                             vector) = *reinterpret_cast<const int4*>(
        result_s + row * kActivationPitch + vector);
  }
}

bool IsFloatType(int32_t type) {
  return type == kLiteRtNvidiaQuantFp16 || type == kLiteRtNvidiaQuantBf16 ||
         type == kLiteRtNvidiaQuantFp32;
}

bool Aligned16(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

// Whether the outputs of every block are 128 consecutive bytes of an
// activation row's results that start on 16 bytes: no regrouping, or one
// that moves whole runs of a multiple of 128 outputs.
bool WholeRows(const LiteRtNvidiaQuantLinearParams& p, const void* output) {
  return Aligned16(output) &&
         (p.swap0 == 1 || p.rows / (p.swap0 * p.swap1) % kTile == 0);
}

template <int Bits>
cudaError_t LaunchGemm(const LiteRtNvidiaQuantLinearParams& p, int32_t m,
                       const int8_t* activation, const uint8_t* weights,
                       const float* multipliers, const int8_t* lut,
                       int8_t* output, cudaStream_t stream) {
  const cudaError_t status = cudaFuncSetAttribute(
      QuantGemmKernel<Bits>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes<Bits>);
  if (status != cudaSuccess) {
    return status;
  }
  QuantGemmKernel<Bits>
      <<<dim3(p.rows / kTile, m / kTile), kThreads, kSharedBytes<Bits>,
         stream>>>(p, WholeRows(p, output), activation, weights, multipliers,
                   lut, output);
  return cudaPeekAtLastError();
}

}  // namespace

extern "C" bool LiteRtNvidiaQuantGemmSupports(
    const LiteRtNvidiaQuantLinearParams* p, int32_t m) {
  return LiteRtNvidiaQuantLinearSupports(p) &&
         p->layout == kLiteRtNvidiaQuantLayoutRows &&
         p->requant != kLiteRtNvidiaQuantRequantLogits && m >= kTile &&
         m % kTile == 0 && p->rows / p->segments % kTile == 0 &&
         p->columns % kChunk == 0 && p->columns <= kMaxColumns;
}

extern "C" bool LiteRtNvidiaQuantGemmAvailable() {
  int device = 0;
  int major = 0;
  int shared = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                device) == cudaSuccess &&
         cudaDeviceGetAttribute(&shared,
                                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                device) == cudaSuccess &&
         major >= 8 && shared >= kSharedBytes<4>;
}

extern "C" size_t LiteRtNvidiaQuantGemmWorkspaceBytes(
    const LiteRtNvidiaQuantLinearParams* p, int32_t m) {
  if (!LiteRtNvidiaQuantGemmSupports(p, m) || !IsFloatType(p->activation)) {
    return 0;
  }
  return static_cast<size_t>(m) * p->segments * p->columns;
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantGemm(
    const LiteRtNvidiaQuantLinearParams* p, int32_t m, const void* activation,
    const uint8_t* weights, const float* multipliers, const int8_t* lut,
    void* output, void* workspace, cudaStream_t stream) {
  if (!LiteRtNvidiaQuantGemmSupports(p, m) || activation == nullptr ||
      weights == nullptr || multipliers == nullptr || output == nullptr ||
      (p->lut != 0 && lut == nullptr)) {
    return cudaErrorInvalidValue;
  }
  const bool float_rows = IsFloatType(p->activation);
  const void* rows = float_rows ? workspace : activation;
  // The copies of a block are 16 bytes each from where they start.
  if (rows == nullptr || !Aligned16(rows) || !Aligned16(weights) ||
      !LiteRtNvidiaQuantGemmAvailable()) {
    return LiteRtNvidiaLaunchQuantLinear(p, m, activation, weights, multipliers,
                                         lut, /*row_scale=*/nullptr, output,
                                         stream);
  }
  if (float_rows) {
    const int64_t count = static_cast<int64_t>(m) * p->segments * p->columns;
    const int64_t blocks = (count + kQuantizeThreads - 1) / kQuantizeThreads;
    QuantizeActivationKernel
        <<<static_cast<unsigned int>(blocks < kQuantizeMaxBlocks
                                         ? blocks
                                         : kQuantizeMaxBlocks),
           kQuantizeThreads, 0, stream>>>(activation, p->activation,
                                          p->quantize_scale, count,
                                          static_cast<int8_t*>(workspace));
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      return status;
    }
  }
  return p->bit_width == 4
             ? LaunchGemm<4>(*p, m, static_cast<const int8_t*>(rows), weights,
                             multipliers, lut, static_cast<int8_t*>(output),
                             stream)
             : LaunchGemm<2>(*p, m, static_cast<const int8_t*>(rows), weights,
                             multipliers, lut, static_cast<int8_t*>(output),
                             stream);
}
