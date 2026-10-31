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

// The INT4-weight GEMM on the int8 tensor cores, which run at twice the fp16
// rate on RTX Blackwell (490 against 245 TFLOPS on the RTX 5080): the
// activations are quantized to int8 per block of 128 input dims with a float
// scale, the weights' nibbles are widened to int8 in registers (the nibble in
// the byte's high half: sixteen times its two's-complement value, taken out
// in the scale), a block accumulates in int32 and the blocks in fp32. The
// kOutliers input dims with the largest magnitude over the rows (a model's
// massive activations sit in fixed channels) leave the int8 path and come
// back exactly through one fp16 MMA chunk, so an outlier does not coarsen its
// block's quantization. Per-channel weight scales only (the blockwise models
// stay on the fp16 kernel); the workspace holds the quantized rows, their
// scales, the column maxima, the outlier set and its gathered weights.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

namespace litert::nvidia::int8_gemm {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlockRows = 128;
constexpr int kBlockCols = 128;
constexpr int kChunk = 128;               // input dims per stage
constexpr int kSlices = kChunk / 32;      // int8 MMAs cover 32 dims
constexpr int kActivationBlock = 128;  // input dims per activation scale (one flush per chunk)
constexpr int kBlocksPerChunk = kChunk / kActivationBlock;
constexpr int kActivationRowBytes = kChunk + 16;  // padded: conflict-free ldmatrix
constexpr int kActivationStageBytes = kBlockRows * kActivationRowBytes;
constexpr int kScaleStageBytes = kBlockRows * kBlocksPerChunk * 4;
constexpr int kPackedRowBytes = kChunk / 2 + 16;  // 64 bytes of nibbles, padded to 80
constexpr int kPackedStageBytes = kBlockCols * kPackedRowBytes;
constexpr int kStages = 3;
// The scales live a chunk longer than the operands: the flush of a chunk is deferred into the
// next one, when the copy of the chunk after next already overwrites the operands' stage.
constexpr int kScaleStages = 4;
constexpr size_t kSharedBytes =
    static_cast<size_t>(kStages) * (kActivationStageBytes + kPackedStageBytes) +
    static_cast<size_t>(kScaleStages) * kScaleStageBytes;
constexpr int kWarpRowGroups = 4;  // 16 rows each: a warp owns 64 rows
constexpr int kWarpColGroups = 4;  // 8 channels each: 32 channels
// The outlier dims: the kOutliers input dims with the largest magnitude over the rows leave the
// int8 path (their activations are zero there) and come back exactly through one fp16 MMA chunk.
constexpr int kOutliers = 32;
constexpr int kOutlierRowBytes = kOutliers * 2 + 16;  // fp16, padded for ldmatrix

__device__ __forceinline__ void MmaF16F32(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                          uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void MmaS8(int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
                                      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void LoadMatrixX4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                             const void* shared_pointer) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

__device__ __forceinline__ void CopyAsync16(void* shared_pointer, const void* global_pointer) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(address), "l"(global_pointer) : "memory");
}
__device__ __forceinline__ void CopyAsync4(void* shared_pointer, const void* global_pointer) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared_pointer));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" : : "r"(address), "l"(global_pointer) : "memory");
}
__device__ __forceinline__ void CommitAsync() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void WaitAsyncButOne() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void WaitAsyncAll() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// Four nibbles (a 16-bit word, nibble i = input dim i of the slice) widened to four int8 bytes
// holding 16 times their two's-complement value: the nibble goes to the byte's high half, so its
// sign bit is the byte's. The factor 16 comes out in the scale.
__device__ __forceinline__ uint32_t WidenNibbles(uint32_t x) {
  const uint32_t lo = (x << 4) & 0xF0F0u;  // nibbles 0, 2 -> bytes 0, 1
  const uint32_t hi = x & 0xF0F0u;         // nibbles 1, 3 -> bytes 0, 1
  uint32_t result;
  asm("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(result) : "r"(lo), "r"(hi));
  return result;
}

__device__ __forceinline__ float Gelu(float x, int gate) {
  if (gate == kLiteRtNvidiaGemmGateGeluTanh) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// The largest magnitude of every input dim over the rows, after the optional norm (the norm's
// per-dim scale multiplies in; the per-row factor is a positive scalar that cannot change which
// dims are largest within a row but can across rows, so it is applied too).
// grid (dims / 256, rows / kColumnMaxRows); column_max starts at zero and takes the maxima by
// atomic max on the bits (non-negative floats order as their bits).
constexpr int kColumnMaxRows = 32;
__global__ void __launch_bounds__(256) ColumnMaxKernel(const __nv_bfloat16* __restrict__ activation,
                                                       int rows, int input_size,
                                                       const float* __restrict__ row_factor,
                                                       const float* __restrict__ norm_scale,
                                                       float* __restrict__ column_max) {
  const int k = blockIdx.x * 256 + threadIdx.x;
  if (k >= input_size) return;
  const int first = blockIdx.y * kColumnMaxRows;
  const int last = min(first + kColumnMaxRows, rows);
  float largest = 0.0f;
  // Eight rows' loads in flight at a time.
  int row = first;
  for (; row + 8 <= last; row += 8) {
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = fabsf(__bfloat162float(activation[static_cast<size_t>(row + i) * input_size + k]));
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (row_factor != nullptr) v[i] *= row_factor[row + i];
      largest = fmaxf(largest, v[i]);
    }
  }
  for (; row < last; ++row) {
    float v = fabsf(__bfloat162float(activation[static_cast<size_t>(row) * input_size + k]));
    if (row_factor != nullptr) v *= row_factor[row];
    largest = fmaxf(largest, v);
  }
  if (norm_scale != nullptr) largest *= fabsf(norm_scale[k]);
  atomicMax(reinterpret_cast<unsigned int*>(column_max + k), __float_as_uint(largest));
}

// The per-row norm factors (rsqrt(mean(x^2) + eps) * multiplier), one block per row.
__global__ void __launch_bounds__(kThreads) RowFactorKernel(const __nv_bfloat16* __restrict__ activation,
                                                            int input_size, float norm_eps,
                                                            float norm_multiplier,
                                                            float* __restrict__ row_factor) {
  __shared__ float squares_s[kWarps];
  const __nv_bfloat162* src =
      reinterpret_cast<const __nv_bfloat162*>(activation + static_cast<size_t>(blockIdx.x) * input_size);
  float squares = 0.0f;
  for (int i = threadIdx.x; i < input_size / 2; i += kThreads) {
    const float2 v = __bfloat1622float2(src[i]);
    squares = fmaf(v.x, v.x, squares);
    squares = fmaf(v.y, v.y, squares);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
  if (threadIdx.x % 32 == 0) squares_s[threadIdx.x / 32] = squares;
  __syncthreads();
  if (threadIdx.x == 0) {
    squares = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) squares += squares_s[w];
    row_factor[blockIdx.x] = rsqrtf(squares / static_cast<float>(input_size) + norm_eps) * norm_multiplier;
  }
}

// One block of 256 threads: the kOutliers dims with the largest column maxima (ties to the lower
// dim), sorted ascending into `dims`, and a flag per dim. Each warp takes the top kOutliers of its
// slice of the dims by repeated argmax over shared memory, without block barriers; warp 0 then
// takes the top kOutliers of the eight warps' candidates.
constexpr int kSelectThreads = 256;
constexpr int kSelectWarps = kSelectThreads / 32;
__device__ __forceinline__ void WarpArgMax(float& best, int& index) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, best, offset);
    const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
    if (other > best || (other == best && other_index < index)) {
      best = other;
      index = other_index;
    }
  }
}
__global__ void __launch_bounds__(kSelectThreads) OutlierDimsKernel(const float* __restrict__ column_max,
                                                                    int input_size,
                                                                    int32_t* __restrict__ dims,
                                                                    uint8_t* __restrict__ flags) {
  extern __shared__ float maxima_s[];
  __shared__ float candidate_value_s[kSelectWarps * kOutliers];
  __shared__ int candidate_index_s[kSelectWarps * kOutliers];
  __shared__ int chosen_s[kOutliers];
  for (int k = threadIdx.x; k < input_size; k += kSelectThreads) {
    maxima_s[k] = column_max[k];
    flags[k] = 0;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // The warp's slice: dims [slice_first, slice_last).
  const int per_warp = (input_size + kSelectWarps - 1) / kSelectWarps;
  const int slice_first = warp * per_warp;
  const int slice_last = min(slice_first + per_warp, input_size);
  for (int round = 0; round < kOutliers; ++round) {
    float best = -1.0f;
    int index = input_size;
    for (int k = slice_first + lane; k < slice_last; k += 32) {
      const float v = maxima_s[k];
      if (v > best) {
        best = v;
        index = k;
      }
    }
    WarpArgMax(best, index);
    if (lane == 0) {
      candidate_value_s[warp * kOutliers + round] = best;
      candidate_index_s[warp * kOutliers + round] = index;
      if (index < input_size) maxima_s[index] = -1.0f;  // taken
    }
    __syncwarp();
  }
  __syncthreads();
  if (warp == 0) {
    for (int round = 0; round < kOutliers; ++round) {
      float best = -1.0f;
      int index = input_size;
      for (int c = lane; c < kSelectWarps * kOutliers; c += 32) {
        const float v = candidate_value_s[c];
        const int i = candidate_index_s[c];
        if (v > best || (v == best && i < index)) {
          best = v;
          index = i;
        }
      }
      WarpArgMax(best, index);
      if (lane == 0) chosen_s[round] = index;
      // Every lane takes the chosen dim out of its own candidates.
      for (int c = lane; c < kSelectWarps * kOutliers; c += 32) {
        if (candidate_index_s[c] == index) candidate_value_s[c] = -1.0f;
      }
      __syncwarp();
    }
    if (lane < kOutliers) {
      // Sorted ascending by rank (the chosen dims are distinct).
      const int mine = chosen_s[lane];
      int rank = 0;
      for (int i = 0; i < kOutliers; ++i) rank += chosen_s[i] < mine ? 1 : 0;
      dims[rank] = mine;
      flags[mine] = 1;
    }
  }
}

// The weights' columns of the outlier dims as fp16 values, [channel][kOutliers]: the nibbles'
// two's-complement values, exact.
__global__ void __launch_bounds__(256) GatherOutlierWeightsKernel(const uint8_t* __restrict__ weights,
                                                                  int channels, int input_size,
                                                                  const int32_t* __restrict__ dims,
                                                                  __half* __restrict__ gathered) {
  __shared__ int32_t dims_s[kOutliers];
  if (threadIdx.x < kOutliers) dims_s[threadIdx.x] = dims[threadIdx.x];
  __syncthreads();
  const int index = blockIdx.x * 256 + threadIdx.x;  // a (channel, outlier) pair
  if (index >= channels * kOutliers) return;
  const int channel = index / kOutliers;
  const int k = dims_s[index % kOutliers];
  const uint8_t byte = weights[static_cast<size_t>(channel) * (input_size / 2) + k / 2];
  const int nibble = (byte >> ((k & 1) * 4)) & 15;
  gathered[index] = __float2half(static_cast<float>((nibble ^ 8) - 8));
}

// Per row: BF16 activations (optionally RMS-normed) to int8 per block of kActivationBlock dims
// with a float scale (max / 127, round to nearest), laid out tile-major like the fp16 path:
// quantized[row tile][chunk][row of the tile][kChunk bytes]; scales[row tile][chunk][row][block].
__global__ void __launch_bounds__(kThreads) QuantizeRowsKernel(
    const __nv_bfloat16* __restrict__ activation, int input_size, const float* __restrict__ norm_scale,
    float norm_eps, float norm_multiplier, const uint8_t* __restrict__ outlier_flags,
    const int32_t* __restrict__ outlier_dims, int8_t* __restrict__ quantized,
    float* __restrict__ scales, __half* __restrict__ outliers) {
  __shared__ float squares_s[kWarps];
  const int row = blockIdx.x;
  const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(activation + static_cast<size_t>(row) * input_size);
  float norm_factor = 1.0f;
  if (norm_scale != nullptr) {
    float squares = 0.0f;
    for (int i = threadIdx.x; i < input_size / 2; i += kThreads) {
      const float2 v = __bfloat1622float2(src[i]);
      squares = fmaf(v.x, v.x, squares);
      squares = fmaf(v.y, v.y, squares);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    if (threadIdx.x % 32 == 0) squares_s[threadIdx.x / 32] = squares;
    __syncthreads();
    squares = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) squares += squares_s[w];
    norm_factor = rsqrtf(squares / static_cast<float>(input_size) + norm_eps) * norm_multiplier;
  }
  // The value at dim 2i, 2i + 1 (the row stays in the cache between the two passes of a block);
  // an outlier dim reads as zero here and goes to the fp16 side.
  auto value_at = [&](int i) -> float2 {
    float2 v = __bfloat1622float2(src[i]);
    if (norm_scale != nullptr) {
      const float2 s = reinterpret_cast<const float2*>(norm_scale)[i];
      v.x *= norm_factor * s.x;
      v.y *= norm_factor * s.y;
    }
    return v;
  };
  auto quantizable_at = [&](int i) -> float2 {
    float2 v = value_at(i);
    if (outlier_flags[2 * i] != 0) v.x = 0.0f;
    if (outlier_flags[2 * i + 1] != 0) v.y = 0.0f;
    return v;
  };
  if (threadIdx.x < kOutliers) {
    const int k = outlier_dims[threadIdx.x];
    const float2 v = value_at(k / 2);
    outliers[static_cast<size_t>(row) * kOutliers + threadIdx.x] = __float2half((k & 1) ? v.y : v.x);
  }
  const int blocks = input_size / kActivationBlock;
  const size_t tile = row / kBlockRows;
  const size_t tile_row = row % kBlockRows;
  // One warp per block of the row at a time: the warp finds the block's max, then quantizes it;
  // a lane handles a pair of dims.
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  constexpr int kPairs = kActivationBlock / 2;
  for (int b = warp; b < blocks; b += kWarps) {
    const int first_pair = b * kPairs;
    float largest = 0.0f;
    for (int i = lane; i < kPairs; i += 32) {
      const float2 v = quantizable_at(first_pair + i);
      largest = fmaxf(largest, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    const float scale = largest > 0.0f && largest < INFINITY ? largest / 127.0f : 0.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    const int chunk = b * kActivationBlock / kChunk;
    const int in_chunk = b * kActivationBlock % kChunk;
    int8_t* out = quantized + (tile * (input_size / kChunk) + chunk) * (kBlockRows * kChunk) + tile_row * kChunk + in_chunk;
    for (int i = lane; i < kPairs; i += 32) {
      const float2 v = quantizable_at(first_pair + i);
      out[2 * i] = static_cast<int8_t>(__float2int_rn(v.x * inverse));
      out[2 * i + 1] = static_cast<int8_t>(__float2int_rn(v.y * inverse));
    }
    if (lane == 0) {
      // The MMA's int8 weights carry 16 times the nibble: the scale takes it out.
      scales[(tile * (input_size / kChunk) + chunk) * (kBlockRows * kBlocksPerChunk) + tile_row * kBlocksPerChunk + in_chunk / kActivationBlock] = scale * (1.0f / 16.0f);
    }
  }
}

// grid (rows / kBlockRows, column tiles). A block multiplies 128 activation rows by 128 weight
// rows (64 gate + 64 up with a gate). Warp w: rows 64 (w / 4), channels 32 (w % 4).
__global__ void __launch_bounds__(kThreads, 1) Int8GemmKernel(
    const int8_t* __restrict__ quantized, const float* __restrict__ quant_scales,
    const uint8_t* __restrict__ weights, const __nv_bfloat16* __restrict__ channel_scales,
    const __half* __restrict__ outliers, const __half* __restrict__ outlier_weights,
    int input_size, int output_size, int gate, __nv_bfloat16* __restrict__ output) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* activation_s = smem;
  unsigned char* packed_s = smem + kStages * kActivationStageBytes;
  float* scales_s = reinterpret_cast<float*>(smem + kStages * (kActivationStageBytes + kPackedStageBytes));
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int m0 = blockIdx.x * kBlockRows;
  const int block_channels = gate != 0 ? kBlockCols / 2 : kBlockCols;
  const int n0 = blockIdx.y * block_channels;
  const int chunks = input_size / kChunk;

  // Copies: 128 rows x 64 B of activations = 512 vectors, two per thread; the 2 or 4 scales of a
  // row = 128 x kBlocksPerChunk floats: a 4-byte copy per thread for the first 128 * kBlocksPerChunk
  // threads; packed weights 128 rows x 32 B = 256 vectors, one per thread.
  const int a_vector = tid % 8;  // 16-byte vector within a row's 128 bytes
  const int a_row = tid / 8;     // rows a_row, + 32, + 64, + 96
  const int8_t* activation_tile = quantized + static_cast<size_t>(blockIdx.x) * chunks * (kBlockRows * kChunk);
  const float* scale_tile = quant_scales + static_cast<size_t>(blockIdx.x) * chunks * (kBlockRows * kBlocksPerChunk);
  const int w_row = tid / 4;  // rows w_row and w_row + 64
  const int w_vector = tid % 4;
  const uint64_t channel_bytes = static_cast<uint64_t>(input_size) / 2;
  auto weight_row_of = [&](int row) {
    const int channel = min(n0 + row % block_channels, output_size - 1);
    const bool up_half = row / block_channels != 0;
    return weights + (up_half ? static_cast<uint64_t>(output_size) * channel_bytes : 0) +
           static_cast<uint64_t>(channel) * channel_bytes + w_vector * 16;
  };
  const uint8_t* weight_row = weight_row_of(w_row);
  const uint8_t* weight_row_2 = weight_row_of(w_row + 64);
  auto copy_stage = [&](int chunk, int stage, int scale_stage) {
    unsigned char* a = activation_s + stage * kActivationStageBytes;
    const int8_t* src = activation_tile + static_cast<size_t>(chunk) * (kBlockRows * kChunk);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      CopyAsync16(a + (a_row + r * 32) * kActivationRowBytes + a_vector * 16, src + (a_row + r * 32) * kChunk + a_vector * 16);
    }
    for (int i = tid; i < kBlockRows * kBlocksPerChunk; i += kThreads) {
      CopyAsync4(scales_s + scale_stage * (kBlockRows * kBlocksPerChunk) + i,
                 scale_tile + static_cast<size_t>(chunk) * (kBlockRows * kBlocksPerChunk) + i);
    }
    CopyAsync16(packed_s + stage * kPackedStageBytes + w_row * kPackedRowBytes + w_vector * 16,
                weight_row + static_cast<size_t>(chunk) * (kChunk / 2));
    CopyAsync16(packed_s + stage * kPackedStageBytes + (w_row + 64) * kPackedRowBytes + w_vector * 16,
                weight_row_2 + static_cast<size_t>(chunk) * (kChunk / 2));
  };

  const int warp_row = (warp / 4) * 64;
  const int warp_col = (warp % 4) * 32;
  float acc[kWarpRowGroups][kWarpColGroups][4];
#pragma unroll
  for (int rg = 0; rg < kWarpRowGroups; ++rg)
#pragma unroll
    for (int cg = 0; cg < kWarpColGroups; ++cg)
      acc[rg][cg][0] = acc[rg][cg][1] = acc[rg][cg][2] = acc[rg][cg][3] = 0.0f;

  // ldmatrix row addresses: lanes 0-7 rows 0-7 at k 0-15, 8-15 rows 8-15 at k 0-15, 16-23 rows
  // 0-7 at k 16-31, 24-31 rows 8-15 at k 16-31 (bytes of the slice).
  const int a_lane_row = warp_row + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_lane_k = (lane >> 4) * 16;
  // The thread's weight rows: channel g of each column group; bytes 2t, 2t+1 (k 4t..4t+3) and
  // 8 + 2t, 8 + 2t + 1 (k 16 + 4t..) of the slice's 16 bytes.
  const int b_row = warp_col + g;

  copy_stage(0, 0, 0);
  CommitAsync();
  if (chunks > 1) copy_stage(1, 1, 1);
  CommitAsync();
  WaitAsyncButOne();
  __syncthreads();

  // Two sets of int32 sums: a chunk's flush into the fp32 accumulators runs while the next
  // chunk's MMAs fill the other set, so the tensor pipe does not drain at the chunk's end.
  int32_t sums[2][kWarpRowGroups][kWarpColGroups][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg)
#pragma unroll
      for (int cg = 0; cg < kWarpColGroups; ++cg)
        sums[b][rg][cg][0] = sums[b][rg][cg][1] = sums[b][rg][cg][2] = sums[b][rg][cg][3] = 0;
  static_assert(kActivationBlock == kChunk, "one flush per chunk");
  auto flush = [&](int32_t (&set)[kWarpRowGroups][kWarpColGroups][4], const float* s_stage) {
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
      const float s_lo = s_stage[(warp_row + rg * 16 + g) * kBlocksPerChunk];
      const float s_hi = s_stage[(warp_row + rg * 16 + g + 8) * kBlocksPerChunk];
#pragma unroll
      for (int cg = 0; cg < kWarpColGroups; ++cg) {
        acc[rg][cg][0] = fmaf(static_cast<float>(set[rg][cg][0]), s_lo, acc[rg][cg][0]);
        acc[rg][cg][1] = fmaf(static_cast<float>(set[rg][cg][1]), s_lo, acc[rg][cg][1]);
        acc[rg][cg][2] = fmaf(static_cast<float>(set[rg][cg][2]), s_hi, acc[rg][cg][2]);
        acc[rg][cg][3] = fmaf(static_cast<float>(set[rg][cg][3]), s_hi, acc[rg][cg][3]);
        set[rg][cg][0] = set[rg][cg][1] = set[rg][cg][2] = set[rg][cg][3] = 0;
      }
    }
  };
  struct Operands {
    uint32_t a[kWarpRowGroups][4];
    uint32_t b[kWarpColGroups][2];
  };
  auto load_operands = [&](const unsigned char* a_stage, const unsigned char* p_stage, int slice, Operands& o) {
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
      LoadMatrixX4(o.a[rg][0], o.a[rg][1], o.a[rg][2], o.a[rg][3],
                   a_stage + (a_lane_row + rg * 16) * kActivationRowBytes + slice * 32 + a_lane_k);
    }
#pragma unroll
    for (int cg = 0; cg < kWarpColGroups; ++cg) {
      const unsigned char* row = p_stage + (b_row + cg * 8) * kPackedRowBytes + slice * 16;
      const uint32_t lo = *reinterpret_cast<const uint16_t*>(row + 2 * t);
      const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + 8 + 2 * t);
      o.b[cg][0] = WidenNibbles(lo);
      o.b[cg][1] = WidenNibbles(hi);
    }
  };
  auto mma_slice = [&](int32_t (&set)[kWarpRowGroups][kWarpColGroups][4], const Operands& o) {
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
#pragma unroll
      for (int cg = 0; cg < kWarpColGroups; ++cg) {
        MmaS8(set[rg][cg][0], set[rg][cg][1], set[rg][cg][2], set[rg][cg][3],
              o.a[rg][0], o.a[rg][1], o.a[rg][2], o.a[rg][3], o.b[cg][0], o.b[cg][1]);
      }
    }
  };

  int stage = 0;
  int scale_stage = 0;
  // The scales of the previous chunk, for its deferred flush.
  const float* previous_scales = scales_s;
  for (int chunk = 0; chunk < chunks; chunk += 2) {
#pragma unroll
    for (int buffer = 0; buffer < 2; ++buffer) {
      const int current = chunk + buffer;
      if (current < chunks) {
        const int next_stage = stage + 1 == kStages ? 0 : stage + 1;
        const int after_stage = next_stage + 1 == kStages ? 0 : next_stage + 1;
        const int after_scale_stage = (scale_stage + 2) % kScaleStages;
        if (current + 2 < chunks) copy_stage(current + 2, after_stage, after_scale_stage);
        CommitAsync();
        const unsigned char* a_stage = activation_s + stage * kActivationStageBytes;
        const unsigned char* p_stage = packed_s + stage * kPackedStageBytes;
        const float* s_stage = scales_s + scale_stage * (kBlockRows * kBlocksPerChunk);
#pragma unroll
        for (int slice = 0; slice < kSlices; ++slice) {
          Operands operands;
          load_operands(a_stage, p_stage, slice, operands);
          mma_slice(sums[buffer], operands);
          if (slice == 0 && current > 0) {
            // The previous chunk's sums, while this chunk's first MMAs run.
            flush(sums[buffer ^ 1], previous_scales);
          }
        }
        previous_scales = s_stage;
        WaitAsyncButOne();
        __syncthreads();
        stage = next_stage;
        scale_stage = scale_stage + 1 == kScaleStages ? 0 : scale_stage + 1;
      }
    }
  }
  // The last chunk's sums (a static index: a dynamic one would put the sums in local memory).
  if (((chunks - 1) & 1) == 0) {
    flush(sums[0], previous_scales);
  } else {
    flush(sums[1], previous_scales);
  }

  // The outlier dims: the rows' fp16 values [128][kOutliers] and the channels' [128][kOutliers]
  // staged in the (now free) operand stages, one fp16 MMA chunk of 32 dims.
  {
    WaitAsyncAll();
    __syncthreads();  // every warp is done with the operand stages
    unsigned char* x_s = smem;
    unsigned char* w_s = smem + kBlockRows * kOutlierRowBytes;
    for (int i = tid; i < kBlockRows * (kOutliers / 8); i += kThreads) {
      const int r = i / (kOutliers / 8);
      const int v = i % (kOutliers / 8);
      *reinterpret_cast<int4*>(x_s + r * kOutlierRowBytes + v * 16) =
          *reinterpret_cast<const int4*>(outliers + static_cast<size_t>(m0 + r) * kOutliers + v * 8);
      const int channel = min(n0 + r % block_channels, output_size - 1) + (r / block_channels != 0 ? output_size : 0);
      *reinterpret_cast<int4*>(w_s + r * kOutlierRowBytes + v * 16) =
          *reinterpret_cast<const int4*>(outlier_weights + static_cast<size_t>(channel) * kOutliers + v * 8);
    }
    __syncthreads();
    // A 16x16 fp16 operand per row group and slice; B (16 dims x 8 channels) per column pair.
    const unsigned char* x_lane = x_s + (warp_row + (lane & 7) + ((lane >> 3) & 1) * 8) * kOutlierRowBytes + (lane >> 4) * 16;
    const unsigned char* w_lane = w_s + (warp_col + (lane & 7) + (lane >> 4) * 8) * kOutlierRowBytes + ((lane >> 3) & 1) * 16;
#pragma unroll
    for (int slice = 0; slice < kOutliers / 16; ++slice) {
      uint32_t a[kWarpRowGroups][4];
#pragma unroll
      for (int rg = 0; rg < kWarpRowGroups; ++rg) {
        LoadMatrixX4(a[rg][0], a[rg][1], a[rg][2], a[rg][3], x_lane + rg * 16 * kOutlierRowBytes + slice * 32);
      }
#pragma unroll
      for (int pair = 0; pair < kWarpColGroups / 2; ++pair) {
        uint32_t b[4];
        LoadMatrixX4(b[0], b[1], b[2], b[3], w_lane + pair * 16 * kOutlierRowBytes + slice * 32);
#pragma unroll
        for (int rg = 0; rg < kWarpRowGroups; ++rg) {
          MmaF16F32(acc[rg][2 * pair][0], acc[rg][2 * pair][1], acc[rg][2 * pair][2], acc[rg][2 * pair][3],
                    a[rg][0], a[rg][1], a[rg][2], a[rg][3], b[0], b[1]);
          MmaF16F32(acc[rg][2 * pair + 1][0], acc[rg][2 * pair + 1][1], acc[rg][2 * pair + 1][2],
                    acc[rg][2 * pair + 1][3], a[rg][0], a[rg][1], a[rg][2], a[rg][3], b[2], b[3]);
        }
      }
    }
    __syncthreads();  // the stages are reused by the gate below
  }

  // Epilogue as the fp16 kernel's: the per-channel scales, the gate through shared memory.
  float* gate_s = reinterpret_cast<float*>(smem);
  if (gate != 0 && warp_col < block_channels) {
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
#pragma unroll
      for (int cg = 0; cg < kWarpColGroups; ++cg) {
        float* lo = gate_s + (warp_row + rg * 16 + g) * block_channels + warp_col + cg * 8 + 2 * t;
        *reinterpret_cast<float2*>(lo) = make_float2(acc[rg][cg][0], acc[rg][cg][1]);
        *reinterpret_cast<float2*>(lo + 8 * block_channels) = make_float2(acc[rg][cg][2], acc[rg][cg][3]);
      }
    }
  }
  if (gate != 0) {
    __syncthreads();
    if (warp_col < block_channels) return;
  }
#pragma unroll
  for (int rg = 0; rg < kWarpRowGroups; ++rg) {
    const int row = warp_row + rg * 16 + g;
    __nv_bfloat16* out_lo = output + static_cast<size_t>(m0 + row) * output_size;
    __nv_bfloat16* out_hi = out_lo + static_cast<size_t>(8) * output_size;
#pragma unroll
    for (int cg = 0; cg < kWarpColGroups; ++cg) {
      const int column = (gate != 0 ? warp_col - block_channels : warp_col) + cg * 8 + 2 * t;
      const int n = n0 + column;
      if (n >= output_size) continue;
      const float2 scale = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(channel_scales + (gate != 0 ? output_size : 0) + n));
      float2 lo = make_float2(acc[rg][cg][0] * scale.x, acc[rg][cg][1] * scale.y);
      float2 hi = make_float2(acc[rg][cg][2] * scale.x, acc[rg][cg][3] * scale.y);
      if (gate != 0) {
        const float2 gate_scale = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(channel_scales + n));
        const float2 gate_lo = *reinterpret_cast<const float2*>(gate_s + row * block_channels + column);
        const float2 gate_hi = *reinterpret_cast<const float2*>(gate_s + (row + 8) * block_channels + column);
        lo.x *= Gelu(gate_lo.x * gate_scale.x, gate);
        lo.y *= Gelu(gate_lo.y * gate_scale.y, gate);
        hi.x *= Gelu(gate_hi.x * gate_scale.x, gate);
        hi.y *= Gelu(gate_hi.y * gate_scale.y, gate);
      }
      *reinterpret_cast<__nv_bfloat162*>(out_lo + n) = __floats2bfloat162_rn(lo.x, lo.y);
      *reinterpret_cast<__nv_bfloat162*>(out_hi + n) = __floats2bfloat162_rn(hi.x, hi.y);
    }
  }
}

}  // namespace litert::nvidia::int8_gemm

namespace litert::nvidia::int8_gemm {
struct Workspace {
  size_t scales, quantized, column_max, row_factor, dims, flags, outliers, outlier_weights, total;
};
Workspace Layout(const LiteRtNvidiaGemmShape* shape) {
  const size_t rows = shape->rows;
  const size_t channels = (shape->gate != 0 ? 2 : 1) * static_cast<size_t>(shape->output_size);
  auto align = [](size_t x) { return (x + 255) / 256 * 256; };
  Workspace w{};
  w.scales = 0;
  w.quantized = align(w.scales + rows * (shape->input_size / kActivationBlock) * sizeof(float));
  w.column_max = align(w.quantized + rows * shape->input_size);
  w.row_factor = align(w.column_max + shape->input_size * sizeof(float));
  w.dims = align(w.row_factor + rows * sizeof(float));
  w.flags = align(w.dims + kOutliers * sizeof(int32_t));
  w.outliers = align(w.flags + shape->input_size);
  w.outlier_weights = align(w.outliers + rows * kOutliers * sizeof(__half));
  w.total = align(w.outlier_weights + channels * kOutliers * sizeof(__half));
  return w;
}
}  // namespace litert::nvidia::int8_gemm

extern "C" bool LiteRtNvidiaInt8GemmSupports(const LiteRtNvidiaGemmShape* shape) {
  using namespace litert::nvidia::int8_gemm;
  return LiteRtNvidiaSubbyteGemmSupports(shape) && shape->scale_block == 0 &&
         shape->input_size % kChunk == 0 && shape->input_size <= 16384;
}

extern "C" size_t LiteRtNvidiaInt8GemmWorkspaceBytes(const LiteRtNvidiaGemmShape* shape) {
  if (!LiteRtNvidiaInt8GemmSupports(shape)) return 0;
  return litert::nvidia::int8_gemm::Layout(shape).total;
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4GemmInt8Normed(
    const LiteRtNvidiaGemmShape* shape, const void* activation, const float* norm_scale,
    float norm_eps, float norm_multiplier, const uint8_t* weights, const void* scales, void* output,
    void* workspace, cudaStream_t stream) {
  using namespace litert::nvidia::int8_gemm;
  if (!LiteRtNvidiaInt8GemmSupports(shape) || activation == nullptr || weights == nullptr ||
      scales == nullptr || output == nullptr || workspace == nullptr ||
      (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  static bool attribute_set = false;
  if (!attribute_set) {
    cudaError_t status = cudaFuncSetAttribute(Int8GemmKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(OutlierDimsKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  16384 * static_cast<int>(sizeof(float)));
    if (status != cudaSuccess) return status;
    attribute_set = true;
  }
  const Workspace w = Layout(shape);
  char* base = static_cast<char*>(workspace);
  float* quant_scales = reinterpret_cast<float*>(base + w.scales);
  int8_t* quantized = reinterpret_cast<int8_t*>(base + w.quantized);
  float* column_max = reinterpret_cast<float*>(base + w.column_max);
  float* row_factor = reinterpret_cast<float*>(base + w.row_factor);
  int32_t* dims = reinterpret_cast<int32_t*>(base + w.dims);
  uint8_t* flags = reinterpret_cast<uint8_t*>(base + w.flags);
  __half* outliers = reinterpret_cast<__half*>(base + w.outliers);
  __half* outlier_weights = reinterpret_cast<__half*>(base + w.outlier_weights);
  const int channels = (shape->gate != 0 ? 2 : 1) * shape->output_size;
  const __nv_bfloat16* typed_activation = static_cast<const __nv_bfloat16*>(activation);
  if (norm_scale != nullptr) {
    RowFactorKernel<<<shape->rows, kThreads, 0, stream>>>(typed_activation, shape->input_size, norm_eps,
                                                           norm_multiplier, row_factor);
  }
  cudaError_t status = cudaMemsetAsync(column_max, 0, shape->input_size * sizeof(float), stream);
  if (status != cudaSuccess) return status;
  ColumnMaxKernel<<<dim3((shape->input_size + 255) / 256, (shape->rows + kColumnMaxRows - 1) / kColumnMaxRows),
                    256, 0, stream>>>(typed_activation, shape->rows, shape->input_size,
                                      norm_scale != nullptr ? row_factor : nullptr, norm_scale, column_max);
  OutlierDimsKernel<<<1, kSelectThreads, shape->input_size * sizeof(float), stream>>>(
      column_max, shape->input_size, dims, flags);
  GatherOutlierWeightsKernel<<<(channels * kOutliers + 255) / 256, 256, 0, stream>>>(
      weights, channels, shape->input_size, dims, outlier_weights);
  QuantizeRowsKernel<<<shape->rows, kThreads, 0, stream>>>(
      typed_activation, shape->input_size, norm_scale, norm_eps, norm_multiplier, flags, dims,
      quantized, quant_scales, outliers);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const int block_channels = shape->gate != 0 ? kBlockCols / 2 : kBlockCols;
  const dim3 grid(shape->rows / kBlockRows, (shape->output_size + block_channels - 1) / block_channels);
  Int8GemmKernel<<<grid, kThreads, kSharedBytes, stream>>>(
      quantized, quant_scales, weights, static_cast<const __nv_bfloat16*>(scales), outliers,
      outlier_weights, shape->input_size, shape->output_size, shape->gate,
      static_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}
