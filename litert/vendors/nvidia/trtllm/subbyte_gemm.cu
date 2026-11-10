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

#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlockRows = 128;  // activation rows of a block
constexpr int kBlockCols = 128;  // weight rows of a block
constexpr int kChunk = 64;  // input dims staged at a time
constexpr int kSlices = kChunk / 16;
constexpr int kChunkVectors = kChunk / 8;  // 16-byte vectors per FP16 row
// The activation rows of a thread: one vector of each per chunk.
constexpr int kThreadRows = kBlockRows * kChunkVectors / kThreads;
constexpr int kThreadRowStep = kThreads / kChunkVectors;

// Row strides are padded so that the eight rows of one ldmatrix fall in
// distinct shared-memory banks. Everything is double buffered: the chunk
// being multiplied and the next one.
constexpr int kRowBytes = kChunk * 2 + 16;
constexpr size_t kActivationBytes = static_cast<size_t>(kBlockRows) * kRowBytes;
constexpr size_t kWeightBytes = static_cast<size_t>(kBlockCols) * kRowBytes;
constexpr size_t kTableBytes = 256 * sizeof(uint32_t);
// The packed weights of a column tile for one chunk.
constexpr size_t kTileChunkBytes = static_cast<size_t>(kBlockCols) * kChunk / 2;
constexpr size_t kSharedBytes = 2 * kActivationBytes + 2 * kWeightBytes +
                                2 * kTileChunkBytes + kTableBytes;
// A warp multiplies 64 activation rows by 32 weight rows.
constexpr int kWarpRowGroups = 4;  // of 16 rows
constexpr int kWarpColGroups = 4;  // of 8 weight rows

// m16n8k16 accumulating in fp16: twice the throughput of fp32 accumulation.
__device__ __forceinline__ void Mma(uint32_t& d0, uint32_t& d1, uint32_t a0,
                                    uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d0), "+r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ldmatrix.x4: lanes 0-7 pass the addresses of the rows of matrix 0 (eight
// 16-bit elements each), lanes 8-15 those of matrix 1, and so on. Every lane
// receives its operand fragment of each matrix.
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

__device__ __forceinline__ void WaitAsync() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ float2 UnpackHalves(uint32_t word) {
  return __half22float2(*reinterpret_cast<const __half2*>(&word));
}

__device__ __forceinline__ uint32_t PackHalves(float lo, float hi) {
  const __half2 packed = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

__device__ __forceinline__ float Gelu(float x, int gate) {
  if (gate == kLiteRtNvidiaGemmGateGeluTanh) {
    return 0.5f * x *
           (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// One block per activation row: row_scale[m] is the power of two that brings
// the largest magnitude of the row into [0.5, 1). The rows times it, in FP16,
// are laid out as scaled[row tile][chunk][row of the tile][kChunk], so that
// a block of the GEMM reads one contiguous tile per chunk.
// With `norm_scale`, the row is first RMS-normalized: x * rsqrt(mean(x^2) +
// norm_eps) * norm_scale[k] * norm_multiplier.
__global__ void __launch_bounds__(kThreads)
    ScaleRowsKernel(const __nv_bfloat16* __restrict__ activation,
                    const int32_t* __restrict__ row_indices, int input_size,
                    const float* __restrict__ norm_scale, float norm_eps,
                    float norm_multiplier, float* __restrict__ row_scale,
                    __half* __restrict__ scaled) {
  __shared__ float max_s[kWarps];
  __shared__ float squares_s[kWarps];
  // A gathered row, a zero row for index -1, or nothing for an index below
  // -1 (a row of an unused tile).
  const int source =
      row_indices == nullptr ? static_cast<int>(blockIdx.x) : row_indices[blockIdx.x];
  if (source < -1) return;
  const __nv_bfloat162* row =
      source < 0 ? nullptr
                 : reinterpret_cast<const __nv_bfloat162*>(
                       activation + static_cast<size_t>(source) * input_size);
  // The norm factor of the row, if any, then the largest magnitude after it.
  float norm_factor = 1.0f;
  if (norm_scale != nullptr && row != nullptr) {
    float squares = 0.0f;
    for (int i = threadIdx.x; i < input_size / 2; i += kThreads) {
      const float2 values = __bfloat1622float2(row[i]);
      squares = fmaf(values.x, values.x, squares);
      squares = fmaf(values.y, values.y, squares);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      squares += __shfl_xor_sync(0xffffffffu, squares, offset);
    }
    if (threadIdx.x % 32 == 0) squares_s[threadIdx.x / 32] = squares;
    __syncthreads();
    squares = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) squares += squares_s[w];
    norm_factor =
        rsqrtf(squares / static_cast<float>(input_size) + norm_eps) *
        norm_multiplier;
  }
  auto value_at = [&](int i) -> float2 {
    float2 values = __bfloat1622float2(row[i]);
    if (norm_scale != nullptr) {
      const float2 scales = reinterpret_cast<const float2*>(norm_scale)[i];
      values.x *= norm_factor * scales.x;
      values.y *= norm_factor * scales.y;
    }
    return values;
  };
  float largest = 0.0f;
  for (int i = threadIdx.x; row != nullptr && i < input_size / 2; i += kThreads) {
    const float2 values = value_at(i);
    largest = fmaxf(largest, fmaxf(fabsf(values.x), fabsf(values.y)));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
  }
  if (threadIdx.x % 32 == 0) {
    max_s[threadIdx.x / 32] = largest;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    largest = fmaxf(largest, max_s[w]);
  }
  // Rows of zeros, and rows with an infinity or a NaN, stay as they are.
  int exponent = 0;
  if (largest > 0.0f && largest < INFINITY) {
    frexpf(largest, &exponent);
  }
  const float scale = ldexpf(1.0f, -exponent);
  if (threadIdx.x == 0) {
    row_scale[blockIdx.x] = scale;
  }
  const size_t tile = blockIdx.x / kBlockRows;
  const size_t tile_row = blockIdx.x % kBlockRows;
  __half2* out = reinterpret_cast<__half2*>(
      scaled + tile * kBlockRows * input_size + tile_row * kChunk);
  for (int i = threadIdx.x; i < input_size / 2; i += kThreads) {
    const float2 values =
        row == nullptr ? make_float2(0.0f, 0.0f) : value_at(i);
    out[static_cast<size_t>(i / (kChunk / 2)) * (kBlockRows * kChunk / 2) +
        i % (kChunk / 2)] =
        __floats2half2_rn(values.x * scale, values.y * scale);
  }
}

// Where a block's weights and scales are: dense row-major weights, or the
// interleaved experts of a grouped GEMM (LiteRtNvidiaGroupedGemmWeights).
struct WeightAddressing {
  const uint8_t* gate;
  const uint8_t* up;
  const void* gate_scales;
  const void* up_scales;
  uint64_t channel_stride;
  uint64_t scale_channel_stride;
  uint64_t expert_stride;
  uint64_t expert_scale_stride;
  const int32_t* tile_experts;
  // Compact scales (see LiteRtNvidiaGroupedGemmWeights).
  const void* gate_scale_max;
  const void* up_scale_max;
  uint64_t max_channel_stride;
  uint64_t max_expert_stride;
  int32_t compact_scales;
};

// grid (rows / kBlockRows, column tiles): the blocks that read the same
// weights are adjacent, so the weights come from memory once. A block
// multiplies kBlockRows activation rows by the kBlockCols weight rows of its
// column tile: the channels [n0, n0 + kBlockCols), or with a gate the
// channels [n0, n0 + kBlockCols / 2) of the gate projection followed by the
// same channels of the up projection. Warp w owns the 64 rows w / 4 and the
// 32 weight rows w % 4 of those: four row groups times four column groups
// per k-slice, from six ldmatrix.
//
// A multiprocessor has four tensor units, one per two warps of the block,
// and they are the bottleneck: a unit that waits costs throughput. The block
// walks the input dims in chunks. While it multiplies one chunk, the
// activations of the next and the packed weights of the one after are
// copied into shared memory, and every thread converts the packed weights
// of the next chunk, one word of eight weights per k-slice.
// kBlockwise: `scales` are FP16 per block of 32 input dims ([channels,
// input_size / 32]) and multiply the weights as they are converted; otherwise
// they are BF16 per channel and multiply the sums.
template <bool kBlockwise>
__global__ void __launch_bounds__(kThreads, 1)
    Int4GemmKernel(const __half* __restrict__ activation,
                   const WeightAddressing addressing,
                   const float* __restrict__ row_scale, int input_size,
                   int output_size, int gate,
                   __nv_bfloat16* __restrict__ output) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* activation_s = smem;
  unsigned char* weight_s = smem + 2 * kActivationBytes;
  unsigned char* packed_s = weight_s + 2 * kWeightBytes;
  uint32_t* table_s = reinterpret_cast<uint32_t*>(packed_s + 2 * kTileChunkBytes);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int m0 = blockIdx.x * kBlockRows;
  const int block_channels = gate != 0 ? kBlockCols / 2 : kBlockCols;
  const int n0 = blockIdx.y * block_channels;
  const int chunks = input_size / kChunk;

  // table[byte] = the two INT4 values of the byte as FP16.
  {
    const int lo = ((tid & 15) ^ 8) - 8;
    const int hi = ((tid >> 4) ^ 8) - 8;
    table_s[tid] = PackHalves(static_cast<float>(lo), static_cast<float>(hi));
  }

  // A thread copies the vectors tid, tid + kThreads, ... of the tile of a
  // chunk, one of each of kThreadRows activation rows, and copies and
  // converts half of the 32 bytes a chunk takes from one weight row. The
  // weight rows of the block are the channels [n0, n0 + block_channels) of
  // the row-major weights, with a gate of the gate projection followed by
  // the up projection. Channels past the last one read the last one; their
  // products are not stored.
  const int a_vector = tid % kChunkVectors;
  const int a_row = tid / kChunkVectors;
  const __half* activation_tile =
      activation + static_cast<size_t>(m0) * input_size + tid * 8;
  const int w_row = tid / 2;
  const int w_channel = min(n0 + w_row % block_channels, output_size - 1);
  const int expert = addressing.tile_experts == nullptr
                         ? 0
                         : addressing.tile_experts[blockIdx.x];
  if (expert < 0) return;  // an unused tile of a grouped GEMM
  const bool up_half = w_row / block_channels != 0;
  const uint8_t* weight_row =
      (up_half ? addressing.up : addressing.gate) +
      expert * addressing.expert_stride +
      static_cast<uint64_t>(w_channel) * addressing.channel_stride +
      (tid % 2) * 16;
  unsigned char* activation_thread =
      activation_s + a_row * kRowBytes + a_vector * 16;
  unsigned char* packed_thread = packed_s + tid * 16;
  unsigned char* weight_thread =
      weight_s + w_row * kRowBytes + (tid % 2) * 64;
  // The thread's 32 weights of a chunk are one block: its scale, for chunk c,
  // is block 2 c + (tid % 2) of the thread's channel.
  const uint64_t scale_offset =
      expert * addressing.expert_scale_stride +
      static_cast<uint64_t>(w_channel) * addressing.scale_channel_stride +
      (tid % 2);
  const void* scale_base =
      up_half ? addressing.up_scales : addressing.gate_scales;
  const __half* block_scales = static_cast<const __half*>(scale_base) + scale_offset;
  const uint8_t* block_fractions =
      static_cast<const uint8_t*>(scale_base) + scale_offset;
  float row_max = 0.0f;
  if constexpr (kBlockwise) {
    if (addressing.compact_scales != 0) {
      const __half* maxima = static_cast<const __half*>(
          up_half ? addressing.up_scale_max : addressing.gate_scale_max);
      row_max = __half2float(
                    maxima[expert * addressing.max_expert_stride +
                           static_cast<uint64_t>(w_channel) *
                               addressing.max_channel_stride]) *
                (1.0f / 255.0f);
    }
  }
  auto scale_of = [&](int chunk) -> __half2 {
    if constexpr (kBlockwise) {
      if (addressing.compact_scales != 0) {
        return __half2half2(__float2half_rn(
            row_max * static_cast<float>(block_fractions[2 * chunk])));
      }
      return __half2half2(block_scales[2 * chunk]);
    } else {
      return __half2half2(__float2half(1.0f));
    }
  };

  // `buffer` is the parity of the chunk.
  auto copy_activation = [&](int chunk, int buffer) {
#pragma unroll
    for (int j = 0; j < kThreadRows; ++j) {
      CopyAsync16(activation_thread + buffer * kActivationBytes +
                      j * kThreadRowStep * kRowBytes,
                  activation_tile +
                      static_cast<size_t>(chunk) * kBlockRows * kChunk +
                      j * kThreads * 8);
    }
  };
  auto copy_packed = [&](int chunk, int buffer) {
    CopyAsync16(packed_thread + buffer * kTileChunkBytes,
                weight_row + static_cast<size_t>(chunk) * (kChunk / 2));
  };
  auto packed = [&](int buffer) {
    return *reinterpret_cast<const int4*>(packed_thread +
                                          buffer * kTileChunkBytes);
  };
  // Four bytes are eight weights: one 16-byte vector of FP16, times the
  // block's scale when the scales are blockwise.
  auto convert = [&](uint32_t word, __half2 scale) {
    int4 values =
        make_int4(table_s[word & 0xff], table_s[(word >> 8) & 0xff],
                  table_s[(word >> 16) & 0xff], table_s[word >> 24]);
    if constexpr (kBlockwise) {
      auto scaled = [&](int value) {
        __half2 pair = *reinterpret_cast<const __half2*>(&value);
        pair = __hmul2(pair, scale);
        return *reinterpret_cast<const int*>(&pair);
      };
      values.x = scaled(values.x);
      values.y = scaled(values.y);
      values.z = scaled(values.z);
      values.w = scaled(values.w);
    }
    return values;
  };
  auto store = [&](int buffer, int index, int4 vector) {
    *reinterpret_cast<int4*>(weight_thread + buffer * kWeightBytes +
                             index * 16) = vector;
  };

  const int warp_row = (warp / 4) * 64;
  const int warp_col = (warp % 4) * 32;
  float acc[kWarpRowGroups][kWarpColGroups][4];
  uint32_t sums[kWarpRowGroups][kWarpColGroups][2];
#pragma unroll
  for (int rg = 0; rg < kWarpRowGroups; ++rg) {
#pragma unroll
    for (int cg = 0; cg < kWarpColGroups; ++cg) {
      acc[rg][cg][0] = acc[rg][cg][1] = acc[rg][cg][2] = acc[rg][cg][3] = 0.0f;
      sums[rg][cg][0] = sums[rg][cg][1] = 0;
    }
  }

  // Per-lane operand addresses. A 16x16 "A" operand is {rows 0-7, rows 8-15}
  // x {dims 0-7, dims 8-15}.
  const unsigned char* activation_lane =
      activation_s +
      (warp_row + (lane & 7) + ((lane >> 3) & 1) * 8) * kRowBytes +
      (lane >> 4) * 16;
  // "B" (dims x weight rows): matrices {0, 1} are weight rows 0-7 at dims
  // {0-7, 8-15} of the slice, matrices {2, 3} weight rows 8-15.
  const unsigned char* weight_lane =
      weight_s + (warp_col + (lane & 7) + (lane >> 4) * 8) * kRowBytes +
      ((lane >> 3) & 1) * 16;

  struct Operands {
    uint32_t a[kWarpRowGroups][4];
    uint32_t b[kWarpColGroups / 2][4];
  };
  auto load_operands = [&](int buffer, int slice, Operands& operands) {
    const unsigned char* a_slice =
        activation_lane + buffer * kActivationBytes + slice * 32;
    const unsigned char* w_slice =
        weight_lane + buffer * kWeightBytes + slice * 32;
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
      LoadMatrixX4(operands.a[rg][0], operands.a[rg][1], operands.a[rg][2],
                   operands.a[rg][3], a_slice + rg * 16 * kRowBytes);
    }
#pragma unroll
    for (int pair = 0; pair < kWarpColGroups / 2; ++pair) {
      LoadMatrixX4(operands.b[pair][0], operands.b[pair][1],
                   operands.b[pair][2], operands.b[pair][3],
                   w_slice + pair * 16 * kRowBytes);
    }
  };

  // Chunks 0 and 1 of the packed weights and chunk 0 of the activations.
  copy_activation(0, 0);
  copy_packed(0, 0);
  copy_packed(1, 1);
  CommitAsync();
  WaitAsync();
  __syncthreads();  // the table too
  {
    const int4 words = packed(0);
    const __half2 scale = scale_of(0);
    store(0, 0, convert(words.x, scale));
    store(0, 1, convert(words.y, scale));
    store(0, 2, convert(words.z, scale));
    store(0, 3, convert(words.w, scale));
  }
  __syncthreads();

  // Nothing in the loop waits for memory. The copies of a chunk start a
  // chunk (activations) or two (packed weights) before the block needs
  // them; the operands of a k-slice are in registers, and the weights of the
  // next chunk converted, a k-slice before.
  Operands operands;
  load_operands(0, 0, operands);
  int4 words = packed(1);
  for (int chunk = 0; chunk < chunks; chunk += 2) {
#pragma unroll
    for (int buffer = 0; buffer < 2; ++buffer) {
      const int current = chunk + buffer;
      const bool more = current + 1 < chunks;
      if (more) {
        copy_activation(current + 1, buffer ^ 1);
      }
      if (current + 2 < chunks) {
        copy_packed(current + 2, buffer);
      }
      CommitAsync();
      const uint32_t next_words[4] = {
          static_cast<uint32_t>(words.x), static_cast<uint32_t>(words.y),
          static_cast<uint32_t>(words.z), static_cast<uint32_t>(words.w)};
      const __half2 next_scale = scale_of(more ? current + 1 : current);
      // With an odd number of chunks the last buffer holds no chunk.
      if (current < chunks) {
#pragma unroll
      for (int slice = 0; slice < kSlices; ++slice) {
        Operands next;
        if (slice + 1 < kSlices) {
          load_operands(buffer, slice + 1, next);
        }
        const int4 converted = convert(next_words[slice], next_scale);
#pragma unroll
        for (int pair = 0; pair < kWarpColGroups / 2; ++pair) {
#pragma unroll
          for (int rg = 0; rg < kWarpRowGroups; ++rg) {
            Mma(sums[rg][2 * pair][0], sums[rg][2 * pair][1],
                operands.a[rg][0], operands.a[rg][1], operands.a[rg][2],
                operands.a[rg][3], operands.b[pair][0], operands.b[pair][1]);
            Mma(sums[rg][2 * pair + 1][0], sums[rg][2 * pair + 1][1],
                operands.a[rg][0], operands.a[rg][1], operands.a[rg][2],
                operands.a[rg][3], operands.b[pair][2], operands.b[pair][3]);
          }
        }
        if (more) {
          store(buffer ^ 1, slice, converted);
        }
        if (slice + 1 < kSlices) {
          operands = next;
        }
      }
      }
      if (buffer == 1) {
        // The fp16 sums cover two chunks: 128 products.
#pragma unroll
        for (int rg = 0; rg < kWarpRowGroups; ++rg) {
#pragma unroll
          for (int cg = 0; cg < kWarpColGroups; ++cg) {
            const float2 lo = UnpackHalves(sums[rg][cg][0]);
            const float2 hi = UnpackHalves(sums[rg][cg][1]);
            acc[rg][cg][0] += lo.x;
            acc[rg][cg][1] += lo.y;
            acc[rg][cg][2] += hi.x;
            acc[rg][cg][3] += hi.y;
            sums[rg][cg][0] = sums[rg][cg][1] = 0;
          }
        }
      }
      WaitAsync();
      __syncthreads();
      if (more) {
        load_operands(buffer ^ 1, 0, operands);
        words = packed(buffer);
      }
    }
  }

  // acc[rg][cg][{0, 1}] are row 16 rg + g, weight rows 8 cg + 2t + {0, 1} of
  // the warp's tile; acc[rg][cg][{2, 3}] row 16 rg + g + 8. With a gate, the
  // warps w and w + 2 of a row half hold the gate and the up projection of
  // the same channels: the gate goes through shared memory.
  float* gate_s = reinterpret_cast<float*>(smem);
  if (gate != 0 && warp_col < block_channels) {
#pragma unroll
    for (int rg = 0; rg < kWarpRowGroups; ++rg) {
#pragma unroll
      for (int cg = 0; cg < kWarpColGroups; ++cg) {
        float* lo = gate_s +
                    (warp_row + rg * 16 + g) * block_channels + warp_col +
                    cg * 8 + 2 * t;
        *reinterpret_cast<float2*>(lo) =
            make_float2(acc[rg][cg][0], acc[rg][cg][1]);
        *reinterpret_cast<float2*>(lo + 8 * block_channels) =
            make_float2(acc[rg][cg][2], acc[rg][cg][3]);
      }
    }
  }
  if (gate != 0) {
    __syncthreads();
    if (warp_col < block_channels) {
      return;
    }
  }
#pragma unroll
  for (int rg = 0; rg < kWarpRowGroups; ++rg) {
    const int row = warp_row + rg * 16 + g;
    const float unscale_lo = 1.0f / row_scale[m0 + row];
    const float unscale_hi = 1.0f / row_scale[m0 + row + 8];
    __nv_bfloat16* out_lo =
        output + static_cast<size_t>(m0 + row) * output_size;
    __nv_bfloat16* out_hi = out_lo + static_cast<size_t>(8) * output_size;
#pragma unroll
    for (int cg = 0; cg < kWarpColGroups; ++cg) {
      const int column = (gate != 0 ? warp_col - block_channels : warp_col) +
                         cg * 8 + 2 * t;
      const int n = n0 + column;
      if (n >= output_size) continue;
      float2 lo = make_float2(acc[rg][cg][0], acc[rg][cg][1]);
      float2 hi = make_float2(acc[rg][cg][2], acc[rg][cg][3]);
      float2 scale = make_float2(1.0f, 1.0f);
      float2 gate_scale = make_float2(1.0f, 1.0f);
      if constexpr (!kBlockwise) {
        // Per-channel scales: channel n of the (up) projection, consecutive.
        const auto* up_or_plain = static_cast<const __nv_bfloat16*>(
            gate != 0 ? addressing.up_scales : addressing.gate_scales);
        scale = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(up_or_plain + n));
        if (gate != 0) {
          gate_scale = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(
                  static_cast<const __nv_bfloat16*>(addressing.gate_scales) +
                  n));
        }
      }
      lo.x *= scale.x * unscale_lo;
      lo.y *= scale.y * unscale_lo;
      hi.x *= scale.x * unscale_hi;
      hi.y *= scale.y * unscale_hi;
      if (gate != 0) {
        const float2 gate_lo = *reinterpret_cast<const float2*>(
            gate_s + row * block_channels + column);
        const float2 gate_hi = *reinterpret_cast<const float2*>(
            gate_s + (row + 8) * block_channels + column);
        lo.x *= Gelu(gate_lo.x * gate_scale.x * unscale_lo, gate);
        lo.y *= Gelu(gate_lo.y * gate_scale.y * unscale_lo, gate);
        hi.x *= Gelu(gate_hi.x * gate_scale.x * unscale_hi, gate);
        hi.y *= Gelu(gate_hi.y * gate_scale.y * unscale_hi, gate);
      }
      *reinterpret_cast<__nv_bfloat162*>(out_lo + n) =
          __floats2bfloat162_rn(lo.x, lo.y);
      *reinterpret_cast<__nv_bfloat162*>(out_hi + n) =
          __floats2bfloat162_rn(hi.x, hi.y);
    }
  }
}

size_t ScaleBytes(int rows) {
  return (static_cast<size_t>(rows) * sizeof(float) + 255) / 256 * 256;
}

int BlockChannels(const LiteRtNvidiaGemmShape& shape) {
  return shape.gate != 0 ? kBlockCols / 2 : kBlockCols;
}

int ColumnTiles(const LiteRtNvidiaGemmShape& shape) {
  return (shape.output_size + BlockChannels(shape) - 1) / BlockChannels(shape);
}

}  // namespace

extern "C" bool LiteRtNvidiaSubbyteGemmSupports(
    const LiteRtNvidiaGemmShape* shape) {
  return shape != nullptr && shape->rows > 0 &&
         shape->rows % kBlockRows == 0 && shape->input_size > 0 &&
         shape->input_size % kChunk == 0 && shape->output_size > 0 &&
         shape->output_size % 2 == 0 &&
         shape->gate >= kLiteRtNvidiaGemmGateNone &&
         shape->gate <= kLiteRtNvidiaGemmGateGeluErf &&
         (shape->scale_block == 0 || shape->scale_block == 32);
}

extern "C" bool LiteRtNvidiaSubbyteGemmAvailable() {
  int device = 0;
  int major = 0;
  int shared_memory = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                device) == cudaSuccess &&
         cudaDeviceGetAttribute(&shared_memory,
                                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                device) == cudaSuccess &&
         major >= 8 && static_cast<size_t>(shared_memory) >= kSharedBytes;
}

extern "C" size_t LiteRtNvidiaSubbyteGemmWeightBytes(
    const LiteRtNvidiaGemmShape* shape) {
  if (!LiteRtNvidiaSubbyteGemmSupports(shape)) {
    return 0;
  }
  return static_cast<size_t>(shape->gate != 0 ? 2 : 1) * shape->output_size *
         (shape->input_size / 2);
}

extern "C" int32_t LiteRtNvidiaSubbyteGemmBlocks(
    const LiteRtNvidiaGemmShape* shape) {
  if (!LiteRtNvidiaSubbyteGemmSupports(shape)) {
    return 0;
  }
  return shape->rows / kBlockRows * ColumnTiles(*shape);
}

extern "C" bool LiteRtNvidiaSubbyteGemmFillsDevice(
    const LiteRtNvidiaGemmShape* shape) {
  int device = 0;
  int multiprocessors = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&multiprocessors,
                                cudaDevAttrMultiProcessorCount,
                                device) == cudaSuccess &&
         multiprocessors > 0 &&
         static_cast<int64_t>(LiteRtNvidiaSubbyteGemmBlocks(shape)) * 2 >=
             static_cast<int64_t>(multiprocessors) * 3;
}

extern "C" size_t LiteRtNvidiaSubbyteGemmWorkspaceBytes(
    const LiteRtNvidiaGemmShape* shape) {
  if (!LiteRtNvidiaSubbyteGemmSupports(shape)) {
    return 0;
  }
  return ScaleBytes(shape->rows) +
         static_cast<size_t>(shape->rows) * shape->input_size * sizeof(__half);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4Gemm(
    const LiteRtNvidiaGemmShape* shape, const void* activation,
    const uint8_t* weights, const void* scales, void* output, void* workspace,
    cudaStream_t stream) {
  return LiteRtNvidiaLaunchBf16Int4GemmNormed(shape, activation, nullptr, 0.0f,
                                              1.0f, weights, scales, output,
                                              workspace, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4GemmNormed(
    const LiteRtNvidiaGemmShape* shape, const void* activation,
    const float* norm_scale, float norm_eps, float norm_multiplier,
    const uint8_t* weights, const void* scales, void* output, void* workspace,
    cudaStream_t stream) {
  if (!LiteRtNvidiaSubbyteGemmSupports(shape) || activation == nullptr ||
      weights == nullptr || scales == nullptr || output == nullptr ||
      workspace == nullptr || (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const bool blockwise = shape->scale_block == 32;
  static bool attribute_set[2] = {false, false};
  if (!attribute_set[blockwise]) {
    const cudaError_t status = cudaFuncSetAttribute(
        blockwise ? Int4GemmKernel<true> : Int4GemmKernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) return status;
    attribute_set[blockwise] = true;
  }
  float* row_scale = static_cast<float*>(workspace);
  __half* scaled = reinterpret_cast<__half*>(static_cast<char*>(workspace) +
                                             ScaleBytes(shape->rows));
  ScaleRowsKernel<<<shape->rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation), nullptr,
      shape->input_size, norm_scale, norm_eps, norm_multiplier, row_scale,
      scaled);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  // Dense row-major weights: the up projection's channels follow the gate's.
  const uint64_t channel_bytes = static_cast<uint64_t>(shape->input_size) / 2;
  const uint64_t scale_stride =
      blockwise ? static_cast<uint64_t>(shape->input_size) / 32 : 1;
  const uint64_t up_channels = shape->gate != 0 ? shape->output_size : 0;
  const WeightAddressing addressing = {
      weights,
      weights + up_channels * channel_bytes,
      scales,
      static_cast<const uint8_t*>(scales) + up_channels * scale_stride * 2,
      channel_bytes,
      scale_stride,
      0,
      0,
      nullptr};
  const dim3 grid(shape->rows / kBlockRows, ColumnTiles(*shape));
  if (blockwise) {
    Int4GemmKernel<true><<<grid, kThreads, kSharedBytes, stream>>>(
        scaled, addressing, row_scale, shape->input_size, shape->output_size,
        shape->gate, static_cast<__nv_bfloat16*>(output));
  } else {
    Int4GemmKernel<false><<<grid, kThreads, kSharedBytes, stream>>>(
        scaled, addressing, row_scale, shape->input_size, shape->output_size,
        shape->gate, static_cast<__nv_bfloat16*>(output));
  }
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchGemmScaleRows(
    const void* activation, const int32_t* row_indices, int32_t rows,
    int32_t input_size, float* row_scale, void* scaled, cudaStream_t stream) {
  return LiteRtNvidiaLaunchGemmScaleRowsNormed(activation, row_indices, rows,
                                               input_size, nullptr, 0.0f, 1.0f,
                                               row_scale, scaled, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchGemmScaleRowsNormed(
    const void* activation, const int32_t* row_indices, int32_t rows,
    int32_t input_size, const float* norm_scale, float norm_eps,
    float norm_multiplier, float* row_scale, void* scaled,
    cudaStream_t stream) {
  if (activation == nullptr || rows <= 0 || rows % kBlockRows != 0 ||
      input_size <= 0 || input_size % kChunk != 0 || row_scale == nullptr ||
      scaled == nullptr || (norm_scale != nullptr && norm_eps < 0.0f)) {
    return cudaErrorInvalidValue;
  }
  ScaleRowsKernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation), row_indices, input_size,
      norm_scale, norm_eps, norm_multiplier, row_scale,
      static_cast<__half*>(scaled));
  return cudaGetLastError();
}

extern "C" cudaError_t LiteRtNvidiaLaunchBf16Int4GroupedGemm(
    const LiteRtNvidiaGemmShape* shape,
    const LiteRtNvidiaGroupedGemmWeights* weights, const void* scaled,
    const float* row_scale, void* output, cudaStream_t stream) {
  if (!LiteRtNvidiaSubbyteGemmSupports(shape) || weights == nullptr ||
      weights->gate == nullptr || weights->gate_scales == nullptr ||
      (shape->gate != 0 &&
       (weights->up == nullptr || weights->up_scales == nullptr)) ||
      scaled == nullptr || row_scale == nullptr || output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const bool blockwise = shape->scale_block == 32;
  static bool attribute_set[2] = {false, false};
  if (!attribute_set[blockwise]) {
    const cudaError_t status = cudaFuncSetAttribute(
        blockwise ? Int4GemmKernel<true> : Int4GemmKernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) return status;
    attribute_set[blockwise] = true;
  }
  const WeightAddressing addressing = {
      weights->gate,           weights->up,
      weights->gate_scales,    weights->up_scales,
      weights->channel_stride, weights->scale_channel_stride,
      weights->expert_stride,  weights->expert_scale_stride,
      weights->tile_experts,   weights->gate_scale_max,
      weights->up_scale_max,   weights->max_channel_stride,
      weights->max_expert_stride, weights->compact_scales};
  const dim3 grid(shape->rows / kBlockRows, ColumnTiles(*shape));
  if (blockwise) {
    Int4GemmKernel<true><<<grid, kThreads, kSharedBytes, stream>>>(
        static_cast<const __half*>(scaled), addressing, row_scale,
        shape->input_size, shape->output_size, shape->gate,
        static_cast<__nv_bfloat16*>(output));
  } else {
    Int4GemmKernel<false><<<grid, kThreads, kSharedBytes, stream>>>(
        static_cast<const __half*>(scaled), addressing, row_scale,
        shape->input_size, shape->output_size, shape->gate,
        static_cast<__nv_bfloat16*>(output));
  }
  return cudaGetLastError();
}
