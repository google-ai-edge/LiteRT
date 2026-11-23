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

#include "litert/vendors/nvidia/trtllm/general_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
// Query rows per CTA: each key and value a CTA loads serves all of them.
constexpr int kRowBlock = 16;
// The CTAs one attention spreads over (about two per multiprocessor of a
// desktop GPU). A constant, so that how a row's keys are split, and with it
// the order of the sums, does not depend on the device.
constexpr int kTargetCtas = 160;
constexpr int kMaxDepth = 512;
constexpr int kMaxSplits = 256;

bool IsFloatType(int32_t type) {
  return type == kLiteRtNvidiaGeneralFloat32 ||
         type == kLiteRtNvidiaGeneralFloat16 ||
         type == kLiteRtNvidiaGeneralBFloat16;
}

bool IsCacheType(int32_t type) {
  return IsFloatType(type) || type == kLiteRtNvidiaGeneralInt8 ||
         type == kLiteRtNvidiaGeneralInt4 || type == kLiteRtNvidiaGeneralUint8;
}

bool IsQuantType(int32_t type) { return IsCacheType(type) && !IsFloatType(type); }

// Keys per tile: a tile of dequantized keys in shared memory stays near
// 32 KB whatever the depth.
int KeyTile(int32_t depth) { return std::min(64, std::max(16, 8192 / depth)); }

__device__ __forceinline__ float LoadFloat(const void* data, int32_t type,
                                           int64_t index) {
  if (type == kLiteRtNvidiaGeneralFloat16) {
    return __half2float(static_cast<const __half*>(data)[index]);
  }
  if (type == kLiteRtNvidiaGeneralBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[index]);
  }
  return static_cast<const float*>(data)[index];
}

__device__ __forceinline__ void StoreFloat(float value, int32_t type, void* data,
                                           int64_t index) {
  if (type == kLiteRtNvidiaGeneralFloat16) {
    static_cast<__half*>(data)[index] = __float2half(value);
  } else if (type == kLiteRtNvidiaGeneralBFloat16) {
    static_cast<__nv_bfloat16*>(data)[index] = __float2bfloat16(value);
  } else {
    static_cast<float*>(data)[index] = value;
  }
}

// The integer a quantized cache stores for `value` (the CPU backend's
// QUANTIZE; see the header).
__device__ __forceinline__ int32_t QuantizeTo(float value, float scale,
                                              int32_t type) {
  if (type == kLiteRtNvidiaGeneralInt4) {
    const float q = roundf(value / scale);
    return static_cast<int32_t>(fminf(fmaxf(q, -8.0f), 7.0f));
  }
  const float q = rintf(value * (1.0f / scale));
  return static_cast<int32_t>(fminf(fmaxf(q, -128.0f), 127.0f));
}

// A float a graph rounds to INT8 with `scale` before the composite reads it.
__device__ __forceinline__ float ThroughInt8(float value, float scale) {
  return scale > 0.0f
             ? static_cast<float>(
                   QuantizeTo(value, scale, kLiteRtNvidiaGeneralInt8)) *
                   scale
             : value;
}

// The value a cache of `type` holds for `value`.
__device__ __forceinline__ float OnCacheGrid(float value, int32_t type,
                                             float scale) {
  if (type == kLiteRtNvidiaGeneralInt8 || type == kLiteRtNvidiaGeneralUint8 ||
      type == kLiteRtNvidiaGeneralInt4) {
    return static_cast<float>(QuantizeTo(value, scale, type)) * scale;
  }
  if (type == kLiteRtNvidiaGeneralFloat16) {
    return __half2float(__float2half(value));
  }
  if (type == kLiteRtNvidiaGeneralBFloat16) {
    return __bfloat162float(__float2bfloat16(value));
  }
  return value;
}

// Element `index` of a cache of `type` (INT4: index counts values), without
// its scale.
__device__ __forceinline__ float CacheElement(const void* data, int32_t type,
                                              int64_t index) {
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
      return static_cast<float>(static_cast<const int8_t*>(data)[index]);
    case kLiteRtNvidiaGeneralUint8:
      return static_cast<float>(
          static_cast<int32_t>(static_cast<const uint8_t*>(data)[index]) - 128);
    case kLiteRtNvidiaGeneralInt4: {
      const uint8_t byte = static_cast<const uint8_t*>(data)[index >> 1];
      const int32_t nibble = (index & 1) ? (byte >> 4) : (byte & 0xF);
      return static_cast<float>(nibble >= 8 ? nibble - 16 : nibble);
    }
    default:
      return LoadFloat(data, type, index);
  }
}

// Eight consecutive cache elements from `index` (a multiple of 8, 16-byte
// aligned data), without their scale.
__device__ __forceinline__ void CacheEight(const void* data, int32_t type,
                                           int64_t index, float (&out)[8]) {
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
    case kLiteRtNvidiaGeneralUint8: {
      const uint2 raw = *reinterpret_cast<const uint2*>(
          static_cast<const uint8_t*>(data) + index);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        out[i] = type == kLiteRtNvidiaGeneralInt8
                     ? static_cast<float>(static_cast<int8_t>(bytes[i]))
                     : static_cast<float>(static_cast<int32_t>(bytes[i]) - 128);
      }
      return;
    }
    case kLiteRtNvidiaGeneralInt4: {
      const uint32_t raw = *reinterpret_cast<const uint32_t*>(
          static_cast<const uint8_t*>(data) + (index >> 1));
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int32_t nibble = static_cast<int32_t>((raw >> (4 * i)) & 0xF);
        out[i] = static_cast<float>(nibble >= 8 ? nibble - 16 : nibble);
      }
      return;
    }
    case kLiteRtNvidiaGeneralFloat32: {
      const float4* at = reinterpret_cast<const float4*>(
          static_cast<const float*>(data) + index);
      const float4 a = at[0];
      const float4 b = at[1];
      out[0] = a.x;
      out[1] = a.y;
      out[2] = a.z;
      out[3] = a.w;
      out[4] = b.x;
      out[5] = b.y;
      out[6] = b.z;
      out[7] = b.w;
      return;
    }
    default: {
      const uint4 raw = *reinterpret_cast<const uint4*>(
          static_cast<const uint16_t*>(data) + index);
      const uint16_t* bits = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        out[i] = type == kLiteRtNvidiaGeneralFloat16
                     ? __half2float(__ushort_as_half(bits[i]))
                     : __bfloat162float(__ushort_as_bfloat16(bits[i]));
      }
      return;
    }
  }
}

struct Layout {
  int key_tile;
  int tiles;
  int splits;
  int tiles_per_split;
  int row_blocks;
};

Layout Plan(const LiteRtNvidiaGeneralAttentionParams& p) {
  Layout layout;
  const int keys = p.cache_len + p.new_len;
  layout.key_tile = KeyTile(p.depth);
  layout.tiles = (keys + layout.key_tile - 1) / layout.key_tile;
  layout.row_blocks = (p.rows + kRowBlock - 1) / kRowBlock;
  const int ctas = p.kv_heads * layout.row_blocks;
  int splits = std::max(1, (kTargetCtas + ctas - 1) / ctas);
  splits = std::min({splits, layout.tiles, kMaxSplits});
  layout.tiles_per_split = (layout.tiles + splits - 1) / splits;
  layout.splits = (layout.tiles + layout.tiles_per_split - 1) /
                  layout.tiles_per_split;
  return layout;
}

size_t SharedBytes(const LiteRtNvidiaGeneralAttentionParams& p, int key_tile) {
  const size_t stride = static_cast<size_t>(p.depth) + 4;
  return (static_cast<size_t>(kRowBlock) + key_tile) * stride * sizeof(float) +
         static_cast<size_t>(kRowBlock) * key_tile * sizeof(float) +
         3 * kRowBlock * sizeof(float);
}

// Whether each bool mask row sees some key: a row that sees none averages
// all of them (see the header).
__global__ void MaskRowsSeeKeyKernel(const bool* mask, int32_t width,
                                     int32_t* sees) {
  const int32_t row = blockIdx.x;
  const bool* at = mask + static_cast<int64_t>(row) * width;
  int any = 0;
  for (int32_t j = threadIdx.x; j < width && !any; j += blockDim.x) {
    any = at[j] ? 1 : 0;
  }
  any = __syncthreads_or(any);
  if (threadIdx.x == 0) sees[row] = any;
}

// One CTA: up to kRowBlock rows of a KV head over a range of key tiles, the
// softmax folded in tile by tile. Writes each row's running maximum, sum and
// unnormalized output for the combine.
__global__ void __launch_bounds__(kThreads)
    GeneralAttendKernel(LiteRtNvidiaGeneralAttentionParams p, int key_tile,
                        int tiles_per_split, int splits, const void* q,
                        const void* k_cache, const void* v_cache,
                        const void* new_k, const void* new_v, const void* mask,
                        const int32_t* sees, float* partial_stats,
                        float* partial_out) {
  extern __shared__ float shared[];
  const int depth = p.depth;
  const int stride = depth + 4;
  float* q_s = shared;                                  // [kRowBlock][stride]
  float* k_s = q_s + kRowBlock * stride;                // [key_tile][stride]
  float* p_s = k_s + key_tile * stride;                 // [kRowBlock][key_tile]
  float* m_s = p_s + kRowBlock * key_tile;              // [kRowBlock]
  float* l_s = m_s + kRowBlock;                         // [kRowBlock]
  float* alpha_s = l_s + kRowBlock;                     // [kRowBlock]

  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int row0 = blockIdx.z * kRowBlock;
  const int rb = min(kRowBlock, p.rows - row0);
  const int keys = p.cache_len + p.new_len;
  const int tid = threadIdx.x;
  const int32_t cache_type = p.caches.type;
  const bool quant_cache = cache_type == kLiteRtNvidiaGeneralInt8 ||
                           cache_type == kLiteRtNvidiaGeneralUint8 ||
                           cache_type == kLiteRtNvidiaGeneralInt4;
  const float k_scale = quant_cache ? p.caches.k_scale : 1.0f;
  const float v_scale = quant_cache ? p.caches.v_scale : 1.0f;

  for (int i = tid; i < rb * depth; i += kThreads) {
    const int r = i / depth;
    const int d = i % depth;
    const int64_t index =
        (static_cast<int64_t>(head) * p.rows + row0 + r) * depth + d;
    q_s[r * stride + d] =
        ThroughInt8(LoadFloat(q, p.q.type, index), p.q.int8_scale);
  }
  if (tid < kRowBlock) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }
  // Each thread keeps the output of two depth positions for every row.
  float acc[kRowBlock][2];
#pragma unroll
  for (int r = 0; r < kRowBlock; ++r) {
    acc[r][0] = 0.0f;
    acc[r][1] = 0.0f;
  }
  const int64_t k_head = static_cast<int64_t>(head) * p.cache_len * depth;
  const int64_t v_head = static_cast<int64_t>(head) * depth * p.cache_len;
  __syncthreads();

  const int first_tile = split * tiles_per_split;
  const int last_tile = min((split + 1) * tiles_per_split,
                            (keys + key_tile - 1) / key_tile);
  for (int tile = first_tile; tile < last_tile; ++tile) {
    const int j0 = tile * key_tile;
    const int kt = min(key_tile, keys - j0);
    // 1. The tile's keys as FP32 values (scaled), eight per thread and step.
    const int groups = depth / 8;
    for (int i = tid; i < kt * groups; i += kThreads) {
      const int kk = i / groups;
      const int d = (i % groups) * 8;
      const int j = j0 + kk;
      float v8[8];
      if (j < p.cache_len) {
        CacheEight(k_cache, cache_type, k_head + static_cast<int64_t>(j) * depth + d,
                   v8);
#pragma unroll
        for (int e = 0; e < 8; ++e) v8[e] *= k_scale;
      } else {
        const int64_t base =
            (static_cast<int64_t>(head) * p.new_len + (j - p.cache_len)) *
                depth + d;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          v8[e] = OnCacheGrid(
              ThroughInt8(LoadFloat(new_k, p.new_kv.type, base + e),
                          p.new_kv.int8_scale),
              cache_type, k_scale);
        }
      }
      float4* dst = reinterpret_cast<float4*>(k_s + kk * stride + d);
      dst[0] = make_float4(v8[0], v8[1], v8[2], v8[3]);
      dst[1] = make_float4(v8[4], v8[5], v8[6], v8[7]);
    }
    __syncthreads();
    // 2. Scores of every (row, key) pair of the tile.
    for (int i = tid; i < rb * kt; i += kThreads) {
      const int r = i / kt;
      const int kk = i % kt;
      const float4* qa = reinterpret_cast<const float4*>(q_s + r * stride);
      const float4* ka = reinterpret_cast<const float4*>(k_s + kk * stride);
      float dot = 0.0f;
      for (int d4 = 0; d4 < depth / 4; ++d4) {
        const float4 a = qa[d4];
        const float4 b = ka[d4];
        dot = fmaf(a.x, b.x, dot);
        dot = fmaf(a.y, b.y, dot);
        dot = fmaf(a.z, b.z, dot);
        dot = fmaf(a.w, b.w, dot);
      }
      float score = dot * p.scale;
      if (p.softcap > 0.0f) score = p.softcap * tanhf(score / p.softcap);
      const int mask_row = (row0 + r) % p.mask_rows;
      const int64_t mask_index = static_cast<int64_t>(mask_row) * keys + j0 + kk;
      if (p.mask_type == kLiteRtNvidiaGeneralMaskBool) {
        if (!sees[mask_row]) {
          score = 0.0f;
        } else if (!static_cast<const bool*>(mask)[mask_index]) {
          score = -INFINITY;
        }
      } else {
        score += static_cast<const float*>(mask)[mask_index];
      }
      p_s[r * key_tile + kk] = score;
    }
    __syncthreads();
    // 3. The softmax of each row folded in: a warp per row.
    const int warp = tid / 32;
    const int lane = tid % 32;
    for (int r = warp; r < rb; r += kThreads / 32) {
      float tile_max = -INFINITY;
      for (int kk = lane; kk < kt; kk += 32) {
        tile_max = fmaxf(tile_max, p_s[r * key_tile + kk]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
      }
      const float m_old = m_s[r];
      const float m_new = fmaxf(m_old, tile_max);
      float sum = 0.0f;
      for (int kk = lane; kk < kt; kk += 32) {
        const float s = p_s[r * key_tile + kk];
        const float e = m_new == -INFINITY ? 0.0f : __expf(s - m_new);
        p_s[r * key_tile + kk] = e;
        sum += e;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
      }
      if (lane == 0) {
        const float alpha =
            m_old == -INFINITY ? (m_new == -INFINITY ? 1.0f : 0.0f)
                               : __expf(m_old - m_new);
        alpha_s[r] = alpha;
        l_s[r] = l_s[r] * alpha + sum;
        m_s[r] = m_new;
      }
    }
    __syncthreads();
    // 4. The values: each thread its depth positions for every row.
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int d = tid + slot * kThreads;
      if (d >= depth) continue;
      float pv[kRowBlock];
#pragma unroll
      for (int r = 0; r < kRowBlock; ++r) {
        pv[r] = 0.0f;
      }
      for (int kk0 = 0; kk0 < kt; kk0 += 8) {
        float v8[8];
        const int j = j0 + kk0;
        const int64_t column = v_head + static_cast<int64_t>(d) * p.cache_len;
        // Eight values from an index that is a multiple of 8 are aligned
        // for the vector loads of every type.
        if (j + 8 <= p.cache_len && kk0 + 8 <= kt && (column + j) % 8 == 0) {
          CacheEight(v_cache, cache_type, column + j, v8);
#pragma unroll
          for (int e = 0; e < 8; ++e) v8[e] *= v_scale;
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int jj = j + e;
            if (kk0 + e >= kt) {
              v8[e] = 0.0f;
            } else if (jj < p.cache_len) {
              v8[e] = CacheElement(v_cache, cache_type, column + jj) * v_scale;
            } else {
              const int t = jj - p.cache_len;
              const int64_t index =
                  p.new_v_token_major
                      ? (static_cast<int64_t>(t) * p.kv_heads + head) * depth + d
                      : (static_cast<int64_t>(head) * p.new_len + t) * depth + d;
              v8[e] = OnCacheGrid(
                  ThroughInt8(LoadFloat(new_v, p.new_kv.type, index),
                              p.new_kv.int8_scale),
                  cache_type, v_scale);
            }
          }
        }
#pragma unroll
        for (int r = 0; r < kRowBlock; ++r) {
          if (r < rb) {
            const float* pr = p_s + r * key_tile + kk0;
            float s = pv[r];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              if (kk0 + e < kt) s = fmaf(pr[e], v8[e], s);
            }
            pv[r] = s;
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRowBlock; ++r) {
        if (r < rb) acc[r][slot] = acc[r][slot] * alpha_s[r] + pv[r];
      }
    }
    __syncthreads();
  }
  // Partial results: [kv_heads][rows][splits] (max, sum) and outputs.
  for (int r = 0; r < rb; ++r) {
    const int64_t at =
        (static_cast<int64_t>(head) * p.rows + row0 + r) * splits + split;
    if (tid == 0) {
      partial_stats[2 * at] = m_s[r];
      partial_stats[2 * at + 1] = l_s[r];
    }
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int d = tid + slot * kThreads;
      if (d < depth) partial_out[at * depth + d] = acc[r][slot];
    }
  }
}

// out[head][row] from the splits' partial results.
__global__ void GeneralCombineKernel(LiteRtNvidiaGeneralAttentionParams p,
                                     int splits, const float* partial_stats,
                                     const float* partial_out, void* out) {
  const int64_t row = blockIdx.x;  // head * rows + r
  const float* stats = partial_stats + 2 * row * splits;
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, stats[2 * s]);
  float total = 0.0f;
  for (int s = 0; s < splits; ++s) {
    if (stats[2 * s] != -INFINITY) {
      total += stats[2 * s + 1] * __expf(stats[2 * s] - m);
    }
  }
  const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
  for (int d = threadIdx.x; d < p.depth; d += blockDim.x) {
    float value = 0.0f;
    for (int s = 0; s < splits; ++s) {
      if (stats[2 * s] != -INFINITY) {
        value += partial_out[(row * splits + s) * p.depth + d] *
                 __expf(stats[2 * s] - m);
      }
    }
    StoreFloat(value * inverse, p.q.type, out, row * p.depth + d);
  }
}

// ---------------------------------------------------------------------------
// Cache writes.
// ---------------------------------------------------------------------------

struct WriteTarget {
  int32_t start;      // slice: the clamped start
  int32_t p0;         // ring
  int32_t first;      // ring: the first token written
  int32_t valid;      // ring: the tokens after the last one written
};

__device__ __forceinline__ WriteTarget Target(
    const LiteRtNvidiaGeneralWriteParams& p, const int32_t* start,
    const int32_t* ring) {
  WriteTarget target{0, 0, 0, 0};
  if (p.mode == kLiteRtNvidiaGeneralWriteSlice) {
    target.start = min(max(*start, 0), p.cache_len - p.tokens);
  } else {
    target.p0 = ring[0];
    const int32_t n = target.p0 < 0 ? 0 : min(max(ring[1] - ring[0], 0), p.tokens);
    target.valid = n;
    target.first = max(0, n - p.cache_len);
  }
  return target;
}

// The cache slot token t goes to, or -1.
__device__ __forceinline__ int32_t SlotOf(const LiteRtNvidiaGeneralWriteParams& p,
                                          const WriteTarget& target, int32_t t) {
  if (p.mode == kLiteRtNvidiaGeneralWriteSlice) return target.start + t;
  if (t < target.first || t >= target.valid) return -1;
  return static_cast<int32_t>(
      (static_cast<int64_t>(target.p0) + t) % p.cache_len);
}

// The token that goes to `slot`, or -1.
__device__ __forceinline__ int32_t TokenOf(const LiteRtNvidiaGeneralWriteParams& p,
                                           const WriteTarget& target,
                                           int32_t slot) {
  if (p.mode == kLiteRtNvidiaGeneralWriteSlice) {
    const int32_t t = slot - target.start;
    return t >= 0 && t < p.tokens ? t : -1;
  }
  if (target.valid <= target.first) return -1;
  const int64_t offset =
      ((static_cast<int64_t>(slot) - target.p0 - target.first) % p.cache_len +
       p.cache_len) %
      p.cache_len;
  const int64_t t = target.first + offset;
  return t < target.valid ? static_cast<int32_t>(t) : -1;
}

__device__ __forceinline__ float SourceValue(
    const LiteRtNvidiaGeneralWriteParams& p, const void* src, bool token_major,
    int32_t head, int32_t t, int32_t d) {
  const int64_t index =
      token_major ? (static_cast<int64_t>(t) * p.heads + head) * p.depth + d
                  : (static_cast<int64_t>(head) * p.tokens + t) * p.depth + d;
  return ThroughInt8(LoadFloat(src, p.src.type, index), p.src.int8_scale);
}

__device__ __forceinline__ void StoreCacheValue(void* cache, int32_t type,
                                                float scale, int64_t index,
                                                float value) {
  switch (type) {
    case kLiteRtNvidiaGeneralInt8:
      static_cast<int8_t*>(cache)[index] =
          static_cast<int8_t>(QuantizeTo(value, scale, type));
      return;
    case kLiteRtNvidiaGeneralUint8:
      static_cast<uint8_t*>(cache)[index] =
          static_cast<uint8_t>(QuantizeTo(value, scale, type) + 128);
      return;
    default:
      StoreFloat(value, type, cache, index);
  }
}

__device__ __forceinline__ uint8_t Nibble(float value, float scale) {
  return static_cast<uint8_t>(
      QuantizeTo(value, scale, kLiteRtNvidiaGeneralInt4) & 0xF);
}

// The keys: token t's row of head h, a thread per element (INT4: per byte).
__global__ void GeneralWriteKeysKernel(LiteRtNvidiaGeneralWriteParams p,
                                       void* cache, const void* src,
                                       const int32_t* start,
                                       const int32_t* ring) {
  const bool packed = p.caches.type == kLiteRtNvidiaGeneralInt4;
  const int32_t per_row = packed ? p.depth / 2 : p.depth;
  const int64_t total = static_cast<int64_t>(p.tokens) * p.heads * per_row;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int32_t e = static_cast<int32_t>(i % per_row);
  const int32_t h = static_cast<int32_t>((i / per_row) % p.heads);
  const int32_t t = static_cast<int32_t>(i / per_row / p.heads);
  const WriteTarget target = Target(p, start, ring);
  const int32_t slot = SlotOf(p, target, t);
  if (slot < 0) return;
  const int64_t row = (static_cast<int64_t>(h) * p.cache_len + slot) * p.depth;
  if (packed) {
    const float scale = p.caches.k_scale;
    const uint8_t low = Nibble(SourceValue(p, src, false, h, t, 2 * e), scale);
    const uint8_t high =
        Nibble(SourceValue(p, src, false, h, t, 2 * e + 1), scale);
    static_cast<uint8_t*>(cache)[row / 2 + e] =
        static_cast<uint8_t>(low | (high << 4));
    return;
  }
  StoreCacheValue(cache, p.caches.type, p.caches.k_scale, row + e,
                  SourceValue(p, src, false, h, t, e));
}

// The values: token t's column of head h, a thread per element. Two slots
// share an INT4 byte: the token of its even slot (or, when none goes there,
// of its odd one) writes both nibbles.
__global__ void GeneralWriteValuesKernel(LiteRtNvidiaGeneralWriteParams p,
                                         void* cache, const void* src,
                                         const int32_t* start,
                                         const int32_t* ring) {
  const int64_t total = static_cast<int64_t>(p.tokens) * p.heads * p.depth;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int32_t d = static_cast<int32_t>(i % p.depth);
  const int32_t h = static_cast<int32_t>((i / p.depth) % p.heads);
  const int32_t t = static_cast<int32_t>(i / p.depth / p.heads);
  const WriteTarget target = Target(p, start, ring);
  const int32_t slot = SlotOf(p, target, t);
  if (slot < 0) return;
  const bool token_major = p.src_v_token_major != 0;
  const int64_t column = (static_cast<int64_t>(h) * p.depth + d) * p.cache_len;
  if (p.caches.type != kLiteRtNvidiaGeneralInt4) {
    StoreCacheValue(cache, p.caches.type, p.caches.v_scale, column + slot,
                    SourceValue(p, src, token_major, h, t, d));
    return;
  }
  const int32_t partner = TokenOf(p, target, slot ^ 1);
  if ((slot & 1) && partner >= 0) return;
  const float scale = p.caches.v_scale;
  uint8_t* byte = static_cast<uint8_t*>(cache) + (column + slot) / 2;
  const uint8_t mine = Nibble(SourceValue(p, src, token_major, h, t, d), scale);
  const uint8_t other =
      partner >= 0
          ? Nibble(SourceValue(p, src, token_major, h, partner, d), scale)
          : static_cast<uint8_t>((*byte >> ((slot & 1) ? 0 : 4)) & 0xF);
  *byte = (slot & 1) ? static_cast<uint8_t>(other | (mine << 4))
                     : static_cast<uint8_t>(mine | (other << 4));
}

}  // namespace

extern "C" bool LiteRtNvidiaGeneralAttentionSupports(
    const LiteRtNvidiaGeneralAttentionParams* params) {
  if (params == nullptr) return false;
  const auto& p = *params;
  const int64_t keys = static_cast<int64_t>(p.cache_len) + p.new_len;
  return p.kv_heads >= 1 && p.rows >= 1 && p.depth >= 8 && p.depth % 8 == 0 &&
         p.depth <= kMaxDepth && p.cache_len >= 0 && p.new_len >= 0 &&
         keys >= 1 && keys <= (1 << 22) && p.mask_rows >= 1 &&
         p.rows % p.mask_rows == 0 &&
         (p.mask_type == kLiteRtNvidiaGeneralMaskBool ||
          p.mask_type == kLiteRtNvidiaGeneralMaskFloat) &&
         IsFloatType(p.q.type) && p.q.int8_scale >= 0.0f &&
         IsCacheType(p.caches.type) &&
         (!IsQuantType(p.caches.type) ||
          (p.caches.k_scale > 0.0f && p.caches.v_scale > 0.0f)) &&
         (p.caches.type != kLiteRtNvidiaGeneralInt4 || p.cache_len % 2 == 0) &&
         (p.new_len == 0 ||
          (IsFloatType(p.new_kv.type) && p.new_kv.int8_scale >= 0.0f)) &&
         std::isfinite(p.scale) && p.softcap >= 0.0f;
}

extern "C" size_t LiteRtNvidiaGeneralAttentionWorkspaceBytes(
    const LiteRtNvidiaGeneralAttentionParams* params) {
  if (!LiteRtNvidiaGeneralAttentionSupports(params)) return 0;
  const Layout layout = Plan(*params);
  const size_t rows = static_cast<size_t>(params->kv_heads) * params->rows;
  const size_t sees = static_cast<size_t>(params->mask_rows) * sizeof(int32_t);
  const size_t stats = rows * layout.splits * 2 * sizeof(float);
  const size_t outs = rows * layout.splits * params->depth * sizeof(float);
  return ((sees + 255) / 256) * 256 + ((stats + 255) / 256) * 256 + outs;
}

extern "C" cudaError_t LiteRtNvidiaLaunchGeneralAttention(
    const LiteRtNvidiaGeneralAttentionParams* params, const void* q,
    const void* k_cache, const void* v_cache, const void* new_k,
    const void* new_v, const void* mask, void* out, void* workspace,
    cudaStream_t stream) {
  if (!LiteRtNvidiaGeneralAttentionSupports(params) || q == nullptr ||
      out == nullptr || mask == nullptr || workspace == nullptr ||
      (params->cache_len > 0 && (k_cache == nullptr || v_cache == nullptr)) ||
      (params->new_len > 0 && (new_k == nullptr || new_v == nullptr))) {
    return cudaErrorInvalidValue;
  }
  const auto& p = *params;
  const Layout layout = Plan(p);
  const size_t rows = static_cast<size_t>(p.kv_heads) * p.rows;
  char* base = static_cast<char*>(workspace);
  int32_t* sees = reinterpret_cast<int32_t*>(base);
  base += ((p.mask_rows * sizeof(int32_t) + 255) / 256) * 256;
  float* stats = reinterpret_cast<float*>(base);
  base += ((rows * layout.splits * 2 * sizeof(float) + 255) / 256) * 256;
  float* outs = reinterpret_cast<float*>(base);

  if (p.mask_type == kLiteRtNvidiaGeneralMaskBool) {
    MaskRowsSeeKeyKernel<<<p.mask_rows, kThreads, 0, stream>>>(
        static_cast<const bool*>(mask), p.cache_len + p.new_len, sees);
  }
  const size_t shared = SharedBytes(p, layout.key_tile);
  static bool configured = false;
  if (!configured) {
    const cudaError_t status = cudaFuncSetAttribute(
        GeneralAttendKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SharedBytes(
            LiteRtNvidiaGeneralAttentionParams{1, 1, kMaxDepth},
            KeyTile(kMaxDepth))));
    if (status != cudaSuccess) return status;
    configured = true;
  }
  const dim3 grid(layout.splits, p.kv_heads, layout.row_blocks);
  GeneralAttendKernel<<<grid, kThreads, shared, stream>>>(
      p, layout.key_tile, layout.tiles_per_split, layout.splits, q, k_cache,
      v_cache, new_k, new_v, mask, sees, stats, outs);
  GeneralCombineKernel<<<static_cast<unsigned>(rows), 128, 0, stream>>>(
      p, layout.splits, stats, outs, out);
  return cudaGetLastError();
}

extern "C" bool LiteRtNvidiaGeneralCacheWriteSupports(
    const LiteRtNvidiaGeneralWriteParams* params) {
  if (params == nullptr) return false;
  const auto& p = *params;
  const bool packed = p.caches.type == kLiteRtNvidiaGeneralInt4;
  return p.heads >= 1 && p.cache_len >= 1 && p.depth >= 1 && p.tokens >= 1 &&
         (p.mode == kLiteRtNvidiaGeneralWriteRing || p.tokens <= p.cache_len) &&
         (p.mode == kLiteRtNvidiaGeneralWriteSlice ||
          p.mode == kLiteRtNvidiaGeneralWriteRing) &&
         IsCacheType(p.caches.type) &&
         (!IsQuantType(p.caches.type) ||
          (p.caches.k_scale > 0.0f && p.caches.v_scale > 0.0f)) &&
         (!packed || (p.depth % 2 == 0 && p.cache_len % 2 == 0)) &&
         IsFloatType(p.src.type) && p.src.int8_scale >= 0.0f;
}

extern "C" cudaError_t LiteRtNvidiaLaunchGeneralCacheWrite(
    const LiteRtNvidiaGeneralWriteParams* params, void* k_cache, void* v_cache,
    const void* src_k, const void* src_v, const int32_t* k_start,
    const int32_t* v_start, const int32_t* ring, cudaStream_t stream) {
  if (!LiteRtNvidiaGeneralCacheWriteSupports(params) || k_cache == nullptr ||
      v_cache == nullptr || src_k == nullptr || src_v == nullptr ||
      (params->mode == kLiteRtNvidiaGeneralWriteSlice
           ? (k_start == nullptr || v_start == nullptr)
           : ring == nullptr)) {
    return cudaErrorInvalidValue;
  }
  const auto& p = *params;
  const int64_t key_threads =
      static_cast<int64_t>(p.tokens) * p.heads *
      (p.caches.type == kLiteRtNvidiaGeneralInt4 ? p.depth / 2 : p.depth);
  const int64_t value_threads = static_cast<int64_t>(p.tokens) * p.heads * p.depth;
  GeneralWriteKeysKernel<<<static_cast<unsigned>((key_threads + kThreads - 1) /
                                                 kThreads),
                           kThreads, 0, stream>>>(p, k_cache, src_k, k_start,
                                                  ring);
  GeneralWriteValuesKernel<<<static_cast<unsigned>(
                                 (value_threads + kThreads - 1) / kThreads),
                             kThreads, 0, stream>>>(p, v_cache, src_v, v_start,
                                                    ring);
  return cudaGetLastError();
}
