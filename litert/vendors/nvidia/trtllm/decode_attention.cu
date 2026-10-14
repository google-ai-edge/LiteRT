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

#include "litert/vendors/nvidia/trtllm/decode_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

// Blocks per head: one per chunk up to kMaxSplits; beyond that the blocks
// loop over chunks, which keeps the partial workspace and the combine small
// (336 measured slower at 30k keys: the blocks are instruction-bound and the
// combine grows with the splits).
constexpr int kMaxSplits = 168;

// Keys per block, 32 or 64. Short sequences have few blocks and each
// block's serial chain (key loads, scores, softmax, values) is what the
// kernel waits for, so shorter blocks finish sooner: 32 keys measured best
// up to 2.7k keys for column caches and for many rows (E4B's 1k local
// layer 10.2 to 7.2 us, the 26B's 8-row global layer at 1k 26.6 to 18.5),
// while a row cache with few rows reads its values a warp per key batch
// and prefers the longer batch (the 26B's local layer: 10.8 us at 64,
// 14.9 at 32). Beyond kMaxSplits chunks the blocks loop over chunks and 64
// wins everywhere.
constexpr int DefaultChunkFor(int seq, bool few_rows_row_cache) {
  return seq > 16 * kMaxSplits || few_rows_row_cache ? 64 : 32;
}

// A small non-negative decimal from an environment value, -1 otherwise (no
// strtol: the CUDA objects would bind the host glibc's newer symbol).
inline int EnvDecimal(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return -1;
  int number = 0;
  for (const char* c = value; *c != '\0'; ++c) {
    if (*c < '0' || *c > '9' || number > 100000) return -1;
    number = number * 10 + (*c - '0');
  }
  return number;
}

// LITERT_NVIDIA_ATTENTION_CHUNK=32|64 pins the chunk (measurements).
inline int ChunkFor(int seq, bool few_rows_row_cache) {
  static const int pinned = [] {
    const int chunk = EnvDecimal("LITERT_NVIDIA_ATTENTION_CHUNK");
    return chunk == 32 || chunk == 64 ? chunk : 0;
  }();
  return pinned != 0 ? pinned : DefaultChunkFor(seq, few_rows_row_cache);
}
constexpr int kKeyBatch = 4;       // keys loaded together per warp (QK)
constexpr int kValueBatch = 8;     // keys loaded together per thread (PV)
constexpr int kMaxRows = 16;

__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ void FromFloat(float v, __half* out) {
  *out = __float2half(v);
}
__device__ __forceinline__ void FromFloat(float v, __nv_bfloat16* out) {
  *out = __float2bfloat16(v);
}

__device__ __forceinline__ float WarpMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Threads per block: 256 when every thread can own at least one value
// element, 128 for depth 128.
// How a kernel reads its key and value caches. FP16 caches hold rows of
// kDepth halves for keys and values alike (the in-place caches of the FP16
// models). INT8 caches (Gemma 4 E4B) hold per-tensor quantized bytes,
// dequantized as (value - zero_point) * scale; their value cache is either
// rows [S, D] or, as the model stores it, columns [D, S], which the kernel
// stages per chunk through shared memory so the reads stay coalesced.
struct Fp16Cache {
  using Element = __half;
  static constexpr bool kValueColumns = false;
  float k_scale = 1.0f, v_scale = 1.0f;
  int k_zero = 0, v_zero = 0;
  __device__ __forceinline__ float Key(__half v) const {
    return __half2float(v);
  }
  __device__ __forceinline__ float Value(__half v) const {
    return __half2float(v);
  }
};

template <bool kColumns>
struct Int8Cache {
  using Element = int8_t;
  static constexpr bool kValueColumns = kColumns;
  float k_scale, v_scale;
  int k_zero, v_zero;
  __device__ __forceinline__ float Key(int8_t v) const {
    return (static_cast<float>(v) - static_cast<float>(k_zero)) * k_scale;
  }
  __device__ __forceinline__ float Value(int8_t v) const {
    return (static_cast<float>(v) - static_cast<float>(v_zero)) * v_scale;
  }
};

template <int kDepth>
__host__ __device__ constexpr int ThreadsFor() {
  return kDepth >= 256 ? 256 : 128;
}

// Probabilities are kept key-major ([kChunk][kRowsPad]) so the value phase
// reads a key's rows with 16-byte loads; the pad keeps rows 16-byte aligned.
template <int kRowsCap>
__host__ __device__ constexpr int RowsPadFor() {
  return kRowsCap == 16 ? 20 : kRowsCap == 8 ? 8 : 4;
}

template <int kDepth, int kRowsCap, bool kValueColumns, int kChunk>
__host__ __device__ constexpr size_t SharedBytesFor() {
  // q as fp32 [rows][E][32], scores/probabilities [kChunk][kRowsPad] fp32,
  // row max and sum, and mask bytes [rows][kChunk]. Column value caches are
  // read straight into registers (below), so no staging tile.
  // Row value caches with few rows also merge the warps' value sums through
  // shared memory ([rows][depth] fp32) at the end.
  return static_cast<size_t>(kRowsCap) * kDepth * sizeof(float) +
         static_cast<size_t>(kChunk) * RowsPadFor<kRowsCap>() * sizeof(float) +
         3 * kRowsCap * sizeof(float) +
         static_cast<size_t>(kRowsCap) * kChunk * sizeof(bool) +
         (!kValueColumns && kRowsCap <= 4
              ? static_cast<size_t>(kRowsCap) * kDepth * sizeof(float)
              : 0);
}

// Column value caches ([depth][seq]): a thread's depth elements of a chunk
// are contiguous along the sequence, so it fetches them as 16-byte vectors,
// all issued together. A chunk tail or a sequence whose rows are not 16-byte
// aligned falls back to element loads.
template <typename Cache, int kThreadElems, int kValueVecs, int kVecKeys>
__device__ __forceinline__ void LoadColumnValues(
    const typename Cache::Element* __restrict__ v_head, int cache_len, int j0,
    int chunk, int tid, bool vectors,
    int4 (&v_vec)[kThreadElems][kValueVecs]) {
  using Element = typename Cache::Element;
#pragma unroll
  for (int e = 0; e < kThreadElems; ++e) {
    const Element* v_row =
        v_head + static_cast<size_t>(tid * kThreadElems + e) * cache_len + j0;
#pragma unroll
    for (int c = 0; c < kValueVecs; ++c) {
      const int jv = c * kVecKeys;
      if (vectors && jv + kVecKeys <= chunk) {
        v_vec[e][c] = reinterpret_cast<const int4*>(v_row)[c];
      } else {
        v_vec[e][c] = make_int4(0, 0, 0, 0);
        Element* tail = reinterpret_cast<Element*>(&v_vec[e][c]);
#pragma unroll
        for (int i = 0; i < kVecKeys; ++i) {
          if (jv + i < chunk) {
            tail[i] = v_row[jv + i];
          }
        }
      }
    }
  }
}

// Clang may retain loops despite the kernel's best-effort unroll hints, and
// reports the failed transformation at the kernel's declaration.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
// One block per (chunk of keys, head). Shared memory: q as [rows][E][32]
// halves (lane-major so each lane reads its own slice without bank
// conflicts), then scores/probabilities [rows][kChunk], then row max and
// sum, then the chunk's mask. Partial results go to the workspace:
// acc [heads][splits][rows][depth], then m [heads][splits][rows], then
// l [heads][splits][rows].
//
// At decode sizes the kernel is latency-bound (a 2 MB cache is microseconds
// of bandwidth, each dependent memory round trip about one), so a block
// reads a chunk in two round trips: the mask, then (if any row sees the
// chunk) all of its values and keys at once.
template <typename T, int kDepth, int kRowsCap, typename Cache, int kChunk>
__global__ void __launch_bounds__(ThreadsFor<kDepth>(), 2)
    DecodeAttentionPartial(const T* __restrict__ q,
                           const typename Cache::Element* __restrict__ k,
                           const typename Cache::Element* __restrict__ v,
                           Cache cache, const bool* __restrict__ mask,
                           int mask_rows, int rows, int seq, int cache_len,
                           const int32_t* __restrict__ window_start, float fill,
                           int splits, float* __restrict__ ws_acc,
                           float* __restrict__ ws_m, float* __restrict__ ws_l) {
  // `seq` keys are read, starting at window_start[0] (0 without a window),
  // out of caches and a mask that hold `cache_len` keys per head.
  using Element = typename Cache::Element;
  constexpr int kThreads = ThreadsFor<kDepth>();
  constexpr int kWarps = kThreads / 32;
  constexpr int kLaneElems = kDepth / 32;     // key elements per lane (QK)
  constexpr int kThreadElems = kDepth / kThreads;  // value elements per thread
  constexpr int kRowsPad = RowsPadFor<kRowsCap>();
  // Keys a warp loads together: a column cache's whole chunk share for a few
  // rows (its values are already in flight, so the keys are the one remaining
  // round trip); row caches keep the smaller batch, which measured faster.
  constexpr int kBatch =
      Cache::kValueColumns && kRowsCap <= 4 ? kChunk / kWarps : kKeyBatch;
  // Column values: keys per 16-byte vector and vectors per chunk row.
  constexpr int kVecKeys = 16 / static_cast<int>(sizeof(Element));
  constexpr int kValueVecs = kChunk / kVecKeys;
  constexpr int kValueRows = Cache::kValueColumns ? kThreadElems : 1;
  constexpr int kValueCols = Cache::kValueColumns ? kValueVecs : 1;
  // Row value caches with few rows: a lane owns kLaneVals depth elements (a
  // 16-byte vector of a key's row) and the warps split the chunk's keys,
  // kVBatch each, loaded together, so a chunk's values take one round trip
  // instead of kChunk / kValueBatch dependent ones; the warps' sums merge
  // through shared memory at the end. Column caches and many rows keep a
  // thread per depth slice over every key.
  constexpr bool kWarpValues = !Cache::kValueColumns && kRowsCap <= 4;
  constexpr int kLaneVals = kWarpValues ? kDepth / 32 : kThreadElems;
  constexpr int kVBatch = kWarpValues ? kChunk / kWarps : kValueBatch;
  extern __shared__ float smem[];
  float* q_s = smem;                                   // rows * depth
  float* p_s = q_s + kRowsCap * kDepth;                // kChunk * kRowsPad
  float* m_s = p_s + kChunk * kRowsPad;                // rows
  float* l_s = m_s + kRowsCap;                         // rows
  float* scale_s = l_s + kRowsCap;                     // rows
  bool* mask_s = reinterpret_cast<bool*>(scale_s + kRowsCap);  // rows*kChunk
  float* merge_s =
      reinterpret_cast<float*>(mask_s + kRowsCap * kChunk);  // rows * depth

  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int num_chunks = (seq + kChunk - 1) / kChunk;
  const int s0 = window_start == nullptr ? 0 : window_start[0];
  const Element* k_head =
      k + (static_cast<size_t>(head) * cache_len + s0) * kDepth;
  const Element* v_head =
      Cache::kValueColumns
          ? v + static_cast<size_t>(head) * kDepth * cache_len + s0
          : v + (static_cast<size_t>(head) * cache_len + s0) * kDepth;
  const bool value_vectors =
      Cache::kValueColumns &&
      (static_cast<size_t>(cache_len) * sizeof(Element)) % 16 == 0 &&
      (static_cast<size_t>(s0) * sizeof(Element)) % 16 == 0;
  int4 v_vec[kValueRows][kValueCols];

  // q[h][r][d] -> q_s[r][ (d / kLaneElems) + 32 * (d % kLaneElems) ]: lane
  // (d / kLaneElems) owns element (d % kLaneElems) of its slice.
  // 16-byte loads (8 elements) issued together; the fills are latency-bound
  // otherwise.
  const T* q_head = q + static_cast<size_t>(head) * rows * kDepth;
  const int q_vectors = rows * kDepth / 8;
#pragma unroll 4
  for (int i = tid; i < q_vectors; i += kThreads) {
    const int4 packed = reinterpret_cast<const int4*>(q_head)[i];
    const T* values = reinterpret_cast<const T*>(&packed);
    const int r = (i * 8) / kDepth;
    const int d0 = (i * 8) % kDepth;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int d = d0 + e;
      q_s[r * kDepth + (d / kLaneElems) + 32 * (d % kLaneElems)] =
          ToFloat(values[e]);
    }
  }
  // Running softmax state: row max/sum in shared memory, weighted values in
  // registers (thread t owns depth slice [t * kThreadElems, +kThreadElems)).
  float acc[kRowsCap][kLaneVals];
#pragma unroll
  for (int r = 0; r < kRowsCap; ++r) {
#pragma unroll
    for (int e = 0; e < kLaneVals; ++e) {
      acc[r][e] = 0.0f;
    }
  }
  if (tid < rows) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }
  __syncthreads();
  for (int c = split; c < num_chunks; c += gridDim.x) {
  const int j0 = c * kChunk;
  const int chunk = min(kChunk, seq - j0);
  // The chunk's mask bytes, loaded once instead of per key. A chunk that no
  // row can see (a sliding window hides most of a long cache) is skipped
  // before any key or value is read, so such layers cost the window, not the
  // context.
  bool visible = mask == nullptr;
  if (tid < kChunk) {
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r < rows) {
        const int jj = tid;
        bool keep = true;
        if (mask != nullptr && jj < chunk) {
          keep = mask[static_cast<size_t>(mask_rows == 1 ? 0 : r) * cache_len +
                      s0 + j0 + jj];
          visible |= keep;
        }
        mask_s[r * kChunk + jj] = keep;
      }
    }
  }
  if (!__syncthreads_or(visible ? 1 : 0)) {
    continue;
  }
  if constexpr (Cache::kValueColumns) {
    LoadColumnValues<Cache, kThreadElems, kValueVecs, kVecKeys>(
        v_head, cache_len, j0, chunk, tid, value_vectors, v_vec);
  }

  // Scores: each warp takes keys warp, warp + kWarps, ... of the chunk,
  // kBatch of them loaded together; keys past the chunk are computed and
  // dropped. A lane's slice of a key is kLaneElems * sizeof(Element) bytes:
  // 16-byte loads where it is at least 16 bytes, one 8- or 4-byte load
  // otherwise. The keys stay packed in registers and are converted per row,
  // which keeps the register count under the launch bound.
  constexpr int kLaneBytes = kLaneElems * static_cast<int>(sizeof(Element));
  constexpr int kVecs = kLaneBytes / 16;
  using Narrow = typename std::conditional<kLaneBytes == 8, int2, int>::type;
  for (int jb = warp; jb < chunk; jb += kWarps * kBatch) {
    int4 k_vec[kBatch][kVecs > 0 ? kVecs : 1];
    Narrow k_narrow[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int jj = min(jb + b * kWarps, chunk - 1);
      const Element* k_row =
          k_head + static_cast<size_t>(j0 + jj) * kDepth + lane * kLaneElems;
      if (kVecs > 0) {
#pragma unroll
        for (int c2 = 0; c2 < kVecs; ++c2) {
          k_vec[b][c2] = reinterpret_cast<const int4*>(k_row)[c2];
        }
      } else {
        k_narrow[b] = reinterpret_cast<const Narrow*>(k_row)[0];
      }
    }
    float partial[kBatch][kRowsCap];
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      float q_reg[kLaneElems];
      if (r < rows) {
        const float* q_row = q_s + r * kDepth + lane;
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e) {
          q_reg[e] = q_row[32 * e];
        }
      }
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        float dot = 0.0f;
        if (r < rows) {
          const Element* h =
              kVecs > 0 ? reinterpret_cast<const Element*>(k_vec[b])
                        : reinterpret_cast<const Element*>(&k_narrow[b]);
#pragma unroll
          for (int e = 0; e < kLaneElems; ++e) {
            dot = fmaf(q_reg[e], cache.Key(h[e]), dot);
          }
        }
        partial[b][r] = dot;
      }
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        if (r < rows) {
          const float score = WarpSum(partial[b][r]);
          if (lane == 0) {
            const int jj = jb + b * kWarps;
            if (jj < chunk) {
              p_s[jj * kRowsPad + r] = mask_s[r * kChunk + jj] ? score : fill;
            }
          }
        }
      }
    }
  }
  __syncthreads();

  // Online softmax: fold the chunk into the running row max/sum; the
  // probabilities are scaled to the new max and the accumulators by
  // scale_s[r] = exp(old max - new max). Warp w handles rows w, w+kWarps...
  for (int r = warp; r < rows; r += kWarps) {
    float m = -INFINITY;
    for (int jj = lane; jj < chunk; jj += 32) {
      m = fmaxf(m, p_s[jj * kRowsPad + r]);
    }
    m = WarpMax(m);
    const float m_old = m_s[r];
    const float m_new = fmaxf(m_old, m);
    float l = 0.0f;
    for (int jj = lane; jj < chunk; jj += 32) {
      // A chunk with every key masked to -inf contributes nothing.
      const float p =
          m_new == -INFINITY ? 0.0f : __expf(p_s[jj * kRowsPad + r] - m_new);
      p_s[jj * kRowsPad + r] = p;
      l += p;
    }
    l = WarpSum(l);
    if (lane == 0) {
      const float scale =
          m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
      m_s[r] = m_new;
      l_s[r] = l_s[r] * scale + l;
      scale_s[r] = scale;
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kRowsCap; ++r) {
    if (r < rows) {
      const float scale = scale_s[r];
#pragma unroll
      for (int e = 0; e < kLaneVals; ++e) {
        acc[r][e] *= scale;
      }
    }
  }

  // Weighted values: thread t owns depth slice [t * kThreadElems, +kThreadElems).
  // Row loops are unrolled to the compile-time cap so `acc` stays in
  // registers (a runtime-indexed register array is demoted to local memory).
  if constexpr (Cache::kValueColumns) {
    // The chunk's values are in registers; a key's probabilities come from
    // one broadcast 16-byte shared-memory read per four rows.
#pragma unroll
    for (int jj = 0; jj < kChunk; ++jj) {
      if (jj >= chunk) {
        break;
      }
      float p_reg[kRowsCap];
#pragma unroll
      for (int r4 = 0; r4 < kRowsCap / 4; ++r4) {
        const float4 packed =
            reinterpret_cast<const float4*>(p_s + jj * kRowsPad)[r4];
        p_reg[r4 * 4] = packed.x;
        p_reg[r4 * 4 + 1] = packed.y;
        p_reg[r4 * 4 + 2] = packed.z;
        p_reg[r4 * 4 + 3] = packed.w;
      }
#pragma unroll
      for (int e = 0; e < kThreadElems; ++e) {
        const Element* vals =
            reinterpret_cast<const Element*>(&v_vec[e][jj / kVecKeys]);
        const float v_f = cache.Value(vals[jj % kVecKeys]);
#pragma unroll
        for (int r = 0; r < kRowsCap; ++r) {
          if (r < rows) {
            acc[r][e] = fmaf(p_reg[r], v_f, acc[r][e]);
          }
        }
      }
    }
  } else if constexpr (kWarpValues) {
    constexpr int kLaneValBytes = kLaneVals * static_cast<int>(sizeof(Element));
    constexpr int kValVecs = kLaneValBytes / 16;
    using NarrowV =
        typename std::conditional<kLaneValBytes == 8, int2, int>::type;
    for (int jb = warp; jb < chunk; jb += kWarps * kVBatch) {
      int4 v_wide[kVBatch][kValVecs > 0 ? kValVecs : 1];
      NarrowV v_short[kVBatch];
#pragma unroll
      for (int b = 0; b < kVBatch; ++b) {
        const int jj = min(jb + b * kWarps, chunk - 1);
        const Element* v_row =
            v_head + static_cast<size_t>(j0 + jj) * kDepth + lane * kLaneVals;
        if (kValVecs > 0) {
#pragma unroll
          for (int c2 = 0; c2 < kValVecs; ++c2) {
            v_wide[b][c2] = reinterpret_cast<const int4*>(v_row)[c2];
          }
        } else {
          v_short[b] = reinterpret_cast<const NarrowV*>(v_row)[0];
        }
      }
#pragma unroll
      for (int b = 0; b < kVBatch; ++b) {
        const int jj = jb + b * kWarps;
        if (jj >= chunk) {
          break;
        }
        float p_reg[kRowsCap];
#pragma unroll
        for (int r4 = 0; r4 < kRowsCap / 4; ++r4) {
          const float4 packed =
              reinterpret_cast<const float4*>(p_s + jj * kRowsPad)[r4];
          p_reg[r4 * 4] = packed.x;
          p_reg[r4 * 4 + 1] = packed.y;
          p_reg[r4 * 4 + 2] = packed.z;
          p_reg[r4 * 4 + 3] = packed.w;
        }
        const Element* vals =
            kValVecs > 0 ? reinterpret_cast<const Element*>(v_wide[b])
                         : reinterpret_cast<const Element*>(&v_short[b]);
#pragma unroll
        for (int e = 0; e < kLaneVals; ++e) {
          const float v_f = cache.Value(vals[e]);
#pragma unroll
          for (int r = 0; r < kRowsCap; ++r) {
            if (r < rows) {
              acc[r][e] = fmaf(p_reg[r], v_f, acc[r][e]);
            }
          }
        }
      }
    }
  } else {
    // Row caches with many rows: a thread owns kThreadElems depth elements
    // and reads them for kValueBatch keys at a time, a round trip per
    // batch. The next batch loads while the current one accumulates, so
    // two round trips overlap (the 26B's global layers at 32k keys loop
    // over chunks and spent most of their time in this chain).
    Element v_cur[kValueBatch][kThreadElems];
    Element v_next[kValueBatch][kThreadElems];
    auto load_batch = [&](int jb, Element (&dst)[kValueBatch][kThreadElems]) {
#pragma unroll
      for (int b = 0; b < kValueBatch; ++b) {
        const int jj = min(jb + b, chunk - 1);
        const Element* v_row =
            v_head + static_cast<size_t>(j0 + jj) * kDepth + tid * kThreadElems;
#pragma unroll
        for (int e = 0; e < kThreadElems; ++e) {
          dst[b][e] = v_row[e];
        }
      }
    };
    load_batch(0, v_cur);
    for (int jb = 0; jb < chunk; jb += kValueBatch) {
      const bool more = jb + kValueBatch < chunk;
      if (more) {
        load_batch(jb + kValueBatch, v_next);
      }
#pragma unroll
      for (int b = 0; b < kValueBatch; ++b) {
        const int jj = jb + b;
        if (jj >= chunk) {
          break;
        }
        float v_f[kThreadElems];
#pragma unroll
        for (int e = 0; e < kThreadElems; ++e) {
          v_f[e] = cache.Value(v_cur[b][e]);
        }
        float p_reg[kRowsCap];
#pragma unroll
        for (int r4 = 0; r4 < kRowsCap / 4; ++r4) {
          const float4 packed =
              reinterpret_cast<const float4*>(p_s + jj * kRowsPad)[r4];
          p_reg[r4 * 4] = packed.x;
          p_reg[r4 * 4 + 1] = packed.y;
          p_reg[r4 * 4 + 2] = packed.z;
          p_reg[r4 * 4 + 3] = packed.w;
        }
#pragma unroll
        for (int r = 0; r < kRowsCap; ++r) {
          if (r < rows) {
#pragma unroll
            for (int e = 0; e < kThreadElems; ++e) {
              acc[r][e] = fmaf(p_reg[r], v_f[e], acc[r][e]);
            }
          }
        }
      }
      if (more) {
#pragma unroll
        for (int b = 0; b < kValueBatch; ++b) {
#pragma unroll
          for (int e = 0; e < kThreadElems; ++e) {
            v_cur[b][e] = v_next[b][e];
          }
        }
      }
    }
  }
  __syncthreads();  // p_s and mask_s are rewritten by the next chunk
  }
  const size_t part = (static_cast<size_t>(head) * splits + split) * rows;
  if constexpr (kWarpValues) {
    // The warps' value sums, merged through shared memory.
    for (int i = tid; i < rows * kDepth; i += kThreads) {
      merge_s[i] = 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r < rows) {
#pragma unroll
        for (int e = 0; e < kLaneVals; ++e) {
          atomicAdd(merge_s + r * kDepth + lane * kLaneVals + e, acc[r][e]);
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < rows * kDepth; i += kThreads) {
      ws_acc[part * kDepth + i] = merge_s[i];
    }
  } else {
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r < rows) {
        float* out_row = ws_acc + (part + r) * kDepth + tid * kThreadElems;
#pragma unroll
        for (int e = 0; e < kThreadElems; ++e) {
          out_row[e] = acc[r][e];
        }
      }
    }
  }
  if (tid < rows) {
    ws_m[part + tid] = m_s[tid];
    ws_l[part + tid] = l_s[tid];
  }
}
#pragma clang diagnostic pop

// One block per (row, head): combine the split partials. The warps take
// interleaved splits (each lane a depth slice) so many partial rows are in
// flight, then the warp sums are reduced through shared memory.
constexpr int kCombineThreads = 256;
constexpr int kCombineWarps = kCombineThreads / 32;

template <typename T, int kDepth>
__global__ void __launch_bounds__(kCombineThreads)
    DecodeAttentionCombine(const float* __restrict__ ws_acc,
                           const float* __restrict__ ws_m,
                           const float* __restrict__ ws_l, int rows,
                           int splits, T* __restrict__ out) {
  constexpr int kLaneElems = kDepth / 32;
  __shared__ float max_s[kCombineWarps];
  __shared__ float sum_s[kCombineWarps];
  __shared__ float acc_s[kCombineWarps][kDepth];
  const int r = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const size_t base = static_cast<size_t>(head) * splits * rows + r;
  // Global row max over the splits.
  float m = -INFINITY;
  for (int s = tid; s < splits; s += kCombineThreads) {
    m = fmaxf(m, ws_m[base + static_cast<size_t>(s) * rows]);
  }
  m = WarpMax(m);
  if (lane == 0) {
    max_s[warp] = m;
  }
  __syncthreads();
  m = max_s[0];
#pragma unroll
  for (int w = 1; w < kCombineWarps; ++w) {
    m = fmaxf(m, max_s[w]);
  }
  // Weighted sums; warp w takes splits w, w + kCombineWarps, ...
  float l = 0.0f;
  float acc[kLaneElems];
#pragma unroll
  for (int e = 0; e < kLaneElems; ++e) {
    acc[e] = 0.0f;
  }
  for (int s = warp; s < splits; s += kCombineWarps) {
    const size_t idx = base + static_cast<size_t>(s) * rows;
    const float ms = ws_m[idx];
    const float w = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l = fmaf(w, ws_l[idx], l);
    const float* part = ws_acc + idx * kDepth + lane * kLaneElems;
#pragma unroll
    for (int e = 0; e < kLaneElems; ++e) {
      acc[e] = fmaf(w, part[e], acc[e]);
    }
  }
  if (lane == 0) {
    sum_s[warp] = l;
  }
#pragma unroll
  for (int e = 0; e < kLaneElems; ++e) {
    acc_s[warp][lane * kLaneElems + e] = acc[e];
  }
  __syncthreads();
  l = 0.0f;
#pragma unroll
  for (int w = 0; w < kCombineWarps; ++w) {
    l += sum_s[w];
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  T* out_row = out + (static_cast<size_t>(head) * rows + r) * kDepth;
  for (int d = tid; d < kDepth; d += kCombineThreads) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kCombineWarps; ++w) {
      total += acc_s[w][d];
    }
    FromFloat(total * inv, out_row + d);
  }
}

int SplitCount(int seq, int chunk) {
  const int chunks = (seq + chunk - 1) / chunk;
  return chunks < kMaxSplits ? chunks : kMaxSplits;
}

// At least SplitCount of every sequence up to `seq` at either chunk:
// workspaces are sized for the longest sequence and reused for shorter ones.
int MaxSplitCount(int seq) {
  const int chunks = (seq + 31) / 32;
  return chunks < kMaxSplits ? chunks : kMaxSplits;
}

template <typename T, int kDepth, int kRowsCap, typename Cache, int kChunk>
cudaError_t LaunchChunk(const T* q, const typename Cache::Element* k,
                        const typename Cache::Element* v, const Cache& cache,
                        const bool* mask, int mask_rows, int heads, int rows,
                        int seq, int cache_len, const int32_t* window_start,
                        float fill, T* out, void* workspace,
                        cudaStream_t stream) {
  const int splits = SplitCount(seq, kChunk);
  float* ws_acc = static_cast<float*>(workspace);
  float* ws_m = ws_acc + static_cast<size_t>(heads) * splits * rows * kDepth;
  float* ws_l = ws_m + static_cast<size_t>(heads) * splits * rows;
  constexpr size_t kSharedBytes =
      SharedBytesFor<kDepth, kRowsCap, Cache::kValueColumns, kChunk>();
  static bool attribute_set = false;  // per instantiation
  if (!attribute_set) {
    cudaError_t status = cudaFuncSetAttribute(
        DecodeAttentionPartial<T, kDepth, kRowsCap, Cache, kChunk>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) {
      return status;
    }
    // The carveout preference, 100 unless LITERT_NVIDIA_ATTENTION_CARVEOUT
    // says otherwise (0 leaves it alone; neither measured different in the
    // 26B's decode step).
    static const int carveout = [] {
      const int value = EnvDecimal("LITERT_NVIDIA_ATTENTION_CARVEOUT");
      return value >= 0 && value <= 100 ? value : 100;
    }();
    if (carveout >= 0) {
      status = cudaFuncSetAttribute(
          DecodeAttentionPartial<T, kDepth, kRowsCap, Cache, kChunk>,
          cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
      if (status != cudaSuccess) {
        return status;
      }
    }
    attribute_set = true;
  }
  DecodeAttentionPartial<T, kDepth, kRowsCap, Cache, kChunk>
      <<<dim3(splits, heads), ThreadsFor<kDepth>(), kSharedBytes, stream>>>(
          q, k, v, cache, mask, mask_rows, rows, seq, cache_len, window_start,
          fill, splits, ws_acc, ws_m, ws_l);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  DecodeAttentionCombine<T, kDepth>
      <<<dim3(rows, heads), kCombineThreads, 0, stream>>>(ws_acc, ws_m, ws_l,
                                                          rows, splits, out);
  return cudaGetLastError();
}

template <typename T, int kDepth, int kRowsCap, typename Cache>
cudaError_t Launch(const T* q, const typename Cache::Element* k,
                   const typename Cache::Element* v, const Cache& cache,
                   const bool* mask, int mask_rows, int heads, int rows,
                   int seq, int cache_len, const int32_t* window_start,
                   float fill, T* out, void* workspace, cudaStream_t stream) {
  switch (ChunkFor(seq, !Cache::kValueColumns && kRowsCap <= 4)) {
    case 32:
      return LaunchChunk<T, kDepth, kRowsCap, Cache, 32>(
          q, k, v, cache, mask, mask_rows, heads, rows, seq, cache_len,
          window_start, fill, out, workspace, stream);
    default:
      return LaunchChunk<T, kDepth, kRowsCap, Cache, 64>(
          q, k, v, cache, mask, mask_rows, heads, rows, seq, cache_len,
          window_start, fill, out, workspace, stream);
  }
}

template <typename T, int kDepth, typename Cache>
cudaError_t LaunchRows(const T* q, const typename Cache::Element* k,
                       const typename Cache::Element* v, const Cache& cache,
                       const bool* mask, int mask_rows, int heads, int rows,
                       int seq, int cache_len, const int32_t* window_start,
                       float fill, T* out, void* workspace,
                       cudaStream_t stream) {
  if (rows <= 4) {
    return Launch<T, kDepth, 4, Cache>(q, k, v, cache, mask, mask_rows, heads,
                                       rows, seq, cache_len, window_start, fill,
                                       out, workspace, stream);
  }
  if (rows <= 8) {
    return Launch<T, kDepth, 8, Cache>(q, k, v, cache, mask, mask_rows, heads,
                                       rows, seq, cache_len, window_start, fill,
                                       out, workspace, stream);
  }
  return Launch<T, kDepth, kMaxRows, Cache>(q, k, v, cache, mask, mask_rows,
                                            heads, rows, seq, cache_len,
                                            window_start, fill, out, workspace,
                                            stream);
}

template <typename T>
cudaError_t LaunchDepth(const T* q, const __half* k, const __half* v,
                        const bool* mask, int mask_rows, int heads, int rows,
                        int seq, int depth, float fill, T* out,
                        void* workspace, cudaStream_t stream) {
  const Fp16Cache cache;
  const int cache_len = seq;
  const int32_t* window_start = nullptr;
  switch (depth) {
    case 128:
      return LaunchRows<T, 128>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    case 256:
      return LaunchRows<T, 256>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    case 512:
      return LaunchRows<T, 512>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <typename T, typename Cache>
cudaError_t LaunchInt8Depth(const T* q, const int8_t* k, const int8_t* v,
                            const Cache& cache, const bool* mask,
                            int mask_rows, int heads, int rows, int seq,
                            int cache_len, const int32_t* window_start,
                            int depth, float fill, T* out, void* workspace,
                            cudaStream_t stream) {
  switch (depth) {
    case 128:
      return LaunchRows<T, 128>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    case 256:
      return LaunchRows<T, 256>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    case 512:
      return LaunchRows<T, 512>(q, k, v, cache, mask, mask_rows, heads, rows,
                                seq, cache_len, window_start, fill, out,
                                workspace, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

// INT8 caches: depths 128, 256 (Gemma 4 E-model local layers) and 512 (their
// global layers), value rows or columns.
template <typename T>
cudaError_t LaunchInt8(const T* q, const int8_t* k, const int8_t* v,
                       const LiteRtNvidiaInt8CacheParams& params,
                       const bool* mask, int mask_rows, int heads, int rows,
                       int seq, int cache_len, const int32_t* window_start,
                       int depth, float fill, T* out, void* workspace,
                       cudaStream_t stream) {
  if (params.value_columns) {
    const Int8Cache<true> cache{params.k_scale, params.v_scale,
                                params.k_zero_point, params.v_zero_point};
    return LaunchInt8Depth(q, k, v, cache, mask, mask_rows, heads, rows, seq,
                           cache_len, window_start, depth, fill, out,
                           workspace, stream);
  }
  const Int8Cache<false> cache{params.k_scale, params.v_scale,
                               params.k_zero_point, params.v_zero_point};
  return LaunchInt8Depth(q, k, v, cache, mask, mask_rows, heads, rows, seq,
                         cache_len, window_start, depth, fill, out, workspace,
                         stream);
}

}  // namespace

extern "C" size_t LiteRtNvidiaDecodeAttentionWorkspaceBytes(int32_t heads,
                                                            int32_t rows,
                                                            int32_t seq,
                                                            int32_t depth) {
  const size_t parts =
      static_cast<size_t>(heads) * MaxSplitCount(seq) * rows;
  return parts * (static_cast<size_t>(depth) + 2) * sizeof(float);
}

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttention(
    const void* q, bool q_bf16, const void* k, const void* v,
    const bool* mask, int32_t mask_rows, int32_t heads, int32_t rows,
    int32_t seq, int32_t depth, float fill, void* out, void* workspace,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      workspace == nullptr || heads <= 0 || rows <= 0 || rows > kMaxRows ||
      seq <= 0 || (mask != nullptr && mask_rows != 1 && mask_rows != rows)) {
    return cudaErrorInvalidValue;
  }
  if (q_bf16) {
    return LaunchDepth(static_cast<const __nv_bfloat16*>(q),
                       static_cast<const __half*>(k),
                       static_cast<const __half*>(v), mask, mask_rows, heads,
                       rows, seq, depth, fill, static_cast<__nv_bfloat16*>(out),
                       workspace, stream);
  }
  return LaunchDepth(static_cast<const __half*>(q),
                     static_cast<const __half*>(k),
                     static_cast<const __half*>(v), mask, mask_rows, heads,
                     rows, seq, depth, fill, static_cast<__half*>(out),
                     workspace, stream);
}

extern "C" bool LiteRtNvidiaDecodeAttentionInt8Supports(int32_t depth) {
  return depth == 128 || depth == 256 || depth == 512;
}

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionInt8(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaInt8CacheParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t seq, int32_t depth,
    float fill, void* out, void* workspace, cudaStream_t stream) {
  if (seq <= 0) {
    return cudaErrorInvalidValue;
  }
  return LiteRtNvidiaLaunchDecodeAttentionInt8Window(
      q, q_bf16, k, v, params, mask, mask_rows, heads, rows, seq, seq, nullptr,
      depth, fill, out, workspace, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionInt8Window(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaInt8CacheParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t window, const int32_t* window_start, int32_t depth, float fill,
    void* out, void* workspace, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || params == nullptr ||
      out == nullptr || workspace == nullptr || heads <= 0 || rows <= 0 ||
      rows > kMaxRows || window <= 0 || cache_len < window ||
      (window_start == nullptr && window != cache_len) ||
      (mask != nullptr && mask_rows != 1 && mask_rows != rows) ||
      !LiteRtNvidiaDecodeAttentionInt8Supports(depth) ||
      !std::isfinite(params->k_scale) || !std::isfinite(params->v_scale)) {
    return cudaErrorInvalidValue;
  }
  if (q_bf16) {
    return LaunchInt8(static_cast<const __nv_bfloat16*>(q),
                      static_cast<const int8_t*>(k),
                      static_cast<const int8_t*>(v), *params, mask, mask_rows,
                      heads, rows, window, cache_len, window_start, depth, fill,
                      static_cast<__nv_bfloat16*>(out), workspace, stream);
  }
  return LaunchInt8(static_cast<const __half*>(q),
                    static_cast<const int8_t*>(k),
                    static_cast<const int8_t*>(v), *params, mask, mask_rows,
                    heads, rows, window, cache_len, window_start, depth, fill,
                    static_cast<__half*>(out), workspace, stream);
}
