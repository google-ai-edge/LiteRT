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

#include <atomic>
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
  // shared memory at the end: half the warps' partials ([warps / 2][rows]
  // [depth] fp32), the other half added in, each slot by one warp, then the
  // slots summed in order (a fixed order: the same bits every run).
  return static_cast<size_t>(kRowsCap) * kDepth * sizeof(float) +
         static_cast<size_t>(kChunk) * RowsPadFor<kRowsCap>() * sizeof(float) +
         3 * kRowsCap * sizeof(float) +
         static_cast<size_t>(kRowsCap) * kChunk * sizeof(bool) +
         (!kValueColumns && kRowsCap <= 4
              ? static_cast<size_t>(ThreadsFor<kDepth>() / 32 / 2) * kRowsCap *
                    kDepth * sizeof(float)
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
  // Many rows (8, 16) of an INT8 cache take two keys per warp per round trip:
  // their keys are converted once per key, not once per row (kManyRows
  // below), and the converted keys and the per-row partial sums of a wider
  // batch would not fit the register budget next to the accumulators. The
  // E4B verify step's 16-row global attention measured 44 to 34 us at 2k keys
  // and 597 to 320 us at 30k, E2B's 8-row global layer at 30k 108 to 90 us.
  // FP16 keys convert in one instruction and keep the wider batch: the 26B's
  // 8-row global layers measured 6% slower at 8k keys with the narrow one.
  constexpr bool kManyRows = kRowsCap >= 8 && sizeof(Element) == 1;
  constexpr int kBatch = Cache::kValueColumns && kRowsCap <= 4 ? kChunk / kWarps
                         : kManyRows                            ? 2
                                                                : kKeyBatch;
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
  float* merge_s = reinterpret_cast<float*>(
      mask_s + kRowsCap * kChunk);  // [warps / 2][rows][depth]

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
  // A column cache's values for the whole chunk are issued now, before the
  // keys, so both round trips overlap; with many rows they wait until the
  // scores are done instead (the registers hold the converted keys and the
  // partial sums meanwhile).
  if constexpr (Cache::kValueColumns && !kManyRows) {
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
    // Many rows convert each key once here (an INT8 element costs three
    // instructions to convert, its FMA one; the row loop used to redo it per
    // row). Few rows keep the keys packed and convert in the row loop, which
    // keeps their register count under the launch bound.
    float k_f[kManyRows ? kBatch : 1][kManyRows ? kLaneElems : 1];
    if constexpr (kManyRows) {
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const Element* h =
            kVecs > 0 ? reinterpret_cast<const Element*>(k_vec[b])
                      : reinterpret_cast<const Element*>(&k_narrow[b]);
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e) {
          k_f[b][e] = cache.Key(h[e]);
        }
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
          if constexpr (kManyRows) {
#pragma unroll
            for (int e = 0; e < kLaneElems; ++e) {
              dot = fmaf(q_reg[e], k_f[b][e], dot);
            }
          } else {
            const Element* h =
                kVecs > 0 ? reinterpret_cast<const Element*>(k_vec[b])
                          : reinterpret_cast<const Element*>(&k_narrow[b]);
#pragma unroll
            for (int e = 0; e < kLaneElems; ++e) {
              dot = fmaf(q_reg[e], cache.Key(h[e]), dot);
            }
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
  if constexpr (Cache::kValueColumns && kManyRows) {
    LoadColumnValues<Cache, kThreadElems, kValueVecs, kVecKeys>(
        v_head, cache_len, j0, chunk, tid, value_vectors, v_vec);
  }
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
    // The warps' value sums, merged through shared memory in a fixed order
    // (shared-memory atomics would add them in whatever order the warps
    // arrive, and the 26B's logits then differed from run to run): the first
    // half of the warps store their partials, the second half add theirs
    // into the slot of their partner warp, and the slots are summed in
    // order.
    constexpr int kHalf = kWarps / 2;
    static_assert(kWarps % 2 == 0, "the merge pairs the warps");
    const int slot = warp < kHalf ? warp : warp - kHalf;
    float* mine = merge_s + static_cast<size_t>(slot) * kRowsCap * kDepth;
    if (warp < kHalf) {
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        if (r < rows) {
#pragma unroll
          for (int e = 0; e < kLaneVals; ++e) {
            mine[r * kDepth + lane * kLaneVals + e] = acc[r][e];
          }
        }
      }
    }
    __syncthreads();
    if (warp >= kHalf) {
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        if (r < rows) {
#pragma unroll
          for (int e = 0; e < kLaneVals; ++e) {
            mine[r * kDepth + lane * kLaneVals + e] += acc[r][e];
          }
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < rows * kDepth; i += kThreads) {
      const int r = i / kDepth;
      const int d = i - r * kDepth;
      float sum = 0.0f;
#pragma unroll
      for (int s = 0; s < kHalf; ++s) {
        sum += merge_s[(static_cast<size_t>(s) * kRowsCap + r) * kDepth + d];
      }
      ws_acc[part * kDepth + i] = sum;
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

// Quantized queries over quantized caches (LiteRtNvidiaQuantAttentionParams):
// the kernels below compute the decomposition of odml.sdpa_transposed with
// the CPU backend's operations; the caches are read as stored, INT8 or INT4
// two per byte, in chunks of kQuantChunk keys.
constexpr int kQuantDepth = 256;
constexpr int kQuantChunk = 64;
constexpr int kQuantThreads = 256;

// QUANTIZE of a FLOAT16 tensor to INT8 as XNNPACK converts it
// (xnn_f16_qs8_vcvt, quant_cache.cu has the account): x times the reciprocal
// of the scale rounded to FLOAT16, rounded half to even, saturated.
__device__ __forceinline__ float QuantizeQuery(float x, float scale) {
  const float reciprocal = __half2float(__float2half(1.0f / scale));
  return fminf(fmaxf(rintf(x * reciprocal), -128.0f), 127.0f);
}

// A float through FP16, where the decomposition holds a tensor in FP16.
__device__ __forceinline__ float ThroughHalf(float x) {
  return __half2float(__float2half(x));
}

// Value `index` of words of kBits-bit signed values, the first in the low
// bits of the first word.
template <int kBits>
__device__ __forceinline__ int PackedValue(const uint32_t* words, int index) {
  return static_cast<int>(words[index * kBits / 32]
                          << (32 - kBits - (index * kBits) % 32)) >>
         (32 - kBits);
}

// One token's key and value rows [heads, depth] that a launch writes into
// the caches before it attends (LiteRtNvidiaQuantCacheToken); no token when
// k is null.
template <typename T>
struct QuantToken {
  const T* k = nullptr;
  const T* v = nullptr;
  const int32_t* k_position = nullptr;
  const int32_t* v_position = nullptr;
};

// QUANTIZE to a cache value, as QuantCacheWriteKernel (quant_cache.cu)
// rounds it: XNNPACK's conversion to INT8, TFLite's division to INT4.
template <int kBits>
__device__ __forceinline__ int QuantizeCacheValue(float x, float scale) {
  if constexpr (kBits == 8) {
    return static_cast<int>(QuantizeQuery(x, scale));
  } else {
    return static_cast<int>(fminf(fmaxf(roundf(x / scale), -8.0f), 7.0f));
  }
}

// The token's key row and value column of `head` into the caches: thread
// `tid` writes its byte of the key row and its depth element's value. Every
// block that reads the head's caches writes them first, the same bytes, so
// what a block reads after its own barrier is the token whatever the other
// blocks have done. An INT4 value shares its byte with a neighbour of its
// own row (cache_len is even), which no block changes.
template <typename T, int kBits>
__device__ __forceinline__ void WriteQuantToken(
    const QuantToken<T>& token, const LiteRtNvidiaQuantAttentionParams& params,
    int8_t* k, int8_t* v, int head, int cache_len, int tid) {
  constexpr int kKeyBytes = kQuantDepth * kBits / 8;
  const int k_pos = min(max(token.k_position[0], 0), cache_len - 1);
  const int v_pos = min(max(token.v_position[0], 0), cache_len - 1);
  const size_t row = static_cast<size_t>(head) * kQuantDepth;
  if (tid < kKeyBytes) {
    int8_t* to =
        k + (static_cast<size_t>(head) * cache_len + k_pos) * kKeyBytes + tid;
    if constexpr (kBits == 8) {
      *to = static_cast<int8_t>(
          QuantizeCacheValue<8>(ToFloat(token.k[row + tid]), params.k_scale));
    } else {
      const int lo = QuantizeCacheValue<4>(ToFloat(token.k[row + 2 * tid]),
                                           params.k_scale);
      const int hi = QuantizeCacheValue<4>(ToFloat(token.k[row + 2 * tid + 1]),
                                           params.k_scale);
      *to = static_cast<int8_t>((lo & 0xf) | ((hi & 0xf) << 4));
    }
  }
  const int value =
      QuantizeCacheValue<kBits>(ToFloat(token.v[row + tid]), params.v_scale);
  const size_t at = (row + tid) * cache_len + v_pos;
  if constexpr (kBits == 8) {
    v[at] = static_cast<int8_t>(value);
  } else {
    const int old = v[at >> 1];
    v[at >> 1] = static_cast<int8_t>((at & 1) != 0
                                         ? (old & 0x0f) | ((value & 0xf) << 4)
                                         : (old & 0xf0) | (value & 0xf));
  }
}

// The query in the FLOAT16 tensor of the decomposition: its QUANTIZE, then its
// DEQUANTIZE with the scale rounded to FLOAT16 (xnn_qs8_f16_vcvt).
__device__ __forceinline__ float DequantizedQuery(float x, float scale) {
  return ThroughHalf(QuantizeQuery(x, scale) *
                     __half2float(__float2half(scale)));
}

// As with DecodeAttentionPartial, Clang may retain loops despite these
// kernels' best-effort unroll hints.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
// The fast kernels: an online softmax in fp32 over chunks of the keys. They
// take the decomposition's operands (the query as the CPU quantizes and
// dequantizes it, the caches as stored) and cast the scores to FLOAT16 as it
// does, but weigh the keys with exp() in fp32 where the graph's SOFTMAX is
// XNNPACK's on a FLOAT16 tensor: every weight is a little different, the
// keys that softmax drops (9.7 below the largest score) keep theirs, and the
// sums are added in another order. The result differs from the CPU's by a
// unit of FLOAT16 in many values.
//
// One block per (split, head); a thread owns one depth element of the
// values. Partial results go to the workspace as DecodeAttentionPartial
// writes them.
template <typename T, int kRowsCap, int kBits>
__global__ void __launch_bounds__(kQuantThreads)
    QuantDecodeAttentionPartial(const T* __restrict__ q, int8_t* k, int8_t* v,
                                LiteRtNvidiaQuantAttentionParams params,
                                const bool* __restrict__ mask, int mask_rows,
                                int rows, int cache_len, float fill,
                                int splits, QuantToken<T> token,
                                float* __restrict__ ws_acc,
                                float* __restrict__ ws_m,
                                float* __restrict__ ws_l) {
  // The dequantized query, [rows][depth / 4][4]: the four threads of a key
  // read dims e, e + 64, e + 128 and e + 192 of a row from one word of every
  // bank. Then the chunk's scores and probabilities, key-major, the running
  // row maximum and sum with the chunk's rescale, and the chunk's mask.
  __shared__ float q_s[kRowsCap * kQuantDepth];
  __shared__ __align__(16) float p_s[kQuantChunk * kRowsCap];
  __shared__ float m_s[kRowsCap];
  __shared__ float l_s[kRowsCap];
  __shared__ float scale_s[kRowsCap];
  __shared__ bool vis_s[kRowsCap * kQuantChunk];
  constexpr int kWarps = kQuantThreads / 32;
  constexpr int kKeyBytes = kQuantDepth * kBits / 8;
  constexpr int kPartDims = kQuantDepth / 4;
  constexpr int kVectorValues = 128 / kBits;  // values of a 16-byte vector

  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int num_chunks = (cache_len + kQuantChunk - 1) / kQuantChunk;
  const int8_t* k_head = k + static_cast<size_t>(head) * cache_len * kKeyBytes;
  // The thread's value row starts at this value of the cache; its values of
  // a chunk are whole 16-byte vectors when every row of the cache is.
  const size_t v_first =
      (static_cast<size_t>(head) * kQuantDepth + tid) * cache_len;
  const bool value_vectors = cache_len % kVectorValues == 0;

  if (token.k != nullptr) {
    // Visible to the block's reads after the barrier below.
    WriteQuantToken<T, kBits>(token, params, k, v, head, cache_len, tid);
  }
  const T* q_head = q + static_cast<size_t>(head) * rows * kQuantDepth;
  for (int i = tid; i < rows * kQuantDepth; i += kQuantThreads) {
    const int d = i % kQuantDepth;
    q_s[(i - d) + (d % kPartDims) * 4 + d / kPartDims] = DequantizedQuery(ToFloat(q_head[i]), params.q_scale);
  }
  float acc[kRowsCap];
#pragma unroll
  for (int r = 0; r < kRowsCap; ++r) {
    acc[r] = 0.0f;
  }
  if (tid < rows) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }
  __syncthreads();
  for (int c = split; c < num_chunks; c += gridDim.x) {
    const int j0 = c * kQuantChunk;
    const int chunk = min(kQuantChunk, cache_len - j0);
    // A chunk no row sees is skipped before any key or value is read.
    bool visible = false;
    if (tid < kQuantChunk) {
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        if (r < rows) {
          const bool keep =
              tid < chunk &&
              (mask == nullptr ||
               mask[static_cast<size_t>(r % mask_rows) * cache_len + j0 +
                    tid]);
          vis_s[r * kQuantChunk + tid] = keep;
          visible |= keep;
        }
      }
    }
    if (!__syncthreads_or(visible ? 1 : 0)) {
      continue;
    }

    // Scores: four threads share a key, a quarter of its dims each, and add
    // their sums up in a fixed order.
    {
      const int key = tid >> 2;
      const int part = tid & 3;
      const int4* k_part = reinterpret_cast<const int4*>(
          k_head + static_cast<size_t>(j0 + min(key, chunk - 1)) * kKeyBytes +
          part * (kKeyBytes / 4));
      int4 loaded[kPartDims / kVectorValues];
#pragma unroll
      for (int i = 0; i < kPartDims / kVectorValues; ++i) {
        loaded[i] = k_part[i];
      }
      float dot[kRowsCap];
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        dot[r] = 0.0f;
      }
#pragma unroll
      for (int e = 0; e < kPartDims; ++e) {
        const float k_f = static_cast<float>(PackedValue<kBits>(
            reinterpret_cast<const uint32_t*>(loaded), e));
#pragma unroll
        for (int r = 0; r < kRowsCap; ++r) {
          if (r < rows) {
            dot[r] = fmaf(q_s[r * kQuantDepth + e * 4 + part], k_f, dot[r]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsCap; ++r) {
        if (r < rows) {
          float score = dot[r];
          score += __shfl_xor_sync(0xffffffffu, score, 1);
          score += __shfl_xor_sync(0xffffffffu, score, 2);
          if (part == 0 && key < chunk) {
            // The decomposition casts its scores to FP16 before the mask.
            p_s[key * kRowsCap + r] = vis_s[r * kQuantChunk + key]
                                          ? ThroughHalf(score * params.k_scale)
                                          : fill;
          }
        }
      }
    }
    __syncthreads();

    // Online softmax, as DecodeAttentionPartial folds a chunk in.
    for (int r = warp; r < rows; r += kWarps) {
      float m = -INFINITY;
      for (int jj = lane; jj < chunk; jj += 32) {
        m = fmaxf(m, p_s[jj * kRowsCap + r]);
      }
      m = WarpMax(m);
      const float m_old = m_s[r];
      const float m_new = fmaxf(m_old, m);
      float l = 0.0f;
      for (int jj = lane; jj < chunk; jj += 32) {
        const float p =
            m_new == -INFINITY ? 0.0f : __expf(p_s[jj * kRowsCap + r] - m_new);
        p_s[jj * kRowsCap + r] = p;
        l += p;
      }
      l = WarpSum(l);
      if (lane == 0) {
        const float scale = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
        m_s[r] = m_new;
        l_s[r] = l_s[r] * scale + l;
        scale_s[r] = scale;
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r < rows) {
        acc[r] *= scale_s[r];
      }
    }

    // Weighted values: the thread's depth element of every key of the chunk,
    // loaded as stored (16 INT8 or 32 INT4 values per vector), or value by
    // value when the rows of the cache are not whole vectors.
    int4 values[kQuantChunk / kVectorValues];
    if (value_vectors) {
      const int4* v_row = reinterpret_cast<const int4*>(
          v + (v_first + j0) / (8 / kBits));
#pragma unroll
      for (int i = 0; i < kQuantChunk / kVectorValues; ++i) {
        if (i * kVectorValues < chunk) {
          values[i] = v_row[i];
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < kQuantChunk; ++jj) {
      if (jj >= chunk) {
        break;
      }
      int value;
      if (value_vectors) {
        value = PackedValue<kBits>(reinterpret_cast<const uint32_t*>(values),
                                   jj);
      } else {
        const size_t at = v_first + j0 + jj;
        if constexpr (kBits == 8) {
          value = v[at];
        } else {
          const uint32_t packed = static_cast<uint8_t>(v[at >> 1]);
          value = static_cast<int>(packed << ((at & 1) != 0 ? 24 : 28)) >> 28;
        }
      }
      const float v_f = static_cast<float>(value);
#pragma unroll
      for (int r4 = 0; r4 < kRowsCap / 4; ++r4) {
        const float4 p =
            reinterpret_cast<const float4*>(p_s + jj * kRowsCap)[r4];
        acc[r4 * 4] = fmaf(p.x, v_f, acc[r4 * 4]);
        acc[r4 * 4 + 1] = fmaf(p.y, v_f, acc[r4 * 4 + 1]);
        acc[r4 * 4 + 2] = fmaf(p.z, v_f, acc[r4 * 4 + 2]);
        acc[r4 * 4 + 3] = fmaf(p.w, v_f, acc[r4 * 4 + 3]);
      }
    }
    __syncthreads();  // p_s and vis_s are rewritten by the next chunk
  }
  const size_t part = (static_cast<size_t>(head) * splits + split) * rows;
#pragma unroll
  for (int r = 0; r < kRowsCap; ++r) {
    if (r < rows) {
      ws_acc[(part + r) * kQuantDepth + tid] = acc[r] * params.v_scale;
    }
  }
  if (tid < rows) {
    ws_m[part + tid] = m_s[tid];
    ws_l[part + tid] = l_s[tid];
  }
}

// The partials and their combine in one launch: a block owns one chunk of
// the keys for one query row of a head, a quarter of the work a block of
// QuantDecodeAttentionPartial does one thread after the other, and the block
// that finishes last of a row's chunks combines them (the others count
// themselves done in `blocks_done`, one counter per row, which the last block
// leaves zero). The arithmetic is that of QuantDecodeAttentionPartial with a
// chunk per split followed by DecodeAttentionCombine, operation for
// operation, and the partials are in the workspace as that kernel writes
// them. On the RTX 5080 the 64 blocks of a decode step over a ring of 512
// keys take 5.8 us against 8.2 us for the two launches; once the blocks
// outnumber the multiprocessors they wait for one another and the two
// launches are faster (2,048 keys: 10 to 12 us against 8.4 us).
template <typename T, int kBits>
__global__ void __launch_bounds__(kQuantThreads)
    QuantDecodeAttentionChunks(const T* __restrict__ q, int8_t* k, int8_t* v,
                               LiteRtNvidiaQuantAttentionParams params,
                               const bool* __restrict__ mask, int mask_rows,
                               int rows, int cache_len, float fill,
                               QuantToken<T> token, float* ws_acc,
                               float* ws_m, float* ws_l,
                               unsigned int* blocks_done,
                               T* __restrict__ out) {
  // The dequantized query, laid out as QuantDecodeAttentionPartial reads it;
  // the chunk's scores, then its probabilities; its maximum and sum.
  __shared__ float q_s[kQuantDepth];
  __shared__ float p_s[kQuantChunk];
  __shared__ float m_s;
  __shared__ float l_s;
  __shared__ bool last_s;
  constexpr int kKeyBytes = kQuantDepth * kBits / 8;
  constexpr int kPartDims = kQuantDepth / 4;
  constexpr int kVectorValues = 128 / kBits;  // values of a 16-byte vector

  const int splits = gridDim.x;
  const int c = blockIdx.x;
  const int r = blockIdx.y % rows;
  const int head = blockIdx.y / rows;
  const int tid = threadIdx.x;
  const int j0 = c * kQuantChunk;
  const int chunk = min(kQuantChunk, cache_len - j0);
  if (token.k != nullptr) {
    // Visible to the block's reads after the barrier below.
    WriteQuantToken<T, kBits>(token, params, k, v, head, cache_len, tid);
  }
  q_s[(tid % kPartDims) * 4 + tid / kPartDims] = DequantizedQuery(
      ToFloat(q[(static_cast<size_t>(head) * rows + r) * kQuantDepth + tid]),
      params.q_scale);
  if (tid == 0) {
    m_s = -INFINITY;
    l_s = 0.0f;
  }
  const bool* mask_row =
      mask == nullptr
          ? nullptr
          : mask + static_cast<size_t>(r % mask_rows) * cache_len + j0;
  // A chunk the row sees no key of is not read: its partial is empty.
  const bool sees = tid < chunk && (mask_row == nullptr || mask_row[tid]);
  float acc = 0.0f;
  if (__syncthreads_or(sees ? 1 : 0)) {
    // Scores: four threads share a key, a quarter of its dims each, and add
    // their sums up in a fixed order.
    {
      const int key = tid >> 2;
      const int part = tid & 3;
      const int4* k_part = reinterpret_cast<const int4*>(
          k +
          (static_cast<size_t>(head) * cache_len + j0 + min(key, chunk - 1)) *
              kKeyBytes +
          part * (kKeyBytes / 4));
      int4 loaded[kPartDims / kVectorValues];
#pragma unroll
      for (int i = 0; i < kPartDims / kVectorValues; ++i) {
        loaded[i] = k_part[i];
      }
      float score = 0.0f;
#pragma unroll
      for (int e = 0; e < kPartDims; ++e) {
        const float k_f = static_cast<float>(PackedValue<kBits>(
            reinterpret_cast<const uint32_t*>(loaded), e));
        score = fmaf(q_s[e * 4 + part], k_f, score);
      }
      score += __shfl_xor_sync(0xffffffffu, score, 1);
      score += __shfl_xor_sync(0xffffffffu, score, 2);
      if (part == 0 && key < chunk) {
        // The decomposition casts its scores to FP16 before the mask.
        p_s[key] = mask_row == nullptr || mask_row[key]
                       ? ThroughHalf(score * params.k_scale)
                       : fill;
      }
    }
    __syncthreads();

    // The chunk's softmax over its own maximum, by the first warp.
    if (tid < 32) {
      float m = -INFINITY;
      for (int jj = tid; jj < chunk; jj += 32) {
        m = fmaxf(m, p_s[jj]);
      }
      m = WarpMax(m);
      float l = 0.0f;
      for (int jj = tid; jj < chunk; jj += 32) {
        const float p = m == -INFINITY ? 0.0f : __expf(p_s[jj] - m);
        p_s[jj] = p;
        l += p;
      }
      l = WarpSum(l);
      if (tid == 0) {
        m_s = m;
        l_s = l;
      }
    }
    __syncthreads();

    // Weighted values: the thread's depth element of every key of the chunk,
    // loaded as stored, or value by value when the rows of the cache are not
    // whole vectors.
    const size_t v_first =
        (static_cast<size_t>(head) * kQuantDepth + tid) * cache_len + j0;
    const bool value_vectors = cache_len % kVectorValues == 0;
    int4 values[kQuantChunk / kVectorValues];
    if (value_vectors) {
      const int4* v_row =
          reinterpret_cast<const int4*>(v + v_first / (8 / kBits));
#pragma unroll
      for (int i = 0; i < kQuantChunk / kVectorValues; ++i) {
        if (i * kVectorValues < chunk) {
          values[i] = v_row[i];
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < kQuantChunk; ++jj) {
      if (jj >= chunk) {
        break;
      }
      int value;
      if (value_vectors) {
        value = PackedValue<kBits>(reinterpret_cast<const uint32_t*>(values),
                                   jj);
      } else {
        const size_t at = v_first + jj;
        if constexpr (kBits == 8) {
          value = v[at];
        } else {
          const uint32_t packed = static_cast<uint8_t>(v[at >> 1]);
          value = static_cast<int>(packed << ((at & 1) != 0 ? 24 : 28)) >> 28;
        }
      }
      acc = fmaf(p_s[jj], static_cast<float>(value), acc);
    }
  }
  const size_t row_parts = static_cast<size_t>(head) * splits * rows + r;
  const size_t part = row_parts + static_cast<size_t>(c) * rows;
  ws_acc[part * kQuantDepth + tid] = acc * params.v_scale;
  if (tid == 0) {
    ws_m[part] = m_s;
    ws_l[part] = l_s;
  }
  // Every thread's part of the partial is in memory before the block counts
  // itself done, so the row's last block reads whole partials.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    last_s = atomicAdd(blocks_done + blockIdx.y, 1u) + 1 == gridDim.x;
  }
  __syncthreads();
  if (!last_s) {
    return;
  }
  if (tid == 0) {
    blocks_done[blockIdx.y] = 0;
  }
  // The combine, as DecodeAttentionCombine's eight warps fold the splits:
  // warp w the splits w, w + 8, ..., then the warps' sums in their order.
  constexpr int kFolds = kCombineThreads / 32;
  float m_row = -INFINITY;
  for (int split = 0; split < splits; ++split) {
    m_row = fmaxf(m_row, ws_m[row_parts + static_cast<size_t>(split) * rows]);
  }
  float l_fold[kFolds];
  float acc_fold[kFolds];
#pragma unroll
  for (int w = 0; w < kFolds; ++w) {
    l_fold[w] = 0.0f;
    acc_fold[w] = 0.0f;
  }
  for (int first = 0; first < splits; first += kFolds) {
#pragma unroll
    for (int w = 0; w < kFolds; ++w) {
      if (first + w < splits) {
        const size_t at = row_parts + static_cast<size_t>(first + w) * rows;
        const float m = ws_m[at];
        const float weight = m == -INFINITY ? 0.0f : __expf(m - m_row);
        l_fold[w] = fmaf(weight, ws_l[at], l_fold[w]);
        acc_fold[w] = fmaf(weight, ws_acc[at * kQuantDepth + tid], acc_fold[w]);
      }
    }
  }
  float l_row = 0.0f;
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kFolds; ++w) {
    l_row += l_fold[w];
    total += acc_fold[w];
  }
  const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
  FromFloat(total * inv,
            out + (static_cast<size_t>(head) * rows + r) * kQuantDepth + tid);
}
#pragma clang diagnostic pop

// Whether QuantDecodeAttentionChunks runs these blocks: a chunk per split,
// and a multiprocessor of the current device for every block.
bool QuantChunkBlocksFit(int cache_len, int splits, int heads, int rows) {
  int device = 0;
  int multiprocessors = 0;
  return (cache_len + kQuantChunk - 1) / kQuantChunk == splits &&
         cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&multiprocessors,
                                cudaDevAttrMultiProcessorCount,
                                device) == cudaSuccess &&
         static_cast<int64_t>(splits) * heads * rows <= multiprocessors;
}

// odml.sdpa_transposed over quantized tensors, tensor by tensor as LiteRT's
// CPU backend computes its decomposition (XNNPACK on x86 with AVX2, FMA3 and
// F16C, the kernels LiteRT-LM's CPU backend runs on the hosts measured; a
// machine with other vector extensions picks other softmax kernels and has
// other bits itself):
//   [1]  DEQUANTIZE the INT8 query to FLOAT16 (xnn_qs8_f16_vcvt): q times
//        the scale rounded to FLOAT16, rounded to FLOAT16.
//   [2]  DEQUANTIZE the keys to fp32: k times the scale.
//   [4]  BATCH_MATMUL (xnn_f32_gemm, fma3): one fused multiply-add per depth
//        element, in order, from zero.
//   [5]  CAST the scores to FLOAT16.
//   [10] SELECT_V2: `fill` where the mask is false.
//   [11] SOFTMAX on FLOAT16 (xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2
//        and xnn_f16_vmulc): exp(x - max) in fp32 by a second-degree
//        polynomial, zero below -9.703125 (where the FLOAT16 result would be
//        denormal: such a key has no weight at all), each rounded to FLOAT16;
//        the fp32 sum of the unrounded values in the kernel's eight lanes;
//        the reciprocal of the sum rounded to FLOAT16; the products rounded
//        to FLOAT16.
//   [12] DEQUANTIZE the values to fp32.
//   [14] BATCH_MATMUL: one fused multiply-add per key, in the cache's order.
//   [15] CAST the result to FLOAT16.
// Two launches. QuantScoresKernel: a block per chunk of keys and head, a
// thread per key and row, [1] to [5] into the workspace. Then
// QuantSoftmaxValuesKernel: a block per row and head, a thread per depth
// element, [10] to [15]; it walks only the keys whose weight is not zero.

template <typename T, int kBits>
__global__ void __launch_bounds__(kQuantThreads)
    QuantScoresKernel(const T* __restrict__ q, int8_t* k, int8_t* v,
                      LiteRtNvidiaQuantAttentionParams params,
                      const bool* __restrict__ mask, int mask_rows, int rows,
                      int cache_len, QuantToken<T> token,
                      float* __restrict__ scores) {
  __shared__ float q_s[kMaxRows * kQuantDepth];
  constexpr int kKeyBytes = kQuantDepth * kBits / 8;
  constexpr int kWordValues = 32 / kBits;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int j0 = blockIdx.x * kQuantChunk;
  const int chunk = min(kQuantChunk, cache_len - j0);
  if (token.k != nullptr) {
    // Visible to every read after the barrier below, and to the next launch.
    WriteQuantToken<T, kBits>(token, params, k, v, head, cache_len, tid);
  }
  const T* q_head = q + static_cast<size_t>(head) * rows * kQuantDepth;
  for (int i = tid; i < rows * kQuantDepth; i += kQuantThreads) {
    q_s[i] = DequantizedQuery(ToFloat(q_head[i]), params.q_scale);
  }
  // A chunk no row sees is left alone: the next launch reads the mask.
  bool visible = mask == nullptr;
  if (mask != nullptr) {
    for (int i = tid; i < mask_rows * chunk; i += kQuantThreads) {
      visible |=
          mask[static_cast<size_t>(i / chunk) * cache_len + j0 + i % chunk];
    }
  }
  if (!__syncthreads_or(visible ? 1 : 0)) {
    return;
  }
  const int8_t* k_head = k + static_cast<size_t>(head) * cache_len * kKeyBytes;
  for (int pair = tid; pair < chunk * rows; pair += kQuantThreads) {
    const int key = pair / rows;
    const int r = pair % rows;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(
        k_head + static_cast<size_t>(j0 + key) * kKeyBytes);
    const float* q_row = q_s + r * kQuantDepth;
    float score = 0.0f;
    for (int w = 0; w < kQuantDepth / kWordValues; ++w) {
      const uint32_t word = words[w];
#pragma unroll
      for (int i = 0; i < kWordValues; ++i) {
        const int value =
            static_cast<int>(word << (32 - kBits - i * kBits)) >> (32 - kBits);
        score = fmaf(q_row[w * kWordValues + i],
                     __fmul_rn(static_cast<float>(value), params.k_scale),
                     score);
      }
    }
    scores[(static_cast<size_t>(head) * rows + r) * cache_len + j0 + key] =
        ThroughHalf(score);
  }
}

// exp(x) of x <= 0 as xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2
// computes it in fp32, operation for operation.
__device__ __forceinline__ float XnnHalfExpMinus(float x) {
  const float magic_bias = 0x1.8000FEp23f;
  float n = fmaf(x, 0x1.715476p0f, magic_bias);
  const float s = __uint_as_float(__float_as_uint(n) << 23);
  n = __fsub_rn(n, magic_bias);
  float t = fmaf(n, -0x1.62E43p-1f, x);
  const float p = fmaf(0x1.FF3A32p-2f, t, 0x1.039E10p+0f);
  t = __fmul_rn(t, s);
  const float f = fmaf(t, p, s);
  return x < -0x1.368000p+3f ? 0.0f : f;
}

template <typename T, int kBits>
__global__ void __launch_bounds__(kQuantThreads)
    QuantSoftmaxValuesKernel(const int8_t* __restrict__ v,
                             LiteRtNvidiaQuantAttentionParams params,
                             const bool* __restrict__ mask, int mask_rows,
                             int rows, int cache_len, float fill,
                             float* __restrict__ scores,
                             float* __restrict__ weights,
                             int32_t* __restrict__ keys, T* __restrict__ out) {
  constexpr int kWarps = kQuantThreads / 32;
  __shared__ float max_s[kWarps];
  __shared__ int count_s[kQuantThreads];
  __shared__ float lane_s[8];
  __shared__ int total_s;
  __shared__ float scale_s;
  const int r = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const size_t row = (static_cast<size_t>(head) * rows + r) * cache_len;
  float* s_row = scores + row;
  float* w_row = weights + row;
  int32_t* k_row = keys + row;
  const bool* mask_row =
      mask == nullptr
          ? nullptr
          : mask + static_cast<size_t>(r % mask_rows) * cache_len;
  // The thread's keys are a run of the cache, so that the threads' runs one
  // after the other are the cache's order.
  const int run = (cache_len + kQuantThreads - 1) / kQuantThreads;
  const int first = min(tid * run, cache_len);
  const int last = min(first + run, cache_len);

  // [10] and the row's maximum.
  float m = -INFINITY;
  for (int j = first; j < last; ++j) {
    const float score = mask_row == nullptr || mask_row[j] ? s_row[j] : fill;
    s_row[j] = score;
    m = fmaxf(m, score);
  }
  m = WarpMax(m);
  if (tid % 32 == 0) {
    max_s[tid / 32] = m;
  }
  __syncthreads();
  m = max_s[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    m = fmaxf(m, max_s[w]);
  }
  if (m == -INFINITY) {
    // The row sees no key and the fill is -inf (the plugin's): the graphs
    // replace such a row or never read it, and it gets zero. With the
    // decomposition's finite fill the row is spread evenly over the whole
    // cache, as on the CPU.
    FromFloat(0.0f, out + (static_cast<size_t>(head) * rows + r) *
                              kQuantDepth + tid);
    return;
  }

  // [11]: exp(x - max), not yet rounded, and the keys that keep a weight, in
  // the cache's order.
  int count = 0;
  for (int j = first; j < last; ++j) {
    const float f = XnnHalfExpMinus(__fsub_rn(s_row[j], m));
    s_row[j] = f;
    count += f != 0.0f ? 1 : 0;
  }
  count_s[tid] = count;
  __syncthreads();
  int at = 0;
  for (int t = 0; t < tid; ++t) {
    at += count_s[t];
  }
  if (tid == kQuantThreads - 1) {
    total_s = at + count;
  }
  for (int j = first; j < last; ++j) {
    const float f = s_row[j];
    if (f != 0.0f) {
      k_row[at] = j;
      w_row[at] = f;
      ++at;
    }
  }
  __syncthreads();
  const int total = total_s;
  // The kernel's sum: lane j % 8 adds its values in order over the whole
  // vectors of eight, the lanes pair up (0 with 4, ...), the last keys of a
  // length that is no multiple of eight join the pairs, then (0 + 2) +
  // (1 + 3).
  const int whole = cache_len & ~7;
  if (tid < 8) {
    float lane = 0.0f;
    for (int i = 0; i < total; ++i) {
      const int j = k_row[i];
      if (j < whole && (j & 7) == tid) {
        lane = __fadd_rn(lane, w_row[i]);
      }
    }
    lane_s[tid] = lane;
  }
  __syncthreads();
  if (tid == 0) {
    float pair[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pair[i] = __fadd_rn(lane_s[i], lane_s[i + 4]);
    }
    float tail[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = total - 1; i >= 0 && k_row[i] >= whole; --i) {
      tail[k_row[i] - whole] = w_row[i];
    }
    const int rest = cache_len - whole;
    int next = 0;
    if ((rest & 4) != 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        pair[i] = __fadd_rn(pair[i], tail[i]);
      }
      next = 4;
    }
    if ((rest & 2) != 0) {
      pair[0] = __fadd_rn(pair[0], tail[next]);
      pair[1] = __fadd_rn(pair[1], tail[next + 1]);
      next += 2;
    }
    if ((rest & 1) != 0) {
      pair[0] = __fadd_rn(pair[0], tail[next]);
    }
    const float sum = __fadd_rn(__fadd_rn(pair[0], pair[2]),
                                __fadd_rn(pair[1], pair[3]));
    scale_s = ThroughHalf(1.0f / sum);
  }
  __syncthreads();
  const float scale = scale_s;
  for (int i = tid; i < total; i += kQuantThreads) {
    w_row[i] = ThroughHalf(__fmul_rn(ThroughHalf(w_row[i]), scale));
  }
  __syncthreads();

  // [12], [14] and [15] for the thread's depth element.
  const size_t v_first =
      (static_cast<size_t>(head) * kQuantDepth + tid) * cache_len;
  float acc = 0.0f;
  for (int i = 0; i < total; ++i) {
    const size_t value_at = v_first + k_row[i];
    int value;
    if constexpr (kBits == 8) {
      value = v[value_at];
    } else {
      const uint32_t packed = static_cast<uint8_t>(v[value_at >> 1]);
      value =
          static_cast<int>(packed << ((value_at & 1) != 0 ? 24 : 28)) >> 28;
    }
    acc = fmaf(w_row[i], __fmul_rn(static_cast<float>(value), params.v_scale),
               acc);
  }
  FromFloat(acc, out + (static_cast<size_t>(head) * rows + r) * kQuantDepth +
                     tid);
}

// Which kernels LaunchQuantRows takes: -1 asks the environment.
std::atomic<int> quant_attention_exact{-1};

bool QuantAttentionExact() {
  const int forced = quant_attention_exact.load(std::memory_order_relaxed);
  if (forced >= 0) {
    return forced != 0;
  }
  static const bool from_environment = [] {
    const char* value = std::getenv("LITERT_NVIDIA_QUANT_ATTENTION_EXACT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return from_environment;
}

// The workspace of the two launches: the scores, the weights and their keys,
// each [heads][rows][cache_len].
size_t QuantWorkspaceBytes(int heads, int rows, int cache_len) {
  return static_cast<size_t>(heads) * rows * cache_len *
         (2 * sizeof(float) + sizeof(int32_t));
}

template <typename T, int kRowsCap, int kBits>
cudaError_t LaunchQuantRows(const T* q, int8_t* k, int8_t* v,
                            const LiteRtNvidiaQuantAttentionParams& params,
                            const bool* mask, int mask_rows, int heads,
                            int rows, int cache_len, float fill,
                            const QuantToken<T>& token, T* out,
                            void* workspace, unsigned int* blocks_done,
                            cudaStream_t stream) {
  if (!QuantAttentionExact()) {
    const int splits = SplitCount(cache_len, kQuantChunk);
    float* ws_acc = static_cast<float*>(workspace);
    float* ws_m =
        ws_acc + static_cast<size_t>(heads) * splits * rows * kQuantDepth;
    float* ws_l = ws_m + static_cast<size_t>(heads) * splits * rows;
    if (blocks_done != nullptr &&
        QuantChunkBlocksFit(cache_len, splits, heads, rows)) {
      QuantDecodeAttentionChunks<T, kBits>
          <<<dim3(splits, heads * rows), kQuantThreads, 0, stream>>>(
              q, k, v, params, mask, mask_rows, rows, cache_len, fill, token,
              ws_acc, ws_m, ws_l, blocks_done, out);
      return cudaGetLastError();
    }
    QuantDecodeAttentionPartial<T, kRowsCap, kBits>
        <<<dim3(splits, heads), kQuantThreads, 0, stream>>>(
            q, k, v, params, mask, mask_rows, rows, cache_len, fill, splits,
            token, ws_acc, ws_m, ws_l);
    const cudaError_t partial_status = cudaGetLastError();
    if (partial_status != cudaSuccess) {
      return partial_status;
    }
    DecodeAttentionCombine<T, kQuantDepth>
        <<<dim3(rows, heads), kCombineThreads, 0, stream>>>(
            ws_acc, ws_m, ws_l, rows, splits, out);
    return cudaGetLastError();
  }
  // The decomposition's own arithmetic: two launches whatever the cache.
  const size_t entries = static_cast<size_t>(heads) * rows * cache_len;
  float* scores = static_cast<float*>(workspace);
  float* weights = scores + entries;
  int32_t* keys = reinterpret_cast<int32_t*>(weights + entries);
  const int chunks = (cache_len + kQuantChunk - 1) / kQuantChunk;
  QuantScoresKernel<T, kBits>
      <<<dim3(chunks, heads), kQuantThreads, 0, stream>>>(
          q, k, v, params, mask, mask_rows, rows, cache_len, token, scores);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  QuantSoftmaxValuesKernel<T, kBits>
      <<<dim3(rows, heads), kQuantThreads, 0, stream>>>(
          v, params, mask, mask_rows, rows, cache_len, fill, scores, weights,
          keys, out);
  return cudaGetLastError();
}

template <typename T>
cudaError_t LaunchQuant(const void* q, void* k, void* v,
                        const LiteRtNvidiaQuantCacheToken* token,
                        const LiteRtNvidiaQuantAttentionParams& params,
                        const bool* mask, int mask_rows, int heads, int rows,
                        int cache_len, float fill, void* out, void* workspace,
                        void* state, cudaStream_t stream) {
  unsigned int* blocks_done = static_cast<unsigned int*>(state);
  QuantToken<T> rows_written;
  if (token != nullptr) {
    rows_written = {static_cast<const T*>(token->k),
                    static_cast<const T*>(token->v), token->k_position,
                    token->v_position};
  }
  const T* q_rows = static_cast<const T*>(q);
  int8_t* k_cache = static_cast<int8_t*>(k);
  int8_t* v_cache = static_cast<int8_t*>(v);
  T* out_rows = static_cast<T*>(out);
  if (params.cache_bits == 8) {
    return rows <= 4 ? LaunchQuantRows<T, 4, 8>(
                           q_rows, k_cache, v_cache, params, mask, mask_rows,
                           heads, rows, cache_len, fill, rows_written,
                           out_rows, workspace, blocks_done, stream)
                     : LaunchQuantRows<T, kMaxRows, 8>(
                           q_rows, k_cache, v_cache, params, mask, mask_rows,
                           heads, rows, cache_len, fill, rows_written,
                           out_rows, workspace, blocks_done, stream);
  }
  return rows <= 4 ? LaunchQuantRows<T, 4, 4>(
                         q_rows, k_cache, v_cache, params, mask, mask_rows,
                         heads, rows, cache_len, fill, rows_written, out_rows,
                         workspace, blocks_done, stream)
                   : LaunchQuantRows<T, kMaxRows, 4>(
                         q_rows, k_cache, v_cache, params, mask, mask_rows,
                         heads, rows, cache_len, fill, rows_written, out_rows,
                         workspace, blocks_done, stream);
}

}  // namespace

extern "C" size_t LiteRtNvidiaDecodeAttentionWorkspaceBytes(int32_t heads,
                                                            int32_t rows,
                                                            int32_t seq,
                                                            int32_t depth) {
  const size_t parts =
      static_cast<size_t>(heads) * MaxSplitCount(seq) * rows;
  const size_t bytes = parts * (static_cast<size_t>(depth) + 2) * sizeof(float);
  // The quantized attention keeps a score, a weight and a key index per key
  // of every row.
  const size_t quant_bytes = QuantWorkspaceBytes(heads, rows, seq);
  return bytes > quant_bytes ? bytes : quant_bytes;
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

extern "C" void LiteRtNvidiaSetQuantAttentionExact(int32_t exact) {
  quant_attention_exact.store(exact, std::memory_order_relaxed);
}

extern "C" bool LiteRtNvidiaQuantDecodeAttentionSupports(int32_t rows,
                                                         int32_t depth,
                                                         int32_t cache_bits) {
  return rows >= 1 && rows <= kMaxRows && depth == kQuantDepth &&
         (cache_bits == 8 || cache_bits == 4);
}

extern "C" bool LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
    int32_t rows, int32_t depth, int32_t cache_bits, int32_t cache_len) {
  return LiteRtNvidiaQuantDecodeAttentionSupports(rows, depth, cache_bits) &&
         cache_len >= 1 && (cache_bits == 8 || cache_len % 2 == 0);
}

extern "C" size_t LiteRtNvidiaQuantDecodeAttentionStateBytes(int32_t heads,
                                                             int32_t rows) {
  return heads <= 0 || rows <= 0
             ? 0
             : static_cast<size_t>(heads) * rows * sizeof(unsigned int);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
    const void* q, bool q_bf16, void* k, void* v,
    const LiteRtNvidiaQuantCacheToken* token,
    const LiteRtNvidiaQuantAttentionParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t depth, float fill, void* out, void* workspace, void* state,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || params == nullptr ||
      out == nullptr || workspace == nullptr || heads <= 0 || cache_len <= 0 ||
      !LiteRtNvidiaQuantDecodeAttentionSupports(rows, depth,
                                                params->cache_bits) ||
      mask_rows <= 0 || rows % mask_rows != 0 ||
      !(params->q_scale > 0.0f) || !std::isfinite(params->q_scale) ||
      !std::isfinite(params->k_scale) || !std::isfinite(params->v_scale)) {
    return cudaErrorInvalidValue;
  }
  if (token != nullptr &&
      (token->k == nullptr || token->v == nullptr ||
       token->k_position == nullptr || token->v_position == nullptr ||
       !(params->k_scale > 0.0f) || !(params->v_scale > 0.0f) ||
       !LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
           rows, depth, params->cache_bits, cache_len))) {
    return cudaErrorInvalidValue;
  }
  if (q_bf16) {
    return LaunchQuant<__nv_bfloat16>(q, k, v, token, *params, mask, mask_rows,
                                      heads, rows, cache_len, fill, out,
                                      workspace, state, stream);
  }
  return LaunchQuant<__half>(q, k, v, token, *params, mask, mask_rows, heads,
                             rows, cache_len, fill, out, workspace, state,
                             stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchQuantDecodeAttention(
    const void* q, bool q_bf16, const void* k, const void* v,
    const LiteRtNvidiaQuantAttentionParams* params, const bool* mask,
    int32_t mask_rows, int32_t heads, int32_t rows, int32_t cache_len,
    int32_t depth, float fill, void* out, void* workspace,
    cudaStream_t stream) {
  // Without a token the launches write nothing to the caches; without a
  // state a second launch combines.
  return LiteRtNvidiaLaunchQuantDecodeAttentionUpdate(
      q, q_bf16, const_cast<void*>(k), const_cast<void*>(v), /*token=*/nullptr,
      params, mask, mask_rows, heads, rows, cache_len, depth, fill, out,
      workspace, /*state=*/nullptr, stream);
}
