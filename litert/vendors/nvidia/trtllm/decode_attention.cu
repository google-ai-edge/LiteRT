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

namespace {

constexpr int kChunk = 64;         // keys per block
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
template <int kDepth>
__host__ __device__ constexpr int ThreadsFor() {
  return kDepth >= 256 ? 256 : 128;
}

// Probabilities are kept key-major ([kChunk][kRowsPad]) so the value phase
// reads a key's rows with 16-byte loads; the pad keeps rows 16-byte aligned.
template <int kRowsCap>
__host__ __device__ constexpr int RowsPadFor() {
  return kRowsCap == 16 ? 20 : 4;
}

template <int kDepth, int kRowsCap>
__host__ __device__ constexpr size_t SharedBytesFor() {
  // q as fp32 [rows][E][32], scores/probabilities [kChunk][kRowsPad] fp32,
  // row max and sum, mask bytes [rows][kChunk].
  return static_cast<size_t>(kRowsCap) * kDepth * sizeof(float) +
         static_cast<size_t>(kChunk) * RowsPadFor<kRowsCap>() * sizeof(float) +
         3 * kRowsCap * sizeof(float) +
         static_cast<size_t>(kRowsCap) * kChunk * sizeof(bool);
}

// One block per (chunk of keys, head). Shared memory: q as [rows][E][32]
// halves (lane-major so each lane reads its own slice without bank
// conflicts), then scores/probabilities [rows][kChunk], then row max and
// sum, then the chunk's mask. Partial results go to the workspace:
// acc [heads][splits][rows][depth], then m [heads][splits][rows], then
// l [heads][splits][rows].
template <typename T, int kDepth, int kRowsCap>
__global__ void __launch_bounds__(ThreadsFor<kDepth>(), 2)
    DecodeAttentionPartial(const T* __restrict__ q, const __half* __restrict__ k,
                           const __half* __restrict__ v,
                           const bool* __restrict__ mask, int mask_rows,
                           int rows, int seq, float fill, int splits,
                           float* __restrict__ ws_acc, float* __restrict__ ws_m,
                           float* __restrict__ ws_l) {
  constexpr int kThreads = ThreadsFor<kDepth>();
  constexpr int kWarps = kThreads / 32;
  constexpr int kLaneElems = kDepth / 32;     // key elements per lane (QK)
  constexpr int kThreadElems = kDepth / kThreads;  // value elements per thread
  constexpr int kRowsPad = RowsPadFor<kRowsCap>();
  extern __shared__ float smem[];
  float* q_s = smem;                                   // rows * depth
  float* p_s = q_s + kRowsCap * kDepth;                // kChunk * kRowsPad
  float* m_s = p_s + kChunk * kRowsPad;                // rows
  float* l_s = m_s + kRowsCap;                         // rows
  float* scale_s = l_s + kRowsCap;                     // rows
  bool* mask_s = reinterpret_cast<bool*>(scale_s + kRowsCap);  // rows*kChunk

  const int split = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int num_chunks = (seq + kChunk - 1) / kChunk;

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
  float acc[kRowsCap][kThreadElems];
#pragma unroll
  for (int r = 0; r < kRowsCap; ++r) {
#pragma unroll
    for (int e = 0; e < kThreadElems; ++e) {
      acc[r][e] = 0.0f;
    }
  }
  if (tid < rows) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }
  __syncthreads();
  const __half* k_head = k + static_cast<size_t>(head) * seq * kDepth;
  const __half* v_head = v + static_cast<size_t>(head) * seq * kDepth;
  for (int c = split; c < num_chunks; c += gridDim.x) {
  const int j0 = c * kChunk;
  const int chunk = min(kChunk, seq - j0);
  // The chunk's mask bytes, loaded once instead of per key.
  if (tid < kChunk) {
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r < rows) {
        const int jj = tid;
        bool keep = true;
        if (mask != nullptr && jj < chunk) {
          keep = mask[static_cast<size_t>(mask_rows == 1 ? 0 : r) * seq + j0 +
                      jj];
        }
        mask_s[r * kChunk + jj] = keep;
      }
    }
  }
  __syncthreads();

  // Scores: each warp takes every kWarps-th key of the chunk, kKeyBatch keys
  // per iteration so their loads are in flight together (the loop is
  // latency-bound otherwise). Keys past the chunk are computed and dropped.
  constexpr int kVecs = kLaneElems / 8;  // 16-byte loads per lane (0 for 4)
  for (int jb = warp; jb < chunk; jb += kWarps * kKeyBatch) {
    int4 k_vec[kKeyBatch][kVecs > 0 ? kVecs : 1];
    int2 k_half[kKeyBatch];
#pragma unroll
    for (int b = 0; b < kKeyBatch; ++b) {
      const int jj = min(jb + b * kWarps, chunk - 1);
      const __half* k_row =
          k_head + static_cast<size_t>(j0 + jj) * kDepth + lane * kLaneElems;
      if (kVecs > 0) {
#pragma unroll
        for (int c = 0; c < kVecs; ++c) {
          k_vec[b][c] = reinterpret_cast<const int4*>(k_row)[c];
        }
      } else {
        k_half[b] = reinterpret_cast<const int2*>(k_row)[0];
      }
    }
    // Convert the batch once, then walk the rows with each query slice read
    // from shared memory once per batch (not once per key).
    float k_reg[kKeyBatch][kLaneElems];
#pragma unroll
    for (int b = 0; b < kKeyBatch; ++b) {
      if (kVecs > 0) {
#pragma unroll
        for (int c = 0; c < kVecs; ++c) {
          const __half* h = reinterpret_cast<const __half*>(&k_vec[b][c]);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            k_reg[b][c * 8 + e] = ToFloat(h[e]);
          }
        }
      } else {
        const __half* h = reinterpret_cast<const __half*>(&k_half[b]);
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e) {
          k_reg[b][e] = ToFloat(h[e]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRowsCap; ++r) {
      if (r >= rows) {
        break;
      }
      const float* q_row = q_s + r * kDepth + lane;
      float q_reg[kLaneElems];
#pragma unroll
      for (int e = 0; e < kLaneElems; ++e) {
        q_reg[e] = q_row[32 * e];
      }
      float partial[kKeyBatch];
#pragma unroll
      for (int b = 0; b < kKeyBatch; ++b) {
        partial[b] = 0.0f;
#pragma unroll
        for (int e = 0; e < kLaneElems; ++e) {
          partial[b] = fmaf(q_reg[e], k_reg[b][e], partial[b]);
        }
        partial[b] = WarpSum(partial[b]);
      }
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < kKeyBatch; ++b) {
          const int jj = jb + b * kWarps;
          if (jj < chunk) {
            p_s[jj * kRowsPad + r] =
                mask_s[r * kChunk + jj] ? partial[b] : fill;
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
      for (int e = 0; e < kThreadElems; ++e) {
        acc[r][e] *= scale;
      }
    }
  }

  // Weighted values: thread t owns depth slice [t * kThreadElems, +kThreadElems).
  // Row loops are unrolled to the compile-time cap so `acc` stays in
  // registers (a runtime-indexed register array is demoted to local memory).
  for (int jb = 0; jb < chunk; jb += kValueBatch) {
    __half v_reg[kValueBatch][kThreadElems];
#pragma unroll
    for (int b = 0; b < kValueBatch; ++b) {
      const int jj = min(jb + b, chunk - 1);
      const __half* v_row =
          v_head + static_cast<size_t>(j0 + jj) * kDepth + tid * kThreadElems;
#pragma unroll
      for (int e = 0; e < kThreadElems; ++e) {
        v_reg[b][e] = v_row[e];
      }
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
        v_f[e] = ToFloat(v_reg[b][e]);
      }
      // The key's probabilities, four rows per 16-byte load.
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
  }
  __syncthreads();  // p_s and mask_s are rewritten by the next chunk
  }
  const size_t part = (static_cast<size_t>(head) * splits + split) * rows;
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
  if (tid < rows) {
    ws_m[part + tid] = m_s[tid];
    ws_l[part + tid] = l_s[tid];
  }
}

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

// Blocks per head: one per chunk up to kMaxSplits; beyond that the blocks
// loop over chunks, which keeps the partial workspace and the combine small.
constexpr int kMaxSplits = 168;
int SplitCount(int seq) {
  const int chunks = (seq + kChunk - 1) / kChunk;
  return chunks < kMaxSplits ? chunks : kMaxSplits;
}


template <typename T, int kDepth, int kRowsCap>
cudaError_t Launch(const T* q, const __half* k, const __half* v,
                   const bool* mask, int mask_rows, int heads, int rows,
                   int seq, float fill, T* out, void* workspace,
                   cudaStream_t stream) {
  const int splits = SplitCount(seq);
  float* ws_acc = static_cast<float*>(workspace);
  float* ws_m = ws_acc + static_cast<size_t>(heads) * splits * rows * kDepth;
  float* ws_l = ws_m + static_cast<size_t>(heads) * splits * rows;
  constexpr size_t kSharedBytes = SharedBytesFor<kDepth, kRowsCap>();
  static bool attribute_set = false;  // per instantiation
  if (!attribute_set) {
    cudaError_t status = cudaFuncSetAttribute(
        DecodeAttentionPartial<T, kDepth, kRowsCap>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) {
      return status;
    }
    // Several blocks per SM hide the per-block latency chains.
    status = cudaFuncSetAttribute(
        DecodeAttentionPartial<T, kDepth, kRowsCap>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (status != cudaSuccess) {
      return status;
    }
    attribute_set = true;
  }
  DecodeAttentionPartial<T, kDepth, kRowsCap>
      <<<dim3(splits, heads), ThreadsFor<kDepth>(), kSharedBytes, stream>>>(
          q, k, v, mask, mask_rows, rows, seq, fill, splits, ws_acc, ws_m,
          ws_l);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  DecodeAttentionCombine<T, kDepth>
      <<<dim3(rows, heads), kCombineThreads, 0, stream>>>(ws_acc, ws_m, ws_l,
                                                          rows, splits, out);
  return cudaGetLastError();
}

template <typename T, int kDepth>
cudaError_t LaunchRows(const T* q, const __half* k, const __half* v,
                       const bool* mask, int mask_rows, int heads, int rows,
                       int seq, float fill, T* out, void* workspace,
                       cudaStream_t stream) {
  if (rows <= 4) {
    return Launch<T, kDepth, 4>(q, k, v, mask, mask_rows, heads, rows, seq,
                                fill, out, workspace, stream);
  }
  return Launch<T, kDepth, kMaxRows>(q, k, v, mask, mask_rows, heads, rows,
                                     seq, fill, out, workspace, stream);
}

template <typename T>
cudaError_t LaunchDepth(const T* q, const __half* k, const __half* v,
                        const bool* mask, int mask_rows, int heads, int rows,
                        int seq, int depth, float fill, T* out,
                        void* workspace, cudaStream_t stream) {
  switch (depth) {
    case 128:
      return LaunchRows<T, 128>(q, k, v, mask, mask_rows, heads, rows, seq,
                                fill, out, workspace, stream);
    case 256:
      return LaunchRows<T, 256>(q, k, v, mask, mask_rows, heads, rows, seq,
                                fill, out, workspace, stream);
    case 512:
      return LaunchRows<T, 512>(q, k, v, mask, mask_rows, heads, rows, seq,
                                fill, out, workspace, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

extern "C" size_t LiteRtNvidiaDecodeAttentionWorkspaceBytes(int32_t heads,
                                                            int32_t rows,
                                                            int32_t seq,
                                                            int32_t depth) {
  const size_t parts = static_cast<size_t>(heads) * SplitCount(seq) * rows;
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
