// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.

#include "litert/vendors/nvidia/dispatch/topk_candidates_kernel.h"

#include <climits>
#include <cstdint>

#include "cuda_fp16.h"

namespace {

constexpr int kThreads = 1024;
constexpr int kMaxK = 1024;
constexpr int kBins = 256;

// An order-preserving key: a larger float gives a larger key. The two zeros
// get one key (the CPU sampler's comparisons hold them equal). A NaN's key
// is whatever its bits give; the kernel flags the row instead.
__device__ __forceinline__ uint32_t FloatKey(float value) {
  if (value == 0.0f) value = 0.0f;
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float KeyFloat(uint32_t key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

__device__ __forceinline__ float ReadLogit(float value) { return value; }
__device__ __forceinline__ float ReadLogit(__half value) {
  return __half2float(value);
}

// Row `blockIdx.x` with one block: radix select of the k-th largest key,
// compaction of the k candidates, bitonic sort (key descending, index
// ascending: the CPU sampler's order), and the candidates written out.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    TopKCandidatesKernel(const T* logits, size_t count, int k, int32_t* ids,
                         float* values, int32_t* flags) {
  __shared__ uint32_t hist[kBins];
  __shared__ uint32_t cand_keys[kMaxK];
  __shared__ int32_t cand_idx[kMaxK];
  __shared__ uint32_t warp_totals[32];
  __shared__ uint32_t s_prefix, s_mask, s_remaining, s_equal, s_count;
  __shared__ int s_nan;

  const T* row = logits + static_cast<size_t>(blockIdx.x) * count;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // 1. Radix select, 8 bits per pass from the top: the key of the k-th
  // largest element, and how many of the elements equal to it are needed.
  // The first pass also looks for NaNs.
  if (tid == 0) {
    s_prefix = 0;
    s_mask = 0;
    s_remaining = static_cast<uint32_t>(k);
    s_equal = 0;
    s_nan = 0;
  }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < kBins) hist[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    const uint32_t mask = s_mask;
    bool nan_seen = false;
    for (size_t i = tid; i < count; i += kThreads) {
      const float value = ReadLogit(row[i]);
      if (shift == 24) nan_seen |= isnan(value);
      const uint32_t key = FloatKey(value);
      if ((key & mask) == prefix) {
        atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
      }
    }
    if (shift == 24 && __any_sync(0xffffffffu, nan_seen) && lane == 0) {
      s_nan = 1;
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t remaining = s_remaining;
      int bin = kBins - 1;
      for (; bin > 0; --bin) {
        if (hist[bin] >= remaining) break;
        remaining -= hist[bin];
      }
      s_remaining = remaining;
      s_prefix = prefix | (static_cast<uint32_t>(bin) << shift);
      s_mask = mask | (0xffu << shift);
      s_equal = hist[bin];
      s_count = 0;
    }
    __syncthreads();
  }
  const uint32_t threshold = s_prefix;
  const uint32_t needed_equal = s_remaining;
  const uint32_t n_above = static_cast<uint32_t>(k) - needed_equal;
  const bool take_all_equal = s_equal == needed_equal;

  // 2. Compaction: every element above the threshold, and the elements equal
  // to it (all of them when they are exactly the ones needed; else the
  // needed number of them in index order).
  for (size_t i = tid; i < count; i += kThreads) {
    const uint32_t key = FloatKey(ReadLogit(row[i]));
    if (key > threshold || (take_all_equal && key == threshold)) {
      const uint32_t slot = atomicAdd(&s_count, 1u);
      if (slot < static_cast<uint32_t>(kMaxK)) {
        cand_keys[slot] = key;
        cand_idx[slot] = static_cast<int32_t>(i);
      }
    }
  }
  __syncthreads();
  bool consistent =
      s_count == (take_all_equal ? static_cast<uint32_t>(k) : n_above);
  if (!take_all_equal) {
    uint32_t taken = 0;
    for (size_t start = 0; start < count && taken < needed_equal;
         start += kThreads) {
      const size_t i = start + tid;
      const bool flag = i < count && FloatKey(ReadLogit(row[i])) == threshold;
      const unsigned ballot = __ballot_sync(0xffffffffu, flag);
      const uint32_t warp_rank = __popc(ballot & ((1u << lane) - 1u));
      if (lane == 0) warp_totals[warp] = __popc(ballot);
      __syncthreads();
      uint32_t warp_base = 0;
      uint32_t chunk_total = 0;
      for (int w = 0; w < 32; ++w) {
        if (w < warp) warp_base += warp_totals[w];
        chunk_total += warp_totals[w];
      }
      if (flag) {
        const uint32_t rank = taken + warp_base + warp_rank;
        if (rank < needed_equal) {
          cand_keys[n_above + rank] = threshold;
          cand_idx[n_above + rank] = static_cast<int32_t>(i);
        }
      }
      taken += chunk_total;
      __syncthreads();
    }
    consistent = consistent && taken >= needed_equal;
  }

  // 3. Bitonic sort of the k candidates, padded to a power of two with
  // entries that sort last.
  int n = 1;
  while (n < k) n <<= 1;
  if (tid >= k && tid < n) {
    cand_keys[tid] = 0;
    cand_idx[tid] = INT32_MAX;
  }
  __syncthreads();
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const int partner = tid ^ stride;
      if (tid < n && partner > tid) {
        const uint32_t ka = cand_keys[tid], kb = cand_keys[partner];
        const int32_t ia = cand_idx[tid], ib = cand_idx[partner];
        const bool a_first = ka > kb || (ka == kb && ia < ib);
        const bool b_first = kb > ka || (kb == ka && ib < ia);
        const bool swap = ((tid & size) == 0) ? b_first : a_first;
        if (swap) {
          cand_keys[tid] = kb;
          cand_keys[partner] = ka;
          cand_idx[tid] = ib;
          cand_idx[partner] = ia;
        }
      }
      __syncthreads();
    }
  }

  // 4. The candidates, and the row's flags.
  if (tid < k) {
    const size_t out = static_cast<size_t>(blockIdx.x) * k + tid;
    ids[out] = cand_idx[tid];
    values[out] = KeyFloat(cand_keys[tid]);
  }
  if (tid == 0) {
    flags[blockIdx.x] = (s_nan ? kLiteRtNvidiaTopKCandidatesNaN : 0) |
                        (consistent ? 0 : kLiteRtNvidiaTopKCandidatesInconsistent);
  }
}

template <typename T>
cudaError_t Launch(const T* logits, size_t rows, size_t count, int k,
                   int32_t* ids, float* values, int32_t* flags,
                   cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || values == nullptr ||
      flags == nullptr || rows == 0 || count == 0 ||
      rows > static_cast<size_t>(INT_MAX) ||
      count > static_cast<size_t>(INT_MAX) || k < 1 || k > kMaxK ||
      static_cast<size_t>(k) > count) {
    return cudaErrorInvalidValue;
  }
  TopKCandidatesKernel<T><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      logits, count, k, ids, values, flags);
  return cudaPeekAtLastError();
}

}  // namespace

extern "C" int LiteRtNvidiaTopKCandidatesMaxK(void) { return kMaxK; }

extern "C" cudaError_t LiteRtNvidiaLaunchF32TopKCandidateRows(
    const float* logits, size_t rows, size_t count, int k, int32_t* ids,
    float* values, int32_t* flags, cudaStream_t stream) {
  return Launch(logits, rows, count, k, ids, values, flags, stream);
}

extern "C" cudaError_t LiteRtNvidiaLaunchF16TopKCandidateRows(
    const uint16_t* logits, size_t rows, size_t count, int k, int32_t* ids,
    float* values, int32_t* flags, cudaStream_t stream) {
  return Launch(reinterpret_cast<const __half*>(logits), rows, count, k, ids,
                values, flags, stream);
}
