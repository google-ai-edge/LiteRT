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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_CACHE_WRITE_SLOTS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_CACHE_WRITE_SLOTS_H_

#include <cstdint>

#include "cuda_runtime_api.h"

// The cache writes of a decode step for several sequence slots in one launch
// (a batched decode step: one sequence per slot, each with its own caches):
// every write puts the rows a step adds into its own cache at its own
// position, in place, and touches nothing else of the cache. A launch takes
// up to kLiteRtNvidiaCacheWriteMaxWrites writes of one shape: a layer's keys
// of every slot, its values, or both.
//
// An FP16 cache gets the rows converted to FP16 (round to nearest even),
// what a cast and TensorRT's KV cache update write. An INT8 cache gets them
// quantized with the write's scale and zero point exactly as
// LaunchInt8CacheWrite (compiler/int8_cache_write_kernel.h) quantizes them.

// The most writes of one launch: the keys and the values of eight slots.
constexpr int kLiteRtNvidiaCacheWriteMaxWrites = 16;

// The element type of the rows being written (Int8CacheWriteSource's values).
constexpr int32_t kLiteRtNvidiaCacheWriteHalf = 0;
constexpr int32_t kLiteRtNvidiaCacheWriteBf16 = 1;
constexpr int32_t kLiteRtNvidiaCacheWriteFloat = 2;

// One write (device pointers): `tokens` rows of `update` [heads, tokens,
// depth] into `cache` at the row position position[0], clamped to [0,
// cache_len - tokens]. The cache holds rows [heads, cache_len, depth], or
// with `columns` [heads, depth, cache_len], where the rows become columns.
struct LiteRtNvidiaCacheWrite {
  void* cache;
  const void* update;
  const int32_t* position;
  int32_t columns;
  int32_t int8;  // 0: an FP16 cache; otherwise INT8 with scale and zero_point
  float scale;
  int32_t zero_point;
};

// `writes` points to `num_writes` (1 to kLiteRtNvidiaCacheWriteMaxWrites)
// writes of the same shape and source type; the table is read during the
// call. Two writes of a launch must not overlap.
extern "C" cudaError_t LiteRtNvidiaLaunchCacheWrites(
    const LiteRtNvidiaCacheWrite* writes, int32_t num_writes, int32_t source,
    int32_t heads, int32_t cache_len, int32_t depth, int32_t tokens,
    cudaStream_t stream);

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_CACHE_WRITE_SLOTS_H_
