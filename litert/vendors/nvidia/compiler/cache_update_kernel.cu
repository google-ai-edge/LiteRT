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

#include "litert/vendors/nvidia/compiler/cache_update_kernel.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "cuda_runtime.h"

namespace litert::nvidia {
namespace {

__global__ void CacheUpdatePatchKernel(
    const uint16_t* cache_k, const uint16_t* cache_v, const uint16_t* update_k,
    const uint16_t* update_v, const int32_t* params, int32_t cache_rows,
    int32_t update_rows, int32_t depth, int32_t patch_rows, int64_t elements,
    bool ring_buffer, bool transposed_value_cache, uint16_t* patch_k,
    uint16_t* patch_v, const uint16_t* forward_input, uint16_t* forward_output,
    int64_t forward_elements) {
  const int32_t write = params[0];
  const int32_t valid_rows = params[3];
  const bool valid = write >= 0 && valid_rows >= 0 && valid_rows <= update_rows;
  const int32_t last_base = cache_rows - update_rows;
  int32_t base = 0;
  if (!ring_buffer && write > 0) {
    base = write > last_base ? last_base : write;
  }
  const int32_t ring_write = valid && ring_buffer ? write % cache_rows : 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int32_t column = i % depth;
    const int64_t head_and_row = i / depth;
    const int32_t row = head_and_row % patch_rows;
    const int64_t destination = static_cast<int64_t>(base) + row;
    int64_t source_row = -1;
    if (valid) {
      if (ring_buffer) {
        source_row = destination - ring_write;
        if (source_row < 0) source_row += cache_rows;
      } else {
        source_row = destination - write;
      }
    }
    const int64_t head = head_and_row / patch_rows;
    const int64_t k_index =
        (head * cache_rows + destination) * depth + column;
    const int64_t v_index =
        transposed_value_cache
            ? k_index
            : (head * depth + column) * cache_rows + destination;
    const int64_t v_patch_index = transposed_value_cache
                                      ? i
                                      : (head * depth + column) * patch_rows + row;
    if (source_row >= 0 && source_row < valid_rows) {
      const int64_t source = (head * update_rows + source_row) * depth + column;
      patch_k[i] = update_k[source];
      patch_v[v_patch_index] = update_v[source];
    } else {
      patch_k[i] = cache_k[k_index];
      patch_v[v_patch_index] = cache_v[v_index];
    }
  }
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < forward_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // Copy representation bits, including NaNs and signed zero, without any
    // FLOAT/HALF/BF16 arithmetic or a second launch.
    forward_output[i] = forward_input[i];
  }
}

bool Overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  const auto a_start = reinterpret_cast<uintptr_t>(a);
  const auto b_start = reinterpret_cast<uintptr_t>(b);
  return a_start <= b_start ? b_start - a_start < a_bytes
                            : a_start - b_start < b_bytes;
}

}  // namespace

cudaError_t LaunchCacheUpdatePatch(
    const void* cache_k, const void* cache_v, const void* update_k,
    const void* update_v, const int32_t* params, int32_t heads,
    int32_t cache_rows, int32_t update_rows, int32_t depth, bool ring_buffer,
    bool transposed_value_cache, void* patch_k, void* patch_v,
    cudaStream_t stream, const void* forward_input, void* forward_output,
    size_t forward_bytes) {
  if (update_k == nullptr || update_v == nullptr || params == nullptr ||
      cache_k == nullptr || cache_v == nullptr || patch_k == nullptr ||
      patch_v == nullptr || patch_k == patch_v || heads <= 0 ||
      cache_rows <= 0 || update_rows <= 0 || update_rows > cache_rows ||
      depth <= 0 ||
      static_cast<int64_t>(heads) * cache_rows >
          std::numeric_limits<int64_t>::max() / sizeof(uint16_t) / depth) {
    return cudaErrorInvalidValue;
  }
  const void* inputs[] = {cache_k, cache_v, update_k, update_v, params};
  for (const void* input : inputs) {
    if (patch_k == input || patch_v == input) return cudaErrorInvalidValue;
  }
  const int32_t patch_rows = ring_buffer ? cache_rows : update_rows;
  const int64_t elements = static_cast<int64_t>(heads) * patch_rows * depth;
  if (forward_bytes == 0) {
    if (forward_input != nullptr || forward_output != nullptr) {
      return cudaErrorInvalidValue;
    }
  } else {
    if (forward_input == nullptr || forward_output == nullptr ||
        forward_bytes % sizeof(uint16_t) != 0 ||
        forward_bytes > static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max()) ||
        reinterpret_cast<uintptr_t>(forward_input) % alignof(uint16_t) != 0 ||
        reinterpret_cast<uintptr_t>(forward_output) % alignof(uint16_t) != 0) {
      return cudaErrorInvalidValue;
    }
    const size_t cache_bytes =
        static_cast<size_t>(heads) * cache_rows * depth * sizeof(uint16_t);
    const size_t update_bytes =
        static_cast<size_t>(heads) * update_rows * depth * sizeof(uint16_t);
    const size_t patch_bytes = static_cast<size_t>(elements) * sizeof(uint16_t);
    const void* read_ptrs[] = {
        cache_k, cache_v, update_k, update_v, params, forward_input};
    const size_t read_bytes[] = {
        cache_bytes, cache_bytes, update_bytes, update_bytes,
        7 * sizeof(int32_t), forward_bytes};
    const void* write_ptrs[] = {patch_k, patch_v, forward_output};
    const size_t write_bytes[] = {patch_bytes, patch_bytes, forward_bytes};
    for (int o = 0; o < 3; ++o) {
      if (reinterpret_cast<uintptr_t>(write_ptrs[o]) >
          std::numeric_limits<uintptr_t>::max() - write_bytes[o]) {
        return cudaErrorInvalidValue;
      }
      for (int p = 0; p < o; ++p) {
        if (Overlaps(write_ptrs[o], write_bytes[o],
                     write_ptrs[p], write_bytes[p])) {
          return cudaErrorInvalidValue;
        }
      }
      for (int i = 0; i < 6; ++i) {
        if (reinterpret_cast<uintptr_t>(read_ptrs[i]) >
                std::numeric_limits<uintptr_t>::max() - read_bytes[i] ||
            Overlaps(write_ptrs[o], write_bytes[o], read_ptrs[i], read_bytes[i])) {
          return cudaErrorInvalidValue;
        }
      }
    }
  }
  const int64_t forward_elements = forward_bytes / sizeof(uint16_t);
  const int64_t work_elements = std::max(elements, forward_elements);
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((work_elements + kThreads - 1) / kThreads, 65535));
  CacheUpdatePatchKernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const uint16_t*>(cache_k),
      static_cast<const uint16_t*>(cache_v),
      static_cast<const uint16_t*>(update_k),
      static_cast<const uint16_t*>(update_v), params, cache_rows, update_rows,
      depth, patch_rows, elements, ring_buffer, transposed_value_cache,
      static_cast<uint16_t*>(patch_k), static_cast<uint16_t*>(patch_v),
      static_cast<const uint16_t*>(forward_input),
      static_cast<uint16_t*>(forward_output), forward_elements);
  return cudaGetLastError();
}

}  // namespace litert::nvidia
