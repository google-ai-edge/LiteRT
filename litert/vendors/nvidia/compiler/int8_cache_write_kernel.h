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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

namespace litert::nvidia {

// The element type of the rows being written.
enum class Int8CacheWriteSource : int32_t { kHalf = 0, kBf16 = 1, kFloat = 2 };

// Writes `tokens` rows of `update` ([1, heads, tokens, depth], float type
// `source`) into the INT8 cache `cache_out` at the row position `*position`
// (clamped to [0, cache_len - tokens]), quantizing them as TensorRT's
// per-tensor quantize layer would (clamp to the representable range, round
// to nearest even, add the zero point). Key caches are [1, heads, cache_len,
// depth]; with `value_columns` the cache is [1, heads, depth, cache_len] and
// the rows become columns. The rest of the cache is left as it is when
// `cache_out` is `cache_in` (the in-place binding LiteRT-LM uses); otherwise
// `cache_bytes` are copied from `cache_in` first.
cudaError_t LaunchInt8CacheWrite(
    void* cache_out, const void* cache_in, size_t cache_bytes,
    const void* update, Int8CacheWriteSource source, const int32_t* position,
    int32_t heads, int32_t cache_len, int32_t depth, int32_t tokens,
    bool value_columns, float scale, int32_t zero_point, cudaStream_t stream);

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_KERNEL_H_
