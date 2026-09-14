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

#ifndef LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_KERNEL_H_
#define LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

namespace litert::nvidia {

// Prepares an out-of-place contiguous patch, copying HALF bits unchanged.
// Ring patches contain cache_rows rows at native write offset zero. Linear
// patches contain update_rows rows at clamp(params[0], 0, cache_rows-update_rows).
// Invalid parameters (negative write or length outside [0, update_rows]) and
// zero length produce the unchanged old patch. Cache inputs are never mutated.
// V cache/patch storage is BHSD when transposed, otherwise flattened BHDS;
// both fresh updates are BHSD. Launching never synchronizes the supplied stream.
// An optional even-sized forward buffer is copied bitwise in that same launch.
// Forward output and patches must not overlap each other or any read buffer.
cudaError_t LaunchCacheUpdatePatch(
    const void* cache_k, const void* cache_v, const void* update_k,
    const void* update_v, const int32_t* params, int32_t heads,
    int32_t cache_rows, int32_t update_rows, int32_t depth, bool ring_buffer,
    bool transposed_value_cache, void* patch_k, void* patch_v,
    cudaStream_t stream, const void* forward_input = nullptr,
    void* forward_output = nullptr, size_t forward_bytes = 0);

}  // namespace litert::nvidia

#endif  // LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_KERNEL_H_
