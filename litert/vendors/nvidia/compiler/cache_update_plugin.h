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

#ifndef LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_PLUGIN_H_
#define LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_PLUGIN_H_

#include "NvInferRuntime.h"

namespace litert::nvidia {

// HALF inputs: K cache [1,H,S,D], V cache [1,H,S,D] when transposed,
// otherwise flattened [1,H*D,S,1], then K/V updates [1,H,U,D]. The fifth
// input is INT32 [1,1,1,7]: params[0] is the write offset and params[3] the
// valid update length. Outputs are separate contiguous K/V patches in the
// cache layouts with S replaced by P (ring: P=S, linear: P=U).
// A following native KVCacheUpdate writes these patches into the original
// caches at zero for ring updates, or clamp(params[0],0,S-U) for linear ones.
// This plugin never mutates inputs and needs no aliasing preview feature.
// Without forward_read, optional FLOAT/HALF/BF16 linear inputs after the first
// five are opaque execution-order dependencies. Pass completed old-cache reader
// outputs so the following native writer cannot overwrite caches before those
// reads. Their ranks are unrestricted and the kernel never reads their values.
// With forward_read, exactly one additional input (index 5) is copied bitwise
// to output 2 in the same kernel launch. Subsequent attention consumers can
// use that output to depend on patch preparation; its shape and type match.
nvinfer1::IPluginV3* CreateCacheUpdatePlugin(
    bool ring_buffer, bool transposed_value_cache,
    bool forward_read = false) noexcept;

void EnsureCacheUpdatePluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // LITERT_VENDORS_NVIDIA_COMPILER_CACHE_UPDATE_PLUGIN_H_
