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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// A TensorRT plugin that writes a step's rows into an INT8 per-tensor KV
// cache without copying the rest of it. Inputs: [0] the cache (INT8,
// [1, H, S, D], or [1, H, D, S] with `value_columns`), [1] the new rows
// (HALF, BF16 or FLOAT, [1, H, T, D]), [2] the row position (INT32 [1]).
// Output: the updated cache, the same shape and type as input 0. The plugin
// quantizes the rows with `scale` and `zero_point` and writes them into its
// output buffer; when that buffer is the cache input's (LiteRT-LM binds a
// cache's input and output to one buffer) nothing else moves, otherwise the
// cache is copied first.
nvinfer1::IPluginV3* CreateInt8CacheWritePlugin(bool value_columns, float scale,
                                                int32_t zero_point) noexcept;

void EnsureInt8CacheWritePluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_INT8_CACHE_WRITE_PLUGIN_H_
