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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_DECODE_ATTENTION_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_DECODE_ATTENTION_PLUGIN_H_

#include "NvInferRuntime.h"

namespace litert::nvidia {

// Fused decode attention over full KV caches:
//   inputs  q [1, H, rows, D] (FP16 or BF16), k [1, H, S, D] FP16,
//           v [1, H, S, D] FP16, mask [1, 1, 1 or rows, S] bool
//   output  softmax(select(mask, q @ k^T, fill)) @ v as [1, H, rows, D] in
//           q's type.
// rows <= 16 and D in {128, 256, 512}; the products accumulate in FP32.
nvinfer1::IPluginV3* CreateDecodeAttentionPlugin(float fill) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureDecodeAttentionPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_DECODE_ATTENTION_PLUGIN_H_
