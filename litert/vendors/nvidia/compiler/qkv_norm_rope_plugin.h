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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QKV_NORM_ROPE_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QKV_NORM_ROPE_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// Gemma 4's odml.qkv_norm_rope composite as one plugin
// (trtllm/qkv_norm_rope.h): the per-head RMS norms and the rotary embedding
// of the fused qkv projection.
//   inputs  qkv BF16 [1, tokens, (heads + 2 kv_heads) * depth],
//           positions INT32 (tokens values), q_scale FP32 [depth],
//           k_scale FP32 [depth]
//   outputs q BF16 [1, heads, tokens, depth], k and v BF16
//           [1, kv_heads, tokens, depth]
nvinfer1::IPluginV3* CreateQkvNormRopePlugin(int32_t heads, int32_t kv_heads,
                                             int32_t depth, int32_t rotary,
                                             bool has_v_norm, float epsilon,
                                             float min_timescale,
                                             float max_timescale) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureQkvNormRopePluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QKV_NORM_ROPE_PLUGIN_H_
