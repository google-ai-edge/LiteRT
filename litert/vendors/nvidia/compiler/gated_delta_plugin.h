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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GATED_DELTA_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GATED_DELTA_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/gated_delta_net.h"

namespace litert::nvidia {

// Qwen 3.5's gated_delta_update custom op as one plugin
// (trtllm/gated_delta.h):
//   inputs  q, k FP32 [1, key_heads, n, key_dim], v FP32 [1, heads, n,
//           value_dim], beta, g FP32 [1, heads, n], state FP32 [1, heads,
//           key_dim, value_dim], then INT32 valid length (a scalar) when
//           has_valid_len
//   outputs out FP32 [1, heads, n, value_dim], state' FP32 like state
// The runtime binds the state's input and output to one buffer; the kernel
// updates it in place safely.
nvinfer1::IPluginV3* CreateGatedDeltaUpdatePlugin(int32_t heads,
                                                  int32_t key_heads,
                                                  int32_t key_dim,
                                                  int32_t value_dim,
                                                  bool has_valid_len) noexcept;

// The odml.short_conv_step composite as one plugin:
//   inputs  x FP32 [1, n, channels], state FP32 [1, channels, taps - 1],
//           weight FP32 [channels, 1, taps], then INT32 valid length (a
//           scalar) when has_valid_len
//   outputs y FP32 [1, n, channels], state' FP32 like state
nvinfer1::IPluginV3* CreateShortConvPlugin(int32_t channels, int32_t taps,
                                           bool silu,
                                           bool has_valid_len) noexcept;

// A whole gated DeltaNet layer between its in-projections and its output
// projection (trtllm/gated_delta_net.h; params.tokens is the runtime's):
//   inputs  proj BF16 [.., n, proj_stride], conv state FP32 [1, C, taps - 1],
//           state FP32 [1, value_heads, head_dim, head_dim], the layer's FP32
//           constants, then INT32 valid length (a scalar) when has_valid_len
//   outputs BF16 [.., n, value_heads * head_dim], conv state' and state'
// The runtime binds each state's input and output to one buffer.
nvinfer1::IPluginV3* CreateGatedDeltaNetPlugin(
    const LiteRtNvidiaGatedDeltaNetParams& params, bool has_valid_len) noexcept;

// Referenced by the dispatch library so the creators' registration objects
// are retained when linking the shared library used for engine
// deserialization.
void EnsureGatedDeltaPluginsRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GATED_DELTA_PLUGIN_H_
