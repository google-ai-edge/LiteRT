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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GENERAL_ATTENTION_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GENERAL_ATTENTION_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// The composites over KV caches in the forms the quantized attention plugin
// (quant_attention_plugin.h) does not take, as one TensorRT plugin over the
// kernels of trtllm/general_attention.h.
//
// A cache crosses the engine as the model stores it: INT8, UINT8 and packed
// INT4 caches as INT8 tensors holding their bytes (INT4: a key cache
// [1, H, S, D / 2], a value cache [1, H, D / 2, S]), float caches in their
// float type. Float inputs are HALF, BF16 or FLOAT; masks BOOL or FLOAT.
// Updated caches are written into the output buffers: in place when the
// runtime binds a cache's input and output to one buffer, otherwise after a
// copy of the cache.
enum class GeneralAttentionMode : int32_t {
  // odml.sdpa_transposed. Inputs: the query [1, H, R, D] (the query heads of
  // a KV head as its rows), the key and the value cache, the mask
  // [1, 1, M, S], M a divisor of R. Output: the attention [1, H, R, D] in the
  // query's type.
  kAttention = 0,
  // odml.fused_sdpa_cache_update. Inputs: kAttention's with the mask
  // [1, 1, T, S + T], then the chunk's keys [1, H, T, D], its values
  // ([1, H, T, D], or [1, T, H, D] with v_token_major) and the cache
  // parameters (INT32, [0] the position of the first token, [1] the position
  // after the last). Outputs: the attention over the caches followed by the
  // chunk, then, with update_cache, the caches with the chunk written into
  // their ring.
  kFusedAttention = 1,
  // odml.cache_update. Inputs: the key and the value cache, the keys
  // [1, H, n, D], the values ([1, H, n, D], or [1, n, H, D] with
  // v_token_major), and the start indices of the two caches (INT32 [4]:
  // element 2 of the keys', element 3 of the values'). Outputs: the caches.
  kCacheUpdate = 2,
};

struct GeneralAttentionOptions {
  GeneralAttentionMode mode = GeneralAttentionMode::kAttention;
  // LiteRtNvidiaGeneralType of the caches, with their scales when quantized.
  int32_t cache_type = 0;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  // The INT8 scales the graph quantizes the query and the step's keys and
  // values with before the composite reads them, or 0.
  float q_int8_scale = 0.0f;
  float new_int8_scale = 0.0f;
  float scale = 1.0f;
  float softcap = 0.0f;
  bool v_token_major = false;
  bool update_cache = true;
};

nvinfer1::IPluginV3* CreateGeneralAttentionPlugin(
    const GeneralAttentionOptions& options) noexcept;

// Referenced so that the plugin creator's registration stays linked.
void EnsureGeneralAttentionPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_GENERAL_ATTENTION_PLUGIN_H_
