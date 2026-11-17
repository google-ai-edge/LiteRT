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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"

namespace litert::nvidia {

// The composites over quantized KV caches as a TensorRT plugin: INT8 queries
// over INT8 caches or INT4 caches (LiteRtNvidiaQuantAttentionParams).
//
// The caches are INT8 tensors that hold the model's bytes: a key cache
// [1, H, S, D] and a value cache [1, H, D, S], or for INT4 caches the packed
// bytes declared [1, H, S, D / 2] and [1, H, D / 2, S]. The float tensors of
// a layer are all HALF or all BF16. A float input the graph quantizes (the
// query, the keys and values a step adds) is the tensor before its QUANTIZE,
// or its dequantized value: the plugin rounds it as TFLite's QUANTIZE does
// (trtllm/quant_cache.h). The scores a mask hides stay out of the softmax,
// and a query row that sees no key gets zero.
//
// An updated cache is written into the plugin's output buffer. When that is
// the cache input's buffer (LiteRT-LM binds a cache's input and output to
// one buffer) only the new entries move; otherwise the cache is copied first.
enum class QuantAttentionMode : int32_t {
  // odml.cache_update. Inputs: the key and the value cache, the keys
  // [1, H, n, D], the values [1, H, n, D] (or [1, n, H, D] with
  // v_token_major), and the start indices of the two caches (INT32 [4]:
  // elements 2 and 3 are the positions). Outputs: the updated caches.
  kCacheUpdate = 0,
  // odml.sdpa_transposed. Inputs: the query [1, H, R, D], the key and the
  // value cache, and the mask [1, 1, M, S] bool, M a divisor of R: query row
  // r reads mask row r % M. Up to 16 rows run the decode kernel, more the
  // tensor-core kernel (R and M multiples of 128, S of 16). Output: the
  // attention [1, H, R, D] in the query's type.
  kAttention = 1,
  // odml.fused_sdpa_cache_update over INT8 ring caches. Inputs: the query
  // [1, H, R, D], the key and the value cache, the mask [1, 1, M, S + M]
  // bool, the chunk's keys [1, H, M, D] and values ([1, H, M, D], or
  // [1, M, H, D] with v_token_major), and the cache parameters (INT32
  // [1, 1, 1, 7]). Outputs: the attention over the cache followed by the
  // chunk's keys, and the caches with the chunk written at its positions
  // modulo S.
  kFusedAttentionUpdate = 2,
};

struct QuantAttentionOptions {
  QuantAttentionMode mode = QuantAttentionMode::kAttention;
  // q_scale is not used by kCacheUpdate.
  LiteRtNvidiaQuantAttentionParams quant = {1.0f, 1.0f, 1.0f, 8};
  bool v_token_major = false;
  // kFusedAttentionUpdate: false leaves the caches as they are.
  bool update_cache = true;
};

nvinfer1::IPluginV3* CreateQuantAttentionPlugin(
    const QuantAttentionOptions& options) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureQuantAttentionPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QUANT_ATTENTION_PLUGIN_H_
