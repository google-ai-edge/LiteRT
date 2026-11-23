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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QSA_ATTENTION_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QSA_ATTENTION_PLUGIN_H_

#include <cstdint>

#include "NvInferRuntime.h"

namespace litert::nvidia {

// odml.qsa_indexer_update as one plugin (trtllm/qsa_indexer.h):
//   inputs  q_idx [1, S, H_idx, D_idx] and k_idx_new [1, S, D_idx] (FP32,
//           FP16 or BF16, both alike), ring [1, 4, D_idx] and pool
//           [1, max_blocks, D_idx] (FP32 or FP16, both alike), k_norm_weight
//           [D_idx], block_cos and block_sin [1, max_blocks, rotary_dim]
//           (FP32), start_pos (INT32, one value), mask [1, 1, S, mask_len]
//           (BOOL, or FP32 additive)
//   outputs topk [1, S, block_topk] INT32, ring' and pool' like ring and pool
// The runtime binds ring and ring', pool and pool' to one buffer each, and the
// plugin updates them in place; given separate buffers it first copies the
// state into the outputs.
nvinfer1::IPluginV3* CreateQsaIndexerPlugin(int32_t block_topk,
                                            float rms_norm_eps,
                                            float scale) noexcept;

// odml.sparse_sdpa_transposed as one plugin (trtllm/qsa_sparse_attention.h):
//   inputs  q [1, H_q, S, D] (FP32, FP16 or BF16), k [1, H_kv, L, D] (FP16
//           or FP32), v of k's type, [1, H_kv, L, D] or, with value_columns,
//           [1, H_kv, D, L], topk [1, S, block_topk] INT32, mask [1, 1, S, L]
//           (BOOL, or FP32 where 0 sees the key)
//   output  [1, S, H_q, D] in q's type
nvinfer1::IPluginV3* CreateQsaAttentionPlugin(float scale,
                                              bool value_columns) noexcept;

// Referenced by the dispatch library so the creators' registration objects
// are retained when linking the shared library used for engine
// deserialization.
void EnsureQsaPluginsRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_QSA_ATTENTION_PLUGIN_H_
