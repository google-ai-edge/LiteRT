/* Copyright 2025-2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_CONFIG_H_

#include <string>
#include <vector>

namespace litert::tensor::examples::gemma3 {

struct Config {
  int vocab_size = 262144;
  int emb_dim = 640;
  int hidden_dim = 2048;
  int head_dim = 256;
  int n_heads = 4;
  int n_layers = 18;
  int n_kv_groups = 1;  // Grouped Query Attention, also only support 1
  int sliding_window = 512;
  float rope_local_base = 10000.0f;
  float rope_global_base = 1000000.0f;
  float query_pre_attn_scalar = 256.0f;
  float rms_norm_eps = 1e-6f;

  // Layer types: every 6th layer is "full_attention", others are
  // "sliding_attention"
  std::vector<std::string> GetLayerTypes() const {
    std::vector<std::string> types;
    for (int i = 0; i < n_layers; ++i) {
      // Pattern: 5 sliding, 1 full (layers 5, 11, 17 are full attention)
      if ((i + 1) % 6 == 0) {
        types.push_back("full_attention");
      } else {
        types.push_back("sliding_attention");
      }
    }
    return types;
  }
};

}  // namespace litert::tensor::examples::gemma3

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_CONFIG_H_
