/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_CONFIG_H_

#include <string>
#include <vector>

namespace litert::tensor::examples {

struct Gemma3Config {
  int vocab_size = 262144;
  int emb_dim = 640;
  int hidden_dim = 2048;
  int head_dim = 256;
  int n_heads = 4;
  int n_layers = 18;
  int n_kv_groups = 1;
  int sliding_window = 512;
  int sliding_window_pattern = 6;
  float rope_local_base = 10000.0f;
  float rope_global_base = 1000000.0f;
  float query_pre_attn_scalar = 256.0f;
  float rms_norm_eps = 1e-6f;
  bool use_matmul_with_cache = false;
  bool bypass_lm_head = false;

  // Layer types: every `sliding_window_pattern` layer is "full_attention",
  // others are "sliding_attention".
  std::vector<std::string> GetLayerTypes() const {
    std::vector<std::string> types;
    types.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
      if (sliding_window_pattern > 0 && (i + 1) % sliding_window_pattern == 0) {
        types.push_back("full_attention");
      } else {
        types.push_back("sliding_attention");
      }
    }
    return types;
  }
};

// Default configuration for Gemma3 270M.
inline Gemma3Config GetGemma3_270M_Config() { return Gemma3Config{}; }

inline Gemma3Config GetGemma3_1B_Config() {
  Gemma3Config config;
  config.vocab_size = 262144;
  config.emb_dim = 1152;
  config.hidden_dim = 6912;
  config.head_dim = 256;
  config.n_heads = 4;
  config.n_layers = 26;
  config.n_kv_groups = 1;
  config.sliding_window = 512;
  config.sliding_window_pattern = 6;
  config.rope_local_base = 10000.0f;
  config.rope_global_base = 1000000.0f;
  config.query_pre_attn_scalar = 256.0f;
  config.rms_norm_eps = 1e-6f;
  return config;
}

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_CONFIG_H_
