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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_CONFIG_H_

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::tensor::examples::gemma4 {

enum class ModelVariant { kE2B, kE4B };

inline bool AbslParseFlag(absl::string_view text, ModelVariant* model_variant,
                          std::string* error) {
  if (text == "e2b" || text == "E2B") {
    *model_variant = ModelVariant::kE2B;
    return true;
  }
  if (text == "e4b" || text == "E4B") {
    *model_variant = ModelVariant::kE4B;
    return true;
  }
  *error = "unknown value for ModelVariant enum (expected e2b or e4b)";
  return false;
}

inline std::string AbslUnparseFlag(ModelVariant model_variant) {
  switch (model_variant) {
    case ModelVariant::kE2B:
      return "E2B";
    case ModelVariant::kE4B:
      return "E4B";
  }
  return "<unknown model variant>";
}

// Configuration parameters for the Gemma 4 model.
struct Config {
  int vocab_size;
  int embed_dim;
  int hidden_dim;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int num_layers;
  float final_logit_softcap;
  float rms_norm_eps;
  std::optional<float> attn_logits_soft_cap;
  int sliding_window_size;
  int global_key_size;
  bool use_post_attn_norm;
  bool use_post_ffw_norm;
  int per_layer_input_dim;

  // RoPE parameters
  float local_base_frequency;
  float global_base_frequency;
  float global_rope_proportion;
  float local_rope_proportion;

  // KV cache sharing
  float frac_shared_layers;
  bool share_global;
  bool share_local;

  // Attention pattern
  int attention_pattern_size;

  enum class LayerType { kGlobal, kLocalSliding };

  LayerType GetLayerType(size_t layer_idx) const {
    if (attention_pattern_size <= 0) {
      return LayerType::kGlobal;
    }
    return (layer_idx + 1) % attention_pattern_size ? LayerType::kLocalSliding
                                                    : LayerType::kGlobal;
  }

  // Returns the maximum number of past tokens that layer `layer_idx` attends
  // to, or 0 when it attends to all of them.
  //
  // A local sliding window layer attends to `sliding_window_size` keys: the
  // current token and the last `sliding_window_size - 1` ones.
  int KvCacheWindow(size_t layer_idx) const {
    if (GetLayerType(layer_idx) == LayerType::kLocalSliding &&
        sliding_window_size > 1) {
      return sliding_window_size - 1;
    }
    return 0;
  }

  // Returns the number of past tokens that layer `layer_idx` attends to after
  // `cache_len` tokens.
  int KvCacheLen(size_t layer_idx, int cache_len) const {
    const int window = KvCacheWindow(layer_idx);
    return window > 0 ? std::min(cache_len, window) : cache_len;
  }

  // Returns the number of past tokens that the local sliding window layers
  // attend to after `cache_len` tokens.
  int SlidingWindowKvCacheLen(int cache_len) const {
    return sliding_window_size > 1
               ? std::min(cache_len, sliding_window_size - 1)
               : cache_len;
  }

  static Config E4B() {
    Config c;
    c.vocab_size = 262144;
    c.embed_dim = 2560;
    c.hidden_dim = 10240;
    c.head_dim = 256;
    c.num_heads = 8;
    c.num_kv_heads = 2;
    c.num_layers = 42;
    c.final_logit_softcap = 30.0f;
    c.rms_norm_eps = 1e-6f;
    c.attn_logits_soft_cap = std::nullopt;
    c.sliding_window_size = 512;
    c.global_key_size = 512;
    c.use_post_attn_norm = true;
    c.use_post_ffw_norm = true;
    c.per_layer_input_dim = 256;
    c.local_base_frequency = 10000.0f;
    c.global_base_frequency = 1000000.0f;
    c.global_rope_proportion = 0.25f;
    c.local_rope_proportion = 1.0f;
    c.frac_shared_layers = 18.0f / 42.0f;
    c.share_global = true;
    c.share_local = true;
    c.attention_pattern_size = 6;  // 5 local, 1 global for E4B
    return c;
  }

  static Config E2B() {
    Config c;
    c.vocab_size = 262144;
    c.embed_dim = 1536;
    c.hidden_dim = 6144;
    c.head_dim = 256;
    c.num_heads = 8;
    c.num_kv_heads = 1;
    c.num_layers = 35;
    c.final_logit_softcap = 30.0f;
    c.rms_norm_eps = 1e-6f;
    c.attn_logits_soft_cap = std::nullopt;
    c.sliding_window_size = 512;
    c.global_key_size = 512;
    c.use_post_attn_norm = true;
    c.use_post_ffw_norm = true;
    c.per_layer_input_dim = 256;
    c.local_base_frequency = 10000.0f;
    c.global_base_frequency = 1000000.0f;
    c.global_rope_proportion = 0.25f;
    c.local_rope_proportion = 1.0f;
    c.frac_shared_layers = 20.0f / 35.0f;
    c.share_global = true;
    c.share_local = true;
    c.attention_pattern_size = 5;  // 4 local, 1 global for E2B
    return c;
  }

  static Config From(ModelVariant variant) {
    switch (variant) {
      case ModelVariant::kE2B:
        return E2B();
      case ModelVariant::kE4B:
        return E4B();
    }
    return {};
  }
};

template <class Sink>
void AbslStringify(Sink& sink, Config::LayerType layer_type) {
  switch (layer_type) {
    case Config::LayerType::kGlobal:
      sink.Append("global");
      break;
    case Config::LayerType::kLocalSliding:
      sink.Append("sliding");
      break;
  }
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_CONFIG_H_
