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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_H_

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/examples/gemma3/config.h"
#include "tensor/examples/gemma3/graph_helpers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma3 {

// Inserts a vector of tensors in a map with the given prefix used as a key.
//
// Each tensor's index is appended to the prefix to form a key.
template <class Map, class TensorT>
void InsertTensors(Map& map, std::vector<TensorT>&& tensors,
                   absl::string_view prefix) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    map.emplace(absl::StrCat(prefix, i), tensors[i]);
  }
}

template <class... Mixins>
void InsertTensor(absl::flat_hash_map<std::string, Tensor<Mixins...>>& map,
                  const Tensor<Mixins...>& tensor) {
  map.emplace(tensor.GetName(), tensor);
}

template <class... Mixins>
void InsertTensor(absl::flat_hash_map<std::string, Tensor<Mixins...>>& map,
                  const std::optional<Tensor<Mixins...>>& tensor) {
  if (tensor.has_value()) {
    map.emplace(tensor->GetName(), tensor.value());
  }
}

template <class... Mixins>
struct Inputs {
  Tensor<Mixins...> input_ids = TensorHandle::Invalid();
  Tensor<Mixins...> embedded_input = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_sin = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_sin = TensorHandle::Invalid();
  Tensor<Mixins...> sliding_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> global_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> slice_index = TensorHandle::Invalid();
  std::optional<Tensor<Mixins...>> key_cache_params;
  std::optional<Tensor<Mixins...>> value_cache_params;
  std::optional<Tensor<Mixins...>> cache_params;
  Tensor<Mixins...> slice_offset_1 = TensorHandle::Invalid();
  Tensor<Mixins...> slice_size_1 = TensorHandle::Invalid();
  Tensor<Mixins...> slice_offset_2 = TensorHandle::Invalid();
  Tensor<Mixins...> slice_size_2 = TensorHandle::Invalid();

  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;
  absl::flat_hash_map<std::string, Tensor<Mixins...>> weights;

  void Check() {
    if (!input_ids.GetStatus().ok() && !embedded_input.GetStatus().ok()) {
      LITERT_ABORT_IF_ERROR(input_ids.GetStatus());
    }
    LITERT_ABORT_IF_ERROR(rope_global_cos.GetStatus());
    LITERT_ABORT_IF_ERROR(rope_global_sin.GetStatus());
    LITERT_ABORT_IF_ERROR(rope_local_cos.GetStatus());
    LITERT_ABORT_IF_ERROR(rope_local_sin.GetStatus());
    LITERT_ABORT_IF_ERROR(sliding_attention_mask.GetStatus());
    LITERT_ABORT_IF_ERROR(global_attention_mask.GetStatus());
    LITERT_ABORT_IF_ERROR(slice_index.GetStatus());
    LITERT_ABORT_IF_ERROR(slice_offset_1.GetStatus());
    LITERT_ABORT_IF_ERROR(slice_size_1.GetStatus());
    LITERT_ABORT_IF_ERROR(slice_offset_2.GetStatus());
    LITERT_ABORT_IF_ERROR(slice_size_2.GetStatus());
  }

  absl::flat_hash_map<std::string, Tensor<Mixins...>> tensors() {
    absl::flat_hash_map<std::string, Tensor<Mixins...>> map;
    if (input_ids.GetStatus().ok()) {
      InsertTensor(map, input_ids);
    }
    if (embedded_input.GetStatus().ok()) {
      InsertTensor(map, embedded_input);
    }
    InsertTensor(map, rope_global_cos);
    InsertTensor(map, rope_global_sin);
    InsertTensor(map, rope_local_cos);
    InsertTensor(map, rope_local_sin);
    InsertTensor(map, sliding_attention_mask);
    InsertTensor(map, global_attention_mask);
    InsertTensor(map, slice_index);
    InsertTensor(map, key_cache_params);
    InsertTensor(map, value_cache_params);
    InsertTensor(map, cache_params);
    InsertTensor(map, slice_offset_1);
    InsertTensor(map, slice_size_1);
    InsertTensor(map, slice_offset_2);
    InsertTensor(map, slice_size_2);
    InsertTensors(map, key_caches, "key_cache_");
    InsertTensors(map, value_caches, "value_cache_");
    map.insert(weights.begin(), weights.end());
    return map;
  }
};

// Model outputs.
template <class... Mixins>
struct Outputs {
  Tensor<Mixins...> output;
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;

  absl::flat_hash_map<std::string, Tensor<Mixins...>> tensors() {
    absl::flat_hash_map<std::string, Tensor<Mixins...>> map;
    map.reserve(key_caches.size() + value_caches.size() + 1);
    map.emplace("output", output);
    InsertTensors(map, key_caches, "output_key_cache_");
    InsertTensors(map, value_caches, "output_value_cache_");
    return map;
  }
};

template <class... Mixins>
Outputs<Mixins...> BuildGraph(Inputs<Mixins...>& inputs, const Config& config) {
  // Input shape is [batch, seq_len, emb_dim].
  //
  // Note: For decode, seq_len == 1.

  // Scale embeddings by sqrt(emb_dim) - Gemma3 specific.
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<Mixins...>(
      {.name = "emb_scale_tensor",
       .type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor hidden_states = Mul(inputs.embedded_input, emb_scale_tensor);

  // Get layer types.
  std::vector<std::string> layer_types = config.GetLayerTypes();

  // Prepare updated KV caches.
  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  updated_key_caches.reserve(config.n_layers);
  updated_value_caches.reserve(config.n_layers);

  for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
    std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);
    bool is_sliding = (layer_types[layer_idx] == "sliding_attention");

    Tensor attention_mask = is_sliding ? inputs.sliding_attention_mask
                                       : inputs.global_attention_mask;
    Tensor cos = is_sliding ? inputs.rope_local_cos : inputs.rope_global_cos;
    Tensor sin = is_sliding ? inputs.rope_local_sin : inputs.rope_global_sin;

    Tensor input_norm_scale = GetWeight(
        inputs.weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_input =
        Gemma3RmsNorm(hidden_states, input_norm_scale, config.rms_norm_eps);

    SelfAttentionOutput attn_output = MakeSelfAttentionLayer(
        normed_input, absl::StrCat(layer_prefix, ".self_attn"), config,
        is_sliding, attention_mask, cos, sin, inputs.slice_offset_1,
        inputs.slice_size_1, inputs.slice_offset_2, inputs.slice_size_2,
        layer_idx < inputs.key_caches.size() ? inputs.key_caches[layer_idx]
                                             : Tensor<Mixins...>(),
        layer_idx < inputs.value_caches.size() ? inputs.value_caches[layer_idx]
                                               : Tensor<Mixins...>(),
        inputs.weights, inputs.key_cache_params);

    updated_key_caches.push_back(attn_output.key_cache);
    updated_value_caches.push_back(attn_output.value_cache);

    // Post-attention RMS normalization.
    Tensor post_attn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_attn_output = Gemma3RmsNorm(
        attn_output.output, post_attn_norm_scale, config.rms_norm_eps);

    // Residual connection after attention.
    hidden_states = Add(hidden_states, normed_attn_output);

    // Pre-FFN RMS normalization.
    Tensor pre_ffn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_for_ffn =
        Gemma3RmsNorm(hidden_states, pre_ffn_norm_scale, config.rms_norm_eps);

    // Feed-forward network.
    Tensor ffn_output =
        MakeFeedForwardLayer(normed_for_ffn, absl::StrCat(layer_prefix, ".mlp"),
                             config, inputs.weights);

    // Post-FFN RMS normalization.
    Tensor post_ffn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_ffn_output =
        Gemma3RmsNorm(ffn_output, post_ffn_norm_scale, config.rms_norm_eps);

    // Residual connection after FFN.
    hidden_states = Add(hidden_states, normed_ffn_output);
    hidden_states.SetName(absl::StrCat(layer_prefix, ".new_hidden_states"));
  }

  // Final RMS normalization.
  Tensor final_norm_scale = GetWeight(inputs.weights, "model.norm.weight",
                                      Type::kFP32, {config.emb_dim});
  Tensor final_output =
      Gemma3RmsNorm(hidden_states, final_norm_scale, config.rms_norm_eps);

  // Output head - need embedding table for tied weights.
  Tensor embedding_table =
      GetWeight(inputs.weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});
  Tensor logits = FullyConnected(final_output, embedding_table, kActNone,
                                 /*keep_num_dims=*/true)
                      .SetName("output");

  return {logits, updated_key_caches, updated_value_caches};
}

}  // namespace litert::tensor::examples::gemma3

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_GEMMA3_H_
