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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_GEMMA4_GRAPH_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_GEMMA4_GRAPH_H_

#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/litert/active_graph_context.h"
#include "tensor/examples/gemma4/litert/bundle_matched_ops.h"
#include "tensor/examples/gemma4/litert/transformer.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4::cpu {

template <class... Mixins>
struct Gemma4Inputs {
  // Use the bundle's FP32 to dynamically quantized INT2 head.
  bool bundle_dynamic_qd8_head = false;
  Tensor<Mixins...> embedded_input = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> per_layer_token_embeddings;
  Tensor<Mixins...> global_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> sliding_attention_mask = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_sin = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_sin = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;
  absl::flat_hash_map<std::string, Tensor<Mixins...>> weights;
};

template <class... Mixins>
struct Gemma4Outputs {
  Tensor<Mixins...> logits;
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;

  // Retained graph values become external outputs only when requested below.
  std::vector<Tensor<Mixins...>> layer_outputs;
  Tensor<Mixins...> final_normalized = TensorHandle::Invalid();

  // Returns model outputs, optionally exposing values for numerical diagnosis.
  std::vector<TensorHandle> GetAllHandles(bool include_intermediates = false) {
    std::vector<TensorHandle> output_handles;
    output_handles.reserve(
        1 + key_caches.size() + value_caches.size() +
        (include_intermediates ? layer_outputs.size() + 1 : 0));
    output_handles.push_back(logits);
    output_handles.insert(output_handles.end(), key_caches.begin(),
                          key_caches.end());
    output_handles.insert(output_handles.end(), value_caches.begin(),
                          value_caches.end());
    if (include_intermediates) {
      output_handles.insert(output_handles.end(), layer_outputs.begin(),
                            layer_outputs.end());
      output_handles.push_back(final_normalized);
    }
    return output_handles;
  }
};

inline std::vector<int> GetKvCacheSharingPatterns(const Config& config) {
  std::vector<int> patterns(config.num_layers);
  std::iota(patterns.begin(), patterns.end(), 0);

  const int num_unshared_layers = static_cast<int>(
      config.num_layers - config.frac_shared_layers * config.num_layers);
  for (int i = num_unshared_layers; i < config.num_layers; ++i) {
    if (config.share_local &&
        config.GetLayerType(i) == Config::LayerType::kLocalSliding) {
      patterns[i] = num_unshared_layers > 2 ? num_unshared_layers - 2 : 0;
    } else if (config.share_global &&
               config.GetLayerType(i) == Config::LayerType::kGlobal) {
      patterns[i] = num_unshared_layers > 1 ? num_unshared_layers - 1 : 0;
    }
  }
  return patterns;
}

template <class... Mixins>
Gemma4Outputs<Mixins...> BuildGemma4Graph(const Gemma4Inputs<Mixins...>& inputs,
                                          const Config& config,
                                          ActiveGraphContext& context) {
  auto* active_graph_context = &context;
  ABSL_VLOG(4) << "Building Gemma4 Graph, seq_len="
               << inputs.embedded_input.GetShape()[1];

  Tensor emb_scale_tensor =
      Tensor<Mixins...>({.type = Type::kFP32,
                         .shape = {1},
                         .buffer = std::sqrt(config.embed_dim)});
  Tensor hidden_states = Mul(inputs.embedded_input, emb_scale_tensor);
  Tensor eps_tensor = Tensor<Mixins...>(
      {.type = Type::kFP32, .shape = {1}, .buffer = config.rms_norm_eps});

  // The bundle stores one quantized global PLE matrix. It consumes the
  // sqrt(embed_dim)-scaled embedding, unlike the source-CT factorized path.
  // Compute it once before layer0 changes hidden_states, then split its last
  // dimension into one contiguous projection for each decoder layer.
  std::vector<Tensor<Mixins...>> bundle_projection_parts;
  auto full_projection =
      inputs.weights.find("model.per_layer_model_projection.weight");
  if (full_projection != inputs.weights.end()) {
    bundle_projection_parts = BundlePerLayerProjectionParts(
        hidden_states, full_projection->second, inputs.weights,
        config.num_layers, config.per_layer_input_dim);
  }

  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  std::vector<Tensor<Mixins...>> layer_outputs;
  updated_key_caches.reserve(config.num_layers);
  updated_value_caches.reserve(config.num_layers);
  layer_outputs.reserve(config.num_layers);

  std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config);
  std::vector<Tensor<Mixins...>> computed_keys(config.num_layers);
  std::vector<Tensor<Mixins...>> computed_values(config.num_layers);

  if (active_graph_context) {
    active_graph_context->initial_hidden = hidden_states;
    active_graph_context->layers.resize(config.num_layers);
    active_graph_context->ple_outputs.resize(config.num_layers);
  }

  for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
    if (active_graph_context) {
      active_graph_context->layer = layer_idx;

      active_graph_context->layers[layer_idx].hidden_input = hidden_states;
    }

    ABSL_VLOG(4) << "  Layer " << layer_idx << " ("
                 << config.GetLayerType(layer_idx) << ")";

    const int shared_idx = sharing_patterns[layer_idx];
    const bool is_shared = (shared_idx != layer_idx);

    Tensor key_cache = layer_idx < inputs.key_caches.size()
                           ? inputs.key_caches[layer_idx]
                           : Tensor<Mixins...>();
    Tensor value_cache = layer_idx < inputs.value_caches.size()
                             ? inputs.value_caches[layer_idx]
                             : Tensor<Mixins...>();

    Tensor<Mixins...> shared_key = TensorHandle::Invalid();
    Tensor<Mixins...> shared_value = TensorHandle::Invalid();

    if (is_shared) {
      shared_key = computed_keys[shared_idx];
      shared_value = computed_values[shared_idx];
    }

    const bool is_global =
        config.GetLayerType(layer_idx) == Config::LayerType::kGlobal;
    Tensor attention_mask = is_global ? inputs.global_attention_mask
                                      : inputs.sliding_attention_mask;
    Tensor cos = is_global ? inputs.rope_global_cos : inputs.rope_local_cos;
    Tensor sin = is_global ? inputs.rope_global_sin : inputs.rope_local_sin;

    Tensor<Mixins...> per_layer_input;
    if (config.per_layer_input_dim > 0 &&
        layer_idx < inputs.per_layer_token_embeddings.size() &&
        inputs.per_layer_token_embeddings[layer_idx].GetStatus().ok()) {
      Tensor<Mixins...> proj_out;
      if (!bundle_projection_parts.empty()) {
        proj_out = bundle_projection_parts[layer_idx];
      } else {
        Tensor proj_w = GetWeight(
            inputs.weights,
            absl::StrCat("model.layers.", layer_idx,
                         ".per_layer_model_projection.weight"),
            Type::kFP32, {config.per_layer_input_dim, config.embed_dim});
        proj_out = MobileFullyConnected(inputs.embedded_input, proj_w,
                                        &inputs.weights);
      }

      Tensor norm_w =
          GetWeight(inputs.weights, "model.per_layer_projection_norm.weight",
                    Type::kFP32, {config.per_layer_input_dim});
      Tensor normed_proj = PrimitiveRmsNorm(proj_out, norm_w, eps_tensor);

      float sqrt_per_layer_dim =
          std::sqrt(static_cast<float>(config.per_layer_input_dim));
      Tensor scaled_tok_emb =
          Mul(inputs.per_layer_token_embeddings[layer_idx], sqrt_per_layer_dim);

      Tensor sum_emb = Add(normed_proj, scaled_tok_emb);

      float rsqrt_2 = 1.0f / std::sqrt(2.0f);
      per_layer_input = Mul(sum_emb, rsqrt_2);
    }

    if (active_graph_context) {
      active_graph_context->ple_outputs[layer_idx] = per_layer_input;

      active_graph_context->layers[layer_idx].ple_input = per_layer_input;
    }

    TransformerLayerOutput<Mixins...> layer_out = TransformerLayer(
        hidden_states, attention_mask, cos, sin, key_cache, value_cache,
        per_layer_input, shared_key, shared_value, config, inputs.weights,
        layer_idx, eps_tensor, active_graph_context);

    hidden_states = layer_out.output;
    if (active_graph_context)
      active_graph_context->layers[layer_idx].output = hidden_states;
    layer_outputs.push_back(hidden_states);

    computed_keys[layer_idx] = layer_out.key_for_attn;
    computed_values[layer_idx] = layer_out.value_for_attn;

    updated_key_caches.push_back(layer_out.key_cache);
    updated_value_caches.push_back(layer_out.value_cache);
  }

  if (active_graph_context) {
    active_graph_context->final_hidden = hidden_states;
  }
  // Final RMSNorm
  Tensor final_norm_scale = GetWeight(inputs.weights, "model.norm.weight",
                                      Type::kFP32, {config.embed_dim});
  Tensor final_output =
      PrimitiveRmsNorm(hidden_states, final_norm_scale, eps_tensor);

  // Mobile CT can provide a separate head even when the original model ties
  // these weights. Preserve the checkpoint head when present.
  Tensor embedding_table =
      GetWeight(inputs.weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.embed_dim});
  auto head_it = inputs.weights.find("lm_head.weight");
  Tensor head =
      head_it == inputs.weights.end() ? embedding_table : head_it->second;
  Tensor logits =
      inputs.bundle_dynamic_qd8_head
          ? FullyConnected(final_output, head,
                           std::optional<Tensor<Mixins...>>{}, kActNone, true,
                           true)
          : MobileFullyConnected(final_output, head, &inputs.weights);

  // Logits Soft Capping
  if (config.final_logit_softcap > 0.0f) {
    Tensor soft_cap_tensor =
        Tensor<Mixins...>({.type = Type::kFP32,
                           .shape = {1},
                           .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                               {config.final_logit_softcap})});
    Tensor<Mixins...> scaled_logits;
    if (inputs.bundle_dynamic_qd8_head) {
      // Preserve the bundle's multiply by its stored FP32 reciprocal; division
      // by the cap rounds differently even when both denote 1 / 30 in real
      // arithmetic. This constant is supplied by the bundle loader.
      auto reciprocal =
          inputs.weights.find("matched.logits_softcap_reciprocal");
      if (reciprocal == inputs.weights.end()) {
        scaled_logits = Tensor<Mixins...>(graph::ErrorTensor(
            absl::InvalidArgumentError("Missing bundle softcap reciprocal")));
      } else {
        auto scale_status = ReadMobileActivationScale(reciprocal->second);
        if (!scale_status.ok()) {
          scaled_logits =
              Tensor<Mixins...>(graph::ErrorTensor(scale_status.status()));
        } else {
          scaled_logits = Mul(logits, reciprocal->second);
        }
      }
    } else {
      scaled_logits = Div(logits, soft_cap_tensor);
    }
    Tensor capped_logits = Tanh(scaled_logits);
    logits = Mul(capped_logits, soft_cap_tensor);
  }

  return {logits, updated_key_caches, updated_value_caches, layer_outputs,
          final_output};
}

}  // namespace litert::tensor::examples::gemma4::cpu

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_GEMMA4_GRAPH_H_
