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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_GEMMA4_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_GEMMA4_GRAPH_H_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/active_graph_context.h"
#include "tensor/examples/experimental/gemma4/bundle_matched_ops.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/experimental/gemma4/transformer.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Graph input tensors and weight map consumed by `BuildGemma4Graph`.
template <class... Mixins>
struct Gemma4Inputs {
  // Use the bundle's FP32-to-dynamically-quantized INT2 vocabulary head and
  // reciprocal softcap constant (`matched.logits_softcap_reciprocal`).
  bool bundle_dynamic_qd8_head = false;
  Tensor<Mixins...> embedded_input = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> per_layer_token_embeddings;
  Tensor<Mixins...> rope_global_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_global_sin = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_cos = TensorHandle::Invalid();
  Tensor<Mixins...> rope_local_sin = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;
  absl::flat_hash_map<std::string, Tensor<Mixins...>> weights;
};

// Graph output tensors returned by `BuildGemma4Graph`.
template <class... Mixins>
struct Gemma4Outputs {
  Tensor<Mixins...> logits = TensorHandle::Invalid();
  std::vector<Tensor<Mixins...>> key_caches;
  std::vector<Tensor<Mixins...>> value_caches;
};

// Authors the full Gemma4 CPU decoder graph (embedding scaling, optional global
// PLE projection, all transformer layers, final RMSNorm, and soft-capped
// logits) and populates `context` with external KV and mask boundary handles.
template <class... Mixins>
Gemma4Outputs<Mixins...> BuildGemma4Graph(
    const Gemma4Inputs<Mixins...>& inputs,
    const examples::gemma4::Config& config, ActiveGraphContext& context) {
  const auto fail = [](absl::Status status) -> Gemma4Outputs<Mixins...> {
    return {.logits = Tensor<Mixins...>(std::move(status))};
  };
  if (config.num_layers <= 0) {
    return fail(
        absl::InvalidArgumentError("Gemma4 config requires num_layers > 0"));
  }
  if (!inputs.embedded_input.GetStatus().ok()) {
    return fail(inputs.embedded_input.GetStatus());
  }
  if (inputs.embedded_input.GetShape().size() >= 2) {
    ABSL_VLOG(4) << "Building Gemma4 Graph, seq_len="
                 << inputs.embedded_input.GetShape()[1];
  }

  const Tensor<Mixins...> emb_scale_tensor(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = std::sqrt(static_cast<float>(config.embed_dim))});
  Tensor<Mixins...> hidden_states =
      Mul(inputs.embedded_input, emb_scale_tensor);
  const Tensor<Mixins...> eps_tensor(
      {.type = Type::kFP32, .shape = {1}, .buffer = config.rms_norm_eps});

  // The bundle stores one quantized global PLE matrix. It consumes the
  // sqrt(embed_dim)-scaled embedding, unlike the source-CT factorized path.
  // Compute it once before layer 0 changes hidden_states, then split its last
  // dimension into one contiguous projection for each decoder layer.
  std::vector<Tensor<Mixins...>> bundle_projection_parts;
  const auto full_projection =
      inputs.weights.find("model.per_layer_model_projection.weight");
  if (full_projection != inputs.weights.end()) {
    bundle_projection_parts = BundlePerLayerProjectionParts(
        hidden_states, full_projection->second, inputs.weights,
        config.num_layers, config.per_layer_input_dim,
        context.dynamic_prefill_rows ? PleRowMode::kDynamic
                                     : PleRowMode::kStatic);
  }

  std::vector<Tensor<Mixins...>> updated_key_caches;
  std::vector<Tensor<Mixins...>> updated_value_caches;
  updated_key_caches.reserve(config.num_layers);
  updated_value_caches.reserve(config.num_layers);

  if (context.owners.empty()) {
    context.owners = examples::gemma4::GetKvCacheSharingPatterns(config);
  }
  context.layers.resize(config.num_layers);

  // Hoist loop-invariant PLE scaling scalars so the FlatBuffer stores one
  // constant tensor per scalar instead of one per layer.
  Tensor<Mixins...> sqrt_per_layer_dim_tensor = TensorHandle::Invalid();
  Tensor<Mixins...> rsqrt_2_tensor = TensorHandle::Invalid();
  Tensor<Mixins...> ple_norm_w = TensorHandle::Invalid();
  if (config.per_layer_input_dim > 0) {
    sqrt_per_layer_dim_tensor = Tensor<Mixins...>(
        {.type = Type::kFP32,
         .shape = {1},
         .buffer = std::sqrt(static_cast<float>(config.per_layer_input_dim))});
    rsqrt_2_tensor = Tensor<Mixins...>(
        {.type = Type::kFP32, .shape = {1}, .buffer = 1.0f / std::sqrt(2.0f)});
    ple_norm_w = GetWeightOrPlaceholder(
        inputs.weights, "model.per_layer_projection_norm.weight", Type::kFP32,
        {config.per_layer_input_dim});
  }

  for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
    context.layer = layer_idx;
    ABSL_VLOG(4) << "  Layer " << layer_idx << " ("
                 << config.GetLayerType(layer_idx) << ")";

    const Tensor<Mixins...> key_cache =
        layer_idx < static_cast<int>(inputs.key_caches.size())
            ? inputs.key_caches[layer_idx]
            : Tensor<Mixins...>(TensorHandle::Invalid());
    const Tensor<Mixins...> value_cache =
        layer_idx < static_cast<int>(inputs.value_caches.size())
            ? inputs.value_caches[layer_idx]
            : Tensor<Mixins...>(TensorHandle::Invalid());

    const bool is_global = config.GetLayerType(layer_idx) ==
                           examples::gemma4::Config::LayerType::kGlobal;
    const Tensor<Mixins...>& cos =
        is_global ? inputs.rope_global_cos : inputs.rope_local_cos;
    const Tensor<Mixins...>& sin =
        is_global ? inputs.rope_global_sin : inputs.rope_local_sin;

    Tensor<Mixins...> per_layer_input = TensorHandle::Invalid();
    if (config.per_layer_input_dim > 0 &&
        layer_idx <
            static_cast<int>(inputs.per_layer_token_embeddings.size()) &&
        inputs.per_layer_token_embeddings[layer_idx] !=
            TensorHandle::Invalid()) {
      const Tensor<Mixins...>& tok_emb =
          inputs.per_layer_token_embeddings[layer_idx];
      if (!tok_emb.GetStatus().ok()) {
        per_layer_input = tok_emb;
      } else {
        Tensor<Mixins...> proj_out = TensorHandle::Invalid();
        if (!bundle_projection_parts.empty()) {
          proj_out = bundle_projection_parts[layer_idx];
        } else {
          const Tensor<Mixins...> proj_w = GetWeightOrPlaceholder(
              inputs.weights,
              absl::StrCat("model.layers.", layer_idx,
                           ".per_layer_model_projection.weight"),
              Type::kFP32, {config.per_layer_input_dim, config.embed_dim});
          proj_out = MobileFullyConnected(inputs.embedded_input, proj_w,
                                          &inputs.weights);
        }
        const Tensor<Mixins...> normed_proj =
            PrimitiveRmsNorm(proj_out, ple_norm_w, eps_tensor);
        if (!normed_proj.GetStatus().ok()) {
          per_layer_input = normed_proj;
        } else {
          const Tensor<Mixins...> scaled_tok_emb =
              Mul(tok_emb, sqrt_per_layer_dim_tensor);
          per_layer_input =
              Mul(Add(normed_proj, scaled_tok_emb), rsqrt_2_tensor);
        }
      }
    }

    const TransformerLayerOutput<Mixins...> layer_out = TransformerLayer(
        hidden_states, cos, sin, key_cache, value_cache, per_layer_input,
        config, inputs.weights, layer_idx, eps_tensor, context);

    hidden_states = layer_out.output;
    updated_key_caches.push_back(layer_out.key_cache);
    updated_value_caches.push_back(layer_out.value_cache);
    if (!hidden_states.GetStatus().ok()) {
      return {hidden_states, updated_key_caches, updated_value_caches};
    }
  }

  // Final RMSNorm.
  const Tensor<Mixins...> final_norm_scale = GetWeightOrPlaceholder(
      inputs.weights, "model.norm.weight", Type::kFP32, {config.embed_dim});
  const Tensor<Mixins...> final_output =
      PrimitiveRmsNorm(hidden_states, final_norm_scale, eps_tensor);
  if (!final_output.GetStatus().ok()) {
    return {final_output, updated_key_caches, updated_value_caches};
  }

  // Mobile CT can provide a separate head even when the original model ties
  // these weights. Preserve the checkpoint head when present without creating
  // an unused embedding placeholder.
  const auto head_it = inputs.weights.find("lm_head.weight");
  const Tensor<Mixins...> head =
      head_it != inputs.weights.end()
          ? head_it->second
          : GetWeightOrPlaceholder(inputs.weights, "model.embed_tokens.weight",
                                   Type::kFP32,
                                   {config.vocab_size, config.embed_dim});
  if (!head.GetStatus().ok()) {
    return {head, updated_key_caches, updated_value_caches};
  }
  Tensor<Mixins...> logits =
      inputs.bundle_dynamic_qd8_head
          ? FullyConnected(final_output, head, kActNone,
                           /*keep_num_dims=*/true,
                           /*asymmetric_quantize_inputs=*/true)
          : MobileFullyConnected(final_output, head, &inputs.weights);
  if (!logits.GetStatus().ok()) {
    return {logits, updated_key_caches, updated_value_caches};
  }

  // Logits Soft Capping.
  if (config.final_logit_softcap > 0.0f) {
    const Tensor<Mixins...> soft_cap_tensor(
        {.type = Type::kFP32,
         .shape = {1},
         .buffer = config.final_logit_softcap});
    Tensor<Mixins...> scaled_logits = TensorHandle::Invalid();
    if (inputs.bundle_dynamic_qd8_head) {
      // Preserve the bundle's multiply by its stored FP32 reciprocal; division
      // by the cap rounds differently even when both denote 1 / 30 in real
      // arithmetic. This constant is supplied by the bundle loader.
      const auto reciprocal =
          inputs.weights.find("matched.logits_softcap_reciprocal");
      if (reciprocal == inputs.weights.end()) {
        return fail(
            absl::InvalidArgumentError("Missing bundle softcap reciprocal"));
      }
      if (absl::StatusOr<float> scale =
              ReadMobileActivationScale(reciprocal->second);
          !scale.ok()) {
        return fail(scale.status());
      }
      scaled_logits = Mul(logits, reciprocal->second);
    } else {
      scaled_logits = Div(logits, soft_cap_tensor);
    }
    logits = Mul(Tanh(scaled_logits), soft_cap_tensor);
  }

  return {logits, updated_key_caches, updated_value_caches};
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_GEMMA4_GRAPH_H_
