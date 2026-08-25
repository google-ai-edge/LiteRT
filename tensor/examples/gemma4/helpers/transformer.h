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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_TRANSFORMER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_TRANSFORMER_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/helpers/attention.h"
#include "tensor/examples/gemma4/helpers/feed_forward_network.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

template <class... Mixins>
struct TransformerLayerOutput {
  Tensor<Mixins...> output;
  Tensor<Mixins...> key_cache;
  Tensor<Mixins...> value_cache;
  Tensor<Mixins...> key_for_attn;
  Tensor<Mixins...> value_for_attn;
};

template <class... Mixins>
TransformerLayerOutput<Mixins...> TransformerLayer(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& attention_mask,
    const Tensor<Mixins...>& cos, const Tensor<Mixins...>& sin,
    const Tensor<Mixins...>& key_cache, const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& per_layer_input,
    const Tensor<Mixins...>& shared_key, const Tensor<Mixins...>& shared_value,
    const Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    int layer_idx, const Tensor<Mixins...>& eps_tensor) {
  std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);

  Tensor pre_attn_norm_scale =
      GetWeight(weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
                Type::kFP32, {config.embed_dim});
  Tensor normed_input = RmsNorm(input, pre_attn_norm_scale, eps_tensor);

  AttentionOutput<Mixins...> attn_out = Attention(
      normed_input, attention_mask, cos, sin, key_cache, value_cache,
      shared_key, shared_value, config, weights,
      absl::StrCat(layer_prefix, ".self_attn"),
      config.GetLayerType(layer_idx) == Config::LayerType::kGlobal, eps_tensor);

  Tensor attn_output = attn_out.output;

  if (config.use_post_attn_norm) {
    Tensor post_attn_norm_scale = GetWeight(
        weights, absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.embed_dim});
    attn_output = RmsNorm(attn_output, post_attn_norm_scale, eps_tensor);
  }

  Tensor attn_residual = Add(attn_output, input);

  Tensor pre_ffn_norm_scale = GetWeight(
      weights, absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
      Type::kFP32, {config.embed_dim});
  Tensor normed_attn_output =
      RmsNorm(attn_residual, pre_ffn_norm_scale, eps_tensor);

  Tensor gate_proj =
      GetWeight(weights, absl::StrCat(layer_prefix, ".mlp.gate_proj.weight"),
                Type::kFP32, {config.hidden_dim, config.embed_dim});
  Tensor up_proj =
      GetWeight(weights, absl::StrCat(layer_prefix, ".mlp.up_proj.weight"),
                Type::kFP32, {config.hidden_dim, config.embed_dim});
  Tensor down_proj =
      GetWeight(weights, absl::StrCat(layer_prefix, ".mlp.down_proj.weight"),
                Type::kFP32, {config.embed_dim, config.hidden_dim});
  Tensor ffn_output =
      FeedForwardNetwork(normed_attn_output, gate_proj, up_proj, down_proj);

  if (config.use_post_ffw_norm) {
    Tensor post_ffn_norm_scale = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.embed_dim});
    ffn_output = RmsNorm(ffn_output, post_ffn_norm_scale, eps_tensor);
  }

  Tensor ffn_residual = Add(ffn_output, attn_residual);

  // This is specific to Gemma 4. More info can be found here.
  // https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/gemma4#per-layer-embeddings-ple
  if (config.per_layer_input_dim > 0 && per_layer_input.GetStatus().ok() &&
      !per_layer_input.GetShape().empty()) {
    Tensor per_layer_input_gate = GetWeight(
        weights, absl::StrCat(layer_prefix, ".per_layer_input_gate.weight"),
        Type::kFP32, {config.per_layer_input_dim, config.embed_dim});

    Tensor per_layer_projection = GetWeight(
        weights, absl::StrCat(layer_prefix, ".per_layer_projection.weight"),
        Type::kFP32, {config.embed_dim, config.per_layer_input_dim});

    Tensor post_per_layer_input_norm = GetWeight(
        weights,
        absl::StrCat(layer_prefix, ".post_per_layer_input_norm.weight"),
        Type::kFP32, {config.embed_dim});

    Tensor gate_val = FullyConnected(ffn_residual, per_layer_input_gate);
    Tensor gated = Mul(Gelu(gate_val, /*approximate=*/true), per_layer_input);
    Tensor projected = FullyConnected(gated, per_layer_projection);
    Tensor normed_projected =
        RmsNorm(projected, post_per_layer_input_norm, eps_tensor);
    ffn_residual = Add(ffn_residual, normed_projected);
  }

  Tensor layer_scalar = GetWeight(
      weights, absl::StrCat(layer_prefix, ".layer_scalar"), Type::kFP32, {1});
  Tensor output = Mul(ffn_residual, layer_scalar);

  return {output, attn_out.key_cache, attn_out.value_cache,
          attn_out.key_for_attn, attn_out.value_for_attn};
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_TRANSFORMER_H_
