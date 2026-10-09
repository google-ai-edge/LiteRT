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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_TRANSFORMER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_TRANSFORMER_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/active_graph_context.h"
#include "tensor/examples/experimental/gemma4/attention.h"
#include "tensor/examples/experimental/gemma4/feed_forward_network.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Output tensors produced by a single Gemma4 decoder layer.
template <class... Mixins>
struct TransformerLayerOutput {
  Tensor<Mixins...> output = TensorHandle::Invalid();
  Tensor<Mixins...> key_cache = TensorHandle::Invalid();
  Tensor<Mixins...> value_cache = TensorHandle::Invalid();
};

// Authors one Gemma4 decoder layer (pre/post-norm self-attention, GeGLU FFN,
// optional Per-Layer Embeddings residual, and layer scalar). Pass
// `TensorHandle::Invalid()` for `per_layer_input` when PLE is absent; a
// non-Invalid error tensor or scalar on `per_layer_input` is propagated as an
// error tensor rather than silently skipping PLE.
template <class... Mixins>
TransformerLayerOutput<Mixins...> TransformerLayer(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& cos,
    const Tensor<Mixins...>& sin, const Tensor<Mixins...>& key_cache,
    const Tensor<Mixins...>& value_cache,
    const Tensor<Mixins...>& per_layer_input,
    const examples::gemma4::Config& config,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    int layer_idx, const Tensor<Mixins...>& eps_tensor,
    ActiveGraphContext& context) {
  const std::string layer_prefix = absl::StrCat("model.layers.", layer_idx);

  const Tensor<Mixins...> pre_attn_norm_scale = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".input_layernorm.weight"),
      Type::kFP32, {config.embed_dim});
  const Tensor<Mixins...> normed_input =
      PrimitiveRmsNorm(input, pre_attn_norm_scale, eps_tensor);

  const AttentionOutput<Mixins...> attn_out =
      Attention(normed_input, cos, sin, key_cache, value_cache, config, weights,
                absl::StrCat(layer_prefix, ".self_attn"),
                config.GetLayerType(layer_idx), eps_tensor, context);
  if (!attn_out.output.GetStatus().ok()) {
    return {attn_out.output, attn_out.key_cache, attn_out.value_cache};
  }

  Tensor<Mixins...> attn_output = attn_out.output;
  if (config.use_post_attn_norm) {
    const Tensor<Mixins...> post_attn_norm_scale = GetWeightOrPlaceholder(
        weights, absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.embed_dim});
    attn_output =
        PrimitiveRmsNorm(attn_output, post_attn_norm_scale, eps_tensor);
  }

  const Tensor<Mixins...> attn_residual = Add(attn_output, input);

  const Tensor<Mixins...> pre_ffn_norm_scale = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
      Type::kFP32, {config.embed_dim});
  const Tensor<Mixins...> normed_attn_output =
      PrimitiveRmsNorm(attn_residual, pre_ffn_norm_scale, eps_tensor);

  const Tensor<Mixins...> gate_proj = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".mlp.gate_proj.weight"), Type::kFP32,
      {config.hidden_dim, config.embed_dim});
  const Tensor<Mixins...> up_proj = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".mlp.up_proj.weight"), Type::kFP32,
      {config.hidden_dim, config.embed_dim});
  const Tensor<Mixins...> down_proj = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".mlp.down_proj.weight"), Type::kFP32,
      {config.embed_dim, config.hidden_dim});
  Tensor<Mixins...> ffn_output = FeedForwardNetwork(
      normed_attn_output, gate_proj, up_proj, down_proj, &weights);
  if (!ffn_output.GetStatus().ok()) {
    return {ffn_output, attn_out.key_cache, attn_out.value_cache};
  }

  if (config.use_post_ffw_norm) {
    const Tensor<Mixins...> post_ffn_norm_scale = GetWeightOrPlaceholder(
        weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.embed_dim});
    ffn_output = PrimitiveRmsNorm(ffn_output, post_ffn_norm_scale, eps_tensor);
  }

  Tensor<Mixins...> ffn_residual = Add(ffn_output, attn_residual);

  // Per-Layer Embeddings (PLE) residual specific to Gemma 4:
  // https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/gemma4#per-layer-embeddings-ple
  if (config.per_layer_input_dim > 0 &&
      per_layer_input != TensorHandle::Invalid()) {
    if (!per_layer_input.GetStatus().ok()) {
      return {per_layer_input, attn_out.key_cache, attn_out.value_cache};
    }
    if (per_layer_input.GetShape().empty()) {
      return {Tensor<Mixins...>(absl::InvalidArgumentError(
                  "Per-layer embedding input must have rank >= 1")),
              attn_out.key_cache, attn_out.value_cache};
    }
    const Tensor<Mixins...> per_layer_input_gate = GetWeightOrPlaceholder(
        weights, absl::StrCat(layer_prefix, ".per_layer_input_gate.weight"),
        Type::kFP32, {config.per_layer_input_dim, config.embed_dim});
    const Tensor<Mixins...> per_layer_projection = GetWeightOrPlaceholder(
        weights, absl::StrCat(layer_prefix, ".per_layer_projection.weight"),
        Type::kFP32, {config.embed_dim, config.per_layer_input_dim});
    const Tensor<Mixins...> post_per_layer_input_norm = GetWeightOrPlaceholder(
        weights,
        absl::StrCat(layer_prefix, ".post_per_layer_input_norm.weight"),
        Type::kFP32, {config.embed_dim});

    const Tensor<Mixins...> gate_val =
        MobileFullyConnected(ffn_residual, per_layer_input_gate, &weights);
    if (!gate_val.GetStatus().ok()) {
      return {gate_val, attn_out.key_cache, attn_out.value_cache};
    }
    const Tensor<Mixins...> gated =
        Mul(Gelu(gate_val, /*approximate=*/true), per_layer_input);
    const Tensor<Mixins...> projected =
        MobileFullyConnected(gated, per_layer_projection, &weights);
    if (!projected.GetStatus().ok()) {
      return {projected, attn_out.key_cache, attn_out.value_cache};
    }
    const Tensor<Mixins...> normed_projected =
        PrimitiveRmsNorm(projected, post_per_layer_input_norm, eps_tensor);
    ffn_residual = Add(ffn_residual, normed_projected);
  }

  const Tensor<Mixins...> layer_scalar = GetWeightOrPlaceholder(
      weights, absl::StrCat(layer_prefix, ".layer_scalar"), Type::kFP32, {1});
  const Tensor<Mixins...> output = Mul(ffn_residual, layer_scalar);

  return {output, attn_out.key_cache, attn_out.value_cache};
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_TRANSFORMER_H_
