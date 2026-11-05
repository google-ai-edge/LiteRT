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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/options/litert_gpu_options.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/arithmetic_graph.h"
#include "third_party/odml/litert/tensor/examples/gemma3/graph_helpers.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/backends/tflite/arithmetic_tflite.h"
#include "third_party/odml/litert/tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/examples/gemma3/config.h"
#include "third_party/odml/litert/tensor/examples/gemma3/gemma3.h"
#include "third_party/odml/litert/tensor/examples/gemma3/safetensor_loader.h"
#include "third_party/odml/litert/tensor/examples/gemma3/tokenizer.h"
#include "third_party/odml/litert/tensor/examples/gemma3/util.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/utils/macros.h"

constexpr absl::string_view kPrefill = "prefill";
constexpr absl::string_view kDecode = "decode";
constexpr size_t kMaxSeqLen = 512;

ABSL_FLAG(std::string, weights_path, "",
          "Path to the safetensors weights file");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer.model file");
ABSL_FLAG(std::string, prompt, "Hello, world!",
          "Input prompt for text generation");
ABSL_FLAG(size_t, max_tokens, 50, "Maximum number of tokens to generate");
ABSL_FLAG(std::string, accelerator, "gpu",
          "Hardware accelerator to use: cpu or gpu");
ABSL_FLAG(::litert::tensor::examples::SafetensorLoader::QuantizedLoadMode,
          weight_mode,
          ::litert::tensor::examples::SafetensorLoader::QuantizedLoadMode::
              kPreserveQuantized,
          "Weight mode (quantized or float)");

namespace litert::tensor::examples::gemma3 {

struct GpuAttnOutput {
  Tensor<TfLiteMixinTag> output;
  Tensor<TfLiteMixinTag> key_cache;
  Tensor<TfLiteMixinTag> value_cache;
};

GpuAttnOutput MakeGpuSelfAttentionLayer(
    const Tensor<TfLiteMixinTag>& input, const std::string& name,
    const Config& config, bool is_sliding_attention,
    const Tensor<TfLiteMixinTag>& attention_mask,
    const Tensor<TfLiteMixinTag>& cos, const Tensor<TfLiteMixinTag>& sin,
    const Tensor<TfLiteMixinTag>& key_cache,
    const Tensor<TfLiteMixinTag>& value_cache,
    const absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>& weights,
    const std::optional<Tensor<TfLiteMixinTag>> cache_params) {
  int qkv_out_dim = config.n_heads * config.head_dim;
  int kv_out_dim = config.n_kv_groups * config.head_dim;

  Tensor q_weight = GetWeight(weights, absl::StrCat(name, ".q_proj.weight"),
                              Type::kFP32, {qkv_out_dim, config.emb_dim});

  Tensor q = FullyConnected(input, q_weight);
  Tensor k = FullyConnected(
      input, GetWeight(weights, absl::StrCat(name, ".k_proj.weight"),
                       Type::kFP32, {kv_out_dim, config.emb_dim}));
  Tensor v = FullyConnected(
      input, GetWeight(weights, absl::StrCat(name, ".v_proj.weight"),
                       Type::kFP32, {kv_out_dim, config.emb_dim}));

  Tensor o_proj = GetWeight(weights, absl::StrCat(name, ".o_proj.weight"),
                            Type::kFP32, {config.emb_dim, qkv_out_dim});

  Tensor q_norm = GetWeight(weights, absl::StrCat(name, ".q_norm.weight"),
                            Type::kFP32, {config.head_dim});
  Tensor k_norm = GetWeight(weights, absl::StrCat(name, ".k_norm.weight"),
                            Type::kFP32, {config.head_dim});

  const Shape& input_shape = input.GetShape();
  int batch_size = input_shape.size() == 3 ? input_shape[0] : 1;
  int seq_len = input_shape.size() == 3 ? input_shape[1] : input_shape[0];

  q = Reshape(q, {batch_size, seq_len, config.n_heads, config.head_dim});
  q = Transpose(q, {0, 2, 1, 3});

  k = Reshape(k, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  k = Transpose(k, {0, 2, 1, 3});
  v = Reshape(v, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  v = Transpose(v, {0, 2, 1, 3});

  q = Gemma3RmsNorm(q, q_norm, config.rms_norm_eps);
  k = Gemma3RmsNorm(k, k_norm, config.rms_norm_eps);

  if (!cache_params.has_value()) {
    // Hardcode RoPE cos/sin in the graph for Prefill!
    size_t rope_elements = 1 * 1 * seq_len * config.head_dim;
    auto cos_buffer = ::litert::tensor::OwningCpuBuffer::Allocate<
        ::litert::tensor::Type::kFP32>(rope_elements);
    auto sin_buffer = ::litert::tensor::OwningCpuBuffer::Allocate<
        ::litert::tensor::Type::kFP32>(rope_elements);

    {
      auto cos_lock = cos_buffer->LockMutable();
      auto cos_span = std::move(cos_lock).As<float>();
      auto sin_lock = sin_buffer->LockMutable();
      auto sin_span = std::move(sin_lock).As<float>();

      float rope_base = is_sliding_attention ? config.rope_local_base
                                             : config.rope_global_base;
      for (int p = 0; p < seq_len; ++p) {
        ::litert::tensor::examples::gemma3::RopeCosSin(
            p, 1, rope_base,
            absl::MakeSpan(cos_span.data() + p * config.head_dim,
                           config.head_dim),
            absl::MakeSpan(sin_span.data() + p * config.head_dim,
                           config.head_dim));
      }
    }

    Tensor<TfLiteMixinTag> hard_cos({.name = absl::StrCat(name, ".hard_cos"),
                                     .type = ::litert::tensor::Type::kFP32,
                                     .shape = {1, 1, seq_len, config.head_dim},
                                     .buffer = cos_buffer});
    Tensor<TfLiteMixinTag> hard_sin({.name = absl::StrCat(name, ".hard_sin"),
                                     .type = ::litert::tensor::Type::kFP32,
                                     .shape = {1, 1, seq_len, config.head_dim},
                                     .buffer = sin_buffer});

    q = ApplyRotaryEmbedding(q, hard_cos, hard_sin);
    k = ApplyRotaryEmbedding(k, hard_cos, hard_sin);
  } else {
    // Use dynamic runtime cos/sin inputs for Decode steps!
    q = ApplyRotaryEmbedding(q, cos, sin);
    k = ApplyRotaryEmbedding(k, cos, sin);
  }

  k.SetName(absl::StrCat(name, ".new_k"));
  v.SetName(absl::StrCat(name, ".new_v"));

  Tensor<TfLiteMixinTag> updated_key_cache = k;
  Tensor<TfLiteMixinTag> updated_value_cache = v;
  Tensor<TfLiteMixinTag> k_for_attn = k;
  Tensor<TfLiteMixinTag> v_for_attn = v;

  if (cache_params.has_value()) {
    updated_key_cache = DynamicUpdateSlice(key_cache, k, *cache_params);
    updated_value_cache = DynamicUpdateSlice(value_cache, v, *cache_params);
    k_for_attn = updated_key_cache;
    v_for_attn = updated_value_cache;
  }
  updated_key_cache.SetName(absl::StrCat(name, ".updated_key_cache"));
  updated_value_cache.SetName(absl::StrCat(name, ".updated_value_cache"));

  int num_groups = config.n_heads / config.n_kv_groups;
  if (num_groups > 1) {
    std::vector<Tensor<TfLiteMixinTag>> tiled_k_heads;
    for (int g = 0; g < config.n_kv_groups; ++g) {
      Tensor head_idx = Tensor<TfLiteMixinTag>(
          {.type = Type::kI32,
           .shape = {1},
           .buffer = OwningCpuBuffer::Copy<Type::kI32>({g})});
      Tensor head = Gather(k_for_attn, head_idx, /*axis=*/1);
      Tensor tiled_head = Tile(head, {1, num_groups, 1, 1});
      tiled_k_heads.push_back(tiled_head);
    }
    k_for_attn = Concatenation(absl::MakeSpan(tiled_k_heads), /*axis=*/1);

    std::vector<Tensor<TfLiteMixinTag>> tiled_v_heads;
    for (int g = 0; g < config.n_kv_groups; ++g) {
      Tensor head_idx = Tensor<TfLiteMixinTag>(
          {.type = Type::kI32,
           .shape = {1},
           .buffer = OwningCpuBuffer::Copy<Type::kI32>({g})});
      Tensor head = Gather(v_for_attn, head_idx, /*axis=*/1);
      Tensor tiled_head = Tile(head, {1, num_groups, 1, 1});
      tiled_v_heads.push_back(tiled_head);
    }
    v_for_attn = Concatenation(absl::MakeSpan(tiled_v_heads), /*axis=*/1);
  }

  Tensor scores = BatchMatMul(q, k_for_attn, /*adj_x=*/false, /*adj_y=*/true);
  float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  Tensor scale_tensor = Tensor<TfLiteMixinTag>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({scale})});
  scores = Mul(scores, scale_tensor);
  if (!cache_params.has_value()) {
    // Compiling a frozen Constant Causal Mask Tensor directly inside the
    // Flatbuffer graph for Prefill!
    size_t mask_elements = 1 * config.n_heads * seq_len * seq_len;
    auto mask_cpu_buffer =
        ::litert::tensor::OwningCpuBuffer::Allocate<Type::kFP32>(mask_elements);
    auto mask_lock = mask_cpu_buffer->LockMutable();
    float* mask_raw_ptr = std::move(mask_lock).As<float>().data();

    for (int h = 0; h < config.n_heads; ++h) {
      for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
          size_t flat_idx = h * seq_len * seq_len + i * seq_len + j;
          if (j <= i) {
            mask_raw_ptr[flat_idx] = 0.0f;
          } else {
            mask_raw_ptr[flat_idx] = -10000.0f;
          }
        }
      }
    }

    Tensor constant_causal_mask = Tensor<TfLiteMixinTag>(
        {.name = absl::StrCat(name, ".constant_causal_mask"),
         .type = Type::kFP32,
         .shape = {1, static_cast<int>(config.n_heads),
                   static_cast<int>(seq_len), static_cast<int>(seq_len)},
         .buffer = std::move(mask_cpu_buffer)});

    scores = Add(scores, constant_causal_mask);
  } else {
    // Fall back to standard dynamic placeholder inputs for Decode generation
    // token steps!
    scores = Add(scores, attention_mask);
  }
  Tensor attn_weights = Softmax(scores);
  attn_weights.SetName(absl::StrCat(name, ".attn_weights"));
  Tensor attn_output = BatchMatMul(attn_weights, v_for_attn);

  attn_output = Transpose(attn_output, {0, 2, 1, 3});
  Tensor attn_output_flat =
      Reshape(attn_output, {batch_size, seq_len, qkv_out_dim});
  attn_output_flat.SetName(absl::StrCat(name, ".attn_output_flat"));

  Tensor output =
      FullyConnected(attn_output_flat, o_proj).SetName("attn_output");

  GpuAttnOutput attn_out;
  attn_out.output = output;
  attn_out.key_cache = updated_key_cache;
  attn_out.value_cache = updated_value_cache;
  return attn_out;
}

struct GpuOutputs {
  Tensor<TfLiteMixinTag> output;
  std::vector<Tensor<TfLiteMixinTag>> key_caches;
  std::vector<Tensor<TfLiteMixinTag>> value_caches;
};

GpuOutputs BuildGpuGraph(Inputs<TfLiteMixinTag>& inputs, const Config& config,
                         bool is_decode) {
  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<TfLiteMixinTag>(
      {.name = "emb_scale_tensor",
       .type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor hidden_states = Mul(inputs.embedded_input, emb_scale_tensor);

  std::vector<std::string> layer_types = config.GetLayerTypes();
  std::vector<Tensor<TfLiteMixinTag>> updated_key_caches;
  std::vector<Tensor<TfLiteMixinTag>> updated_value_caches;
  updated_key_caches.reserve(config.n_layers);
  updated_value_caches.reserve(config.n_layers);

  GpuOutputs gpu_outputs;

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

    auto attn_output = MakeGpuSelfAttentionLayer(
        normed_input, absl::StrCat(layer_prefix, ".self_attn"), config,
        is_sliding, attention_mask, cos, sin,
        layer_idx < inputs.key_caches.size() ? inputs.key_caches[layer_idx]
                                             : Tensor<TfLiteMixinTag>(),
        layer_idx < inputs.value_caches.size() ? inputs.value_caches[layer_idx]
                                               : Tensor<TfLiteMixinTag>(),
        inputs.weights, inputs.cache_params);

    updated_key_caches.push_back(attn_output.key_cache);
    updated_value_caches.push_back(attn_output.value_cache);

    Tensor post_attn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".post_attention_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_attn_output = Gemma3RmsNorm(
        attn_output.output, post_attn_norm_scale, config.rms_norm_eps);
    hidden_states = Add(hidden_states, normed_attn_output);

    Tensor pre_ffn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".pre_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_for_ffn =
        Gemma3RmsNorm(hidden_states, pre_ffn_norm_scale, config.rms_norm_eps);

    Tensor ffn_output =
        MakeFeedForwardLayer(normed_for_ffn, absl::StrCat(layer_prefix, ".mlp"),
                             config, inputs.weights);

    Tensor post_ffn_norm_scale = GetWeight(
        inputs.weights,
        absl::StrCat(layer_prefix, ".post_feedforward_layernorm.weight"),
        Type::kFP32, {config.emb_dim});
    Tensor normed_ffn_output =
        Gemma3RmsNorm(ffn_output, post_ffn_norm_scale, config.rms_norm_eps);

    hidden_states = Add(hidden_states, normed_ffn_output);
  }

  Tensor final_norm_scale = GetWeight(inputs.weights, "model.norm.weight",
                                      Type::kFP32, {config.emb_dim});
  Tensor final_output =
      Gemma3RmsNorm(hidden_states, final_norm_scale, config.rms_norm_eps);

  Tensor embedding_table =
      GetWeight(inputs.weights, "model.embed_tokens.weight", Type::kFP32,
                {config.vocab_size, config.emb_dim});

  if (!is_decode) {
    final_output = Gather(final_output, inputs.slice_index, /*axis=*/1);
  }
  Tensor final_output_4d =
      Reshape(final_output, {1, 1, 1, static_cast<int>(config.emb_dim)});

  Tensor logits_4d = FullyConnected(final_output_4d, embedding_table, kActNone,
                                    /*keep_num_dims=*/true);
  Tensor logits =
      Reshape(logits_4d, {1, 1, static_cast<int>(config.vocab_size)});
  Tensor output_token_id = ArgMax(logits, /*axis=*/-1, Type::kI32);
  output_token_id.SetName("output");
  gpu_outputs.output = output_token_id;

  gpu_outputs.key_caches = updated_key_caches;
  gpu_outputs.value_caches = updated_value_caches;

  return gpu_outputs;
}

}  // namespace litert::tensor::examples::gemma3

absl::Status SetInputOptional(::litert::tensor::LitertDynamicRunner& runner,
                              const std::string& signature,
                              const std::string& name,
                              absl::Span<const uint8_t> data) {
  auto status = runner.SetInput(signature, name, data);
  if (status.code() == absl::StatusCode::kNotFound) {
    // Silenced to avoid device pipe flooding!
    return absl::OkStatus();
  }
  return status;
}

absl::Status RunGemma3Inference(
    const std::string& weights_path, const std::string& tokenizer_path,
    const std::string& prompt_text, size_t max_tokens,
    ::litert::tensor::examples::SafetensorLoader::QuantizedLoadMode
        weight_mode) {
  ::litert::tensor::examples::gemma3::Config config;
  config.vocab_size = 262144;

  absl::flat_hash_map<std::string, ::litert::tensor::TensorHandle> weights;
  std::vector<int32_t> input_tokens;
  absl::StatusOr<::litert::tensor::examples::GemmaTokenizerSP> tokenizer_or;

  tokenizer_or =
      ::litert::tensor::examples::GemmaTokenizerSP::Load(tokenizer_path);
  if (!tokenizer_or.ok())
    return absl::InternalError("Failed to load tokenizer!");

  auto loader_res =
      ::litert::tensor::examples::SafetensorLoader::Load(weights_path);
  if (!loader_res.ok()) {
    return absl::InternalError("Failed to load safetensors file!");
  }
  auto loader = std::move(*loader_res);
  auto weights_res = loader.LoadAllTensors(weight_mode);
  if (!weights_res.ok()) {
    return absl::InternalError("Failed to load tensors from safetensors!");
  }
  weights = std::move(*weights_res);
  input_tokens = (*tokenizer_or).Encode(prompt_text);

  auto locked_emb_table = weights.find("model.embed_tokens.weight")
                              ->second.GetBuffer()
                              .value()
                              .Lock()
                              .As<const float>();

  int raw_seq_len = static_cast<int>(input_tokens.size());
  int padded_seq_len = 1;
  while (padded_seq_len < raw_seq_len) padded_seq_len *= 2;
  while (input_tokens.size() < padded_seq_len) {
    input_tokens.push_back(0);
  }
  const int seq_len = padded_seq_len;

  if (seq_len >= kMaxSeqLen)
    return absl::InvalidArgumentError("Prompt exceeds max tokens bounds!");

  std::vector<float> prefill_embeddings(seq_len * config.emb_dim, 0.0f);
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    int32_t token_id = input_tokens[i];
    const float* src_row = locked_emb_table.data() + token_id * config.emb_dim;
    std::copy(src_row, src_row + config.emb_dim,
              prefill_embeddings.data() + i * config.emb_dim);
  }

  std::vector<float> prefill_mask(seq_len * seq_len,
                                  std::numeric_limits<float>::lowest());
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (i < raw_seq_len && j < raw_seq_len) {
        prefill_mask[i * seq_len + j] = 0.0f;
      }
    }
  }

  std::vector<float> rope_global_cos(seq_len * config.head_dim);
  std::vector<float> rope_global_sin(seq_len * config.head_dim);
  std::vector<float> rope_local_cos(seq_len * config.head_dim);
  std::vector<float> rope_local_sin(seq_len * config.head_dim);

  ::litert::tensor::examples::gemma3::RopeCosSin(
      0, seq_len, config.rope_global_base, absl::MakeSpan(rope_global_cos),
      absl::MakeSpan(rope_global_sin));
  float rope_local_base =
      config.rope_local_base > 0 ? config.rope_local_base : 10000.0f;
  ::litert::tensor::examples::gemma3::RopeCosSin(
      0, seq_len, rope_local_base, absl::MakeSpan(rope_local_cos),
      absl::MakeSpan(rope_local_sin));

  // 4. Setup Multi-Signature model JIT serialization factory compiler contexts
  ::litert::tensor::ModelFactory model_factory;

  auto SetupGraphInputs = [&](bool is_decode) {
    ::litert::tensor::examples::gemma3::Inputs<::litert::tensor::TfLiteMixinTag>
        inputs;

    int seq = is_decode ? 1 : static_cast<int>(kMaxSeqLen);

    inputs.embedded_input =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "embedded_input",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, seq, config.emb_dim}});

    inputs.rope_global_cos =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_global_cos",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, config.head_dim}});
    inputs.rope_global_sin =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_global_sin",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, config.head_dim}});
    inputs.rope_local_cos =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_local_cos",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, config.head_dim}});
    inputs.rope_local_sin =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_local_sin",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, config.head_dim}});

    if (is_decode) {
      inputs.sliding_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "sliding_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, 1, static_cast<int>(kMaxSeqLen)}});
      inputs.global_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "global_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, 1, static_cast<int>(kMaxSeqLen)}});
      inputs.cache_params =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "start_indices",
               .type = ::litert::tensor::Type::kI32,
               .shape = {4}});

      for (int i = 0; i < config.n_layers; ++i) {
        inputs.key_caches.push_back(
            ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
                {.name = absl::StrCat("key_cache_", i),
                 .type = ::litert::tensor::Type::kFP32,
                 .shape = {1, static_cast<int>(config.n_kv_groups), kMaxSeqLen,
                           config.head_dim}}));
        inputs.value_caches.push_back(
            ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
                {.name = absl::StrCat("value_cache_", i),
                 .type = ::litert::tensor::Type::kFP32,
                 .shape = {1, static_cast<int>(config.n_kv_groups), kMaxSeqLen,
                           config.head_dim}}));
      }
    } else {
      inputs.sliding_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "sliding_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, static_cast<int>(kMaxSeqLen),
                         static_cast<int>(kMaxSeqLen)}});
      inputs.global_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "global_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, static_cast<int>(kMaxSeqLen),
                         static_cast<int>(kMaxSeqLen)}});
    }

    inputs.slice_index =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "slice_index",
             .type = ::litert::tensor::Type::kI32,
             .shape = {1}});

    inputs.weights.insert(weights.begin(), weights.end());
    return inputs;
  };

  // [SIGNATURE 1]: Staging the Prefill Subgraph
  auto prefill_inputs = SetupGraphInputs(/*is_decode=*/false);
  auto prefill_outputs =
      BuildGpuGraph(prefill_inputs, config, /*is_decode=*/false);

  std::vector<::litert::tensor::TensorHandle> prefill_sig_outputs = {
      prefill_outputs.output};
  for (int i = 0; i < config.n_layers; ++i) {
    prefill_sig_outputs.push_back(prefill_outputs.key_caches[i]);
    prefill_sig_outputs.push_back(prefill_outputs.value_caches[i]);
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      model_factory.AddSignature(prefill_sig_outputs, std::string(kPrefill)));

  // [SIGNATURE 2]: Staging the Autoregressive Decode Subgraph
  auto decode_inputs = SetupGraphInputs(/*is_decode=*/true);
  auto decode_outputs =
      BuildGpuGraph(decode_inputs, config, /*is_decode=*/true);

  std::vector<::litert::tensor::TensorHandle> decode_sig_outputs = {
      decode_outputs.output};
  for (int i = 0; i < config.n_layers; ++i) {
    decode_sig_outputs.push_back(decode_outputs.key_caches[i]);
    decode_sig_outputs.push_back(decode_outputs.value_caches[i]);
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      model_factory.AddSignature(decode_sig_outputs, std::string(kDecode)));

  // 5. JIT Compilation Model Save Pass
  auto gpu_tflite_file = weights_path == "dummy"
                             ? "/tmp/dummy.tflite"
                             : std::string(weights_path) + ".tflite";
  LRT_TENSOR_RETURN_IF_ERROR(model_factory.Save(gpu_tflite_file));

  // 6. Initializing the pristine hardware accelerated environment options!
  auto env_or = ::litert::Environment::Create({});
  if (!env_or.HasValue())
    return absl::InternalError(
        "Failed to instantiate LiteRT Environment context!");
  auto env = std::move(env_or.Value());

  auto options_or = ::litert::Options::Create();
  if (!options_or.HasValue())
    return absl::InternalError("Failed to create LiteRT Options context!");
  auto options = std::move(options_or.Value());
  std::string acc_flag = absl::GetFlag(FLAGS_accelerator);
  if (acc_flag == "cpu") {
    options.SetHardwareAccelerators(::litert::HwAccelerators::kCpu);
  } else {
    options.SetHardwareAccelerators(::litert::HwAccelerators::kGpu);
    auto gpu_options_or = options.GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      gpu_options_or->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      gpu_options_or->SetBufferStorageType(
          ::litert::GpuOptions::BufferStorageType::kBuffer);
      gpu_options_or->EnableExternalTensorsMode(true);
      gpu_options_or->AddExternalTensorPattern("key_cache_.*");
      gpu_options_or->AddExternalTensorPattern("value_cache_.*");
    }
  }

  auto runner_res = ::litert::tensor::LitertDynamicRunner::Create(
      env, gpu_tflite_file, options);
  if (!runner_res.ok())
    return absl::InternalError(
        "Failed to initialize dynamic hardware execution engine!");
  auto runner = std::move(*runner_res);

  auto StageRuntimeInputs =
      [&](const std::string& signature, const std::vector<float>& emb,
          const std::vector<float>& mask, const std::vector<float>& r_g_cos,
          const std::vector<float>& r_g_sin, const std::vector<float>& r_l_cos,
          const std::vector<float>& r_l_sin, int slice_idx) -> absl::Status {
    auto emb_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(emb.data()),
                            emb.size() * sizeof(float));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "embedded_input", emb_span));

    auto mask_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(mask.data()),
                            mask.size() * sizeof(float));
    LRT_TENSOR_RETURN_IF_ERROR(SetInputOptional(
        runner, signature, "sliding_attention_mask", mask_span));
    LRT_TENSOR_RETURN_IF_ERROR(SetInputOptional(
        runner, signature, "global_attention_mask", mask_span));

    auto r_g_cos_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(r_g_cos.data()),
                            r_g_cos.size() * sizeof(float));
    auto r_g_sin_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(r_g_sin.data()),
                            r_g_sin.size() * sizeof(float));
    auto r_l_cos_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(r_l_cos.data()),
                            r_l_cos.size() * sizeof(float));
    auto r_l_sin_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(r_l_sin.data()),
                            r_l_sin.size() * sizeof(float));

    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "rope_global_cos", r_g_cos_span));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "rope_global_sin", r_g_sin_span));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "rope_local_cos", r_l_cos_span));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "rope_local_sin", r_l_sin_span));

    std::vector<int32_t> slice_vec = {slice_idx};
    auto slice_span =
        absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(slice_vec.data()),
                            slice_vec.size() * sizeof(int32_t));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, signature, "slice_index", slice_span));
    return absl::OkStatus();
  };

  // 7. Running the Prefill Parallel Phase pass
  LRT_TENSOR_RETURN_IF_ERROR(StageRuntimeInputs(
      std::string(kPrefill), prefill_embeddings, prefill_mask, rope_global_cos,
      rope_global_sin, rope_local_cos, rope_local_sin, raw_seq_len - 1));
  ::litert::tensor::examples::Timer::Get("run_prefill").StartLap();
  LRT_TENSOR_RETURN_IF_ERROR(runner.Run(std::string(kPrefill)));

  int32_t current_token = 0;
  {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto prefill_out, runner.GetOutput(std::string(kPrefill), "output"));
    LRT_TENSOR_ASSIGN_OR_RETURN(::litert::tensor::Buffer & prefill_buf,
                                prefill_out.GetBuffer());
    auto locked_prefill_id = prefill_buf.Lock().As<const int32_t>();
    current_token = locked_prefill_id.data()[0];
  }
  ::litert::tensor::examples::Timer::Get("run_prefill").StopLap();

  size_t cache_size = config.n_kv_groups * kMaxSeqLen * config.head_dim;
  std::vector<std::vector<float>> persistent_key_caches(
      config.n_layers, std::vector<float>(cache_size, 0.0f));
  std::vector<std::vector<float>> persistent_value_caches(
      config.n_layers, std::vector<float>(cache_size, 0.0f));

  for (int layer = 0; layer < config.n_layers; ++layer) {
    auto k_out_or = runner.GetOutput(
        std::string(kPrefill),
        absl::StrCat("model.layers.", layer, ".self_attn.updated_key_cache"));
    if (k_out_or.ok()) {
      auto k_buf_or = k_out_or->GetBuffer();
      if (k_buf_or.ok()) {
        auto k_lock = k_buf_or->Lock().As<const float>();
        int prefill_seq_len = seq_len;
        for (int g = 0; g < config.n_kv_groups; ++g) {
          const float* src =
              k_lock.data() + g * prefill_seq_len * config.head_dim;
          float* dst = persistent_key_caches[layer].data() +
                       g * kMaxSeqLen * config.head_dim;
          std::copy(src, src + prefill_seq_len * config.head_dim, dst);
        }
      }
    }

    auto v_out_or = runner.GetOutput(
        std::string(kPrefill),
        absl::StrCat("model.layers.", layer, ".self_attn.updated_value_cache"));
    if (v_out_or.ok()) {
      auto v_buf_or = v_out_or->GetBuffer();
      if (v_buf_or.ok()) {
        auto v_lock = v_buf_or->Lock().As<const float>();
        int prefill_seq_len = seq_len;
        for (int g = 0; g < config.n_kv_groups; ++g) {
          const float* src =
              v_lock.data() + g * prefill_seq_len * config.head_dim;
          float* dst = persistent_value_caches[layer].data() +
                       g * kMaxSeqLen * config.head_dim;
          std::copy(src, src + prefill_seq_len * config.head_dim, dst);
        }
      }
    }
  }

  // 8. Autoregressive token generation loop steps!
  std::cout << "[Inference Output Start] " << prompt_text << std::flush;
  std::vector<float> token_embedding(config.emb_dim);

  // Set initial caches from persistent CPU vectors!
  for (int layer = 0; layer < config.n_layers; ++layer) {
    auto k_span = absl::MakeConstSpan(
        reinterpret_cast<const uint8_t*>(persistent_key_caches[layer].data()),
        persistent_key_caches[layer].size() * sizeof(float));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, std::string(kDecode),
                         absl::StrCat("key_cache_", layer), k_span));

    auto v_span = absl::MakeConstSpan(
        reinterpret_cast<const uint8_t*>(persistent_value_caches[layer].data()),
        persistent_value_caches[layer].size() * sizeof(float));
    LRT_TENSOR_RETURN_IF_ERROR(
        SetInputOptional(runner, std::string(kDecode),
                         absl::StrCat("value_cache_", layer), v_span));
  }
  ::litert::tensor::examples::DecodeTiming decode_timing;
  absl::Duration total_get_output_time = absl::ZeroDuration();
  absl::Duration total_set_input_time = absl::ZeroDuration();
  decode_timing.decode.StartLap();

  // Pre-allocate reusable static host buffers outside the generative hot loop!
  std::vector<float> decode_mask(kMaxSeqLen, -10000.0f);
  for (int j = 0; j < raw_seq_len; ++j) {
    decode_mask[j] = 0.0f;
  }
  std::vector<float> dec_rope_global_cos(config.head_dim);
  std::vector<float> dec_rope_global_sin(config.head_dim);
  std::vector<float> dec_rope_local_cos(config.head_dim);
  std::vector<float> dec_rope_local_sin(config.head_dim);
  std::vector<int32_t> start_indices = {0, 0, 0, 0};

  std::vector<::litert::tensor::TensorHandle> cached_k_out_handles;
  std::vector<::litert::tensor::TensorHandle> cached_v_out_handles;
  std::vector<std::string> cached_k_in_names;
  std::vector<std::string> cached_v_in_names;
  cached_k_out_handles.reserve(config.n_layers);
  cached_v_out_handles.reserve(config.n_layers);
  cached_k_in_names.reserve(config.n_layers);
  cached_v_in_names.reserve(config.n_layers);
  for (int layer = 0; layer < config.n_layers; ++layer) {
    cached_k_in_names.push_back(absl::StrCat("key_cache_", layer));
    cached_v_in_names.push_back(absl::StrCat("value_cache_", layer));
    auto k_out_or = runner.GetOutput(
        std::string(kDecode),
        absl::StrCat("model.layers.", layer, ".self_attn.updated_key_cache"));
    if (k_out_or.ok()) cached_k_out_handles.push_back(*k_out_or);
    auto v_out_or = runner.GetOutput(
        std::string(kDecode),
        absl::StrCat("model.layers.", layer, ".self_attn.updated_value_cache"));
    if (v_out_or.ok()) cached_v_out_handles.push_back(*v_out_or);
  }

  for (size_t step = 0; step < max_tokens; ++step) {
    if (current_token == 1) break;  // Generative completion EOS token hit!
    std::string word = (*tokenizer_or).DecodeToken(current_token);
    std::cout << word << std::flush;

    decode_timing.cpu_prep.StartLap();
    const float* src_row = locked_emb_table.data() +
                           (current_token % config.vocab_size) * config.emb_dim;
    std::copy(src_row, src_row + config.emb_dim, token_embedding.data());

    int cache_len = raw_seq_len + step;
    decode_mask[cache_len] = 0.0f;  // Unmasks exactly the newly added column!

    ::litert::tensor::examples::gemma3::RopeCosSin(
        cache_len, 1, config.rope_global_base,
        absl::MakeSpan(dec_rope_global_cos),
        absl::MakeSpan(dec_rope_global_sin));
    ::litert::tensor::examples::gemma3::RopeCosSin(
        cache_len, 1, rope_local_base, absl::MakeSpan(dec_rope_local_cos),
        absl::MakeSpan(dec_rope_local_sin));

    start_indices[2] = cache_len;
    auto start_indices_span = absl::MakeConstSpan(
        reinterpret_cast<const uint8_t*>(start_indices.data()),
        start_indices.size() * sizeof(int32_t));
    LRT_TENSOR_RETURN_IF_ERROR(SetInputOptional(
        runner, std::string(kDecode), "start_indices", start_indices_span));

    LRT_TENSOR_RETURN_IF_ERROR(StageRuntimeInputs(
        std::string(kDecode), token_embedding, decode_mask, dec_rope_global_cos,
        dec_rope_global_sin, dec_rope_local_cos, dec_rope_local_sin, 0));

    decode_timing.cpu_prep.StopLap();

    decode_timing.run.StartLap();
    auto status = runner.Run(std::string(kDecode));
    if (!status.ok()) return status;

    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto decode_out, runner.GetOutput(std::string(kDecode), "output"));
    LRT_TENSOR_ASSIGN_OR_RETURN(::litert::tensor::Buffer & decode_buf,
                                decode_out.GetBuffer());
    auto locked_decode_id = decode_buf.Lock().As<const int32_t>();
    decode_timing.run.StopLap();

    decode_timing.argmax.StartLap();
    current_token = locked_decode_id.data()[0];
    decode_timing.argmax.StopLap();

    decode_timing.cache_readback.StartLap();
    for (int layer = 0; layer < config.n_layers; ++layer) {
      if (layer < cached_k_out_handles.size()) {
        auto t2 = absl::Now();
        LRT_TENSOR_RETURN_IF_ERROR(
            runner.SetInput(std::string(kDecode), cached_k_in_names[layer],
                            cached_k_out_handles[layer]));
        auto t3 = absl::Now();
        total_set_input_time += (t3 - t2);
      }
      if (layer < cached_v_out_handles.size()) {
        auto t6 = absl::Now();
        LRT_TENSOR_RETURN_IF_ERROR(
            runner.SetInput(std::string(kDecode), cached_v_in_names[layer],
                            cached_v_out_handles[layer]));
        auto t7 = absl::Now();
        total_set_input_time += (t7 - t6);
      }
    }
    decode_timing.cache_readback.StopLap();
  }
  decode_timing.decode.StopLap();

  std::cout << std::endl
            << "[Inference End] Finished successfully." << std::endl;

  absl::Duration prefill_duration =
      ::litert::tensor::examples::Timer::Get("run_prefill").Duration();
  absl::Duration decode_duration = decode_timing.decode.Duration();
  std::cerr << "Prefilled " << input_tokens.size() << " tokens in "
            << absl::ToDoubleMilliseconds(prefill_duration) << " ms ("
            << (absl::ToDoubleMilliseconds(prefill_duration) /
                input_tokens.size())
            << " ms/tok).\n";
  std::cerr << "Decoded " << max_tokens << " tokens in "
            << absl::ToDoubleMilliseconds(decode_duration) << " ms ("
            << absl::ToDoubleMilliseconds(decode_duration / max_tokens)
            << " ms/tok).\n"
            << "  Total GetOutput overhead: "
            << absl::ToDoubleMilliseconds(total_get_output_time) << " ms ("
            << absl::ToDoubleMilliseconds(total_get_output_time) / max_tokens
            << " ms/tok)\n"
            << "  Total SetInput overhead: "
            << absl::ToDoubleMilliseconds(total_set_input_time) << " ms ("
            << absl::ToDoubleMilliseconds(total_set_input_time) / max_tokens
            << " ms/tok)\n"
            << decode_timing.Stats() << "\n";

  cached_k_out_handles.clear();
  cached_v_out_handles.clear();
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  size_t max_tokens = absl::GetFlag(FLAGS_max_tokens);

  if (weights_path.empty() || tokenizer_path.empty()) {
    std::cerr << "Error: --weights_path and --tokenizer_path are required"
              << std::endl;
    std::cerr << "Usage: " << argv[0]
              << " --weights_path=/path/to/model.safetensors"
              << " --tokenizer_path=/path/to/tokenizer.model" << std::endl;
    return 1;
  }

  ::litert::tensor::examples::SafetensorLoader::QuantizedLoadMode weight_mode =
      absl::GetFlag(FLAGS_weight_mode);

  absl::Status status = RunGemma3Inference(weights_path, tokenizer_path, prompt,
                                           max_tokens, weight_mode);
  if (!status.ok()) {
    std::cerr << "Error: " << status << std::endl;
    return 1;
  }
  return 0;
}
