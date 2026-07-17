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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/gloop/base/init_google.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/config.h"
#include "tensor/examples/gemma3/gemma3.h"
#include "tensor/examples/gemma3/graph_helpers.h"
#include "tensor/examples/gemma3/tflite_loader.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

constexpr absl::string_view kPrefill = "prefill";
constexpr absl::string_view kDecode = "decode";
ABSL_FLAG(std::string, weights_path, "", "Path to the TFLite weights file");
ABSL_FLAG(std::string, tokenizer_path, "", "Path to the tokenizer.model file");
ABSL_FLAG(std::string, prompt, "Hello, world!",
          "Input prompt for text generation");
ABSL_FLAG(size_t, max_tokens, 50, "Maximum number of tokens to generate");
ABSL_FLAG(std::string, accelerator, "gpu",
          "Hardware accelerator to use: cpu or gpu");
ABSL_FLAG(::litert::tensor::examples::TfliteLoader::QuantizedLoadMode,
          weight_mode,
          ::litert::tensor::examples::TfliteLoader::QuantizedLoadMode::
              kPreserveQuantized,
          "Weight mode (quantized or float)");
ABSL_FLAG(bool, benchmark, false, "Run in benchmark mode");
ABSL_FLAG(size_t, benchmark_prefill_tokens, 32,
          "Number of prefill tokens in benchmark mode");
ABSL_FLAG(size_t, benchmark_decode_tokens, 50,
          "Number of decode tokens in benchmark mode");
ABSL_FLAG(bool, fp16, true,
          "Whether to use half-precision (FP16) calculations on the GPU.");

namespace litert::tensor::examples::gemma3 {

struct GpuAttnOutput {
  Tensor<TfLiteMixinTag> output;
  Tensor<TfLiteMixinTag> key_cache;
  Tensor<TfLiteMixinTag> value_cache;
};

GpuAttnOutput MakeGpuSelfAttentionLayer(
    const Tensor<TfLiteMixinTag>& input, absl::string_view name,
    const Config& config, bool is_sliding_attention,
    const Tensor<TfLiteMixinTag>& attention_mask,
    const Tensor<TfLiteMixinTag>& cos, const Tensor<TfLiteMixinTag>& sin,
    const Tensor<TfLiteMixinTag>& slice_offset_1,
    const Tensor<TfLiteMixinTag>& slice_size_1,
    const Tensor<TfLiteMixinTag>& slice_offset_2,
    const Tensor<TfLiteMixinTag>& slice_size_2,
    const Tensor<TfLiteMixinTag>& key_cache,
    const Tensor<TfLiteMixinTag>& value_cache,
    const absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>& weights,
    const std::optional<Tensor<TfLiteMixinTag>> key_cache_params,
    const std::optional<Tensor<TfLiteMixinTag>> value_cache_params,
    bool is_decode) {
  int qkv_out_dim = config.n_heads * config.head_dim;
  int kv_out_dim = config.n_kv_groups * config.head_dim;

  const Shape& input_shape = input.GetShape();
  int batch_size = input_shape.size() == 3 ? input_shape[0] : 1;
  int seq_len = input_shape.size() == 3 ? input_shape[1] : input_shape[0];

  Tensor q_weight = GetWeight(weights, absl::StrCat(name, ".q_proj.weight"),
                              Type::kFP32, {qkv_out_dim, config.emb_dim});
  Tensor k_weight = GetWeight(weights, absl::StrCat(name, ".k_proj.weight"),
                              Type::kFP32, {kv_out_dim, config.emb_dim});
  Tensor v_weight = GetWeight(weights, absl::StrCat(name, ".v_proj.weight"),
                              Type::kFP32, {kv_out_dim, config.emb_dim});

  Tensor q = FullyConnected(input, q_weight);
  Tensor k = FullyConnected(input, k_weight);
  Tensor v = FullyConnected(input, v_weight);

  Tensor o_proj = GetWeight(weights, absl::StrCat(name, ".o_proj.weight"),
                            Type::kFP32, {config.emb_dim, qkv_out_dim});

  Tensor q_norm = GetWeight(weights, absl::StrCat(name, ".q_norm.weight"),
                            Type::kFP32, {config.head_dim});
  Tensor k_norm = GetWeight(weights, absl::StrCat(name, ".k_norm.weight"),
                            Type::kFP32, {config.head_dim});
  if (is_decode) {
    // ========= HIGH-PERFORMANCE DECODE ATTENTION PATH (0 TRANSPOSES!)
    // =========
    q = Reshape(q, {batch_size, seq_len, config.n_heads, config.head_dim});
    k = Reshape(k, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
    v = Reshape(v, {batch_size, seq_len, config.n_kv_groups, config.head_dim});

    q = Gemma3RmsNorm(q, q_norm, config.rms_norm_eps);
    k = Gemma3RmsNorm(k, k_norm, config.rms_norm_eps);

    q = ApplyRotaryEmbedding(q, cos, sin, slice_offset_1, slice_size_1,
                             slice_offset_2, slice_size_2);
    k = ApplyRotaryEmbedding(k, cos, sin, slice_offset_1, slice_size_1,
                             slice_offset_2, slice_size_2);

    Tensor updated_key_cache =
        DynamicUpdateSlice(key_cache, k, *key_cache_params);
    Tensor v_reshaped =
        Reshape(v, {batch_size, config.n_kv_groups, config.head_dim, seq_len});
    Tensor updated_value_cache =
        DynamicUpdateSlice(value_cache, v_reshaped, *value_cache_params);

    updated_key_cache.SetName(absl::StrCat(name, ".updated_key_cache"));
    updated_value_cache.SetName(absl::StrCat(name, ".updated_value_cache"));

    // Perform BatchMatMul Q * K^T directly (adj_y = true)
    Tensor scores_flat =
        BatchMatMul(q, updated_key_cache, /*adj_x=*/false, /*adj_y=*/true);

    float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
    Tensor scale_tensor = Tensor<TfLiteMixinTag>(
        {.type = Type::kFP32,
         .shape = {1},
         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({scale})});
    scores_flat = Mul(scores_flat, scale_tensor);

    Tensor scores = Add(scores_flat, attention_mask);
    Tensor attn_weights = Softmax(scores);

    // Standard BatchMatMul Attn * V (adj_y = true)
    Tensor attn_output_flat = BatchMatMul(attn_weights, updated_value_cache,
                                          /*adj_x=*/false, /*adj_y=*/true);
    Tensor<TfLiteMixinTag> attn_output_flat_proj;
    if (input_shape.size() == 2) {
      attn_output_flat_proj = Reshape(attn_output_flat, {seq_len, qkv_out_dim});
    } else {
      attn_output_flat_proj =
          Reshape(attn_output_flat, {batch_size, seq_len, qkv_out_dim});
    }
    attn_output_flat_proj.SetName(absl::StrCat(name, ".attn_output_flat"));

    Tensor output = FullyConnected(attn_output_flat_proj, o_proj)
                        .SetName(absl::StrCat(name, ".attn_output"));

    GpuAttnOutput attn_out;
    attn_out.output = output;
    attn_out.key_cache = updated_key_cache;
    attn_out.value_cache = updated_value_cache;
    return attn_out;
  }

  // ========= HIGH-PERFORMANCE PREFILL PATH (WITH DYNAMIC BROADCAST ROPE)
  // =========
  q = Reshape(q, {batch_size, seq_len, config.n_heads, config.head_dim});
  q = Transpose(q, {0, 2, 1, 3});

  k = Reshape(k, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  k = Transpose(k, {0, 2, 1, 3});

  v = Reshape(v, {batch_size, seq_len, config.n_kv_groups, config.head_dim});
  v = Transpose(v, {0, 2, 1, 3});

  q = Gemma3RmsNorm(q, q_norm, config.rms_norm_eps);
  k = Gemma3RmsNorm(k, k_norm, config.rms_norm_eps);

  q = ApplyRotaryEmbedding(q, cos, sin, slice_offset_1, slice_size_1,
                           slice_offset_2, slice_size_2);
  k = ApplyRotaryEmbedding(k, cos, sin, slice_offset_1, slice_size_1,
                           slice_offset_2, slice_size_2);

  k.SetName(absl::StrCat(name, ".new_k"));
  v.SetName(absl::StrCat(name, ".new_v"));

  Tensor updated_key_cache =
      DynamicUpdateSlice(key_cache, k, *key_cache_params);

  Tensor v_transposed = Transpose(v, {0, 1, 3, 2});
  Tensor updated_value_cache =
      DynamicUpdateSlice(value_cache, v_transposed, *value_cache_params);

  Tensor<TfLiteMixinTag> k_for_attn = k;
  Tensor<TfLiteMixinTag> v_for_attn = v;
  updated_key_cache.SetName(absl::StrCat(name, ".updated_key_cache"));
  updated_value_cache.SetName(absl::StrCat(name, ".updated_value_cache"));

  int num_groups = config.n_heads / config.n_kv_groups;
  if (num_groups > 1) {
    if (config.n_kv_groups == 1) {
      k_for_attn = Tile(k_for_attn, {1, num_groups, 1, 1});
      v_for_attn = Tile(v_for_attn, {1, num_groups, 1, 1});
    } else {
      std::vector<Tensor<TfLiteMixinTag>> tiled_k_heads;
      tiled_k_heads.reserve(config.n_kv_groups);
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
      tiled_v_heads.reserve(config.n_kv_groups);
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
  }

  Tensor scores = BatchMatMul(q, k_for_attn, /*adj_x=*/false, /*adj_y=*/true);

  float scale = 1.0f / std::sqrt(config.query_pre_attn_scalar);
  Tensor scale_tensor = Tensor<TfLiteMixinTag>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({scale})});
  scores = Mul(scores, scale_tensor);

  scores = Add(scores, attention_mask);

  Tensor attn_weights = Softmax(scores);
  attn_weights.SetName(absl::StrCat(name, ".attn_weights"));

  // Standard BatchMatMul Attn * V (adj_y = false)
  Tensor attn_output = BatchMatMul(attn_weights, v_for_attn,
                                   /*adj_x=*/false, /*adj_y=*/false);

  attn_output = Transpose(attn_output, {0, 2, 1, 3});
  Tensor<TfLiteMixinTag> attn_output_flat_proj;
  if (input_shape.size() == 2) {
    attn_output_flat_proj = Reshape(attn_output, {seq_len, qkv_out_dim});
  } else {
    attn_output_flat_proj =
        Reshape(attn_output, {batch_size, seq_len, qkv_out_dim});
  }
  attn_output_flat_proj.SetName(absl::StrCat(name, ".attn_output_flat"));

  Tensor output = FullyConnected(attn_output_flat_proj, o_proj)
                      .SetName(absl::StrCat(name, ".attn_output"));

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
  Tensor<TfLiteMixinTag> embedded_input;
};

GpuOutputs BuildGpuGraph(Inputs<TfLiteMixinTag>& inputs, const Config& config,
                         bool is_decode) {
  auto emb_it = inputs.weights.find("model.embed_tokens.weight");
  Type emb_type =
      emb_it != inputs.weights.end() ? emb_it->second.GetType() : Type::kFP32;
  Tensor embedding_table = GetWeight(
      inputs.weights, "model.embed_tokens.weight", emb_type,
      {static_cast<int>(config.vocab_size), static_cast<int>(config.emb_dim)});

  Tensor embedded_input = EmbeddingLookup(inputs.input_ids, embedding_table);
  embedded_input.SetName("embedded_input");

  float emb_scale = std::sqrt(static_cast<float>(config.emb_dim));
  Tensor emb_scale_tensor = Tensor<TfLiteMixinTag>(
      {.name = "emb_scale_tensor",
       .type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({emb_scale})});
  Tensor hidden_states = Mul(embedded_input, emb_scale_tensor);

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
        is_sliding, attention_mask, cos, sin, inputs.slice_offset_1,
        inputs.slice_size_1, inputs.slice_offset_2, inputs.slice_size_2,
        layer_idx < inputs.key_caches.size() ? inputs.key_caches[layer_idx]
                                             : Tensor<TfLiteMixinTag>(),
        layer_idx < inputs.value_caches.size() ? inputs.value_caches[layer_idx]
                                               : Tensor<TfLiteMixinTag>(),
        inputs.weights, inputs.key_cache_params, inputs.value_cache_params,
        is_decode);

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

  if (!is_decode) {
    final_output = Gather(final_output, inputs.slice_index, /*axis=*/0);
  }
  Tensor logits = FullyConnected(final_output, embedding_table, kActNone,
                                 /*keep_num_dims=*/true);
  if (is_decode) {
    Tensor argmax_out = ArgMax(logits, 1, Type::kI32);
    argmax_out.SetName("output");
    gpu_outputs.output = argmax_out;
  } else {
    logits.SetName("output");
    gpu_outputs.output = logits;
  }

  gpu_outputs.key_caches = updated_key_caches;
  gpu_outputs.value_caches = updated_value_caches;
  gpu_outputs.embedded_input = embedded_input;

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

template <typename T>
absl::Status WriteOptional(
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>& map,
    ::litert::StringView name, absl::Span<const T> data) {
  auto it = map.find(name);
  if (it != map.end()) {
    auto res = it->second.Write(data);
    if (!res.HasValue()) {
      return absl::InternalError(res.Error().Message());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, ::litert::tensor::TensorHandle>>
LoadGemma3Weights(
    const std::string& weights_path,
    const ::litert::tensor::examples::gemma3::Config& config,
    ::litert::tensor::examples::TfliteLoader::QuantizedLoadMode weight_mode) {
  auto loader_res =
      ::litert::tensor::examples::TfliteLoader::Load(weights_path);
  if (!loader_res.ok()) {
    return absl::InternalError("Failed to load TFLite weights file!");
  }
  auto loader = std::move(*loader_res);

  auto mapping =
      ::litert::tensor::examples::GetGemma3TfliteWeightMapping(config.n_layers);
  auto weights_res = loader.LoadWeightsWithMapping(mapping, weight_mode);
  if (!weights_res.ok()) {
    return absl::InternalError("Failed to load tensors from TFLite weights!");
  }
  auto weights = std::move(*weights_res);

  auto emb_entry = mapping["model.embed_tokens.weight"];
  auto emb_res = loader.LoadTensor(emb_entry.tflite_tensor_name,
                                   ::litert::tensor::examples::TfliteLoader::
                                       QuantizedLoadMode::kPreserveQuantized);
  if (!emb_res.ok()) {
    return absl::InternalError(absl::StrCat("Failed to load embedding table: ",
                                            emb_res.status().ToString()));
  }
  auto emb = std::move(*emb_res);
  emb.SetName("model.embed_tokens.weight");

  weights["model.embed_tokens.weight"] = std::move(emb);

  const auto& loaded_emb = weights["model.embed_tokens.weight"];
  ABSL_LOG(WARNING) << "CPU Loaded Embedding table type: "
                    << ToString(loaded_emb.GetType());
  auto emb_buf_or = loaded_emb.GetBuffer();
  if (!emb_buf_or.ok()) {
    ABSL_LOG(ERROR) << "Failed to get embedding table buffer!";
  } else {
    auto locked_span = emb_buf_or.value().Lock();
    const uint8_t* raw_data =
        reinterpret_cast<const uint8_t*>(locked_span.data());
    size_t raw_bytes = locked_span.size();
    double abs_sum = 0.0;
    for (size_t i = 0; i < raw_bytes; ++i) {
      abs_sum += std::abs((int)raw_data[i]);
    }
    ABSL_LOG(WARNING) << "CPU Loaded Embedding table raw bytes sum: " << abs_sum
                      << " over " << raw_bytes << " bytes.";

    std::string bytes_str = "";
    for (size_t i = 0; i < std::min(raw_bytes, (size_t)20); ++i) {
      absl::StrAppend(&bytes_str, (int)raw_data[i], " ");
    }
    ABSL_LOG(WARNING) << "CPU Loaded Embedding table first 20 bytes: [ "
                      << bytes_str << "]";
  }

  return weights;
}

void TokenizeInput(
    const std::string& prompt_text,
    const ::litert::tensor::examples::GemmaTokenizerSP& tokenizer,
    bool benchmark, size_t benchmark_prefill, size_t benchmark_decode,
    std::vector<int32_t>& input_tokens, size_t& max_tokens) {
  if (benchmark) {
    input_tokens.resize(benchmark_prefill, 99);  // Fill with dummy Token ID 99
    if (!input_tokens.empty()) {
      input_tokens[0] = 2;  // <bos>
    }
    max_tokens = benchmark_decode;
  } else {
    input_tokens = tokenizer.Encode(prompt_text, /*add_bos=*/true);
  }
}

absl::Status BuildAndSaveJitModel(
    const std::string& gpu_tflite_file,
    const ::litert::tensor::examples::gemma3::Config& config,
    const absl::flat_hash_map<std::string, ::litert::tensor::TensorHandle>&
        weights,
    int seq_len, int max_seq_len) {
  ::litert::tensor::ModelFactory model_factory;

  auto SetupGraphInputs = [&](bool is_decode) {
    ::litert::tensor::examples::gemma3::Inputs<::litert::tensor::TfLiteMixinTag>
        inputs;

    int seq = is_decode ? 1 : seq_len;

    inputs.input_ids =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "input_ids",
             .type = ::litert::tensor::Type::kI32,
             .shape = {seq}});

    inputs.rope_global_cos =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_global_cos",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, static_cast<int>(config.head_dim)}});
    inputs.rope_global_sin =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_global_sin",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, static_cast<int>(config.head_dim)}});
    inputs.rope_local_cos =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_local_cos",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, static_cast<int>(config.head_dim)}});
    inputs.rope_local_sin =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "rope_local_sin",
             .type = ::litert::tensor::Type::kFP32,
             .shape = {1, 1, seq, static_cast<int>(config.head_dim)}});
    std::vector<int32_t> off1 = {0, 0, 0, 0};
    std::vector<int32_t> sz1 = {-1, -1, -1,
                                static_cast<int32_t>(config.head_dim / 2)};
    std::vector<int32_t> off2 = {0, 0, 0,
                                 static_cast<int32_t>(config.head_dim / 2)};
    std::vector<int32_t> sz2 = {-1, -1, -1,
                                static_cast<int32_t>(config.head_dim / 2)};

    inputs.slice_offset_1 =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "slice_offset_1",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4},
             .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                 ::litert::tensor::Type::kI32>(off1)});
    inputs.slice_size_1 =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "slice_size_1",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4},
             .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                 ::litert::tensor::Type::kI32>(sz1)});
    inputs.slice_offset_2 =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "slice_offset_2",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4},
             .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                 ::litert::tensor::Type::kI32>(off2)});
    inputs.slice_size_2 =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "slice_size_2",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4},
             .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                 ::litert::tensor::Type::kI32>(sz2)});

    for (int i = 0; i < config.n_layers; ++i) {
      inputs.key_caches.push_back(
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = absl::StrCat("key_cache_", i),
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, static_cast<int>(config.n_kv_groups), max_seq_len,
                         static_cast<int>(config.head_dim)}}));
      inputs.value_caches.push_back(
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = absl::StrCat("value_cache_", i),
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, static_cast<int>(config.n_kv_groups),
                         static_cast<int>(config.head_dim), max_seq_len}}));
    }

    inputs.key_cache_params =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "key_cache_start_indices",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4}});
    inputs.value_cache_params =
        ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
            {.name = "value_cache_start_indices",
             .type = ::litert::tensor::Type::kI32,
             .shape = {4}});

    if (is_decode) {
      inputs.sliding_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "sliding_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, 1, max_seq_len}});
      inputs.global_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "global_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, 1, max_seq_len}});
    } else {
      inputs.sliding_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "sliding_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, seq_len, seq_len}});
      inputs.global_attention_mask =
          ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>(
              {.name = "global_attention_mask",
               .type = ::litert::tensor::Type::kFP32,
               .shape = {1, 1, seq_len, seq_len}});
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
      prefill_outputs.output, prefill_outputs.embedded_input};
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

  LRT_TENSOR_RETURN_IF_ERROR(model_factory.Save(gpu_tflite_file));

  return absl::OkStatus();
}

absl::StatusOr<::litert::CompiledModel> CreateCompiledModel(
    ::litert::Environment& env, const std::string& gpu_tflite_file,
    const std::string& acc_flag, bool fp16) {
  auto options_or = ::litert::Options::Create();
  if (!options_or.HasValue())
    return absl::InternalError("Failed to create LiteRT Options context!");
  auto options = std::move(options_or.Value());

  if (acc_flag == "cpu") {
    options.SetHardwareAccelerators(::litert::HwAccelerators::kCpu);
  } else if (acc_flag == "gpu") {
    options.SetHardwareAccelerators(::litert::HwAccelerators::kGpu);
    auto gpu_options_or = options.GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      if (fp16) {
        gpu_options_or->SetPrecision(::litert::GpuOptions::Precision::kFp16);
      } else {
        gpu_options_or->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      }
      gpu_options_or->SetBufferStorageType(
          ::litert::GpuOptions::BufferStorageType::kBuffer);
    }
  } else {
    ABSL_LOG(INFO) << "=== RUNNING IN PURE CPU REFERENCE INTERPRETER MODE ===";
  }

  auto compiled_model_or =
      ::litert::CompiledModel::Create(env, gpu_tflite_file, options);
  if (!compiled_model_or.HasValue()) {
    return absl::InternalError(compiled_model_or.Error().Message());
  }
  return std::move(compiled_model_or.Value());
}

struct Gemma3KVCache {
  std::vector<std::string> cached_k_in_names;
  std::vector<std::string> cached_v_in_names;
  std::vector<std::string> cached_k_out_names;
  std::vector<std::string> cached_v_out_names;

  std::vector<::litert::TensorBuffer> key_caches_a;
  std::vector<::litert::TensorBuffer> key_caches_b;
  std::vector<::litert::TensorBuffer> value_caches_a;
  std::vector<::litert::TensorBuffer> value_caches_b;
};

absl::StatusOr<Gemma3KVCache> SetupKVCache(
    ::litert::CompiledModel& compiled_model,
    const ::litert::tensor::examples::gemma3::Config& config) {
  Gemma3KVCache cache;
  cache.cached_k_in_names.reserve(config.n_layers);
  cache.cached_v_in_names.reserve(config.n_layers);
  cache.cached_k_out_names.reserve(config.n_layers);
  cache.cached_v_out_names.reserve(config.n_layers);

  for (int layer = 0; layer < config.n_layers; ++layer) {
    cache.cached_k_in_names.push_back(absl::StrCat("key_cache_", layer));
    cache.cached_v_in_names.push_back(absl::StrCat("value_cache_", layer));
    cache.cached_k_out_names.push_back(
        absl::StrCat("model.layers.", layer, ".self_attn.updated_key_cache"));
    cache.cached_v_out_names.push_back(
        absl::StrCat("model.layers.", layer, ".self_attn.updated_value_cache"));
  }

  cache.key_caches_a.reserve(config.n_layers);
  cache.key_caches_b.reserve(config.n_layers);
  cache.value_caches_a.reserve(config.n_layers);
  cache.value_caches_b.reserve(config.n_layers);

  for (int i = 0; i < config.n_layers; ++i) {
    auto k_buf_A_or =
        compiled_model.CreateInputBuffer(kDecode, cache.cached_k_in_names[i]);
    if (!k_buf_A_or.HasValue()) {
      return absl::InternalError(k_buf_A_or.Error().Message());
    }
    auto k_buf_B_or =
        compiled_model.CreateInputBuffer(kDecode, cache.cached_k_in_names[i]);
    if (!k_buf_B_or.HasValue()) {
      return absl::InternalError(k_buf_B_or.Error().Message());
    }
    auto v_buf_A_or =
        compiled_model.CreateInputBuffer(kDecode, cache.cached_v_in_names[i]);
    if (!v_buf_A_or.HasValue()) {
      return absl::InternalError(v_buf_A_or.Error().Message());
    }
    auto v_buf_B_or =
        compiled_model.CreateInputBuffer(kDecode, cache.cached_v_in_names[i]);
    if (!v_buf_B_or.HasValue()) {
      return absl::InternalError(v_buf_B_or.Error().Message());
    }

    cache.key_caches_a.push_back(std::move(k_buf_A_or.Value()));
    cache.key_caches_b.push_back(std::move(k_buf_B_or.Value()));
    cache.value_caches_a.push_back(std::move(v_buf_A_or.Value()));
    cache.value_caches_b.push_back(std::move(v_buf_B_or.Value()));
  }

  return cache;
}

absl::Status SetupStandardBuffers(
    ::litert::CompiledModel& compiled_model, absl::string_view signature_name,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        inputs_map,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        outputs_map,
    bool& has_key_cache_params, bool& has_value_cache_params) {
  static constexpr absl::string_view kStandardInputs[] = {
      "input_ids",       "sliding_attention_mask", "global_attention_mask",
      "rope_global_cos", "rope_global_sin",        "rope_local_cos",
      "rope_local_sin"};

  for (absl::string_view name : kStandardInputs) {
    auto buf_or = compiled_model.CreateInputBuffer(signature_name, name);
    if (!buf_or.HasValue()) {
      return absl::InternalError(buf_or.Error().Message());
    }
    inputs_map[name] = std::move(buf_or.Value());
  }

  has_key_cache_params = false;
  has_value_cache_params = false;

  auto key_indices_buf_or = compiled_model.CreateInputBuffer(
      signature_name, "key_cache_start_indices");
  if (key_indices_buf_or.HasValue()) {
    inputs_map["key_cache_start_indices"] =
        std::move(key_indices_buf_or.Value());
    has_key_cache_params = true;
  }
  auto value_indices_buf_or = compiled_model.CreateInputBuffer(
      signature_name, "value_cache_start_indices");
  if (value_indices_buf_or.HasValue()) {
    inputs_map["value_cache_start_indices"] =
        std::move(value_indices_buf_or.Value());
    has_value_cache_params = true;
  }

  auto out_or = compiled_model.CreateOutputBuffer(signature_name, "output");
  if (!out_or.HasValue()) {
    return absl::InternalError(out_or.Error().Message());
  }
  outputs_map["output"] = std::move(out_or.Value());

  return absl::OkStatus();
}

absl::Status SetupPrefillBuffers(
    ::litert::CompiledModel& compiled_model,
    const ::litert::tensor::examples::gemma3::Config& config,
    size_t max_seq_len, Gemma3KVCache& kv_cache,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        prefill_inputs_map,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        prefill_outputs_map,
    bool& prefill_has_key_cache_params, bool& prefill_has_value_cache_params) {
  // Zero-initialize the B caches (which will be used as inputs to prefill)
  size_t cache_size = config.n_kv_groups * max_seq_len * config.head_dim;
  std::vector<float> zeros(cache_size, 0.0f);
  for (int i = 0; i < config.n_layers; ++i) {
    auto status = kv_cache.key_caches_b[i].Write<float>(absl::MakeSpan(zeros));
    if (!status.HasValue()) {
      return absl::InternalError(status.Error().Message());
    }
    status = kv_cache.value_caches_b[i].Write<float>(absl::MakeSpan(zeros));
    if (!status.HasValue()) {
      return absl::InternalError(status.Error().Message());
    }
  }

  for (int i = 0; i < config.n_layers; ++i) {
    auto k_in_dup_or = kv_cache.key_caches_b[i].Duplicate();
    if (!k_in_dup_or.HasValue()) {
      return absl::InternalError(k_in_dup_or.Error().Message());
    }
    prefill_inputs_map[kv_cache.cached_k_in_names[i]] =
        std::move(k_in_dup_or.Value());

    auto v_in_dup_or = kv_cache.value_caches_b[i].Duplicate();
    if (!v_in_dup_or.HasValue()) {
      return absl::InternalError(v_in_dup_or.Error().Message());
    }
    prefill_inputs_map[kv_cache.cached_v_in_names[i]] =
        std::move(v_in_dup_or.Value());
  }

  auto status = SetupStandardBuffers(
      compiled_model, kPrefill, prefill_inputs_map, prefill_outputs_map,
      prefill_has_key_cache_params, prefill_has_value_cache_params);
  if (!status.ok()) return status;

  auto prefill_slice_in_or =
      compiled_model.CreateInputBuffer(kPrefill, "slice_index");
  if (!prefill_slice_in_or.HasValue()) {
    return absl::InternalError(prefill_slice_in_or.Error().Message());
  }
  prefill_inputs_map["slice_index"] = std::move(prefill_slice_in_or.Value());

  if (prefill_has_key_cache_params) {
    std::vector<int32_t> zero_indices = {0, 0, 0, 0};
    auto write_status =
        prefill_inputs_map["key_cache_start_indices"].Write<int32_t>(
            absl::MakeSpan(zero_indices));
    if (!write_status.HasValue()) {
      return absl::InternalError(write_status.Error().Message());
    }
  }
  if (prefill_has_value_cache_params) {
    std::vector<int32_t> zero_indices = {0, 0, 0, 0};
    auto write_status =
        prefill_inputs_map["value_cache_start_indices"].Write<int32_t>(
            absl::MakeSpan(zero_indices));
    if (!write_status.HasValue()) {
      return absl::InternalError(write_status.Error().Message());
    }
  }

  {
    auto emb_type_exp =
        compiled_model.GetOutputTensorType(kPrefill, "embedded_input");
    if (emb_type_exp.HasValue()) {
      auto emb_type = emb_type_exp.Value();
      std::string dims_str = absl::StrJoin(emb_type.Layout().Dimensions(), " ");
      ABSL_LOG(WARNING) << "[JETSKI] Host CompiledModel: embedded_input shape: "
                        << dims_str << ", Rank: " << emb_type.Layout().Rank();
    } else {
      ABSL_LOG(ERROR)
          << "[JETSKI] Failed to get output tensor type for embedded_input";
    }
  }

  auto prefill_emb_out_or =
      compiled_model.CreateOutputBuffer(kPrefill, "embedded_input");
  if (!prefill_emb_out_or.HasValue()) {
    return absl::InternalError(prefill_emb_out_or.Error().Message());
  }
  prefill_outputs_map["embedded_input"] = std::move(prefill_emb_out_or.Value());

  for (int i = 0; i < config.n_layers; ++i) {
    auto k_out_dup_or = kv_cache.key_caches_a[i].Duplicate();
    if (!k_out_dup_or.HasValue()) {
      return absl::InternalError(k_out_dup_or.Error().Message());
    }
    prefill_outputs_map[kv_cache.cached_k_out_names[i]] =
        std::move(k_out_dup_or.Value());

    auto v_out_dup_or = kv_cache.value_caches_a[i].Duplicate();
    if (!v_out_dup_or.HasValue()) {
      return absl::InternalError(v_out_dup_or.Error().Message());
    }
    prefill_outputs_map[kv_cache.cached_v_out_names[i]] =
        std::move(v_out_dup_or.Value());
  }

  return absl::OkStatus();
}

absl::Status SetupDecodeBuffers(
    ::litert::CompiledModel& compiled_model,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        decode_inputs_map_common,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        decode_outputs_map_common,
    bool& has_key_cache_params, bool& has_value_cache_params) {
  return SetupStandardBuffers(compiled_model, kDecode, decode_inputs_map_common,
                              decode_outputs_map_common, has_key_cache_params,
                              has_value_cache_params);
}

absl::StatusOr<int32_t> ExecutePrefill(
    ::litert::CompiledModel& compiled_model,
    const ::litert::tensor::examples::gemma3::Config& config,
    const std::vector<int32_t>& input_tokens,
    const std::vector<float>& prefill_mask,
    const std::vector<float>& rope_global_cos,
    const std::vector<float>& rope_global_sin,
    const std::vector<float>& rope_local_cos,
    const std::vector<float>& rope_local_sin,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        prefill_inputs_map,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        prefill_outputs_map,
    ::litert::tensor::examples::PrefillTiming& prefill_timing) {
  prefill_timing.prefill.StartLap();
  prefill_timing.uploads.StartLap();

  auto StagePrefillInputs =
      [&](const std::vector<int32_t>& ids, const std::vector<float>& mask,
          const std::vector<float>& r_g_cos, const std::vector<float>& r_g_sin,
          const std::vector<float>& r_l_cos, const std::vector<float>& r_l_sin,
          int slice_idx) -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(
        WriteOptional(prefill_inputs_map, "input_ids", absl::MakeSpan(ids)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "sliding_attention_mask", absl::MakeSpan(mask)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "global_attention_mask", absl::MakeSpan(mask)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "rope_global_cos", absl::MakeSpan(r_g_cos)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "rope_global_sin", absl::MakeSpan(r_g_sin)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "rope_local_cos", absl::MakeSpan(r_l_cos)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        prefill_inputs_map, "rope_local_sin", absl::MakeSpan(r_l_sin)));
    const std::vector<int32_t> slice_vec = {slice_idx};
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(prefill_inputs_map, "slice_index",
                                             absl::MakeSpan(slice_vec)));
    return absl::OkStatus();
  };

  LRT_TENSOR_RETURN_IF_ERROR(
      StagePrefillInputs(input_tokens, prefill_mask, rope_global_cos,
                         rope_global_sin, rope_local_cos, rope_local_sin,
                         static_cast<int>(input_tokens.size()) - 1));
  prefill_timing.uploads.StopLap();

  prefill_timing.run.StartLap();
  ::litert::tensor::examples::Timer::Get("run_prefill").StartLap();
  auto run_status =
      compiled_model.Run(kPrefill, prefill_inputs_map, prefill_outputs_map);
  if (!run_status.HasValue()) {
    return absl::InternalError(run_status.Error().Message());
  }
  prefill_timing.run.StopLap();

  prefill_timing.readback.StartLap();
  int32_t current_token = 0;
  {
    auto host_mem_or = prefill_outputs_map["output"].Lock(
        ::litert::TensorBuffer::LockMode::kRead);
    if (!host_mem_or.HasValue()) {
      return absl::InternalError(host_mem_or.Error().Message());
    }
    void* host_mem = host_mem_or.Value();
    auto unlock = absl::MakeCleanup(
        [&prefill_outputs_map] { prefill_outputs_map["output"].Unlock(); });
    const float* locked_prefill_logits =
        reinterpret_cast<const float*>(host_mem);
    std::string logits_str = "";
    for (int i = 0; i < 10; ++i) {
      absl::StrAppend(&logits_str, locked_prefill_logits[i], " ");
    }
    ABSL_LOG(INFO) << "Prefill logits (first 10): " << logits_str;

    float max_logit = locked_prefill_logits[0];
    int max_idx = 0;
    for (int i = 1; i < config.vocab_size; ++i) {
      if (locked_prefill_logits[i] > max_logit) {
        max_logit = locked_prefill_logits[i];
        max_idx = i;
      }
    }
    ABSL_LOG(INFO) << "Prefill max logit: " << max_logit << " at index "
                   << max_idx;

    current_token = max_idx;
  }
  {
    auto host_mem_or = prefill_outputs_map["embedded_input"].Lock(
        ::litert::TensorBuffer::LockMode::kRead);
    if (!host_mem_or.HasValue()) {
      return absl::InternalError(host_mem_or.Error().Message());
    }
    void* host_mem = host_mem_or.Value();
    auto unlock = absl::MakeCleanup([&prefill_outputs_map] {
      prefill_outputs_map["embedded_input"].Unlock();
    });
    const float* locked_prefill_emb = reinterpret_cast<const float*>(host_mem);
    std::string emb_norms_str = "";
    int seq_len = static_cast<int>(input_tokens.size());
    for (int t = 0; t < seq_len; ++t) {
      float abs_sum = 0.0f;
      for (int d = 0; d < config.emb_dim; ++d) {
        abs_sum += std::abs(locked_prefill_emb[t * config.emb_dim + d]);
      }
      absl::StrAppend(&emb_norms_str, "tok_", t, "(id=", input_tokens[t],
                      ")=", abs_sum, " ");
    }
    std::string first_10_str = "";
    for (size_t i = 0; i < 10; ++i) {
      absl::StrAppend(&first_10_str, locked_prefill_emb[i], " ");
    }
    ABSL_LOG(WARNING) << "GPU Prefill embedded_input absolute sums: [ "
                      << emb_norms_str << "]";
    ABSL_LOG(WARNING) << "GPU Prefill embedded_input (first 10): [ "
                      << first_10_str << "]";
  }
  prefill_timing.readback.StopLap();
  ::litert::tensor::examples::Timer::Get("run_prefill").StopLap();

  prefill_timing.prefill.StopLap();
  return current_token;
}

absl::Status ExecuteDecodeLoop(
    ::litert::CompiledModel& compiled_model,
    const ::litert::tensor::examples::gemma3::Config& config,
    int32_t& current_token, size_t max_tokens, int raw_seq_len, int max_seq_len,
    float rope_local_base, bool benchmark,
    const ::litert::tensor::examples::GemmaTokenizerSP& tokenizer,
    Gemma3KVCache& kv_cache,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        decode_inputs_map_common,
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>&
        decode_outputs_map_common,
    bool has_key_cache_params, bool has_value_cache_params, size_t& step,
    ::litert::tensor::examples::DecodeTiming& decode_timing) {
  decode_timing.decode.StartLap();

  // Pre-allocate reusable static host buffers outside the generative hot loop!
  std::vector<float> decode_mask(max_seq_len, -10000.0f);
  for (int j = 0; j < raw_seq_len; ++j) {
    decode_mask[j] = 0.0f;
  }
  std::vector<float> dec_rope_global_cos(config.head_dim);
  std::vector<float> dec_rope_global_sin(config.head_dim);
  std::vector<float> dec_rope_local_cos(config.head_dim);
  std::vector<float> dec_rope_local_sin(config.head_dim);

  auto StageDecodeInputs =
      [&](const std::vector<int32_t>& ids, const std::vector<float>& mask,
          const std::vector<float>& r_g_cos, const std::vector<float>& r_g_sin,
          const std::vector<float>& r_l_cos,
          const std::vector<float>& r_l_sin) -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(decode_inputs_map_common,
                                             "input_ids", absl::MakeSpan(ids)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(decode_inputs_map_common,
                                             "sliding_attention_mask",
                                             absl::MakeSpan(mask)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(decode_inputs_map_common,
                                             "global_attention_mask",
                                             absl::MakeSpan(mask)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        decode_inputs_map_common, "rope_global_cos", absl::MakeSpan(r_g_cos)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        decode_inputs_map_common, "rope_global_sin", absl::MakeSpan(r_g_sin)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        decode_inputs_map_common, "rope_local_cos", absl::MakeSpan(r_l_cos)));
    LRT_TENSOR_RETURN_IF_ERROR(WriteOptional(
        decode_inputs_map_common, "rope_local_sin", absl::MakeSpan(r_l_sin)));
    return absl::OkStatus();
  };

  step = 0;
  for (; step < max_tokens; ++step) {
    if (!benchmark && (current_token == 1 || current_token == 106)) {
      break;  // Generative completion EOS/EOT token hit!
    }
    if (!benchmark) {
      std::string word = tokenizer.DecodeToken(current_token);
      std::cout << word << std::flush;
    }

    decode_timing.cpu_prep.StartLap();

    int cache_len = raw_seq_len + step;
    decode_mask[cache_len] = 0.0f;  // Unmasks exactly the newly added column!

    ::litert::tensor::examples::gemma3::RopeCosSin(
        cache_len, 1, config.rope_global_base,
        absl::MakeSpan(dec_rope_global_cos),
        absl::MakeSpan(dec_rope_global_sin));
    ::litert::tensor::examples::gemma3::RopeCosSin(
        cache_len, 1, rope_local_base, absl::MakeSpan(dec_rope_local_cos),
        absl::MakeSpan(dec_rope_local_sin));

    std::vector<int32_t> key_start_indices = {
        0, 0, static_cast<int32_t>(cache_len), 0};
    std::vector<int32_t> value_start_indices = {
        0, 0, 0, static_cast<int32_t>(cache_len)};

    if (has_key_cache_params) {
      auto status =
          decode_inputs_map_common["key_cache_start_indices"].Write<int32_t>(
              absl::MakeSpan(key_start_indices));
      if (!status.HasValue()) {
        return absl::InternalError(status.Error().Message());
      }
    }
    if (has_value_cache_params) {
      auto status =
          decode_inputs_map_common["value_cache_start_indices"].Write<int32_t>(
              absl::MakeSpan(value_start_indices));
      if (!status.HasValue()) {
        return absl::InternalError(status.Error().Message());
      }
    }

    std::vector<int32_t> current_token_vec = {
        static_cast<int32_t>(current_token)};
    LRT_TENSOR_RETURN_IF_ERROR(StageDecodeInputs(
        current_token_vec, decode_mask, dec_rope_global_cos,
        dec_rope_global_sin, dec_rope_local_cos, dec_rope_local_sin));

    // Double buffering ping-pong swap!
    bool is_even = (step % 2 == 0);
    auto& current_input_keys =
        is_even ? kv_cache.key_caches_a : kv_cache.key_caches_b;
    auto& current_input_values =
        is_even ? kv_cache.value_caches_a : kv_cache.value_caches_b;
    auto& current_output_keys =
        is_even ? kv_cache.key_caches_b : kv_cache.key_caches_a;
    auto& current_output_values =
        is_even ? kv_cache.value_caches_b : kv_cache.value_caches_a;

    // Build temporary maps for this step by duplicating handles
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
        step_inputs;
    ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
        step_outputs;

    for (const auto& [name, buffer] : decode_inputs_map_common) {
      auto dup_or = buffer.Duplicate();
      if (!dup_or.HasValue()) {
        return absl::InternalError(dup_or.Error().Message());
      }
      step_inputs[name] = std::move(dup_or.Value());
    }
    for (const auto& [name, buffer] : decode_outputs_map_common) {
      auto dup_or = buffer.Duplicate();
      if (!dup_or.HasValue()) {
        return absl::InternalError(dup_or.Error().Message());
      }
      step_outputs[name] = std::move(dup_or.Value());
    }

    for (int i = 0; i < config.n_layers; ++i) {
      auto k_in_dup_or = current_input_keys[i].Duplicate();
      if (!k_in_dup_or.HasValue()) {
        return absl::InternalError(k_in_dup_or.Error().Message());
      }
      step_inputs[kv_cache.cached_k_in_names[i]] =
          std::move(k_in_dup_or.Value());

      auto v_in_dup_or = current_input_values[i].Duplicate();
      if (!v_in_dup_or.HasValue()) {
        return absl::InternalError(v_in_dup_or.Error().Message());
      }
      step_inputs[kv_cache.cached_v_in_names[i]] =
          std::move(v_in_dup_or.Value());

      auto k_out_dup_or = current_output_keys[i].Duplicate();
      if (!k_out_dup_or.HasValue()) {
        return absl::InternalError(k_out_dup_or.Error().Message());
      }
      step_outputs[kv_cache.cached_k_out_names[i]] =
          std::move(k_out_dup_or.Value());

      auto v_out_dup_or = current_output_values[i].Duplicate();
      if (!v_out_dup_or.HasValue()) {
        return absl::InternalError(v_out_dup_or.Error().Message());
      }
      step_outputs[kv_cache.cached_v_out_names[i]] =
          std::move(v_out_dup_or.Value());
    }

    decode_timing.cpu_prep.StopLap();

    decode_timing.run.StartLap();
    auto run_status = compiled_model.Run(kDecode, step_inputs, step_outputs);
    if (!run_status.HasValue()) {
      return absl::InternalError(run_status.Error().Message());
    }

    // Retrieve token ID directly
    auto host_mem_or = decode_outputs_map_common["output"].Lock(
        ::litert::TensorBuffer::LockMode::kRead);
    if (!host_mem_or.HasValue()) {
      return absl::InternalError(host_mem_or.Error().Message());
    }
    void* host_mem = host_mem_or.Value();
    auto unlock = absl::MakeCleanup([&decode_outputs_map_common] {
      decode_outputs_map_common["output"].Unlock();
    });
    const int32_t* locked_decode_token =
        reinterpret_cast<const int32_t*>(host_mem);
    decode_timing.run.StopLap();

    decode_timing.argmax.StartLap();
    int32_t token_id = locked_decode_token[0];
    current_token = token_id;
    decode_timing.argmax.StopLap();
  }
  decode_timing.decode.StopLap();

  return absl::OkStatus();
}

absl::Status RunGemma3Inference(
    const std::string& weights_path, const std::string& tokenizer_path,
    const std::string& prompt_text, size_t max_tokens,
    ::litert::tensor::examples::TfliteLoader::QuantizedLoadMode weight_mode) {
  volatile int dummy_recompile_token = 99199;
  static_cast<void>(dummy_recompile_token);
  ::litert::tensor::examples::gemma3::Config config;
  config.vocab_size = 262144;
  config.emb_dim = 1152;
  config.hidden_dim = 6912;
  config.head_dim = 256;
  config.n_heads = 4;
  config.n_layers = 26;
  config.n_kv_groups = 1;
  config.sliding_window = 512;
  config.rope_local_base = 10000.0f;
  config.rope_global_base = 1000000.0f;
  config.query_pre_attn_scalar = 256.0f;
  config.rms_norm_eps = 1e-6f;

  std::vector<int32_t> input_tokens;
  ::litert::tensor::examples::PrefillTiming prefill_timing;

  auto tokenizer_or =
      ::litert::tensor::examples::GemmaTokenizerSP::Load(tokenizer_path);
  if (!tokenizer_or.ok()) {
    return absl::InternalError("Failed to load tokenizer!");
  }
  auto tokenizer = std::move(*tokenizer_or);

  auto weights_or = LoadGemma3Weights(weights_path, config, weight_mode);
  if (!weights_or.ok()) return weights_or.status();
  auto weights = std::move(*weights_or);

  bool benchmark = absl::GetFlag(FLAGS_benchmark);
  size_t benchmark_prefill = absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  size_t benchmark_decode = absl::GetFlag(FLAGS_benchmark_decode_tokens);

  TokenizeInput(prompt_text, tokenizer, benchmark, benchmark_prefill,
                benchmark_decode, input_tokens, max_tokens);

  const int seq_len = static_cast<int>(input_tokens.size());
  const int raw_seq_len = seq_len;
  const int max_seq_len = seq_len + max_tokens;

  prefill_timing.prefill.StartLap();
  prefill_timing.cpu_prep.StartLap();

  std::vector<float> prefill_mask(seq_len * seq_len, -10000.0f);
  for (int i = 0; i < raw_seq_len; ++i) {
    std::fill_n(prefill_mask.data() + i * seq_len, i + 1, 0.0f);
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
  prefill_timing.cpu_prep.StopLap();
  prefill_timing.prefill.StopLap();

  auto gpu_tflite_file = absl::StrCat(weights_path, ".tflite");
  auto build_status = BuildAndSaveJitModel(gpu_tflite_file, config, weights,
                                           seq_len, max_seq_len);
  if (!build_status.ok()) return build_status;
  weights.clear();

  auto env_or = ::litert::Environment::Create({});
  if (!env_or.HasValue()) {
    return absl::InternalError(
        "Failed to instantiate LiteRT Environment context!");
  }
  auto env = std::move(env_or.Value());

  std::string acc_flag = absl::GetFlag(FLAGS_accelerator);
  bool fp16 = absl::GetFlag(FLAGS_fp16);
  auto compiled_model_or =
      CreateCompiledModel(env, gpu_tflite_file, acc_flag, fp16);
  if (!compiled_model_or.ok()) return compiled_model_or.status();
  auto compiled_model = std::move(*compiled_model_or);

  auto kv_cache_or = SetupKVCache(compiled_model, config);
  if (!kv_cache_or.ok()) return kv_cache_or.status();
  auto kv_cache = std::move(*kv_cache_or);

  ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
      prefill_inputs_map;
  ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
      prefill_outputs_map;
  bool prefill_has_key_cache_params = false;
  bool prefill_has_value_cache_params = false;

  auto setup_prefill_status = SetupPrefillBuffers(
      compiled_model, config, max_seq_len, kv_cache, prefill_inputs_map,
      prefill_outputs_map, prefill_has_key_cache_params,
      prefill_has_value_cache_params);
  if (!setup_prefill_status.ok()) return setup_prefill_status;

  ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
      decode_inputs_map_common;
  ::litert::FlatHashMap<::litert::StringView, ::litert::TensorBuffer>
      decode_outputs_map_common;
  bool has_key_cache_params = false;
  bool has_value_cache_params = false;

  auto setup_decode_status = SetupDecodeBuffers(
      compiled_model, decode_inputs_map_common, decode_outputs_map_common,
      has_key_cache_params, has_value_cache_params);
  if (!setup_decode_status.ok()) return setup_decode_status;

  auto current_token_or = ExecutePrefill(
      compiled_model, config, input_tokens, prefill_mask, rope_global_cos,
      rope_global_sin, rope_local_cos, rope_local_sin, prefill_inputs_map,
      prefill_outputs_map, prefill_timing);
  if (!current_token_or.ok()) return current_token_or.status();
  int32_t current_token = *current_token_or;

  if (!benchmark) {
    std::cout << "[Inference Output Start] " << prompt_text << std::flush;
  } else {
    std::cout << "[Benchmark Start] Prefill=" << benchmark_prefill
              << " Decode=" << benchmark_decode << std::endl;
  }

  ::litert::tensor::examples::DecodeTiming decode_timing;
  size_t step = 0;
  auto decode_status = ExecuteDecodeLoop(
      compiled_model, config, current_token, max_tokens, raw_seq_len,
      max_seq_len, rope_local_base, benchmark, tokenizer, kv_cache,
      decode_inputs_map_common, decode_outputs_map_common, has_key_cache_params,
      has_value_cache_params, step, decode_timing);
  if (!decode_status.ok()) return decode_status;

  std::cout << std::endl;
  if (!benchmark) {
    std::cout << "[Inference End] Finished successfully." << std::endl;
  } else {
    std::cout << "[Benchmark End] Finished successfully." << std::endl;
  }

  absl::Duration prefill_duration = prefill_timing.prefill.Duration();
  absl::Duration decode_duration = decode_timing.decode.Duration();
  double prefill_sec = absl::ToDoubleSeconds(prefill_duration);
  double prefill_tok_per_sec =
      prefill_sec > 0 ? (static_cast<double>(input_tokens.size()) / prefill_sec)
                      : 0.0;

  std::cerr << "Prefilled " << input_tokens.size() << " tokens in "
            << absl::ToDoubleMilliseconds(prefill_duration) << " ms ("
            << (absl::ToDoubleMilliseconds(prefill_duration) /
                input_tokens.size())
            << " ms/tok, " << prefill_tok_per_sec << " tokens/sec).\n";
  size_t denom = step > 0 ? step : 1;
  double decode_sec = absl::ToDoubleSeconds(decode_duration);
  double decode_tok_per_sec =
      decode_sec > 0 ? (static_cast<double>(step) / decode_sec) : 0.0;

  absl::Duration total_get_output_time = absl::ZeroDuration();
  absl::Duration total_set_input_time = absl::ZeroDuration();

  std::cerr << "Decoded " << step << " tokens in "
            << absl::ToDoubleMilliseconds(decode_duration) << " ms ("
            << absl::ToDoubleMilliseconds(decode_duration / denom)
            << " ms/tok, " << decode_tok_per_sec << " tokens/sec).\n"
            << "  Total GetOutput overhead: "
            << absl::ToDoubleMilliseconds(total_get_output_time) << " ms ("
            << absl::ToDoubleMilliseconds(total_get_output_time) / denom
            << " ms/tok)\n"
            << "  Total SetInput overhead: "
            << absl::ToDoubleMilliseconds(total_set_input_time) << " ms ("
            << absl::ToDoubleMilliseconds(total_set_input_time) / denom
            << " ms/tok)\n"
            << prefill_timing.Stats() << "\n"
            << decode_timing.Stats() << "\n";

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, /*remove_flags=*/true);
  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer_path);
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  size_t max_tokens = absl::GetFlag(FLAGS_max_tokens);

  if (weights_path.empty() || tokenizer_path.empty()) {
    std::cerr << "Error: --weights_path and --tokenizer_path are required"
              << std::endl;
    std::cerr << "Usage: " << argv[0] << " --weights_path=/path/to/model.tflite"
              << " --tokenizer_path=/path/to/tokenizer.model" << std::endl;
    return 1;
  }

  ::litert::tensor::examples::TfliteLoader::QuantizedLoadMode weight_mode =
      absl::GetFlag(FLAGS_weight_mode);

  absl::Status status = RunGemma3Inference(weights_path, tokenizer_path, prompt,
                                           max_tokens, weight_mode);
  if (!status.ok()) {
    std::cerr << "Error: " << status << std::endl;
    return 1;
  }
  return 0;
}
