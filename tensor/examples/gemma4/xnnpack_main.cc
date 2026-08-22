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
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/gemma4/safetensor_loader.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/examples/utils/initialization.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto
#include "tflite/delegates/xnnpack/weight_cache.h"

ABSL_FLAG(std::string, weights, "/tmp/gemma4/model.safetensors",
          "Path to safetensor weights file or directory.");
ABSL_FLAG(std::string, tokenizer, "/tmp/gemma4/tokenizer.model",
          "Path to SentencePiece tokenizer model file.");
ABSL_FLAG(std::string, prompt, "Write a short poem about coding.",
          "Prompt to run.");
ABSL_FLAG(int, max_tokens, 100, "Maximum number of tokens to generate.");
ABSL_FLAG(int, num_threads, 4, "Number of threads for XNNPack.");
ABSL_FLAG(bool, verbose, false, "Verbose logging.");
ABSL_FLAG(litert::tensor::examples::TokenPrinter::Kind, print,
          litert::tensor::examples::TokenPrinter::Kind::kTokens,
          "Output mode (tokens or progress).");
ABSL_FLAG(std::string, weight_cache, "", "Path to XNNPack weight cache file.");
ABSL_FLAG(std::string, perfetto_output, "",
          "Path to output Perfetto trace file.");

namespace litert::tensor::examples::gemma4 {
namespace {

constexpr int32_t kStartOfTurnToken = 105;
constexpr int32_t kEndOfTurnToken = 106;

using ::litert::tensor::PerfettoSession;
using ::litert::tensor::examples::DecodeTiming;
using ::litert::tensor::examples::GemmaTokenizerSP;
using ::litert::tensor::examples::PrefillTiming;
using ::litert::tensor::examples::SafetensorLoader;
using ::litert::tensor::examples::Timer;
using ::litert::tensor::examples::TokenPrinter;

using XnnTensor = Tensor<XnnpackMixinTag>;

absl::Status MapGemma4WeightIdentifiers(
    tflite::xnnpack::MMapWeightCacheProvider& cache_provider,
    const absl::flat_hash_map<std::string, TensorHandle>& weights_handle) {
  TRACE_EVENT(kTensorApiCategory, "MapGemma4WeightIdentifiers");
  for (const auto& [name, tensor] : weights_handle) {
    LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    auto locked = buffer.Lock();
    uint64_t identifier = static_cast<uint64_t>(std::hash<std::string>{}(name));
    if (!cache_provider.MapBufferIdentifier(locked.data(), locked.size(),
                                            identifier)) {
      return absl::InternalError(
          absl::StrCat("Failed to map weight identifier for ", name));
    }
  }
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, std::string> GetGemma4WeightMapping(
    int n_layers) {
  absl::flat_hash_map<std::string, std::string> mapping;

  // Embedding
  mapping["model.language_model.embed_tokens.weight"] =
      "model.embed_tokens.weight";

  // Final norm
  mapping["model.language_model.norm.weight"] = "model.norm.weight";

  // Per-layer input weights
  mapping["model.language_model.embed_tokens_per_layer.weight"] =
      "model.embed_tokens_per_layer.weight";
  mapping["model.language_model.per_layer_model_projection.weight"] =
      "model.per_layer_model_projection.weight";
  mapping["model.language_model.per_layer_projection_norm.weight"] =
      "model.per_layer_projection_norm.weight";

  // Per-layer weights
  for (int i = 0; i < n_layers; ++i) {
    std::string hf_prefix = absl::StrCat("model.language_model.layers.", i);
    std::string model_prefix = absl::StrCat("model.layers.", i);

    // Attention weights
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.v_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.v_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.o_proj.weight")] =
        absl::StrCat(model_prefix, ".self_attn.o_proj.weight");

    // QK normalization
    mapping[absl::StrCat(hf_prefix, ".self_attn.q_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.q_norm.weight");
    mapping[absl::StrCat(hf_prefix, ".self_attn.k_norm.weight")] =
        absl::StrCat(model_prefix, ".self_attn.k_norm.weight");

    // MLP weights
    mapping[absl::StrCat(hf_prefix, ".mlp.gate_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.gate_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.up_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.up_proj.weight");
    mapping[absl::StrCat(hf_prefix, ".mlp.down_proj.weight")] =
        absl::StrCat(model_prefix, ".mlp.down_proj.weight");

    // Layer norms
    mapping[absl::StrCat(hf_prefix, ".input_layernorm.weight")] =
        absl::StrCat(model_prefix, ".input_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_attention_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_attention_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".pre_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".pre_feedforward_layernorm.weight");
    mapping[absl::StrCat(hf_prefix, ".post_feedforward_layernorm.weight")] =
        absl::StrCat(model_prefix, ".post_feedforward_layernorm.weight");

    // Per-layer input integration weights
    mapping[absl::StrCat(hf_prefix, ".per_layer_input_gate.weight")] =
        absl::StrCat(model_prefix, ".per_layer_input_gate.weight");
    mapping[absl::StrCat(hf_prefix, ".per_layer_projection.weight")] =
        absl::StrCat(model_prefix, ".per_layer_projection.weight");
    mapping[absl::StrCat(hf_prefix, ".post_per_layer_input_norm.weight")] =
        absl::StrCat(model_prefix, ".post_per_layer_input_norm.weight");

    // Layer Scalar (replaces Gemma3 skip_scale)
    mapping[absl::StrCat(hf_prefix, ".layer_scalar")] =
        absl::StrCat(model_prefix, ".layer_scalar");
  }

  return mapping;
}

// Slices the combined per-layer model projection weight matrix
// ("model.per_layer_model_projection.weight") of shape
// [num_layers, per_layer_input_dim, embed_dim] into individual per-layer 2D
// weight tensors ("model.layers.<l>.per_layer_model_projection.weight") of
// shape [per_layer_input_dim, embed_dim] for each layer `l`.
absl::Status SlicePerLayerModelProjectionWeights(
    const Config& config,
    absl::flat_hash_map<std::string, TensorHandle>& weights_handle) {
  TRACE_EVENT(kTensorApiCategory, "SlicePerLayerModelProjectionWeights");
  auto proj_w_it =
      weights_handle.find("model.per_layer_model_projection.weight");
  if (proj_w_it == weights_handle.end()) {
    return absl::OkStatus();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & proj_w_buf,
                              proj_w_it->second.GetBuffer());
  auto proj_locked = proj_w_buf.Lock();
  const std::byte* proj_w_bytes = proj_locked.data();
  if (proj_w_bytes == nullptr) {
    return absl::InternalError(
        "Null buffer data for model.per_layer_model_projection.weight");
  }

  const size_t layer_w_bytes = static_cast<size_t>(config.per_layer_input_dim) *
                               config.embed_dim * sizeof(float);

  for (int l = 0; l < config.num_layers; ++l) {
    const std::byte* layer_bytes = proj_w_bytes + l * layer_w_bytes;
    std::string name =
        absl::StrCat("model.layers.", l, ".per_layer_model_projection.weight");
    weights_handle[name] = Tensor({
        .name = name,
        .type = Type::kFP32,
        .shape = {config.per_layer_input_dim, config.embed_dim},
        .buffer = std::make_shared<SpanCpuBuffer>(layer_bytes, layer_w_bytes),
    });
  }
  return absl::OkStatus();
}

absl::StatusOr<LockedBufferSpan<const float>> GetOrDequantizeEmbeddingTableFp32(
    const XnnTensor& tensor) {
  TRACE_EVENT(kTensorApiCategory, "GetOrDequantizeEmbeddingTableFp32");
  LRT_TENSOR_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
  LockedBufferSpan<const float> data = LockedBufferSpan<const float>::Empty();
  if (tensor.GetQuantization() != nullptr) {
    XnnTensor dequantized = Dequantize(tensor);
    LRT_TENSOR_ASSIGN_OR_RETURN(auto runner,
                                XnnpackRunner::Create({dequantized}));
    LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
    return runner.ReadOutputAs<float>(dequantized);
  }
  return buffer.Lock().As<const float>();
}

std::vector<float> EmbeddingLookupCpu(const std::vector<int32_t>& tokens,
                                      absl::Span<const float> embedding_table,
                                      const int vocab_size, const int emb_dim) {
  const size_t seq_len = tokens.size();
  std::vector<float> embeddings;
  embeddings.reserve(seq_len * emb_dim);
  for (size_t i = 0; i < seq_len; ++i) {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= vocab_size) {
      ABSL_LOG(WARNING) << "Token ID " << token_id << " out of range [0, "
                        << vocab_size << "), using 0";
      token_id = 0;
    }
    const absl::Span<const float> src =
        embedding_table.subspan(token_id * emb_dim, emb_dim);
    embeddings.insert(embeddings.end(), src.begin(), src.end());
  }
  return embeddings;
}

std::vector<std::vector<float>> GetPerLayerTokenEmbeddingsCpu(
    const std::vector<int32_t>& tokens,
    absl::Span<const float> emb_per_layer_table, const int vocab_size,
    const int num_layers, const int per_layer_input_dim) {
  size_t seq_len = tokens.size();
  size_t total_per_layer_dim =
      static_cast<size_t>(num_layers) * per_layer_input_dim;
  std::vector<std::vector<float>> result(
      num_layers, std::vector<float>(seq_len * per_layer_input_dim));

  for (size_t s = 0; s < seq_len; ++s) {
    int32_t token_id = tokens[s];
    if (token_id < 0 || token_id >= vocab_size) {
      token_id = 0;
    }
    const float* src = emb_per_layer_table.data() +
                       static_cast<size_t>(token_id) * total_per_layer_dim;
    for (int l = 0; l < num_layers; ++l) {
      std::copy(src + l * per_layer_input_dim,
                src + (l + 1) * per_layer_input_dim,
                result[l].data() + s * per_layer_input_dim);
    }
  }
  return result;
}

absl::Status FillAttentionMask(const Shape& shape, const absl::Span<float> mask,
                               const bool is_local,
                               const int sliding_window_size) {
  if (shape.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask output shape must have at least 2 dims, got %zu",
        shape.size()));
  }

  const int64_t seq_q = shape[shape.size() - 2];
  const int64_t seq_k = shape[shape.size() - 1];
  if (seq_q <= 0 || seq_k <= 0 || seq_q != seq_k) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FillAttentionMask expects output shape [..., S, S] with S>0; got [%s]",
        absl::StrJoin(shape, ", ")));
  }
  if (std::any_of(shape.begin(), shape.end() - 2,
                  [](auto d) { return d <= 0; })) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask does not support non-positive "
                        "leading dims. Got shape [%s]",
                        absl::StrJoin(shape, ", ")));
  }
  const int64_t leading =
      std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<>());
  const int64_t matrix_size = seq_q * seq_k;
  const int64_t tensor_size = leading * matrix_size;

  if (mask.size() != tensor_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FillAttentionMask output mask should hold %" PRIi64
                        " elements but holds %zu",
                        tensor_size, mask.size()));
  }

  const float neg_inf = std::numeric_limits<float>::lowest();

  for (int64_t b = 0; b < leading; ++b) {
    const int64_t base = b * matrix_size;
    for (int64_t i = 0; i < seq_q; ++i) {
      const int64_t I = i * seq_q;
      for (int64_t j = 0; j < seq_q; ++j) {
        const bool is_causal_masked = (j > i);
        const bool is_sliding_masked = is_local && (sliding_window_size > 0) &&
                                       (i - j >= sliding_window_size);
        mask[static_cast<size_t>(base + I + j)] =
            (is_causal_masked || is_sliding_masked) ? neg_inf : 0.0f;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<XnnTensor> AttentionMask(Shape shape, const bool is_local,
                                        const int sliding_window_size) {
  auto buffer = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  XnnTensor mask({.name = "attention_mask",
                  .type = Type::kFP32,
                  .shape = std::move(shape),
                  .buffer = buffer});
  LRT_TENSOR_RETURN_IF_ERROR(FillAttentionMask(
      mask.GetShape(), buffer->Span<float>(), is_local, sliding_window_size));
  return mask;
}

void AppendTokenToKvCache(std::vector<float>& cache_buf,
                          absl::Span<const float> new_kv, int num_kv_heads,
                          int cache_len, int head_dim) {
  if (num_kv_heads == 1) {
    std::copy_n(new_kv.data(), head_dim,
                cache_buf.data() + cache_len * head_dim);
  } else {
    for (int h = num_kv_heads - 1; h >= 0; --h) {
      if (h > 0) {
        std::copy_backward(cache_buf.data() + h * cache_len * head_dim,
                           cache_buf.data() + (h + 1) * cache_len * head_dim,
                           cache_buf.data() + h * (cache_len + 1) * head_dim +
                               cache_len * head_dim);
      }
      std::copy_n(new_kv.data() + h * head_dim, head_dim,
                  cache_buf.data() + h * (cache_len + 1) * head_dim +
                      cache_len * head_dim);
    }
  }
}

struct LoadedTensors {
  absl::flat_hash_map<std::string, TensorHandle> weights_handle;
  LockedBufferSpan<const float> embedding_table_fp32;
  LockedBufferSpan<const float> emb_per_layer_table_fp32;
};

absl::StatusOr<LoadedTensors> LoadWeightsAndPrepareTensors(
    const SafetensorLoader& loader, const Config& config) {
  TRACE_EVENT(kTensorApiCategory, "LoadWeightsAndPrepareTensors");
  auto weight_mapping = GetGemma4WeightMapping(config.num_layers);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto weights_handle,
                              loader.LoadWeightsWithMapping(weight_mapping));
  LRT_TENSOR_RETURN_IF_ERROR(
      SlicePerLayerModelProjectionWeights(config, weights_handle));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      LockedBufferSpan<const float> embedding_table_fp32,
      GetOrDequantizeEmbeddingTableFp32(
          weights_handle["model.embed_tokens.weight"]));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      LockedBufferSpan<const float> emb_per_layer_table_fp32,
      GetOrDequantizeEmbeddingTableFp32(
          weights_handle["model.embed_tokens_per_layer.weight"]));
  return LoadedTensors{std::move(weights_handle),
                       std::move(embedding_table_fp32),
                       std::move(emb_per_layer_table_fp32)};
}

struct BuiltGraphs {
  Gemma4Inputs<XnnpackMixinTag> prefill_inputs;
  Gemma4Outputs<XnnpackMixinTag> prefill_outputs;
  Gemma4Inputs<XnnpackMixinTag> decode_inputs;
  Gemma4Outputs<XnnpackMixinTag> decode_outputs;
};

absl::StatusOr<Gemma4Inputs<XnnpackMixinTag>> CreateGemma4Inputs(
    const Config& config, int input_seq_len, int kv_cache_len,
    const absl::flat_hash_map<std::string, TensorHandle>& weights_handle,
    bool verbose) {
  const int batch_size = 1;
  Gemma4Inputs<XnnpackMixinTag> inputs;

  inputs.embedded_input =
      XnnTensor({.name = "embedded_input",
                 .type = Type::kFP32,
                 .shape = {batch_size, input_seq_len, config.embed_dim}});

  if (input_seq_len > 1) {
    std::tie(inputs.rope_global_cos, inputs.rope_global_sin) =
        RopeCosSin(input_seq_len, config.global_key_size,
                   config.global_base_frequency, config.global_rope_proportion);
    inputs.rope_global_cos.SetName("rope_global_cos");
    inputs.rope_global_sin.SetName("rope_global_sin");

    std::tie(inputs.rope_local_cos, inputs.rope_local_sin) =
        RopeCosSin(input_seq_len, config.head_dim, config.local_base_frequency,
                   config.local_rope_proportion);
    inputs.rope_local_cos.SetName("rope_local_cos");
    inputs.rope_local_sin.SetName("rope_local_sin");

    LRT_TENSOR_ASSIGN_OR_RETURN(
        inputs.global_attention_mask,
        AttentionMask({1, 1, input_seq_len, input_seq_len},
                      /*is_local=*/false, config.sliding_window_size));
    inputs.global_attention_mask.SetName("global_attention_mask");

    LRT_TENSOR_ASSIGN_OR_RETURN(
        inputs.sliding_attention_mask,
        AttentionMask({1, 1, input_seq_len, input_seq_len},
                      /*is_local=*/true, config.sliding_window_size));
    inputs.sliding_attention_mask.SetName("sliding_attention_mask");
  } else {
    inputs.rope_global_cos.Set({.name = "rope_global_cos",
                                .type = Type::kFP32,
                                .shape = {1, 1, 1, config.global_key_size}});
    inputs.rope_global_sin.Set({.name = "rope_global_sin",
                                .type = Type::kFP32,
                                .shape = {1, 1, 1, config.global_key_size}});
    inputs.rope_local_cos.Set({.name = "rope_local_cos",
                               .type = Type::kFP32,
                               .shape = {1, 1, 1, config.head_dim}});
    inputs.rope_local_sin.Set({.name = "rope_local_sin",
                               .type = Type::kFP32,
                               .shape = {1, 1, 1, config.head_dim}});
    inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                       .type = Type::kFP32,
                                       .shape = {1, 1, 1, kv_cache_len + 1}});
    inputs.global_attention_mask.Set({.name = "global_attention_mask",
                                      .type = Type::kFP32,
                                      .shape = {1, 1, 1, kv_cache_len + 1}});
  }

  inputs.key_caches.reserve(config.num_layers);
  inputs.value_caches.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
    int head_dim = is_global ? config.global_key_size : config.head_dim;
    inputs.key_caches.push_back(XnnTensor(
        {.name = absl::StrCat("key_cache_", i),
         .type = Type::kFP32,
         .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}}));
    inputs.value_caches.push_back(XnnTensor(
        {.name = absl::StrCat("value_cache_", i),
         .type = Type::kFP32,
         .shape = {batch_size, config.num_kv_heads, kv_cache_len, head_dim}}));
  }

  inputs.per_layer_token_embeddings.reserve(config.num_layers);
  for (int l = 0; l < config.num_layers; ++l) {
    inputs.per_layer_token_embeddings.push_back(XnnTensor(
        {.name = absl::StrCat("per_layer_token_embedding_", l),
         .type = Type::kFP32,
         .shape = {batch_size, input_seq_len, config.per_layer_input_dim}}));
  }

  for (auto& [name, xnnpack_tensor] : weights_handle) {
    if (xnnpack_tensor.GetBuffer().ok()) {
      inputs.weights.emplace(name, xnnpack_tensor);
      if (verbose) {
        ABSL_LOG(INFO) << "Added weight: " << name << " shape: ["
                       << absl::StrJoin(xnnpack_tensor.GetShape(), ", ") << "]";
      }
    }
  }

  return inputs;
}

absl::StatusOr<BuiltGraphs> BuildModelGraphs(
    const Config& config, int seq_len,
    const absl::flat_hash_map<std::string, TensorHandle>& weights_handle,
    bool verbose) {
  TRACE_EVENT(kTensorApiCategory, "BuildModelGraphs");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Inputs<XnnpackMixinTag> prefill_inputs,
      CreateGemma4Inputs(config, /*input_seq_len=*/seq_len, /*kv_cache_len=*/0,
                         weights_handle, verbose));
  Gemma4Outputs<XnnpackMixinTag> prefill_outputs =
      BuildGemma4Graph(prefill_inputs, config);

  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Inputs<XnnpackMixinTag> decode_inputs,
      CreateGemma4Inputs(config, /*input_seq_len=*/1, /*kv_cache_len=*/seq_len,
                         weights_handle, /*verbose=*/false));
  Gemma4Outputs<XnnpackMixinTag> decode_outputs =
      BuildGemma4Graph(decode_inputs, config);

  return BuiltGraphs{std::move(prefill_inputs), std::move(prefill_outputs),
                     std::move(decode_inputs), std::move(decode_outputs)};
}

struct CompiledRunners {
  XnnpackRunner prefill_runner;
  XnnpackRunner decode_runner;
};

absl::StatusOr<CompiledRunners> CompileRunners(
    BuiltGraphs& graphs, int num_threads, bool use_weight_cache,
    tflite::xnnpack::MMapWeightCacheProvider* weight_cache_provider) {
  TRACE_EVENT(kTensorApiCategory, "CompileRunners");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto runner,
      XnnpackRunner::Create(graphs.prefill_outputs.GetAllHandles()));
  runner.SetNumThreads(num_threads);
  if (use_weight_cache && weight_cache_provider != nullptr) {
    runner.SetWeightsCache(&weight_cache_provider->GetCacheProvider());
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto decode_runner,
      XnnpackRunner::Create(graphs.decode_outputs.GetAllHandles()));
  decode_runner.SetNumThreads(num_threads);
  if (use_weight_cache && weight_cache_provider != nullptr) {
    decode_runner.SetWeightsCache(&weight_cache_provider->GetCacheProvider());
  }

  if (use_weight_cache && weight_cache_provider != nullptr &&
      weight_cache_provider->CanStartBuildStep()) {
    ABSL_LOG(INFO) << "Building cache.";
    if (!weight_cache_provider->StartBuildStep()) {
      return absl::InternalError(
          "Failed to start build step for XNNPack weight cache.");
    }
    LRT_TENSOR_RETURN_IF_ERROR(runner.PrepareRuntime());
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.PrepareRuntime());
    if (!weight_cache_provider->StopBuildStep()) {
      ABSL_LOG(ERROR) << "Failed to stop build step for XNNPack weight cache.";
    }
    weight_cache_provider->StopBuild();
  }

  return CompiledRunners{std::move(runner), std::move(decode_runner)};
}

absl::StatusOr<int32_t> ExecutePrefillPass(
    XnnpackRunner& runner, Gemma4Inputs<XnnpackMixinTag>& inputs,
    Gemma4Outputs<XnnpackMixinTag>& outputs, const Config& config,
    const std::vector<int32_t>& input_tokens,
    const absl::Span<const float> embedding_table_fp32,
    const absl::Span<const float> emb_per_layer_table_fp32,
    PrefillTiming& prefill_timing, bool verbose) {
  TRACE_EVENT(kTensorApiCategory, "Prefill");
  Timer::LapScope lap_scope = prefill_timing.prefill.Lap();

  int seq_len = static_cast<int>(input_tokens.size());
  std::vector<float> embedded_input;
  std::vector<std::vector<float>> per_layer_tok_embs;
  {
    TRACE_EVENT(kTensorApiCategory, "CpuPrep");
    Timer::LapScope cpu_prep_scope = prefill_timing.cpu_prep.Lap();
    embedded_input = EmbeddingLookupCpu(input_tokens, embedding_table_fp32,
                                        config.vocab_size, config.embed_dim);

    per_layer_tok_embs = GetPerLayerTokenEmbeddingsCpu(
        input_tokens, emb_per_layer_table_fp32, config.vocab_size,
        config.num_layers, config.per_layer_input_dim);
  }

  {
    TRACE_EVENT(kTensorApiCategory, "Uploads");
    Timer::LapScope uploads_scope = prefill_timing.uploads.Lap();
    LRT_TENSOR_RETURN_IF_ERROR(
        runner.SetInput(inputs.embedded_input, embedded_input));

    for (int l = 0; l < config.num_layers; ++l) {
      LRT_TENSOR_RETURN_IF_ERROR(runner.SetInput(
          inputs.per_layer_token_embeddings[l], per_layer_tok_embs[l]));
    }
  }

  {
    TRACE_EVENT(kTensorApiCategory, "Run");
    Timer::LapScope run_scope = prefill_timing.run.Lap();
    LRT_TENSOR_RETURN_IF_ERROR(runner.Run());
  }

  LockedBufferSpan<const float> initial_output_locked =
      LockedBufferSpan<const float>::Empty();
  {
    TRACE_EVENT(kTensorApiCategory, "Readback");
    Timer::LapScope readback_scope = prefill_timing.readback.Lap();
    LRT_TENSOR_ASSIGN_OR_RETURN(initial_output_locked,
                                runner.ReadOutputAs<float>(outputs.logits));
  }

  absl::Span<const float> prefill_logits(
      initial_output_locked.begin() + (seq_len - 1) * config.vocab_size,
      config.vocab_size);

  if (prefill_logits.empty()) {
    return absl::InternalError("Prefill logits span is empty.");
  }

  int32_t current_token =
      absl::c_max_element(prefill_logits) - prefill_logits.begin();
  return current_token;
}

absl::StatusOr<int32_t> ExecuteDecodeStep(
    XnnpackRunner& decode_runner, Gemma4Inputs<XnnpackMixinTag>& decode_inputs,
    Gemma4Outputs<XnnpackMixinTag>& decode_outputs, const Config& config,
    int32_t current_token, int cache_len,
    const absl::Span<const float> embedding_table_fp32,
    const absl::Span<const float> emb_per_layer_table_fp32,
    std::vector<float>& global_cos, std::vector<float>& global_sin,
    std::vector<float>& local_cos, std::vector<float>& local_sin,
    DecodeTiming& decode_timing) {
  TRACE_EVENT(kTensorApiCategory, "Decode");
  Timer::LapScope lap_scope = decode_timing.decode.Lap();

  std::vector<int32_t> token_vec = {current_token};
  std::vector<float> token_embedding;
  std::vector<std::vector<float>> token_per_layer_embs;
  const int32_t seq_k = cache_len + 1;
  std::vector<float> sliding_mask(static_cast<size_t>(seq_k), 0.0f);
  std::vector<float> global_mask(static_cast<size_t>(seq_k), 0.0f);

  {
    TRACE_EVENT(kTensorApiCategory, "CpuPrep");
    Timer::LapScope cpu_prep_scope = decode_timing.cpu_prep.Lap();
    token_embedding = EmbeddingLookupCpu(token_vec, embedding_table_fp32,
                                         config.vocab_size, config.embed_dim);

    token_per_layer_embs = GetPerLayerTokenEmbeddingsCpu(
        token_vec, emb_per_layer_table_fp32, config.vocab_size,
        config.num_layers, config.per_layer_input_dim);

    RopeCosSin(/*start=*/cache_len, /*seq_len=*/1, config.global_key_size,
               config.global_base_frequency, config.global_rope_proportion,
               absl::Span<float>(global_cos), absl::Span<float>(global_sin));
    RopeCosSin(/*start=*/cache_len, /*seq_len=*/1, config.head_dim,
               config.local_base_frequency, config.local_rope_proportion,
               absl::Span<float>(local_cos), absl::Span<float>(local_sin));

    if (config.sliding_window_size > 0) {
      const int32_t min_allowed_pos =
          std::max<int32_t>(0, seq_k - config.sliding_window_size);
      const float neg_inf = std::numeric_limits<float>::lowest();
      for (int32_t j = 0; j < min_allowed_pos; ++j) {
        sliding_mask[static_cast<size_t>(j)] = neg_inf;
      }
    }
    std::array<int32_t, 4> mask_shape = {1, 1, 1, seq_k};
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
        decode_inputs.sliding_attention_mask, mask_shape));

    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
        decode_inputs.global_attention_mask, mask_shape));
  }

  {
    TRACE_EVENT(kTensorApiCategory, "Uploads");
    Timer::LapScope uploads_scope = decode_timing.uploads.Lap();
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.embedded_input, token_embedding));

    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_global_cos, global_cos));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_global_sin, global_sin));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_local_cos, local_cos));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_local_sin, local_sin));

    for (int l = 0; l < config.num_layers; ++l) {
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner.SetInput(decode_inputs.per_layer_token_embeddings[l],
                                 token_per_layer_embs[l]));
    }

    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.WriteInput(
        decode_inputs.sliding_attention_mask, 0,
        absl::MakeConstSpan(
            reinterpret_cast<const std::byte*>(sliding_mask.data()),
            sliding_mask.size() * sizeof(float))));
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.WriteInput(
        decode_inputs.global_attention_mask, 0,
        absl::MakeConstSpan(
            reinterpret_cast<const std::byte*>(global_mask.data()),
            global_mask.size() * sizeof(float))));
  }

  {
    TRACE_EVENT(kTensorApiCategory, "Run");
    Timer::LapScope lap(decode_timing.run);
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.Run());
  }

  LockedBufferSpan<const float> logits_locked =
      LockedBufferSpan<const float>::Empty();
  {
    TRACE_EVENT(kTensorApiCategory, "Readback");
    Timer::LapScope readback_scope = decode_timing.readback.Lap();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        logits_locked,
        decode_runner.ReadOutputAs<float>(decode_outputs.logits));
  }

  TRACE_EVENT(kTensorApiCategory, "Argmax");
  Timer::LapScope argmax_scope = decode_timing.argmax.Lap();
  if (logits_locked.size() == 0) {
    return absl::InternalError("Decode logits span is empty.");
  }
  return absl::c_max_element(logits_locked) - logits_locked.begin();
}

absl::Status UpdateKvCache(XnnpackRunner& decode_runner,
                           Gemma4Inputs<XnnpackMixinTag>& decode_inputs,
                           Gemma4Outputs<XnnpackMixinTag>& decode_outputs,
                           const Config& config, int cache_len, int batch_size,
                           const std::vector<int>& sharing_patterns,
                           std::vector<std::vector<float>>& host_key_caches,
                           std::vector<std::vector<float>>& host_value_caches,
                           DecodeTiming& decode_timing) {
  TRACE_EVENT(kTensorApiCategory, "UpdateKvCache");
  for (int i = 0; i < config.num_layers; ++i) {
    if (sharing_patterns[i] != i) {
      continue;
    }
    bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
    int head_dim = is_global ? config.global_key_size : config.head_dim;

    LockedBufferSpan<const float> new_key_locked =
        LockedBufferSpan<const float>::Empty();
    LockedBufferSpan<const float> new_value_locked =
        LockedBufferSpan<const float>::Empty();

    {
      TRACE_EVENT(kTensorApiCategory, "KvCache::Readback");
      Timer::LapScope readback_scope = decode_timing.cache_readback.Lap();
      LRT_TENSOR_ASSIGN_OR_RETURN(
          new_key_locked,
          decode_runner.ReadOutputAs<float>(decode_outputs.key_caches[i]));

      LRT_TENSOR_ASSIGN_OR_RETURN(
          new_value_locked,
          decode_runner.ReadOutputAs<float>(decode_outputs.value_caches[i]));
    }

    {
      TRACE_EVENT(kTensorApiCategory, "KvCache::AppendAndUpload");
      Timer::LapScope upload_scope = decode_timing.cache_upload.Lap();
      AppendTokenToKvCache(host_key_caches[i], new_key_locked,
                           config.num_kv_heads, cache_len, head_dim);
      AppendTokenToKvCache(host_value_caches[i], new_value_locked,
                           config.num_kv_heads, cache_len, head_dim);

      std::array<int32_t, 4> next_cache_shape = {
          batch_size, config.num_kv_heads, cache_len + 1, head_dim};
      const size_t next_cache_elements =
          static_cast<size_t>(config.num_kv_heads) * (cache_len + 1) * head_dim;

      LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
          decode_inputs.key_caches[i], next_cache_shape));
      absl::Span<const float> next_key_span =
          absl::MakeConstSpan(host_key_caches[i].data(), next_cache_elements);
      absl::Span<const float> next_val_span =
          absl::MakeConstSpan(host_value_caches[i].data(), next_cache_elements);
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner.SetInput(decode_inputs.key_caches[i], next_key_span));

      LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
          decode_inputs.value_caches[i], next_cache_shape));
      LRT_TENSOR_RETURN_IF_ERROR(
          decode_runner.SetInput(decode_inputs.value_caches[i], next_val_span));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ModelVariant> DeduceModelVariant(
    const SafetensorLoader& loader) {
  static constexpr absl::string_view kEmbedKeys[] = {
      "model.embed_tokens.weight",
      "model.language_model.embed_tokens.weight",
  };
  for (const absl::string_view key : kEmbedKeys) {
    if (auto info_or = loader.GetTensorInfo(key); info_or.ok()) {
      if (info_or->shape.size() >= 2) {
        int64_t embed_dim = info_or->shape[1];
        if (embed_dim == 1536) {
          return ModelVariant::kE2B;
        } else if (embed_dim == 2560) {
          return ModelVariant::kE4B;
        }
      }
    }
  }
  return absl::InvalidArgumentError(
      "Failed to deduce Gemma 4 model variant from safetensor metadata.");
}

absl::Status Run(const std::string& weights_path,
                 const std::string& tokenizer_path,
                 const std::string& raw_prompt, int max_tokens, bool verbose) {
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return absl::InternalError("Failed to initialize XNNPACK");
  }
  const std::string& perfetto_out = absl::GetFlag(FLAGS_perfetto_output);
  std::unique_ptr<PerfettoSession> perfetto_session;
  if (!perfetto_out.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(perfetto_session,
                                PerfettoSession::Create(perfetto_out));
  }

  TRACE_EVENT_BEGIN(kTensorApiCategory, "Load tokenizer");
  LRT_TENSOR_ASSIGN_OR_RETURN(GemmaTokenizerSP tokenizer,
                              GemmaTokenizerSP::Load(tokenizer_path));
  TRACE_EVENT_END(kTensorApiCategory);

  TRACE_EVENT_BEGIN(kTensorApiCategory, "Load weights");
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(weights_path));
  TRACE_EVENT_END(kTensorApiCategory);
  LRT_TENSOR_ASSIGN_OR_RETURN(ModelVariant model_variant,
                              DeduceModelVariant(loader));

  const Config config = Config::From(model_variant);

  std::string prompt = raw_prompt;
  if (model_variant == ModelVariant::kE4B &&
      !absl::StrContains(raw_prompt, "<start_of_turn>")) {
    prompt = absl::StrCat("<start_of_turn>user\n", raw_prompt,
                          "<end_of_turn>\n<start_of_turn>model\n");
  }

  ABSL_LOG(INFO) << "Using Gemma4 " << AbslUnparseFlag(model_variant)
                 << " config"
                 << " layers=" << config.num_layers
                 << " emb_dim=" << config.embed_dim
                 << " hidden_dim=" << config.hidden_dim
                 << " head_dim=" << config.head_dim
                 << " n_heads=" << config.num_heads
                 << " n_kv_heads=" << config.num_kv_heads
                 << " vocab_size=" << config.vocab_size;

  LRT_TENSOR_ASSIGN_OR_RETURN(LoadedTensors loaded_tensors,
                              LoadWeightsAndPrepareTensors(loader, config));

  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache);
  tflite::xnnpack::MMapWeightCacheProvider weight_cache_provider;
  const bool use_weight_cache = !weight_cache_path.empty();
  if (use_weight_cache) {
    TRACE_EVENT(kTensorApiCategory, "MapWeightCache");
    LRT_TENSOR_RETURN_IF_ERROR(MapGemma4WeightIdentifiers(
        weight_cache_provider, loaded_tensors.weights_handle));
    if (!weight_cache_provider.LoadOrStartBuild(weight_cache_path.c_str())) {
      return absl::InternalError(absl::StrCat(
          "Failed to load or start build for XNNPack weight cache file: ",
          weight_cache_path));
    }
  }

  TRACE_EVENT_BEGIN(kTensorApiCategory, "TokenizerEncode");
  std::vector<int32_t> input_tokens =
      tokenizer.Encode(prompt, /*add_bos=*/true);
  TRACE_EVENT_END(kTensorApiCategory);
  int seq_len = static_cast<int>(input_tokens.size());

  if (verbose) {
    ABSL_LOG(INFO) << "Input prompt: \"" << prompt << "\"";
    ABSL_LOG(INFO) << "Tokenized to " << seq_len << " tokens";
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      BuiltGraphs graphs,
      BuildModelGraphs(config, seq_len, loaded_tensors.weights_handle,
                       verbose));

  LRT_TENSOR_RETURN_IF_ERROR(graphs.prefill_outputs.logits.GetStatus())
      << "Output logits tensor isn't valid.";

  LRT_TENSOR_ASSIGN_OR_RETURN(
      CompiledRunners runners,
      CompileRunners(graphs, absl::GetFlag(FLAGS_num_threads), use_weight_cache,
                     &weight_cache_provider));

  ABSL_LOG(INFO) << "Running initial forward pass (prefill)...";

  int32_t current_token;
  PrefillTiming prefill_timing;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      current_token,
      ExecutePrefillPass(
          runners.prefill_runner, graphs.prefill_inputs, graphs.prefill_outputs,
          config, input_tokens,
          absl::MakeSpan(loaded_tensors.embedding_table_fp32),
          absl::MakeSpan(loaded_tensors.emb_per_layer_table_fp32),
          prefill_timing, verbose));

  std::cout << prompt << std::flush;

  if (seq_len > 0) {
    prefill_timing.prefill.SetCountPerLap(seq_len);
    ABSL_LOG(INFO) << "Prefill " << seq_len << " tokens in "
                   << prefill_timing.prefill.Duration();
    ABSL_LOG(INFO) << prefill_timing.Stats();
  }

  if (current_token == GemmaTokenizerSP::kEosToken ||
      current_token == kEndOfTurnToken || current_token == kStartOfTurnToken) {
    if (verbose) {
      ABSL_LOG(INFO) << "Stop token predicted from prefill (token="
                     << current_token << ")";
    }
    std::cout << std::endl;
    if (perfetto_session) {
      LRT_TENSOR_RETURN_IF_ERROR(perfetto_session->StopAndSave());
    }
    return absl::OkStatus();
  }

  // Initialize decode runner KV caches with prefill K/V
  int cache_len = seq_len;
  const int max_cache_len = seq_len + max_tokens;
  const int batch_size = 1;
  DecodeTiming decode_timing;
  std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config);
  std::vector<std::vector<float>> host_key_caches(config.num_layers);
  std::vector<std::vector<float>> host_value_caches(config.num_layers);

  {
    TRACE_EVENT(kTensorApiCategory, "InitKvCacheFromPrefill");
    for (int i = 0; i < config.num_layers; ++i) {
      if (sharing_patterns[i] != i) {
        continue;
      }
      bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
      int head_dim = is_global ? config.global_key_size : config.head_dim;

      const size_t max_cache_elements =
          static_cast<size_t>(config.num_kv_heads) * max_cache_len * head_dim;
      host_key_caches[i].resize(max_cache_elements, 0.0f);
      host_value_caches[i].resize(max_cache_elements, 0.0f);

      std::array<int32_t, 4> current_cache_shape = {
          batch_size, config.num_kv_heads, cache_len, head_dim};

      LRT_TENSOR_RETURN_IF_ERROR(runners.decode_runner.ReshapeInput(
          graphs.decode_inputs.key_caches[i], current_cache_shape));
      LRT_TENSOR_RETURN_IF_ERROR(runners.decode_runner.ReshapeInput(
          graphs.decode_inputs.value_caches[i], current_cache_shape));

      LRT_TENSOR_ASSIGN_OR_RETURN(auto key_locked,
                                  runners.prefill_runner.ReadOutputAs<float>(
                                      graphs.prefill_outputs.key_caches[i]));
      LRT_TENSOR_ASSIGN_OR_RETURN(auto value_locked,
                                  runners.prefill_runner.ReadOutputAs<float>(
                                      graphs.prefill_outputs.value_caches[i]));

      const size_t initial_elements =
          static_cast<size_t>(config.num_kv_heads) * cache_len * head_dim;
      std::copy_n(key_locked.begin(), initial_elements,
                  host_key_caches[i].begin());
      std::copy_n(value_locked.begin(), initial_elements,
                  host_value_caches[i].begin());

      absl::Span<const float> key_span =
          absl::MakeConstSpan(host_key_caches[i].data(), initial_elements);
      absl::Span<const float> val_span =
          absl::MakeConstSpan(host_value_caches[i].data(), initial_elements);
      LRT_TENSOR_RETURN_IF_ERROR(runners.decode_runner.SetInput(
          graphs.decode_inputs.key_caches[i], key_span));
      LRT_TENSOR_RETURN_IF_ERROR(runners.decode_runner.SetInput(
          graphs.decode_inputs.value_caches[i], val_span));
    }
  }

  TokenPrinter printer(absl::GetFlag(FLAGS_print), max_tokens);
  printer.Push(tokenizer.DecodeToken(current_token));

  std::vector<float> global_cos(config.global_key_size);
  std::vector<float> global_sin(config.global_key_size);
  std::vector<float> local_cos(config.head_dim);
  std::vector<float> local_sin(config.head_dim);

  int tokens_generated = 0;
  for (int step = 0; step < max_tokens; ++step) {
    TRACE_EVENT(kTensorApiCategory, "DecodeStep");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        current_token,
        ExecuteDecodeStep(
            runners.decode_runner, graphs.decode_inputs, graphs.decode_outputs,
            config, current_token, cache_len,
            absl::MakeSpan(loaded_tensors.embedding_table_fp32),
            absl::MakeSpan(loaded_tensors.emb_per_layer_table_fp32), global_cos,
            global_sin, local_cos, local_sin, decode_timing));

    LRT_TENSOR_RETURN_IF_ERROR(UpdateKvCache(
        runners.decode_runner, graphs.decode_inputs, graphs.decode_outputs,
        config, cache_len, batch_size, sharing_patterns, host_key_caches,
        host_value_caches, decode_timing));

    cache_len += 1;

    if (current_token == GemmaTokenizerSP::kEosToken ||
        current_token == kEndOfTurnToken ||
        current_token == kStartOfTurnToken) {
      if (verbose) {
        ABSL_LOG(INFO) << "Stop token generated at step " << step
                       << " (token=" << current_token << ")";
      }
      break;
    }

    printer.Push(tokenizer.DecodeToken(current_token));
    tokens_generated++;
  }
  printer.Flush();

  ABSL_LOG(INFO) << "Decoded " << tokens_generated << " tokens in "
                 << decode_timing.decode.Duration();
  ABSL_LOG(INFO) << decode_timing.Stats();

  return absl::OkStatus();
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4

int main(int argc, char** argv) {
  litert::tensor::Initialize("gemma4", argc, argv, true);

  absl::Status status = litert::tensor::examples::gemma4::Run(
      absl::GetFlag(FLAGS_weights), absl::GetFlag(FLAGS_tokenizer),
      absl::GetFlag(FLAGS_prompt), absl::GetFlag(FLAGS_max_tokens),
      absl::GetFlag(FLAGS_verbose));

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run Gemma4 model: " << status;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
