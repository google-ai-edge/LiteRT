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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"      // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"               // from @com_google_absl
#include "absl/log/absl_log.h"             // from @com_google_absl
#include "absl/status/status.h"            // from @com_google_absl
#include "absl/strings/ascii.h"            // from @com_google_absl
#include "absl/strings/match.h"            // from @com_google_absl
#include "absl/strings/numbers.h"          // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/strings/str_format.h"       // from @com_google_absl
#include "absl/strings/str_join.h"         // from @com_google_absl
#include "absl/strings/str_split.h"        // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "absl/types/span.h"               // from @com_google_absl
#include "perfetto/tracing/track_event.h"  // from @perfetto
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/examples/utils/initialization.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/delegates/xnnpack/weight_cache.h"
#include "xnnpack.h"  // from @XNNPACK

ABSL_FLAG(std::string, weights, "",
          "Path to safetensor weights file or directory.");
ABSL_FLAG(std::string, tokenizer, "",
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

ABSL_FLAG(bool, consistent_arithmetic, false,
          "Use XNNPACK's slower consistent-arithmetic paths for single-KV-head "
          "E2B diagnostics; does not guarantee cross-platform equality.");
ABSL_FLAG(
    std::string, token_ids, "",
    "Comma-separated input token IDs; bypass tokenizer and prompt wrapping.");
ABSL_FLAG(
    std::string, dump_logits, "",
    "Output path prefix for little-endian float32 logits and a JSON report.");
ABSL_FLAG(bool, dump_intermediates, false,
          "Also dump full layer outputs and final normalized hidden states; "
          "requires --dump_logits.");

namespace litert::tensor::examples::gemma4 {
namespace {

constexpr int32_t kStartOfTurnToken = 105;
constexpr int32_t kEndOfTurnToken = 106;
// Google's mobile checkpoints also stop before a tool response.
constexpr int32_t kToolResponseToken = 50;
constexpr absl::string_view kAutoWeightCacheFlag = ":auto";

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

absl::StatusOr<std::vector<int32_t>> ParseTokenIds(absl::string_view text) {
  std::vector<int32_t> ids;
  for (absl::string_view field : absl::StrSplit(text, ',')) {
    int32_t token;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(field), &token) ||
        token < 0) {
      return absl::InvalidArgumentError(
          "--token_ids must contain comma-separated nonnegative int32 IDs");
    }
    ids.push_back(token);
  }
  return ids;
}

std::string PassSuffix(int generated_index) {
  return generated_index == 0
             ? ".prefill"
             : absl::StrFormat(".decode_%04d", generated_index);
}

std::string LogitsSuffix(int generated_index) {
  return absl::StrCat(PassSuffix(generated_index), ".f32");
}

absl::Status WriteRawFloats(absl::string_view path,
                            absl::Span<const float> values) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
  const uint16_t byte_order = 1;
  if (*reinterpret_cast<const unsigned char*>(&byte_order) == 1) {
    output.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(float));
  } else {
    for (float value : values) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      const char bytes[] = {
          static_cast<char>(bits), static_cast<char>(bits >> 8),
          static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)};
      output.write(bytes, sizeof(bytes));
    }
  }
  output.close();
  if (!output) {
    return absl::InternalError(
        absl::StrCat("Failed to write float32 output: ", path));
  }
  return absl::OkStatus();
}

absl::Status DumpIntermediates(XnnpackRunner& runner,
                               const Gemma4Outputs<XnnpackMixinTag>& outputs,
                               const Config& config, int seq_len,
                               absl::string_view prefix, int generated_index) {
  const size_t expected_elements =
      static_cast<size_t>(seq_len) * config.embed_dim;
  if (outputs.layer_outputs.size() != config.num_layers) {
    return absl::InternalError(
        "Intermediate layer count disagrees with config");
  }
  auto dump = [&](const TensorHandle& tensor,
                  absl::string_view suffix) -> absl::Status {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto values,
                                runner.ReadOutputAs<float>(tensor));
    if (tensor.GetType() != Type::kFP32 || values.size() != expected_elements) {
      return absl::InternalError(
          "Intermediate output type or size disagrees with config");
    }
    return WriteRawFloats(
        absl::StrCat(prefix, PassSuffix(generated_index), suffix),
        absl::MakeConstSpan(values.data(), values.size()));
  };
  for (size_t layer = 0; layer < outputs.layer_outputs.size(); ++layer) {
    LRT_TENSOR_RETURN_IF_ERROR(dump(outputs.layer_outputs[layer],
                                    absl::StrFormat(".layer_%03d.f32", layer)));
  }
  return dump(outputs.final_normalized, ".norm.f32");
}

absl::StatusOr<int32_t> SelectTokenAndDump(absl::Span<const float> logits,
                                           absl::string_view dump_prefix,
                                           int generated_index) {
  if (logits.empty() ||
      std::any_of(logits.begin(), logits.end(),
                  [](float value) { return !std::isfinite(value); })) {
    return absl::InternalError("Model logits must be nonempty and finite");
  }
  if (!dump_prefix.empty()) {
    LRT_TENSOR_RETURN_IF_ERROR(WriteRawFloats(
        absl::StrCat(dump_prefix, LogitsSuffix(generated_index)), logits));
  }
  return static_cast<int32_t>(absl::c_max_element(logits) - logits.begin());
}

absl::string_view StopReason(int32_t token) {
  if (token == GemmaTokenizerSP::kEosToken) return "eos_token";
  if (token == kEndOfTurnToken) return "end_of_turn";
  if (token == kToolResponseToken) return "tool_response_token";
  if (token == kStartOfTurnToken) return "start_of_turn";
  return "";
}

absl::Status WriteGenerationReport(
    absl::string_view prefix, ModelVariant variant, const Config& config,
    absl::Span<const int32_t> inputs, absl::Span<const int32_t> generated,
    int max_tokens, absl::string_view stop_reason, bool dump_intermediates,
    uint32_t runtime_flags) {
  if (prefix.empty()) return absl::OkStatus();
  const std::string path = absl::StrCat(prefix, ".json");
  const bool consistent_arithmetic =
      (runtime_flags & XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC) != 0;
  std::ofstream output(path, std::ios::trunc);
  output << "{\n  \"model_variant\": \"" << AbslUnparseFlag(variant)
         << "\",\n  \"dtype\": \"float32\",\n  \"byte_order\": \"little\",\n"
         << "  \"logits_shape\": [" << config.vocab_size << "],\n"
         << "  \"prefill_logits\": \"last_prompt_position\",\n"
         << "  \"xnnpack_runtime_flags\": " << runtime_flags << ",\n"
         << "  \"consistent_arithmetic\": "
         << (consistent_arithmetic ? "true" : "false") << ",\n"
         << "  \"input_token_ids\": [" << absl::StrJoin(inputs, ", ") << "],\n"
         << "  \"generated_token_ids\": [" << absl::StrJoin(generated, ", ")
         << "],\n  \"max_tokens\": " << max_tokens << ",\n"
         << "  \"stop_reason\": \"" << stop_reason << "\",\n"
         << "  \"logits_suffixes\": [";
  for (int i = 0; i < generated.size(); ++i) {
    if (i != 0) output << ", ";
    output << '\"' << LogitsSuffix(i) << '\"';
  }
  output << ']';
  if (dump_intermediates) {
    output << ",\n  \"intermediates\": [\n";
    bool first = true;
    for (int step = 0; step < generated.size(); ++step) {
      const size_t seq_len = step == 0 ? inputs.size() : 1;
      for (int layer = 0; layer <= config.num_layers; ++layer) {
        if (!first) output << ",\n";
        first = false;
        const std::string suffix = absl::StrCat(
            PassSuffix(step), layer == config.num_layers
                                  ? ".norm.f32"
                                  : absl::StrFormat(".layer_%03d.f32", layer));
        output << "    {\"suffix\": \"" << suffix << "\", \"shape\": [1, "
               << seq_len << ", " << config.embed_dim
               << "], \"dtype\": \"float32\"}";
      }
    }
    output << "\n  ]";
  }
  output << "\n}\n";
  output.close();
  if (!output) {
    return absl::InternalError(absl::StrCat("Failed to write report: ", path));
  }
  return absl::OkStatus();
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

  const TensorHandle& projection = proj_w_it->second;
  const Shape expected_shape = {config.num_layers, config.per_layer_input_dim,
                                config.embed_dim};
  const Shape flattened_shape = {config.num_layers * config.per_layer_input_dim,
                                 config.embed_dim};
  if (projection.GetType() != Type::kFP32 ||
      (projection.GetShape() != expected_shape &&
       projection.GetShape() != flattened_shape)) {
    return absl::InvalidArgumentError(
        "model.per_layer_model_projection.weight must be FP32 with shape "
        "[num_layers, per_layer_input_dim, embed_dim] or "
        "[num_layers * per_layer_input_dim, embed_dim]");
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

  if (proj_locked.size() != layer_w_bytes * config.num_layers) {
    return absl::InvalidArgumentError(
        "model.per_layer_model_projection.weight buffer size disagrees with "
        "shape");
  }

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
  std::unique_ptr<GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table;
};

absl::StatusOr<LoadedTensors> LoadWeightsAndPrepareTensors(
    const SafetensorLoader& loader, const Config& config) {
  TRACE_EVENT(kTensorApiCategory, "LoadWeightsAndPrepareTensors");
  auto weight_mapping = GetGemma4WeightMapping(config.num_layers);
  LRT_TENSOR_ASSIGN_OR_RETURN(auto weights_handle,
                              loader.LoadWeightsWithMapping(weight_mapping));
  LRT_TENSOR_RETURN_IF_ERROR(
      SlicePerLayerModelProjectionWeights(config, weights_handle));
  if (!weights_handle.contains("model.embed_tokens.weight") ||
      !weights_handle.contains("model.embed_tokens_per_layer.weight")) {
    return absl::InvalidArgumentError(
        "Required Gemma4 embedding weights are missing");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> token_embedding,
      GemmaEmbeddingTable::Create(weights_handle["model.embed_tokens.weight"],
                                  config.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table,
      GemmaEmbeddingTable::Create(
          weights_handle["model.embed_tokens_per_layer.weight"],
          config.num_layers * config.per_layer_input_dim));
  return LoadedTensors{std::move(weights_handle), std::move(token_embedding),
                       std::move(emb_per_layer_table)};
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

  // A prefill can contain only the BOS token; cache history distinguishes it
  // from a decode step.
  if (kv_cache_len == 0) {
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
    tflite::xnnpack::MMapWeightCacheProvider* weight_cache_provider,
    bool dump_intermediates, uint32_t runtime_flags) {
  TRACE_EVENT(kTensorApiCategory, "CompileRunners");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto runner, XnnpackRunner::Create(
                       graphs.prefill_outputs.GetAllHandles(dump_intermediates),
                       runtime_flags));
  runner.SetNumThreads(num_threads);
  if (use_weight_cache && weight_cache_provider != nullptr) {
    runner.SetWeightsCache(&weight_cache_provider->GetCacheProvider());
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto decode_runner,
      XnnpackRunner::Create(
          graphs.decode_outputs.GetAllHandles(dump_intermediates),
          runtime_flags));
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
    const GemmaEmbeddingTable& token_embedding,
    const GemmaEmbeddingTable& emb_per_layer_table,
    PrefillTiming& prefill_timing, absl::string_view dump_prefix,
    bool dump_intermediates) {
  TRACE_EVENT(kTensorApiCategory, "Prefill");
  Timer::LapScope lap_scope = prefill_timing.prefill.Lap();

  int seq_len = static_cast<int>(input_tokens.size());
  std::vector<float> embedded_input(static_cast<size_t>(seq_len) *
                                    config.embed_dim);
  std::vector<std::vector<float>> per_layer_tok_embs(
      config.num_layers, std::vector<float>(static_cast<size_t>(seq_len) *
                                            config.per_layer_input_dim));
  {
    TRACE_EVENT(kTensorApiCategory, "CpuPrep");
    Timer::LapScope cpu_prep_scope = prefill_timing.cpu_prep.Lap();
    LRT_TENSOR_RETURN_IF_ERROR(
        token_embedding.Lookup(input_tokens, absl::MakeSpan(embedded_input)));

    LRT_TENSOR_RETURN_IF_ERROR(emb_per_layer_table.LookupPerLayer(
        input_tokens, config.num_layers, config.per_layer_input_dim,
        absl::MakeSpan(per_layer_tok_embs)));
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

  if (dump_intermediates) {
    LRT_TENSOR_RETURN_IF_ERROR(DumpIntermediates(
        runner, outputs, config, seq_len, dump_prefix, /*generated_index=*/0));
  }

  LockedBufferSpan<const float> initial_output_locked =
      LockedBufferSpan<const float>::Empty();
  {
    TRACE_EVENT(kTensorApiCategory, "Readback");
    Timer::LapScope readback_scope = prefill_timing.readback.Lap();
    LRT_TENSOR_ASSIGN_OR_RETURN(initial_output_locked,
                                runner.ReadOutputAs<float>(outputs.logits));
  }

  if (initial_output_locked.size() !=
      static_cast<size_t>(seq_len) * config.vocab_size) {
    return absl::InternalError("Prefill logits size disagrees with config");
  }
  absl::Span<const float> prefill_logits(
      initial_output_locked.begin() +
          static_cast<size_t>(seq_len - 1) * config.vocab_size,
      config.vocab_size);

  return SelectTokenAndDump(prefill_logits, dump_prefix, /*generated_index=*/0);
}

absl::StatusOr<int32_t> ExecuteDecodeStep(
    XnnpackRunner& decode_runner, Gemma4Inputs<XnnpackMixinTag>& decode_inputs,
    Gemma4Outputs<XnnpackMixinTag>& decode_outputs, const Config& config,
    int32_t current_token, int cache_len,
    const GemmaEmbeddingTable& token_embedding_table,
    const GemmaEmbeddingTable& emb_per_layer_table,
    std::vector<float>& global_cos, std::vector<float>& global_sin,
    std::vector<float>& local_cos, std::vector<float>& local_sin,
    DecodeTiming& decode_timing, absl::string_view dump_prefix,
    int generated_index, bool dump_intermediates) {
  TRACE_EVENT(kTensorApiCategory, "Decode");
  Timer::LapScope lap_scope = decode_timing.decode.Lap();

  const int32_t seq_k = cache_len + 1;
  std::vector<float> sliding_mask(static_cast<size_t>(seq_k), 0.0f);
  std::vector<float> global_mask(static_cast<size_t>(seq_k), 0.0f);

  LockedBufferSpan<const float> token_embeddings =
      LockedBufferSpan<const float>::Empty();
  std::vector<LockedBufferSpan<const float>> token_per_layer_embs;

  {
    TRACE_EVENT(kTensorApiCategory, "CpuPrep");
    Timer::LapScope cpu_prep_scope = decode_timing.cpu_prep.Lap();

    LRT_TENSOR_ASSIGN_OR_RETURN(token_embeddings,
                                token_embedding_table.Lookup(current_token));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        token_per_layer_embs,
        emb_per_layer_table.LookupPerLayer(current_token, config.num_layers,
                                           config.per_layer_input_dim));

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
    absl::Span<const float> token_embeddings_span =
        absl::MakeConstSpan(token_embeddings.data(), token_embeddings.size());
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.SetInput(
        decode_inputs.embedded_input, token_embeddings_span));

    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_global_cos, global_cos));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_global_sin, global_sin));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_local_cos, local_cos));
    LRT_TENSOR_RETURN_IF_ERROR(
        decode_runner.SetInput(decode_inputs.rope_local_sin, local_sin));

    for (int l = 0; l < config.num_layers; ++l) {
      absl::Span<const float> layer_ple_span = absl::MakeConstSpan(
          token_per_layer_embs[l].data(), token_per_layer_embs[l].size());
      LRT_TENSOR_RETURN_IF_ERROR(decode_runner.SetInput(
          decode_inputs.per_layer_token_embeddings[l], layer_ple_span));
    }

    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.SetInput(
        decode_inputs.sliding_attention_mask, sliding_mask));
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.SetInput(
        decode_inputs.global_attention_mask, global_mask));
  }

  {
    TRACE_EVENT(kTensorApiCategory, "Run");
    Timer::LapScope lap(decode_timing.run);
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.Run());
  }

  if (dump_intermediates) {
    LRT_TENSOR_RETURN_IF_ERROR(DumpIntermediates(decode_runner, decode_outputs,
                                                 config, /*seq_len=*/1,
                                                 dump_prefix, generated_index));
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
  if (logits_locked.size() != config.vocab_size) {
    return absl::InternalError("Decode logits size disagrees with config");
  }
  return SelectTokenAndDump(
      absl::MakeConstSpan(logits_locked.data(), logits_locked.size()),
      dump_prefix, generated_index);
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

    const size_t new_elements =
        static_cast<size_t>(config.num_kv_heads) * head_dim;
    const size_t required_elements = new_elements * (cache_len + 1);
    if (new_key_locked.size() != new_elements ||
        new_value_locked.size() != new_elements ||
        host_key_caches[i].size() < required_elements ||
        host_value_caches[i].size() < required_elements) {
      return absl::InternalError("Decode KV sizes disagree with model config");
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
  static constexpr absl::string_view kNormKeys[] = {
      "model.norm.weight",
      "model.language_model.norm.weight",
      "model.layers.0.input_layernorm.weight",
      "model.language_model.layers.0.input_layernorm.weight",
  };
  for (const absl::string_view key : kNormKeys) {
    if (auto info_or = loader.GetTensorInfo(key); info_or.ok()) {
      if (!info_or->shape.empty()) {
        const int64_t dim = info_or->shape[0];
        if (dim == 1536) {
          return ModelVariant::kE2B;
        } else if (dim == 2560) {
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
  if (max_tokens <= 0) {
    return absl::InvalidArgumentError("max_tokens must be positive");
  }
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  if (num_threads <= 0) {
    return absl::InvalidArgumentError(
        "--num_threads must be greater than zero");
  }
  if (weights_path.empty()) {
    return absl::InvalidArgumentError("--weights is required");
  }
  const std::string dump_prefix = absl::GetFlag(FLAGS_dump_logits);
  const bool dump_intermediates = absl::GetFlag(FLAGS_dump_intermediates);
  const uint32_t runtime_flags = absl::GetFlag(FLAGS_consistent_arithmetic)
                                     ? XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC
                                     : 0;
  if (dump_intermediates && dump_prefix.empty()) {
    return absl::InvalidArgumentError(
        "--dump_intermediates requires --dump_logits");
  }
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
  const std::string token_ids = absl::GetFlag(FLAGS_token_ids);
  std::optional<GemmaTokenizerSP> tokenizer;
  std::vector<int32_t> input_tokens;
  if (!token_ids.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(input_tokens, ParseTokenIds(token_ids));
  } else {
    if (tokenizer_path.empty()) {
      return absl::InvalidArgumentError(
          "--tokenizer is required unless --token_ids is supplied");
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(auto loaded_tokenizer,
                                GemmaTokenizerSP::Load(tokenizer_path));
    tokenizer.emplace(std::move(loaded_tokenizer));
  }

  TRACE_EVENT_END(kTensorApiCategory);

  TRACE_EVENT_BEGIN(kTensorApiCategory, "Load weights");
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(weights_path));
  TRACE_EVENT_END(kTensorApiCategory);
  LRT_TENSOR_ASSIGN_OR_RETURN(ModelVariant model_variant,
                              DeduceModelVariant(loader));

  const Config config = Config::From(model_variant);

  std::string prompt = raw_prompt;
  if (tokenizer.has_value() && model_variant == ModelVariant::kE4B &&
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

  if (runtime_flags != 0 && config.num_kv_heads != 1) {
    return absl::InvalidArgumentError(
        "--consistent_arithmetic currently requires a single KV head (E2B); "
        "multi-KV-head tiling still requires XNNPACK broadcast rewriting");
  }
  if (loaded_tensors.token_embedding->VocabSize() != config.vocab_size ||
      loaded_tensors.emb_per_layer_table->VocabSize() != config.vocab_size) {
    return absl::InvalidArgumentError(
        "Embedding vocabulary size disagrees with model config");
  }
  ABSL_LOG(INFO) << "XNNPACK runtime flags=" << runtime_flags;

  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache);
  if (weight_cache_path == kAutoWeightCacheFlag) {
    weight_cache_path = absl::StrCat(weights_path, ".cache");
  }
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
  if (tokenizer.has_value()) {
    input_tokens = tokenizer->Encode(prompt, /*add_bos=*/true);
  }
  TRACE_EVENT_END(kTensorApiCategory);
  if (input_tokens.empty() ||
      input_tokens.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(
        "Input must contain between 1 and INT_MAX tokens");
  }
  const int seq_len = static_cast<int>(input_tokens.size());
  if (max_tokens > std::numeric_limits<int>::max() - seq_len) {
    return absl::InvalidArgumentError(
        "Prompt and generation length overflow tensor dimensions");
  }
  for (int32_t token : input_tokens) {
    if (token < 0 || token >= config.vocab_size) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input token %d is outside model vocabulary [0, %d)",
                          token, config.vocab_size));
    }
  }

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

  LRT_TENSOR_RETURN_IF_ERROR(graphs.decode_outputs.logits.GetStatus())
      << "Decode logits tensor isn't valid.";
  LRT_TENSOR_ASSIGN_OR_RETURN(
      CompiledRunners runners,
      CompileRunners(graphs, num_threads, use_weight_cache,
                     &weight_cache_provider, dump_intermediates,
                     runtime_flags));

  ABSL_LOG(INFO) << "Running initial forward pass (prefill)...";

  int32_t current_token;
  PrefillTiming prefill_timing;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      current_token,
      ExecutePrefillPass(runners.prefill_runner, graphs.prefill_inputs,
                         graphs.prefill_outputs, config, input_tokens,
                         *loaded_tensors.token_embedding,
                         *loaded_tensors.emb_per_layer_table, prefill_timing,
                         dump_prefix, dump_intermediates));
  std::vector<int32_t> generated_tokens{current_token};

  if (tokenizer.has_value()) std::cout << prompt << std::flush;

  if (seq_len > 0) {
    prefill_timing.prefill.SetCountPerLap(seq_len);
    ABSL_LOG(INFO) << "Prefill " << seq_len << " tokens in "
                   << prefill_timing.prefill.Duration();
    ABSL_LOG(INFO) << prefill_timing.Stats();
  }

  TokenPrinter printer(absl::GetFlag(FLAGS_print), max_tokens);
  auto print_token = [&](int32_t token) {
    printer.Push(tokenizer.has_value() ? tokenizer->DecodeToken(token)
                                       : absl::StrCat(token, " "));
  };
  absl::string_view stop_reason = StopReason(current_token);
  auto finish = [&]() -> absl::Status {
    LRT_TENSOR_RETURN_IF_ERROR(WriteGenerationReport(
        dump_prefix, model_variant, config, input_tokens, generated_tokens,
        max_tokens, stop_reason.empty() ? "max_tokens" : stop_reason,
        dump_intermediates, runtime_flags));
    if (perfetto_session) {
      LRT_TENSOR_RETURN_IF_ERROR(perfetto_session->StopAndSave());
    }
    return absl::OkStatus();
  };
  if (stop_reason.empty()) print_token(current_token);
  // Prefill already predicted the first generated token.
  if (!stop_reason.empty() || max_tokens == 1) {
    printer.Flush();
    return finish();
  }

  // Initialize decode runner KV caches with prefill K/V
  int cache_len = seq_len;
  const int max_cache_len = seq_len + max_tokens - 1;
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
      if (key_locked.size() != initial_elements ||
          value_locked.size() != initial_elements) {
        return absl::InternalError(
            "Prefill KV output size disagrees with config");
      }
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

  std::vector<float> global_cos(config.global_key_size);
  std::vector<float> global_sin(config.global_key_size);
  std::vector<float> local_cos(config.head_dim);
  std::vector<float> local_sin(config.head_dim);

  // The token predicted by prefill counts toward --max_tokens.
  for (int step = 1; step < max_tokens; ++step) {
    TRACE_EVENT(kTensorApiCategory, "DecodeStep");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        current_token,
        ExecuteDecodeStep(runners.decode_runner, graphs.decode_inputs,
                          graphs.decode_outputs, config, current_token,
                          cache_len, *loaded_tensors.token_embedding,
                          *loaded_tensors.emb_per_layer_table, global_cos,
                          global_sin, local_cos, local_sin, decode_timing,
                          dump_prefix, step, dump_intermediates));
    generated_tokens.push_back(current_token);
    stop_reason = StopReason(current_token);
    if (!stop_reason.empty()) {
      if (verbose) {
        ABSL_LOG(INFO) << "Stop token generated at step " << step
                       << " (token=" << current_token << ")";
      }
      break;
    }
    print_token(current_token);
    if (step + 1 < max_tokens) {
      LRT_TENSOR_RETURN_IF_ERROR(UpdateKvCache(
          runners.decode_runner, graphs.decode_inputs, graphs.decode_outputs,
          config, cache_len, batch_size, sharing_patterns, host_key_caches,
          host_value_caches, decode_timing));
      ++cache_len;
    }
  }
  printer.Flush();
  ABSL_LOG(INFO) << "Generated " << generated_tokens.size()
                 << " tokens (including the prefill prediction)";
  ABSL_LOG(INFO) << decode_timing.Stats();
  return finish();
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
