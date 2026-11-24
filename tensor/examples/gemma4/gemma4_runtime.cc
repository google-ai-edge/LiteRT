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

#include "tensor/examples/gemma4/gemma4_runtime.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/gemma4/host_kv_cache.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

namespace litert::tensor::examples::gemma4 {

bool IsStopToken(const int32_t token) {
  return token == GemmaTokenizerSP::kEosToken || token == kEndOfTurnToken ||
         token == kStartOfTurnToken;
}

std::string FormatPrompt(const GemmaTokenizerSP& tokenizer,
                         const absl::string_view raw_prompt,
                         const bool instruction_tuned) {
  const std::string start_of_turn = tokenizer.DecodeToken(kStartOfTurnToken);
  if (!instruction_tuned || absl::StrContains(raw_prompt, start_of_turn)) {
    return std::string(raw_prompt);
  }
  return absl::StrCat(start_of_turn, "user\n", raw_prompt,
                      tokenizer.DecodeToken(kEndOfTurnToken), "\n",
                      start_of_turn, "model\n");
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

absl::StatusOr<TensorHandle> AttentionMask(Shape shape, const bool is_local,
                                           const int sliding_window_size) {
  auto buffer = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  TensorHandle mask({.name = "attention_mask",
                     .type = Type::kFP32,
                     .shape = std::move(shape),
                     .buffer = buffer});
  LRT_TENSOR_RETURN_IF_ERROR(FillAttentionMask(
      mask.GetShape(), buffer->Span<float>(), is_local, sliding_window_size));
  return mask;
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

absl::StatusOr<LoadedTensors> LoadWeightsAndPrepareTensors(
    SafetensorLoader loader, const Config& config,
    absl::FunctionRef<void(LazyTensorMapping&)> register_backend_hooks) {
  TRACE_EVENT(kTensorApiCategory, "LoadWeightsAndPrepareTensors");
  LazyTensorMapping weights(GetGemma4WeightMapping(config.num_layers),
                            std::move(loader));
  weights.Register<Gemma4WeightHooks>(config)
      .Register<FallbackBF16ToFp32Hooks>();
  register_backend_hooks(weights);
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle embed_tokens,
                              weights.Get("model.embed_tokens.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> token_embedding,
      GemmaEmbeddingTable::Create(embed_tokens, config.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorHandle embed_tokens_per_layer,
      weights.Get("model.embed_tokens_per_layer.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table,
      GemmaEmbeddingTable::Create(
          embed_tokens_per_layer,
          config.num_layers * config.per_layer_input_dim));

  return LoadedTensors{std::move(weights), std::move(token_embedding),
                       std::move(emb_per_layer_table)};
}

absl::StatusOr<Gemma4Inputs<>> CreateGemma4Inputs(const Config& config,
                                                  const int input_seq_len,
                                                  const int kv_cache_len) {
  const int batch_size = 1;
  Gemma4Inputs<> inputs;

  inputs.embedded_input.Set(
      {.name = "embedded_input",
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
    // Every past token bound to the local layers is inside their sliding
    // window.
    const int sliding_kv_cache_len =
        config.SlidingWindowKvCacheLen(kv_cache_len);
    inputs.sliding_attention_mask.Set(
        {.name = "sliding_attention_mask",
         .type = Type::kFP32,
         .shape = {1, 1, 1, sliding_kv_cache_len + 1}});
    inputs.global_attention_mask.Set({.name = "global_attention_mask",
                                      .type = Type::kFP32,
                                      .shape = {1, 1, 1, kv_cache_len + 1}});
  }

  inputs.key_caches.reserve(config.num_layers);
  inputs.value_caches.reserve(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    bool is_global = config.GetLayerType(i) == Config::LayerType::kGlobal;
    int head_dim = is_global ? config.global_key_size : config.head_dim;
    const int layer_kv_cache_len = config.KvCacheLen(i, kv_cache_len);
    inputs.key_caches.emplace_back(
        TensorInit{.name = absl::StrCat("key_cache_", i),
                   .type = Type::kFP32,
                   .shape = {batch_size, config.num_kv_heads,
                             layer_kv_cache_len, head_dim}});
    inputs.value_caches.emplace_back(
        TensorInit{.name = absl::StrCat("value_cache_", i),
                   .type = Type::kFP32,
                   .shape = {batch_size, config.num_kv_heads,
                             layer_kv_cache_len, head_dim}});
  }

  inputs.per_layer_token_embeddings.reserve(config.num_layers);
  for (int l = 0; l < config.num_layers; ++l) {
    inputs.per_layer_token_embeddings.emplace_back(TensorInit{
        .name = absl::StrCat("per_layer_token_embedding_", l),
        .type = Type::kFP32,
        .shape = {batch_size, input_seq_len, config.per_layer_input_dim}});
  }

  return inputs;
}

// Runs the prefill graph and returns the first predicted token.
absl::StatusOr<int32_t> ExecutePrefillPass(
    NnpackRunner& runner, Gemma4Inputs<>& inputs, Gemma4Outputs<>& outputs,
    const Config& config, const std::vector<int32_t>& input_tokens,
    const GemmaEmbeddingTable& token_embedding,
    const GemmaEmbeddingTable& emb_per_layer_table,
    PrefillTiming& prefill_timing) {
  TRACE_EVENT(kTensorApiCategory, "Prefill");
  Timer::LapScope lap_scope = prefill_timing.prefill.Lap();

  int seq_len = static_cast<int>(input_tokens.size());
  std::vector<float> embedded_input(seq_len * config.embed_dim);
  std::vector<std::vector<float>> per_layer_tok_embs(
      config.num_layers,
      std::vector<float>(seq_len * config.per_layer_input_dim));
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

  LockedBufferSpan<const float> initial_output_locked =
      LockedBufferSpan<const float>::Empty();
  {
    TRACE_EVENT(kTensorApiCategory, "Readback");
    Timer::LapScope readback_scope = prefill_timing.readback.Lap();
    LRT_TENSOR_ASSIGN_OR_RETURN(initial_output_locked,
                                runner.ReadOutputAs<float>(outputs.logits));
  }

  // The prefill graph only computes the logits of the last prompt token.
  absl::Span<const float> prefill_logits(initial_output_locked.data(),
                                         initial_output_locked.size());
  if (prefill_logits.size() != static_cast<size_t>(config.vocab_size)) {
    return absl::InternalError(absl::StrFormat(
        "Prefill logits should hold %d values (one token) but hold %zu.",
        config.vocab_size, prefill_logits.size()));
  }

  int32_t current_token =
      absl::c_max_element(prefill_logits) - prefill_logits.begin();
  return current_token;
}

// Runs one decode step for `current_token` and returns the next token.
absl::StatusOr<int32_t> ExecuteDecodeStep(
    NnpackRunner& decode_runner, Gemma4Inputs<>& decode_inputs,
    Gemma4Outputs<>& decode_outputs, const Config& config,
    int32_t current_token, int cache_len,
    const GemmaEmbeddingTable& token_embedding_table,
    const GemmaEmbeddingTable& emb_per_layer_table,
    std::vector<float>& global_cos, std::vector<float>& global_sin,
    std::vector<float>& local_cos, std::vector<float>& local_sin,
    DecodeTiming& decode_timing) {
  TRACE_EVENT(kTensorApiCategory, "Decode");
  Timer::LapScope lap_scope = decode_timing.decode.Lap();

  // The local layers only attend to their sliding window and every key bound to
  // them is inside it (see `UpdateKvCache`), so both masks are all zeros.
  const int32_t global_seq_k = cache_len + 1;
  const int32_t sliding_seq_k = config.SlidingWindowKvCacheLen(cache_len) + 1;
  std::vector<float> sliding_mask(static_cast<size_t>(sliding_seq_k), 0.0f);
  std::vector<float> global_mask(static_cast<size_t>(global_seq_k), 0.0f);

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

    const std::array<int32_t, 4> sliding_mask_shape = {1, 1, 1, sliding_seq_k};
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
        decode_inputs.sliding_attention_mask, sliding_mask_shape));

    const std::array<int32_t, 4> global_mask_shape = {1, 1, 1, global_seq_k};
    LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(
        decode_inputs.global_attention_mask, global_mask_shape));
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

namespace {

// Returns the head size of layer `layer_idx`.
int HeadDim(const Config& config, int layer_idx) {
  return config.GetLayerType(layer_idx) == Config::LayerType::kGlobal
             ? config.global_key_size
             : config.head_dim;
}

// Checks that the cache of layer `layer_idx` holds the number of tokens that
// the decode graph inputs and attention masks assume after `cache_len` tokens.
absl::Status CheckKvCacheLen(const Config& config, const int layer_idx,
                             const int cache_len, const HostKvCache& cache) {
  const int expected_len = config.KvCacheLen(layer_idx, cache_len);
  if (cache.len() != expected_len) {
    return absl::InternalError(absl::StrFormat(
        "The KV cache of layer %d holds %d tokens after %d tokens but the "
        "decode graph expects %d.",
        layer_idx, cache.len(), cache_len, expected_len));
  }
  return absl::OkStatus();
}

// Binds `cache` ([num_kv_heads, len, head_dim]) to the decode graph input
// `input` without a copy.
absl::Status BindKvCache(NnpackRunner& decode_runner, const TensorHandle& input,
                         const HostKvCache& cache, const int batch_size,
                         const int num_kv_heads, const int head_dim) {
  const std::array<int32_t, 4> shape = {batch_size, num_kv_heads, cache.len(),
                                        head_dim};
  LRT_TENSOR_RETURN_IF_ERROR(decode_runner.ReshapeInput(input, shape));
  const absl::Span<const float> data = cache.data();
  return decode_runner.SetInput(input, data);
}

}  // namespace

absl::Status InitKvCacheFromPrefill(
    NnpackRunner& prefill_runner, Gemma4Outputs<>& prefill_outputs,
    NnpackRunner& decode_runner, Gemma4Inputs<>& decode_inputs,
    const Config& config, int cache_len, int max_cache_len, int batch_size,
    const std::vector<int>& sharing_patterns,
    std::vector<std::optional<HostKvCache>>& host_key_caches,
    std::vector<std::optional<HostKvCache>>& host_value_caches,
    const HostKvCache::RingMode ring_mode) {
  TRACE_EVENT(kTensorApiCategory, "InitKvCacheFromPrefill");
  for (int i = 0; i < config.num_layers; ++i) {
    if (sharing_patterns[i] != i) {
      continue;
    }
    const int head_dim = HeadDim(config, i);
    const int window_len = config.KvCacheWindow(i);

    LRT_TENSOR_ASSIGN_OR_RETURN(
        host_key_caches[i],
        HostKvCache::Create(config.num_kv_heads, head_dim, max_cache_len,
                            window_len, ring_mode));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        host_value_caches[i],
        HostKvCache::Create(config.num_kv_heads, head_dim, max_cache_len,
                            window_len, ring_mode));

    LRT_TENSOR_ASSIGN_OR_RETURN(
        LockedBufferSpan<const float> key_locked,
        prefill_runner.ReadOutputAs<float>(prefill_outputs.key_caches[i]));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        LockedBufferSpan<const float> value_locked,
        prefill_runner.ReadOutputAs<float>(prefill_outputs.value_caches[i]));
    LRT_TENSOR_RETURN_IF_ERROR(host_key_caches[i]->InitFromPrefill(
        absl::MakeConstSpan(key_locked.data(), key_locked.size()), cache_len));
    LRT_TENSOR_RETURN_IF_ERROR(host_value_caches[i]->InitFromPrefill(
        absl::MakeConstSpan(value_locked.data(), value_locked.size()),
        cache_len));
    LRT_TENSOR_RETURN_IF_ERROR(
        CheckKvCacheLen(config, i, cache_len, *host_key_caches[i]));

    LRT_TENSOR_RETURN_IF_ERROR(BindKvCache(
        decode_runner, decode_inputs.key_caches[i], *host_key_caches[i],
        batch_size, config.num_kv_heads, head_dim));
    LRT_TENSOR_RETURN_IF_ERROR(BindKvCache(
        decode_runner, decode_inputs.value_caches[i], *host_value_caches[i],
        batch_size, config.num_kv_heads, head_dim));
  }
  return absl::OkStatus();
}

absl::Status UpdateKvCache(
    NnpackRunner& decode_runner, Gemma4Inputs<>& decode_inputs,
    Gemma4Outputs<>& decode_outputs, const Config& config, int cache_len,
    int batch_size, const std::vector<int>& sharing_patterns,
    std::vector<std::optional<HostKvCache>>& host_key_caches,
    std::vector<std::optional<HostKvCache>>& host_value_caches,
    DecodeTiming& decode_timing) {
  TRACE_EVENT(kTensorApiCategory, "UpdateKvCache");
  for (int i = 0; i < config.num_layers; ++i) {
    if (sharing_patterns[i] != i) {
      continue;
    }
    if (!host_key_caches[i].has_value() || !host_value_caches[i].has_value()) {
      return absl::FailedPreconditionError(
          absl::StrFormat("The KV cache of layer %d isn't initialized.", i));
    }
    const int head_dim = HeadDim(config, i);

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
      LRT_TENSOR_RETURN_IF_ERROR(host_key_caches[i]->Push(
          absl::MakeConstSpan(new_key_locked.data(), new_key_locked.size())));
      LRT_TENSOR_RETURN_IF_ERROR(host_value_caches[i]->Push(absl::MakeConstSpan(
          new_value_locked.data(), new_value_locked.size())));
      LRT_TENSOR_RETURN_IF_ERROR(
          CheckKvCacheLen(config, i, cache_len + 1, *host_key_caches[i]));

      // The local layers' windows move in memory as they slide, so the caches
      // are bound again after every step. It's only a pointer update.
      LRT_TENSOR_RETURN_IF_ERROR(BindKvCache(
          decode_runner, decode_inputs.key_caches[i], *host_key_caches[i],
          batch_size, config.num_kv_heads, head_dim));
      LRT_TENSOR_RETURN_IF_ERROR(BindKvCache(
          decode_runner, decode_inputs.value_caches[i], *host_value_caches[i],
          batch_size, config.num_kv_heads, head_dim));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4
