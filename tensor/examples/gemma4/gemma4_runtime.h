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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_RUNTIME_H_

// Backend-agnostic Gemma 4 inference runtime.
//
// Everything that doesn't depend on the execution backend lives here so that
// `xnnpack_main.cc` and `ynnpack_main.cc` only hold backend specific setup.
//
// - Runner operations go through the common `NnpackRunner` base class.
// - Graph inputs/outputs are templated on the backend lowering mixin `Tag`.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/tokenizer.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/utils/perfetto_session.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/common_nnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "perfetto/tracing/track_event.h"  // from @perfetto

namespace litert::tensor::examples::gemma4 {

inline constexpr int32_t kStartOfTurnToken = 105;
inline constexpr int32_t kEndOfTurnToken = 106;

// Returns true if `token` ends the generation.
bool IsStopToken(int32_t token);

// Wraps `raw_prompt` with the user/model turn markers when `instruction_tuned`
// is true and the prompt doesn't already contain a turn marker.
std::string FormatPrompt(const GemmaTokenizerSP& tokenizer,
                         absl::string_view raw_prompt, bool instruction_tuned);

// Fills `mask` with a causal (and optionally sliding window) attention mask of
// the given `shape` ([..., S, S]).
absl::Status FillAttentionMask(const Shape& shape, absl::Span<float> mask,
                               bool is_local, int sliding_window_size);

// Returns a constant tensor holding a causal (and optionally sliding window)
// attention mask of the given `shape` ([..., S, S]).
absl::StatusOr<TensorHandle> AttentionMask(Shape shape, bool is_local,
                                           int sliding_window_size);

// Appends one token worth of K or V data (`new_kv`, [num_kv_heads, head_dim])
// to a host cache laid out as [num_kv_heads, cache_len, head_dim], growing it
// in place to [num_kv_heads, cache_len + 1, head_dim].
void AppendTokenToKvCache(std::vector<float>& cache_buf,
                          absl::Span<const float> new_kv, int num_kv_heads,
                          int cache_len, int head_dim);

// Deduces the model variant from the safetensor metadata.
absl::StatusOr<ModelVariant> DeduceModelVariant(const SafetensorLoader& loader);

struct LoadedTensors {
  LazyTensorMapping weights;
  std::unique_ptr<GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<GemmaEmbeddingTable> emb_per_layer_table;
};

// Creates the weight mapping and the embedding tables.
//
// `register_backend_hooks` is called to register backend specific hooks after
// the common Gemma 4 ones and before any weight is loaded.
absl::StatusOr<LoadedTensors> LoadWeightsAndPrepareTensors(
    SafetensorLoader loader, const Config& config,
    absl::FunctionRef<void(LazyTensorMapping&)> register_backend_hooks);

// Returns handles to the same graph tensors with different backend mixins.
template <class... ToMixins, class... Mixins>
Gemma4Inputs<ToMixins...> ChangeMixinsTo(
    const Gemma4Inputs<Mixins...>& inputs) {
  return Gemma4Inputs<ToMixins...>{
      .embedded_input = inputs.embedded_input,
      .per_layer_token_embeddings = {inputs.per_layer_token_embeddings.begin(),
                                     inputs.per_layer_token_embeddings.end()},
      .global_attention_mask = inputs.global_attention_mask,
      .sliding_attention_mask = inputs.sliding_attention_mask,
      .rope_global_cos = inputs.rope_global_cos,
      .rope_global_sin = inputs.rope_global_sin,
      .rope_local_cos = inputs.rope_local_cos,
      .rope_local_sin = inputs.rope_local_sin,
      .key_caches = {inputs.key_caches.begin(), inputs.key_caches.end()},
      .value_caches = {inputs.value_caches.begin(), inputs.value_caches.end()},
  };
}

// Returns handles to the same graph tensors with different backend mixins.
template <class... ToMixins, class... Mixins>
Gemma4Outputs<ToMixins...> ChangeMixinsTo(
    const Gemma4Outputs<Mixins...>& outputs) {
  return Gemma4Outputs<ToMixins...>{
      .logits = outputs.logits,
      .key_caches = {outputs.key_caches.begin(), outputs.key_caches.end()},
      .value_caches = {outputs.value_caches.begin(),
                       outputs.value_caches.end()},
  };
}

// Input and output handles of the prefill and decode graphs.
struct BuiltGraphs {
  Gemma4Inputs<> prefill_inputs;
  Gemma4Outputs<> prefill_outputs;
  Gemma4Inputs<> decode_inputs;
  Gemma4Outputs<> decode_outputs;
};

template <class Runner>
struct CompiledRunners {
  Runner prefill_runner;
  Runner decode_runner;
};

// Creates the graph inputs for a sequence of `input_seq_len` tokens and a KV
// cache holding `kv_cache_len` tokens.
//
// The returned tensors don't have a backend lowering mixin. Use
// `ChangeMixinsTo<Tag>()` to build a graph from them.
absl::StatusOr<Gemma4Inputs<>> CreateGemma4Inputs(const Config& config,
                                                  int input_seq_len,
                                                  int kv_cache_len);

// Builds the prefill and decode graphs with the `Tag` backend lowering.
template <class Tag>
absl::StatusOr<BuiltGraphs> BuildModelGraphs(const Config& config, int seq_len,
                                             TensorMapping& weights) {
  TRACE_EVENT(kTensorApiCategory, "BuildModelGraphs");
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Inputs<> prefill_inputs,
      CreateGemma4Inputs(config, /*input_seq_len=*/seq_len,
                         /*kv_cache_len=*/0));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Outputs<Tag> prefill_outputs,
      BuildGemma4Graph(ChangeMixinsTo<Tag>(prefill_inputs), weights, config));

  LRT_TENSOR_ASSIGN_OR_RETURN(Gemma4Inputs<> decode_inputs,
                              CreateGemma4Inputs(config, /*input_seq_len=*/1,
                                                 /*kv_cache_len=*/seq_len));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Gemma4Outputs<Tag> decode_outputs,
      BuildGemma4Graph(ChangeMixinsTo<Tag>(decode_inputs), weights, config));

  return BuiltGraphs{std::move(prefill_inputs), ChangeMixinsTo(prefill_outputs),
                     std::move(decode_inputs), ChangeMixinsTo(decode_outputs)};
}

// Runs the prefill graph and returns the first predicted token.
absl::StatusOr<int32_t> ExecutePrefillPass(
    NnpackRunner& runner, Gemma4Inputs<>& inputs, Gemma4Outputs<>& outputs,
    const Config& config, const std::vector<int32_t>& input_tokens,
    const GemmaEmbeddingTable& token_embedding,
    const GemmaEmbeddingTable& emb_per_layer_table,
    PrefillTiming& prefill_timing);

// Runs one decode step for `current_token` and returns the next token.
absl::StatusOr<int32_t> ExecuteDecodeStep(
    NnpackRunner& decode_runner, Gemma4Inputs<>& decode_inputs,
    Gemma4Outputs<>& decode_outputs, const Config& config,
    int32_t current_token, int cache_len,
    const GemmaEmbeddingTable& token_embedding_table,
    const GemmaEmbeddingTable& emb_per_layer_table,
    std::vector<float>& global_cos, std::vector<float>& global_sin,
    std::vector<float>& local_cos, std::vector<float>& local_sin,
    DecodeTiming& decode_timing);

// Initializes the host KV caches from the prefill outputs and uploads them to
// the decode runner.
absl::Status InitKvCacheFromPrefill(
    NnpackRunner& prefill_runner, Gemma4Outputs<>& prefill_outputs,
    NnpackRunner& decode_runner, Gemma4Inputs<>& decode_inputs,
    const Config& config, int cache_len, int max_cache_len, int batch_size,
    const std::vector<int>& sharing_patterns,
    std::vector<std::vector<float>>& host_key_caches,
    std::vector<std::vector<float>>& host_value_caches);

// Appends the K/V produced by the last decode step to the host caches and
// uploads them to the decode runner.
absl::Status UpdateKvCache(NnpackRunner& decode_runner,
                           Gemma4Inputs<>& decode_inputs,
                           Gemma4Outputs<>& decode_outputs,
                           const Config& config, int cache_len, int batch_size,
                           const std::vector<int>& sharing_patterns,
                           std::vector<std::vector<float>>& host_key_caches,
                           std::vector<std::vector<float>>& host_value_caches,
                           DecodeTiming& decode_timing);

struct GenerateOptions {
  std::string weights_path;
  std::string tokenizer_path;
  std::string prompt;
  // Maximum number of generated tokens, including the one predicted by the
  // prefill pass.
  int max_tokens = 100;
  bool instruction_tuned = true;
  bool verbose = false;
  TokenPrinter::Kind print = TokenPrinter::Kind::kTokens;
};

// Runs Gemma 4.
//
// Loads the tokenizer and weights, builds the prefill and decode graphs, runs
// prefill and then decodes until a stop token is predicted or
// `options.max_tokens` tokens have been generated.
//
// - `register_backend_hooks`: registers backend specific weight loading hooks.
// - `compile_runners`: creates the backend runners for the built graphs.
template <class Tag, class Runner>
absl::Status Generate(
    const GenerateOptions& options,
    absl::FunctionRef<void(LazyTensorMapping&)> register_backend_hooks,
    absl::FunctionRef<absl::StatusOr<CompiledRunners<Runner>>(BuiltGraphs&)>
        compile_runners) {
  const int max_tokens = options.max_tokens;
  const bool verbose = options.verbose;
  if (max_tokens <= 0) {
    return absl::InvalidArgumentError("max_tokens must be positive");
  }

  TRACE_EVENT_BEGIN(kTensorApiCategory, "Load tokenizer");
  LRT_TENSOR_ASSIGN_OR_RETURN(GemmaTokenizerSP tokenizer,
                              GemmaTokenizerSP::Load(options.tokenizer_path));
  TRACE_EVENT_END(kTensorApiCategory);

  TRACE_EVENT_BEGIN(kTensorApiCategory, "Load weights");
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(options.weights_path));
  TRACE_EVENT_END(kTensorApiCategory);
  LRT_TENSOR_ASSIGN_OR_RETURN(ModelVariant model_variant,
                              DeduceModelVariant(loader));

  const Config config = Config::From(model_variant);

  const std::string prompt =
      FormatPrompt(tokenizer, options.prompt, options.instruction_tuned);

  ABSL_LOG(INFO) << "Using Gemma4 " << AbslUnparseFlag(model_variant)
                 << " config"
                 << " layers=" << config.num_layers
                 << " emb_dim=" << config.embed_dim
                 << " hidden_dim=" << config.hidden_dim
                 << " head_dim=" << config.head_dim
                 << " n_heads=" << config.num_heads
                 << " n_kv_heads=" << config.num_kv_heads
                 << " vocab_size=" << config.vocab_size;

  LRT_TENSOR_ASSIGN_OR_RETURN(
      LoadedTensors loaded_tensors,
      LoadWeightsAndPrepareTensors(std::move(loader), config,
                                   register_backend_hooks));

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
      BuildModelGraphs<Tag>(config, seq_len, loaded_tensors.weights));

  LRT_TENSOR_RETURN_IF_ERROR(graphs.prefill_outputs.logits.GetStatus())
      << "Output logits tensor isn't valid.";

  LRT_TENSOR_ASSIGN_OR_RETURN(CompiledRunners<Runner> runners,
                              compile_runners(graphs));

  ABSL_LOG(INFO) << "Running initial forward pass (prefill)...";

  PrefillTiming prefill_timing;
  LRT_TENSOR_ASSIGN_OR_RETURN(
      int32_t current_token,
      ExecutePrefillPass(runners.prefill_runner, graphs.prefill_inputs,
                         graphs.prefill_outputs, config, input_tokens,
                         *loaded_tensors.token_embedding,
                         *loaded_tensors.emb_per_layer_table, prefill_timing));

  std::cout << prompt << std::flush;

  if (seq_len > 0) {
    prefill_timing.prefill.SetCountPerLap(seq_len);
    ABSL_LOG(INFO) << "Prefill " << seq_len << " tokens in "
                   << prefill_timing.prefill.Duration();
    ABSL_LOG(INFO) << prefill_timing.Stats();
  }

  if (IsStopToken(current_token)) {
    if (verbose) {
      ABSL_LOG(INFO) << "Stop token predicted from prefill (token="
                     << current_token << ")";
    }
    std::cout << std::endl;
    return absl::OkStatus();
  }

  TokenPrinter printer(options.print, max_tokens);
  printer.Push(tokenizer.DecodeToken(current_token));
  // Prefill already predicted the first generated token.
  if (max_tokens == 1) {
    printer.Flush();
    return absl::OkStatus();
  }

  // Initialize decode runner KV caches with prefill K/V
  int cache_len = seq_len;
  const int max_cache_len = seq_len + max_tokens - 1;
  const int batch_size = 1;
  DecodeTiming decode_timing;
  std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config);
  std::vector<std::vector<float>> host_key_caches(config.num_layers);
  std::vector<std::vector<float>> host_value_caches(config.num_layers);

  LRT_TENSOR_RETURN_IF_ERROR(InitKvCacheFromPrefill(
      runners.prefill_runner, graphs.prefill_outputs, runners.decode_runner,
      graphs.decode_inputs, config, cache_len, max_cache_len, batch_size,
      sharing_patterns, host_key_caches, host_value_caches));

  std::vector<float> global_cos(config.global_key_size);
  std::vector<float> global_sin(config.global_key_size);
  std::vector<float> local_cos(config.head_dim);
  std::vector<float> local_sin(config.head_dim);

  int tokens_generated = 0;
  for (int step = 1; step < max_tokens; ++step) {
    TRACE_EVENT(kTensorApiCategory, "DecodeStep");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        current_token,
        ExecuteDecodeStep(runners.decode_runner, graphs.decode_inputs,
                          graphs.decode_outputs, config, current_token,
                          cache_len, *loaded_tensors.token_embedding,
                          *loaded_tensors.emb_per_layer_table, global_cos,
                          global_sin, local_cos, local_sin, decode_timing));

    LRT_TENSOR_RETURN_IF_ERROR(UpdateKvCache(
        runners.decode_runner, graphs.decode_inputs, graphs.decode_outputs,
        config, cache_len, batch_size, sharing_patterns, host_key_caches,
        host_value_caches, decode_timing));

    cache_len += 1;

    if (IsStopToken(current_token)) {
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

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_GEMMA4_RUNTIME_H_
