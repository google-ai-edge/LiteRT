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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"  // IWYU pragma: keep
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/util.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/examples/gemma4/host_kv_cache.h"
#include "tensor/examples/gemma4/test_backends.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/examples/utils/ring_buffer.h"
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::IsOk;
using ::testing::FloatNear;
using ::testing::Pointwise;

// A tiny Gemma 4 that exercises the sliding window KV caches:
// - layers 0 and 2 are local sliding window layers, 1 and 3 are global;
// - layers 2 and 3 reuse the K/V of layers 0 and 1;
// - local layers keep `sliding_window_size - 1 = 3` cached tokens;
// - 2 KV heads, so the multi-head window update is used.
Config TinyConfig() {
  Config config = Config::E4B();
  config.vocab_size = 16;
  config.embed_dim = 8;
  config.hidden_dim = 16;
  config.head_dim = 4;
  config.global_key_size = 8;
  config.num_heads = 2;
  config.num_kv_heads = 2;
  config.num_layers = 4;
  config.per_layer_input_dim = 4;
  config.sliding_window_size = 4;
  config.attention_pattern_size = 2;
  config.frac_shared_layers = 0.5f;
  return config;
}

// Returns deterministic pseudo-random values in [offset - scale,
// offset + scale].
std::vector<float> PseudoRandomValues(size_t size, uint32_t seed, float scale,
                                      float offset) {
  std::vector<float> values(size);
  uint32_t state = seed * 2654435761u + 12345u;
  for (float& value : values) {
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>(state >> 8) / (1 << 24);  // [0, 1)
    value = offset + scale * (2.0f * unit - 1.0f);
  }
  return values;
}

// Creates every weight of `config` with pseudo-random values.
absl::flat_hash_map<std::string, TensorHandle> CreateWeights(
    const Config& config) {
  absl::flat_hash_map<std::string, TensorHandle> weights;
  uint32_t seed = 0;
  auto add = [&](const std::string& name, Shape shape, float scale = 0.3f,
                 float offset = 0.0f) {
    size_t size = 1;
    for (int dim : shape) {
      size *= dim;
    }
    weights.emplace(name, TensorInit{.name = name,
                                     .type = Type::kFP32,
                                     .shape = std::move(shape),
                                     .buffer = PseudoRandomValues(
                                         size, ++seed, scale, offset)});
  };
  const int embed = config.embed_dim;
  const int ple = config.per_layer_input_dim;
  for (int l = 0; l < config.num_layers; ++l) {
    const std::string prefix = absl::StrCat("model.layers.", l);
    const int head_dim = config.GetLayerType(l) == Config::LayerType::kGlobal
                             ? config.global_key_size
                             : config.head_dim;
    const int q_dim = config.num_heads * head_dim;
    const int kv_dim = config.num_kv_heads * head_dim;
    add(absl::StrCat(prefix, ".input_layernorm.weight"), {embed}, 0.1f, 1.0f);
    add(absl::StrCat(prefix, ".self_attn.q_proj.weight"), {q_dim, embed});
    add(absl::StrCat(prefix, ".self_attn.k_proj.weight"), {kv_dim, embed});
    add(absl::StrCat(prefix, ".self_attn.v_proj.weight"), {kv_dim, embed});
    add(absl::StrCat(prefix, ".self_attn.o_proj.weight"), {embed, q_dim});
    add(absl::StrCat(prefix, ".self_attn.q_norm.weight"), {head_dim}, 0.1f,
        1.0f);
    add(absl::StrCat(prefix, ".self_attn.k_norm.weight"), {head_dim}, 0.1f,
        1.0f);
    add(absl::StrCat(prefix, ".post_attention_layernorm.weight"), {embed}, 0.1f,
        1.0f);
    add(absl::StrCat(prefix, ".pre_feedforward_layernorm.weight"), {embed},
        0.1f, 1.0f);
    add(absl::StrCat(prefix, ".post_feedforward_layernorm.weight"), {embed},
        0.1f, 1.0f);
    add(absl::StrCat(prefix, ".mlp.gate_proj.weight"),
        {config.hidden_dim, embed});
    add(absl::StrCat(prefix, ".mlp.up_proj.weight"),
        {config.hidden_dim, embed});
    add(absl::StrCat(prefix, ".mlp.down_proj.weight"),
        {embed, config.hidden_dim});
    add(absl::StrCat(prefix, ".per_layer_model_projection.weight"),
        {ple, embed});
    add(absl::StrCat(prefix, ".per_layer_input_gate.weight"), {ple, embed});
    add(absl::StrCat(prefix, ".per_layer_projection.weight"), {embed, ple});
    add(absl::StrCat(prefix, ".post_per_layer_input_norm.weight"), {embed},
        0.1f, 1.0f);
    add(absl::StrCat(prefix, ".layer_scalar"), {1}, 0.1f, 1.0f);
  }
  add("model.per_layer_projection_norm.weight", {ple}, 0.1f, 1.0f);
  add("model.norm.weight", {embed}, 0.1f, 1.0f);
  add("model.embed_tokens.weight", {config.vocab_size, embed}, 1.0f);
  add("model.embed_tokens_per_layer.weight",
      {config.vocab_size, config.num_layers * ple}, 1.0f);
  return weights;
}

struct TinyModel {
  LazyTensorMapping weights;
  std::unique_ptr<GemmaEmbeddingTable> token_embedding;
  std::unique_ptr<GemmaEmbeddingTable> per_layer_embedding;
};

absl::StatusOr<TinyModel> CreateTinyModel(const Config& config) {
  LazyTensorMapping weights =
      LazyTensorMapping(CreateWeights(config)).Register<Gemma4WeightHooks>();
  LRT_TENSOR_ASSIGN_OR_RETURN(TensorHandle embed_tokens,
                              weights.Get("model.embed_tokens.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> token_embedding,
      GemmaEmbeddingTable::Create(embed_tokens, config.embed_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      TensorHandle embed_tokens_per_layer,
      weights.Get("model.embed_tokens_per_layer.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::unique_ptr<GemmaEmbeddingTable> per_layer_embedding,
      GemmaEmbeddingTable::Create(
          embed_tokens_per_layer,
          config.num_layers * config.per_layer_input_dim));
  return TinyModel{std::move(weights), std::move(token_embedding),
                   std::move(per_layer_embedding)};
}

// Runs the prefill graph over `tokens` and returns the logits of the last
// token. The prefill applies the sliding window with its [S, S] attention
// mask, independently of the decode KV caches.
template <class Tag, class Runner>
absl::StatusOr<std::vector<float>> LastTokenLogits(
    const Config& config, const std::vector<int32_t>& tokens,
    TinyModel& model) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      BuiltGraphs graphs,
      BuildModelGraphs<Tag>(config, static_cast<int>(tokens.size()),
                            model.weights));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      Runner runner, Runner::Create(graphs.prefill_outputs.GetAllHandles()));
  PrefillTiming timing;
  LRT_TENSOR_RETURN_IF_ERROR(
      ExecutePrefillPass(runner, graphs.prefill_inputs, graphs.prefill_outputs,
                         config, tokens, *model.token_embedding,
                         *model.per_layer_embedding, timing)
          .status());
  LRT_TENSOR_ASSIGN_OR_RETURN(
      LockedBufferSpan<const float> logits,
      runner.template ReadOutputAs<float>(graphs.prefill_outputs.logits));
  return std::vector<float>(logits.begin(), logits.end());
}

template <class Backend>
class Gemma4RuntimeTest : public ::testing::Test {};
TYPED_TEST_SUITE(Gemma4RuntimeTest, TestBackends, TestBackendNames);

TYPED_TEST(Gemma4RuntimeTest, SlidingWindowDecodeMatchesFullSequencePrefill) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  const Config config = TinyConfig();
  // The prompt is longer than the window, so the KV caches of the local layers
  // only keep its last tokens, and the decode steps wrap the window more than
  // once.
  const std::vector<int32_t> prompt = {2, 5, 7, 1, 9, 3};
  const std::vector<int32_t> decoded = {4, 8, 6, 10, 11, 12, 13};
  const int prompt_len = static_cast<int>(prompt.size());
  const int max_cache_len = prompt_len + static_cast<int>(decoded.size());

  std::vector<HostKvCache::RingMode> ring_modes = {
      HostKvCache::RingMode::kCompacting};
  if (RingBufferStorage::IsMirroredModeSupported()) {
    ring_modes.push_back(HostKvCache::RingMode::kMirrored);
  }
  for (const HostKvCache::RingMode ring_mode : ring_modes) {
    SCOPED_TRACE(ring_mode == HostKvCache::RingMode::kMirrored ? "mirrored"
                                                               : "compacting");
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(TinyModel model, CreateTinyModel(config));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        BuiltGraphs graphs,
        BuildModelGraphs<Tag>(config, prompt_len, model.weights));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        Runner prefill_runner,
        Runner::Create(graphs.prefill_outputs.GetAllHandles()));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        Runner decode_runner,
        Runner::Create(graphs.decode_outputs.GetAllHandles()));

    PrefillTiming prefill_timing;
    ASSERT_THAT(ExecutePrefillPass(prefill_runner, graphs.prefill_inputs,
                                   graphs.prefill_outputs, config, prompt,
                                   *model.token_embedding,
                                   *model.per_layer_embedding, prefill_timing),
                IsOk());

    const std::vector<int> sharing_patterns = GetKvCacheSharingPatterns(config);
    std::vector<std::optional<HostKvCache>> host_key_caches(config.num_layers);
    std::vector<std::optional<HostKvCache>> host_value_caches(
        config.num_layers);
    ASSERT_THAT(InitKvCacheFromPrefill(
                    prefill_runner, graphs.prefill_outputs, decode_runner,
                    graphs.decode_inputs, config, prompt_len, max_cache_len,
                    /*batch_size=*/1, sharing_patterns, host_key_caches,
                    host_value_caches, ring_mode),
                IsOk());
    // Local layer 0 only keeps the last 3 tokens of the prompt.
    EXPECT_EQ(host_key_caches[0]->len(), 3);
    EXPECT_EQ(host_key_caches[1]->len(), prompt_len);

    std::vector<float> global_cos(config.global_key_size);
    std::vector<float> global_sin(config.global_key_size);
    std::vector<float> local_cos(config.head_dim);
    std::vector<float> local_sin(config.head_dim);
    DecodeTiming decode_timing;
    std::vector<int32_t> sequence = prompt;
    int cache_len = prompt_len;
    for (const int32_t token : decoded) {
      SCOPED_TRACE(absl::StrCat("position ", cache_len));
      ASSERT_THAT(
          ExecuteDecodeStep(decode_runner, graphs.decode_inputs,
                            graphs.decode_outputs, config, token, cache_len,
                            *model.token_embedding, *model.per_layer_embedding,
                            global_cos, global_sin, local_cos, local_sin,
                            decode_timing),
          IsOk());
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          LockedBufferSpan<const float> decode_logits,
          decode_runner.template ReadOutputAs<float>(
              graphs.decode_outputs.logits));
      const std::vector<float> actual(decode_logits.begin(),
                                      decode_logits.end());

      sequence.push_back(token);
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          std::vector<float> expected,
          (LastTokenLogits<Tag, Runner>(config, sequence, model)));
      EXPECT_THAT(actual, Pointwise(FloatNear(1e-4f), expected));

      ASSERT_THAT(
          UpdateKvCache(decode_runner, graphs.decode_inputs,
                        graphs.decode_outputs, config, cache_len,
                        /*batch_size=*/1, sharing_patterns, host_key_caches,
                        host_value_caches, decode_timing),
          IsOk());
      ++cache_len;
      EXPECT_EQ(host_key_caches[0]->len(), 3);
      EXPECT_EQ(host_key_caches[1]->len(), cache_len);
    }
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
