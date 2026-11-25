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

#include "tensor/examples/experimental/gemma4/model_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/gemma4_graph.h"
#include "tensor/examples/experimental/gemma4/model_builder_internal.h"
#include "tensor/examples/experimental/gemma4/runner.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/schema/schema_generated.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Bypass only checkpoint I/O. Signature discovery and all runtime operations
// are identical to Runner::Create, including input binding and KV transactions.
class RunnerTestPeer {
 public:
  static absl::StatusOr<std::unique_ptr<Runner>> Create(
      Environment& env, LitertDynamicRunner model, LoadedTensors embeddings,
      ActiveKvBank bank, int alignment, bool dynamic_rows, bool strict_shapes,
      const examples::gemma4::Config& config) {
    std::unique_ptr<Runner> runner(new Runner(
        env, std::move(model), std::move(embeddings), std::move(bank),
        alignment, dynamic_rows, strict_shapes, config));
    LRT_TENSOR_RETURN_IF_ERROR(runner->InitializeSignatures());
    return runner;
  }
};

namespace {
using ::litert::tensor::examples::gemma4::Config;
using ::litert::tensor::examples::gemma4::GemmaEmbeddingTable;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

Config TinyConfig() {
  Config config = Config::E2B();
  config.vocab_size = 64;
  config.embed_dim = 32;
  config.hidden_dim = 64;
  // Four-token alignment times the 16-byte local head preserves 64-byte
  // host-buffer alignment when the sliding-window view advances.
  config.head_dim = 16;
  config.global_key_size = 32;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.num_layers = 6;
  config.per_layer_input_dim = 4;
  config.sliding_window_size = 4;
  config.attention_pattern_size = 2;
  config.frac_shared_layers = 2.0f / 6.0f;
  return config;
}

std::vector<KvOwnerSpec> OwnerSpecs(const Config& config, int count) {
  std::vector<KvOwnerSpec> specs;
  for (int owner = 0; owner < count; ++owner) {
    specs.push_back({owner,
                     config.GetLayerType(owner) == Config::LayerType::kGlobal
                         ? config.global_key_size
                         : config.head_dim,
                     0.03125f, 0.03125f, config.num_kv_heads});
  }
  return specs;
}

// Small deterministic, nonuniform weights exercise attention, MLP, PLE, RoPE,
// shared caches and the head. No downloaded checkpoint or tokenizer is needed.
absl::StatusOr<LoadedTensors> TinyWeights(const Config& c) {
  LoadedTensors loaded;
  int seed = 0;
  const auto add = [&](const std::string& name, Shape shape,
                       std::optional<float> constant = std::nullopt) {
    int count = 1;
    for (int dim : shape) count *= dim;
    std::vector<float> values(count);
    ++seed;
    for (int i = 0; i < count; ++i) {
      values[i] = constant.value_or(0.1f * std::sin(i * 0.73f + seed * 1.31f));
    }
    loaded.weights_handle.emplace(
        name,
        TensorHandle({.name = name,
                      .type = Type::kFP32,
                      .shape = std::move(shape),
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)}));
  };
  add("model.embed_tokens.weight", {c.vocab_size, c.embed_dim});
  add("model.embed_tokens_per_layer.weight",
      {c.vocab_size, c.num_layers * c.per_layer_input_dim});
  add("model.norm.weight", {c.embed_dim}, 1.0f);
  add("model.per_layer_projection_norm.weight", {c.per_layer_input_dim}, 1.0f);
  add("lm_head.weight", {c.vocab_size, c.embed_dim});
  add("matched.logits_softcap_reciprocal", {}, 0.03333333507180214f);
  const std::string projection = "model.per_layer_model_projection";
  std::vector<int8_t> codes(c.num_layers * c.per_layer_input_dim * c.embed_dim);
  for (int i = 0; i < codes.size(); ++i) codes[i] = (i * 7 + i / 11) % 15 - 7;
  loaded.weights_handle.emplace(
      projection + ".weight",
      TensorHandle(
          {.name = projection + ".weight",
           .type = Type::kI8,
           .shape = {c.num_layers * c.per_layer_input_dim, c.embed_dim},
           .buffer = OwningCpuBuffer::Copy<Type::kI8>(codes),
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               std::vector<float>(c.num_layers * c.per_layer_input_dim,
                                  0.015625f),
               std::vector<int64_t>{0}, 0)}));
  add(projection + ".input_scale", {}, 0.03125f);
  add(projection + ".output_scale", {}, 0.015625f);
  for (int layer = 0; layer < c.num_layers; ++layer) {
    const std::string prefix = absl::StrCat("model.layers.", layer);
    const int dim = c.GetLayerType(layer) == Config::LayerType::kGlobal
                        ? c.global_key_size
                        : c.head_dim;
    for (const char* norm :
         {"input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "post_per_layer_input_norm"}) {
      add(absl::StrCat(prefix, ".", norm, ".weight"), {c.embed_dim}, 1.0f);
    }
    for (const char* norm : {"q_norm", "k_norm"}) {
      add(absl::StrCat(prefix, ".self_attn.", norm, ".weight"), {dim}, 1.0f);
    }
    add(prefix + ".self_attn.q_proj.weight", {c.num_heads * dim, c.embed_dim});
    add(prefix + ".self_attn.k_proj.weight",
        {c.num_kv_heads * dim, c.embed_dim});
    add(prefix + ".self_attn.v_proj.weight",
        {c.num_kv_heads * dim, c.embed_dim});
    add(prefix + ".self_attn.o_proj.weight", {c.embed_dim, c.num_heads * dim});
    add(prefix + ".mlp.gate_proj.weight", {c.hidden_dim, c.embed_dim});
    add(prefix + ".mlp.up_proj.weight", {c.hidden_dim, c.embed_dim});
    add(prefix + ".mlp.down_proj.weight", {c.embed_dim, c.hidden_dim});
    add(prefix + ".per_layer_input_gate.weight",
        {c.per_layer_input_dim, c.embed_dim});
    add(prefix + ".per_layer_projection.weight",
        {c.embed_dim, c.per_layer_input_dim});
    add(prefix + ".layer_scalar", {1}, 0.8f + layer * 0.02f);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      loaded.token_embedding,
      GemmaEmbeddingTable::Create(
          loaded.weights_handle.at("model.embed_tokens.weight")));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      loaded.emb_per_layer_table,
      GemmaEmbeddingTable::Create(
          loaded.weights_handle.at("model.embed_tokens_per_layer.weight")));
  return loaded;
}

TEST(ModelBuilderValidationTest, RejectsInvalidPrefillRows) {
  for (int rows : {0, 1, 31, 33, 1025, 2048}) {
    SCOPED_TRACE(rows);
    ModelFactory factory;
    EXPECT_THAT(
        AddGemma4Signatures(factory, {}, {}, {.prefill_rows = rows}).message(),
        HasSubstr("power-of-two"));
  }
}

TEST(ModelBuilderValidationTest, RejectsMissingOrInvalidOwners) {
  ModelFactory factory;
  EXPECT_THAT(AddGemma4Signatures(factory, {}, {}).message(),
              HasSubstr("15 E2B"));
  auto specs = OwnerSpecs(Config::E2B(), 15);
  specs[2].owner = 1;
  EXPECT_THAT(AddGemma4Signatures(factory, {}, specs).message(),
              HasSubstr("metadata"));
  specs = OwnerSpecs(Config::E2B(), 15);
  specs[0].key_scale = 0;
  EXPECT_THAT(AddGemma4Signatures(factory, {}, specs).message(),
              HasSubstr("metadata"));
}

TEST(ModelBuilderValidationTest, RejectsMissingWeightsBeforeAuthoring) {
  ModelFactory factory;
  EXPECT_THAT(
      AddGemma4Signatures(factory, {}, OwnerSpecs(Config::E2B(), 15)).message(),
      HasSubstr("Missing model weight"));
}

TEST(ModelBuilderTest, CacheSharingUsesLastLocalAndGlobalOwners) {
  EXPECT_THAT(GetKvCacheSharingPatterns(TinyConfig()),
              ElementsAre(0, 1, 2, 3, 2, 3));
  for (const Config& config : {Config::E2B(), Config::E4B()}) {
    const auto owners = GetKvCacheSharingPatterns(config);
    const int count = config.num_layers == 35 ? 15 : 24;
    for (int i = 0; i < config.num_layers; ++i) {
      EXPECT_EQ(owners[i],
                i < count ? i
                : config.GetLayerType(i) == Config::LayerType::kGlobal
                    ? count - 1
                    : count - 2);
    }
  }
}

// (dynamic prefill rows, explicit shape signatures / strict resizing)
class ModelBuilderExecutionTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    auto env = Environment::Create({});
    ASSERT_TRUE(env) << env.Error().Message();
    env_.emplace(std::move(*env));
    auto options = Options::Create();
    ASSERT_TRUE(options) << options.Error().Message();
    options_.emplace(std::move(*options));
    ASSERT_TRUE(options_->SetHardwareAccelerators(HwAccelerators::kCpu));
    auto cpu = options_->GetCpuOptions();
    ASSERT_TRUE(cpu);
    ASSERT_TRUE(cpu->SetNumThreads(1));
    ASSERT_TRUE(cpu->SetXNNPackFlags(
        TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
    ASSERT_TRUE(cpu->SetHintFullyDelegatedToSingleDelegate(true));
  }

  absl::StatusOr<Runner*> NewRunner(int capacity = 32) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto weights, TinyWeights(config_));
    const auto specs = OwnerSpecs(config_, 4);
    ModelFactory factory;
    const ModelAuthoringOptions authoring{
        .prefill_rows = 4,
        .dynamic_prefill_rows = std::get<0>(GetParam()),
        .explicit_dynamic_shapes = std::get<1>(GetParam())};
    LRT_TENSOR_RETURN_IF_ERROR(model_builder_internal::AddSignatures(
        factory, config_, weights, specs, authoring));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto bytes, factory.CreateFlatbuffer());
    models_.push_back(std::make_unique<std::vector<char>>(std::move(bytes)));
    const auto& model = *models_.back();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto compiled,
        LitertDynamicRunner::Create(
            *env_,
            absl::Span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(model.data()), model.size()),
            *options_));
    LRT_TENSOR_ASSIGN_OR_RETURN(auto bank,
                                ActiveKvBank::Create(specs, capacity));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto runner, RunnerTestPeer::Create(
                         *env_, std::move(compiled), std::move(weights),
                         std::move(bank), 4, authoring.dynamic_prefill_rows,
                         authoring.explicit_dynamic_shapes, config_));
    runners_.push_back(std::move(runner));
    return runners_.back().get();
  }

  void ExpectSameBank(const ActiveKvBank& a, const ActiveKvBank& b) {
    ASSERT_EQ(a.length(), b.length());
    for (int owner = 0; owner < 4; ++owner) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto ak, a.Keys(owner, 0, a.length()));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bk, b.Keys(owner, 0, b.length()));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto av, a.Values(owner, 0, a.length()));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bv, b.Values(owner, 0, b.length()));
      EXPECT_THAT(ak, testing::ElementsAreArray(bk)) << "key owner=" << owner;
      EXPECT_THAT(av, testing::ElementsAreArray(bv)) << "value owner=" << owner;
    }
  }

  const Config config_ = TinyConfig();
  std::optional<Environment> env_;
  std::optional<Options> options_;
  // Model bytes and environment must outlive all compiled runners.
  std::vector<std::unique_ptr<std::vector<char>>> models_;
  std::vector<std::unique_ptr<Runner>> runners_;
};

TEST_P(ModelBuilderExecutionTest, ChunkedPrefillMatchesTokenByTokenDecode) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * chunked, NewRunner());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference, NewRunner());
  const std::vector<int32_t> tokens{2, 5, 11, 17, 23, 7, 9, 3, 31, 4, 29, 13};
  // Reuse the same compiled models through shorter/longer prompts, empty
  // history, partial chunks and both sides of the sliding-window boundary.
  for (int prefix : {0, 1, 3, 4, 5, 11, 2}) {
    SCOPED_TRACE(prefix);
    chunked->Reset();
    reference->Reset();
    ASSERT_THAT(
        chunked->Prefill(absl::MakeConstSpan(tokens).subspan(0, prefix)),
        IsOk());
    for (int i = 0; i < prefix; ++i)
      ASSERT_TRUE(reference->Decode(tokens[i]).ok());
    EXPECT_EQ(chunked->bank().length(), prefix);
    ExpectSameBank(chunked->bank(), reference->bank());
    for (int token : {tokens[prefix], 19}) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto actual, chunked->Decode(token));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto expected, reference->Decode(token));
      ASSERT_EQ(actual.size(), config_.vocab_size);
      ASSERT_EQ(expected.size(), actual.size());
      for (int i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(std::isfinite(actual.data()[i]));
        EXPECT_NEAR(actual.data()[i], expected.data()[i], 1e-4f)
            << "logit=" << i;
      }
      EXPECT_GT(std::abs(actual.data()[0] - actual.data()[1]), 1e-4f);
    }
    ExpectSameBank(chunked->bank(), reference->bank());
  }
}

TEST_P(ModelBuilderExecutionTest,
       InvalidTokenAndCapacityFailurePreserveCommittedCache) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference, NewRunner(4));
  const std::vector<int32_t> prefix{2, 5, 11};
  ASSERT_THAT(runner->Prefill(prefix), IsOk());
  ASSERT_THAT(reference->Prefill(prefix), IsOk());
  EXPECT_FALSE(runner->Decode(-1).ok());
  EXPECT_FALSE(runner->Decode(config_.vocab_size).ok());
  const std::vector<int32_t> bad_chunk{7, -1};
  EXPECT_FALSE(runner->Prefill(bad_chunk).ok());
  ExpectSameBank(runner->bank(), reference->bank());
  ASSERT_TRUE(runner->Decode(17).ok());
  ASSERT_TRUE(reference->Decode(17).ok());
  EXPECT_FALSE(runner->Decode(23).ok());
  ExpectSameBank(runner->bank(), reference->bank());
  runner->Reset();
  EXPECT_EQ(runner->bank().length(), 0);
  EXPECT_TRUE(runner->Decode(23).ok());
  EXPECT_EQ(runner->bank().length(), 1);
}

TEST_P(ModelBuilderExecutionTest,
       SignaturesExposeOnlyNeededCachesAndDynamicDimensions) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, NewRunner());
  EXPECT_EQ(runner->prefill_rows(), 4);
  const auto* model = tflite::GetModel(models_.back()->data());
  ASSERT_EQ(model->signature_defs()->size(), 2);
  for (const auto* sig : *model->signature_defs()) {
    const bool decode = sig->signature_key()->str() == "decode";
    const auto* graph = model->subgraphs()->Get(sig->subgraph_index());
    EXPECT_EQ(sig->outputs()->size(), decode ? 9 : 8);
    bool saw_last_cache = false;
    bool saw_embedding = false;
    for (const auto* entry : *sig->inputs()) {
      const std::string name = entry->name()->str();
      const auto* tensor = graph->tensors()->Get(entry->tensor_index());
      EXPECT_EQ(name.find(".weight"), std::string::npos) << name;
      if (name == "model.layers.3.self_attn.past_k") saw_last_cache = true;
      EXPECT_EQ(name.find("model.layers.4.self_attn.past_"), std::string::npos);
      EXPECT_EQ(name.find("model.layers.5.self_attn.past_"), std::string::npos);
      if (name == "embedded_input") {
        saw_embedding = true;
        if (std::get<1>(GetParam())) {
          const auto* shape = tensor->shape_signature();
          if (shape == nullptr || shape->size() == 0) shape = tensor->shape();
          EXPECT_EQ(shape->Get(1), !decode && std::get<0>(GetParam()) ? -1
                                   : decode                           ? 1
                                                                      : 4);
        }
      }
      if (std::get<1>(GetParam()) &&
          name.find(".past_k") != std::string::npos) {
        ASSERT_NE(tensor->shape_signature(), nullptr);
        EXPECT_EQ(tensor->shape_signature()->Get(2), -1);
      }
    }
    EXPECT_TRUE(saw_embedding);
    EXPECT_EQ(saw_last_cache, decode);
  }
}

INSTANTIATE_TEST_SUITE_P(ShapeModes, ModelBuilderExecutionTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
