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

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/bundle_loader.h"
#include "tensor/examples/experimental/gemma4/kv_bank.h"
#include "tensor/examples/experimental/gemma4/kv_cache.h"
#include "tensor/examples/experimental/gemma4/runner.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_graph.h"
#include "tensor/examples/gemma4/helpers/quantized_embedding.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tensor/utils/matchers.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/schema/mutable/schema_generated.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::litert::tensor::examples::gemma4::Config;
using ::litert::tensor::examples::gemma4::GemmaEmbeddingTable;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::Not;

constexpr int kTinyVocabSize = 64;
constexpr int kTinyNumLayers = 6;
constexpr int kTinyNumOwners = 4;
constexpr int kTinyPrefillRows = 4;
constexpr int kTinySlidingWindow = 4;
constexpr int kTinyAlignment = 4;

struct ShapeMode {
  std::string name;
  bool dynamic_prefill_rows;
  bool explicit_dynamic_shapes;
};

// Six-layer synthetic Gemma4 config with 4 unique KV owners (layers 0..3),
// 2 shared consumer layers (layers 4->2 local, 5->3 global), vocab_size = 64,
// and sliding_window_size = 4.
Config SixLayerFourOwnerConfig() {
  Config config = Config::E2B();
  config.vocab_size = kTinyVocabSize;
  config.embed_dim = 32;
  config.hidden_dim = 64;
  // Four-token alignment times the 16-byte local head preserves 64-byte
  // host-buffer alignment when the sliding-window view advances.
  config.head_dim = 16;
  config.global_key_size = 32;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.num_layers = kTinyNumLayers;
  config.per_layer_input_dim = 4;
  config.sliding_window_size = kTinySlidingWindow;
  config.attention_pattern_size = 2;
  config.frac_shared_layers = 2.0f / 6.0f;
  return config;
}

std::vector<KvOwnerSpec> OwnerSpecs(const Config& config, int count) {
  std::vector<KvOwnerSpec> specs;
  specs.reserve(count);
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
absl::StatusOr<LoadedTensors> TinyWeights(const Config& config) {
  LoadedTensors loaded;
  int seed = 0;
  const auto add = [&](absl::string_view name, Shape shape,
                       std::optional<float> constant = std::nullopt) {
    int count = 1;
    for (int dim : shape) {
      count *= dim;
    }
    std::vector<float> values(count);
    ++seed;
    for (int i = 0; i < count; ++i) {
      values[i] = constant.value_or(0.1f * std::sin(i * 0.73f + seed * 1.31f));
    }
    const std::string key(name);
    loaded.weights_handle.emplace(
        key,
        TensorHandle({.name = key,
                      .type = Type::kFP32,
                      .shape = std::move(shape),
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)}));
  };
  add("model.embed_tokens.weight", {config.vocab_size, config.embed_dim});
  add("model.embed_tokens_per_layer.weight",
      {config.vocab_size, config.num_layers * config.per_layer_input_dim});
  add("model.norm.weight", {config.embed_dim}, 1.0f);
  add("model.per_layer_projection_norm.weight", {config.per_layer_input_dim},
      1.0f);
  add("lm_head.weight", {config.vocab_size, config.embed_dim});
  add("matched.logits_softcap_reciprocal", {}, 0.03333333507180214f);
  constexpr absl::string_view kProjection = "model.per_layer_model_projection";
  std::vector<int8_t> codes(config.num_layers * config.per_layer_input_dim *
                            config.embed_dim);
  for (int i = 0; i < static_cast<int>(codes.size()); ++i) {
    codes[i] = (i * 7 + i / 11) % 15 - 7;
  }
  const std::string projection_weight_name =
      absl::StrCat(kProjection, ".weight");
  loaded.weights_handle.emplace(
      projection_weight_name,
      TensorHandle(
          {.name = projection_weight_name,
           .type = Type::kI8,
           .shape = {config.num_layers * config.per_layer_input_dim,
                     config.embed_dim},
           .buffer = OwningCpuBuffer::Copy<Type::kI8>(codes),
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               std::vector<float>(
                   config.num_layers * config.per_layer_input_dim, 0.015625f),
               std::vector<int64_t>{0}, 0)}));
  add(absl::StrCat(kProjection, ".input_scale"), {}, 0.03125f);
  add(absl::StrCat(kProjection, ".output_scale"), {}, 0.015625f);
  for (int layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = absl::StrCat("model.layers.", layer);
    const int dim = config.GetLayerType(layer) == Config::LayerType::kGlobal
                        ? config.global_key_size
                        : config.head_dim;
    for (absl::string_view norm :
         {"input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "post_per_layer_input_norm"}) {
      add(absl::StrCat(prefix, ".", norm, ".weight"), {config.embed_dim}, 1.0f);
    }
    for (absl::string_view norm : {"q_norm", "k_norm"}) {
      add(absl::StrCat(prefix, ".self_attn.", norm, ".weight"), {dim}, 1.0f);
    }
    add(absl::StrCat(prefix, ".self_attn.q_proj.weight"),
        {config.num_heads * dim, config.embed_dim});
    add(absl::StrCat(prefix, ".self_attn.k_proj.weight"),
        {config.num_kv_heads * dim, config.embed_dim});
    add(absl::StrCat(prefix, ".self_attn.v_proj.weight"),
        {config.num_kv_heads * dim, config.embed_dim});
    add(absl::StrCat(prefix, ".self_attn.o_proj.weight"),
        {config.embed_dim, config.num_heads * dim});
    add(absl::StrCat(prefix, ".mlp.gate_proj.weight"),
        {config.hidden_dim, config.embed_dim});
    add(absl::StrCat(prefix, ".mlp.up_proj.weight"),
        {config.hidden_dim, config.embed_dim});
    add(absl::StrCat(prefix, ".mlp.down_proj.weight"),
        {config.embed_dim, config.hidden_dim});
    add(absl::StrCat(prefix, ".per_layer_input_gate.weight"),
        {config.per_layer_input_dim, config.embed_dim});
    add(absl::StrCat(prefix, ".per_layer_projection.weight"),
        {config.embed_dim, config.per_layer_input_dim});
    add(absl::StrCat(prefix, ".layer_scalar"), {1}, 0.8f + layer * 0.02f);
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

// Populates lightweight placeholder handles for every expected E2B weight name
// so E2B validation and authoring delegation can be tested without allocating
// gigabytes of weight buffers.
LoadedTensors PlaceholderE2BWeights(bool include_softcap_reciprocal) {
  LoadedTensors loaded;
  const Config config = Config::E2B();
  for (const auto& [name, expected] :
       published_bundle_detail::ExpectedTensors(config)) {
    const Type type = expected.dtype == "float32" ? Type::kFP32
                      : expected.dtype == "int8"  ? Type::kI8
                                                  : Type::kI4;
    if (absl::EndsWith(name, ".input_scale") ||
        absl::EndsWith(name, ".output_scale")) {
      loaded.weights_handle.emplace(name, TensorHandle({.name = name,
                                                        .type = Type::kFP32,
                                                        .shape = {1},
                                                        .buffer = 0.03125f}));
    } else if (type == Type::kFP32) {
      loaded.weights_handle.emplace(
          name,
          TensorHandle({.name = name, .type = type, .shape = expected.shape}));
    } else {
      const int rows = expected.shape.empty() ? 1 : expected.shape[0];
      loaded.weights_handle.emplace(
          name,
          TensorHandle(
              {.name = name,
               .type = type,
               .shape = expected.shape,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   std::vector<float>(rows, 0.03125f),
                   std::vector<int64_t>(rows, 0), 0)}));
    }
  }
  if (include_softcap_reciprocal) {
    loaded.weights_handle.emplace(
        "matched.logits_softcap_reciprocal",
        TensorHandle({.name = "matched.logits_softcap_reciprocal",
                      .type = Type::kFP32,
                      .shape = {},
                      .buffer = 0.03333333507180214f}));
  }
  return loaded;
}

absl::Status CompareBanks(const ActiveKvBank& actual,
                          const ActiveKvBank& expected, int num_owners) {
  if (actual.length() != expected.length()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Bank length mismatch: ", actual.length(), " vs ", expected.length()));
  }
  for (int owner = 0; owner < num_owners; ++owner) {
    LRT_TENSOR_ASSIGN_OR_RETURN(absl::Span<const int8_t> actual_keys,
                                actual.Keys(owner, 0, actual.length()));
    LRT_TENSOR_ASSIGN_OR_RETURN(absl::Span<const int8_t> expected_keys,
                                expected.Keys(owner, 0, expected.length()));
    if (actual_keys != expected_keys) {
      return absl::FailedPreconditionError(
          absl::StrCat("Key cache mismatch for owner ", owner));
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(absl::Span<const int8_t> actual_values,
                                actual.Values(owner, 0, actual.length()));
    LRT_TENSOR_ASSIGN_OR_RETURN(absl::Span<const int8_t> expected_values,
                                expected.Values(owner, 0, expected.length()));
    if (actual_values != expected_values) {
      return absl::FailedPreconditionError(
          absl::StrCat("Value cache mismatch for owner ", owner));
    }
  }
  return absl::OkStatus();
}

struct SignatureSnapshot {
  int output_count = 0;
  absl::flat_hash_map<std::string, std::vector<int32_t>> input_shapes;
  absl::flat_hash_map<std::string, std::vector<int32_t>> input_signatures;
  absl::flat_hash_map<std::string, std::vector<int32_t>> output_signatures;
};

absl::flat_hash_map<std::string, SignatureSnapshot> ParseSignatureSnapshots(
    const std::vector<char>& flatbuffer_bytes) {
  absl::flat_hash_map<std::string, SignatureSnapshot> result;
  const tflite::Model* model = tflite::GetModel(flatbuffer_bytes.data());
  if (model == nullptr || model->signature_defs() == nullptr) {
    return result;
  }
  const auto read_vec = [](const flatbuffers::Vector<int32_t>* vec) {
    std::vector<int32_t> out;
    if (vec != nullptr) {
      out.assign(vec->begin(), vec->end());
    }
    return out;
  };
  for (const tflite::SignatureDef* sig : *model->signature_defs()) {
    const tflite::SubGraph* graph =
        model->subgraphs()->Get(sig->subgraph_index());
    SignatureSnapshot snap;
    snap.output_count = static_cast<int>(sig->outputs()->size());
    for (const tflite::TensorMap* entry : *sig->inputs()) {
      const std::string name = entry->name()->str();
      const tflite::Tensor* tensor =
          graph->tensors()->Get(entry->tensor_index());
      snap.input_shapes.emplace(name, read_vec(tensor->shape()));
      snap.input_signatures.emplace(name, read_vec(tensor->shape_signature()));
    }
    for (const tflite::TensorMap* entry : *sig->outputs()) {
      const std::string name = entry->name()->str();
      const tflite::Tensor* tensor =
          graph->tensors()->Get(entry->tensor_index());
      snap.output_signatures.emplace(name, read_vec(tensor->shape_signature()));
    }
    result.emplace(sig->signature_key()->str(), std::move(snap));
  }
  return result;
}

struct InvalidPrefillRowsCase {
  std::string name;
  int rows;
};

class ModelBuilderInvalidPrefillRowsTest
    : public testing::TestWithParam<InvalidPrefillRowsCase> {};

TEST_P(ModelBuilderInvalidPrefillRowsTest, RejectsInvalidPrefillRows) {
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, {}, {}, {.prefill_rows = GetParam().rows}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("power-of-two")));
}

INSTANTIATE_TEST_SUITE_P(
    PrefillRows, ModelBuilderInvalidPrefillRowsTest,
    testing::Values(InvalidPrefillRowsCase{"Zero", 0},
                    InvalidPrefillRowsCase{"One", 1},
                    InvalidPrefillRowsCase{"BelowMin31", 31},
                    InvalidPrefillRowsCase{"NonPowerOfTwo33", 33},
                    InvalidPrefillRowsCase{"AboveMax1025", 1025},
                    InvalidPrefillRowsCase{"AboveMaxPowerOfTwo2048", 2048}),
    [](const testing::TestParamInfo<InvalidPrefillRowsCase>& info) {
      return info.param.name;
    });

struct InvalidOwnerSpecCase {
  std::string name;
  std::function<void(std::vector<KvOwnerSpec>&)> mutate;
  std::string expected_error;
};

class ModelBuilderInvalidOwnerSpecTest
    : public testing::TestWithParam<InvalidOwnerSpecCase> {};

TEST_P(ModelBuilderInvalidOwnerSpecTest, RejectsInvalidOwnerSpec) {
  std::vector<KvOwnerSpec> specs = OwnerSpecs(Config::E2B(), 15);
  GetParam().mutate(specs);
  ModelFactory factory;

  EXPECT_THAT(AddGemma4Signatures(factory, {}, specs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(GetParam().expected_error)));
}

INSTANTIATE_TEST_SUITE_P(
    OwnerSpecs, ModelBuilderInvalidOwnerSpecTest,
    testing::Values(
        InvalidOwnerSpecCase{"EmptyOwners",
                             [](std::vector<KvOwnerSpec>& s) { s.clear(); },
                             "15 E2B"},
        InvalidOwnerSpecCase{
            "MismatchedOwnerId",
            [](std::vector<KvOwnerSpec>& s) { s[2].owner = 1; }, "metadata"},
        InvalidOwnerSpecCase{
            "MultiHeadOwner",
            [](std::vector<KvOwnerSpec>& s) { s[0].num_heads = 2; },
            "metadata"},
        InvalidOwnerSpecCase{
            "WrongHeadDim",
            [](std::vector<KvOwnerSpec>& s) { s[0].head_dim = 512; },
            "metadata"},
        InvalidOwnerSpecCase{
            "ZeroKeyScale",
            [](std::vector<KvOwnerSpec>& s) { s[0].key_scale = 0.0f; },
            "metadata"},
        InvalidOwnerSpecCase{
            "NegativeKeyScale",
            [](std::vector<KvOwnerSpec>& s) { s[0].key_scale = -0.5f; },
            "metadata"},
        InvalidOwnerSpecCase{"InfKeyScale",
                             [](std::vector<KvOwnerSpec>& s) {
                               s[0].key_scale =
                                   std::numeric_limits<float>::infinity();
                             },
                             "metadata"},
        InvalidOwnerSpecCase{"NanKeyScale",
                             [](std::vector<KvOwnerSpec>& s) {
                               s[0].key_scale =
                                   std::numeric_limits<float>::quiet_NaN();
                             },
                             "metadata"},
        InvalidOwnerSpecCase{
            "ZeroValueScale",
            [](std::vector<KvOwnerSpec>& s) { s[0].value_scale = 0.0f; },
            "metadata"},
        InvalidOwnerSpecCase{
            "NegativeValueScale",
            [](std::vector<KvOwnerSpec>& s) { s[0].value_scale = -0.5f; },
            "metadata"},
        InvalidOwnerSpecCase{"InfValueScale",
                             [](std::vector<KvOwnerSpec>& s) {
                               s[0].value_scale =
                                   std::numeric_limits<float>::infinity();
                             },
                             "metadata"},
        InvalidOwnerSpecCase{"NanValueScale",
                             [](std::vector<KvOwnerSpec>& s) {
                               s[0].value_scale =
                                   std::numeric_limits<float>::quiet_NaN();
                             },
                             "metadata"}),
    [](const testing::TestParamInfo<InvalidOwnerSpecCase>& info) {
      return info.param.name;
    });

TEST(ModelBuilderValidationTest, RejectsMissingExpectedWeightBeforeAuthoring) {
  ModelFactory factory;

  EXPECT_THAT(AddGemma4Signatures(factory, {}, OwnerSpecs(Config::E2B(), 15)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing model weight")));
}

TEST(ModelBuilderValidationTest,
     RejectsMissingSoftcapReciprocalWhenAllExpectedE2BWeightsPresent) {
  const LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/false);
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("matched.logits_softcap_reciprocal")));
}

TEST(ModelBuilderValidationTest,
     RejectsExpectedWeightWithWrongShapeBeforeAuthoring) {
  LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  weights.weights_handle.insert_or_assign(
      "model.norm.weight",
      TensorHandle(
          {.name = "model.norm.weight", .type = Type::kFP32, .shape = {1}}));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unexpected shape for model weight")));
}

TEST(ModelBuilderValidationTest,
     RejectsExpectedWeightWithWrongDtypeBeforeAuthoring) {
  LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  weights.weights_handle.insert_or_assign(
      "model.norm.weight", TensorHandle({.name = "model.norm.weight",
                                         .type = Type::kI8,
                                         .shape = {Config::E2B().embed_dim}}));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unexpected dtype for model weight")));
}

TEST(ModelBuilderValidationTest,
     RejectsExpectedWeightWithErrorStatusBeforeAuthoring) {
  LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  weights.weights_handle.insert_or_assign(
      "model.norm.weight",
      TensorHandle(absl::FailedPreconditionError("corrupt weight")));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid model weight")));
}

TEST(ModelBuilderValidationTest,
     RejectsNonPositiveSoftcapReciprocalBeforeAuthoring) {
  LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  weights.weights_handle.insert_or_assign(
      "matched.logits_softcap_reciprocal",
      TensorHandle({.name = "matched.logits_softcap_reciprocal",
                    .type = Type::kFP32,
                    .shape = {},
                    .buffer = 0.0f}));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("finite and positive")));
}

TEST(ModelBuilderValidationTest,
     AcceptsRepackedInt2WeightForExpectedInt4Tensor) {
  LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  const Config e2b = Config::E2B();
  weights.weights_handle.insert_or_assign(
      "lm_head.weight",
      TensorHandle(
          {.name = "lm_head.weight",
           .type = Type::kI2,
           .shape = {e2b.vocab_size, e2b.embed_dim},
           .quantization = std::make_shared<PerChannelAffineQuantization>(
               std::vector<float>(e2b.vocab_size, 0.03125f),
               std::vector<int64_t>(e2b.vocab_size, 0), 0)}));
  ModelFactory factory;

  ASSERT_THAT(AddGemma4Signatures(factory, weights, OwnerSpecs(e2b, 15),
                                  {.prefill_rows = 32}),
              IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> flatbuffer,
                                  factory.CreateFlatbuffer());
  EXPECT_THAT(ParseSignatureSnapshots(flatbuffer), Contains(Key("decode")));
}

TEST(ModelBuilderValidationTest,
     AuthorsValidE2BSignaturesThroughPublishedEntryPoint) {
  const LoadedTensors weights =
      PlaceholderE2BWeights(/*include_softcap_reciprocal=*/true);
  ModelFactory factory;

  ASSERT_THAT(
      AddGemma4Signatures(factory, weights, OwnerSpecs(Config::E2B(), 15),
                          {.prefill_rows = 32}),
      IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> flatbuffer,
                                  factory.CreateFlatbuffer());
  const absl::flat_hash_map<std::string, SignatureSnapshot> snaps =
      ParseSignatureSnapshots(flatbuffer);

  ASSERT_THAT(snaps, Contains(Key("prefill")));
  ASSERT_THAT(snaps, Contains(Key("decode")));
  EXPECT_EQ(snaps.at("prefill").output_count, 30);
  EXPECT_EQ(snaps.at("decode").output_count, 31);
  EXPECT_THAT(snaps.at("prefill").input_shapes.at("embedded_input"),
              ElementsAre(1, 32, 1536));
  EXPECT_THAT(snaps.at("decode").input_shapes.at("embedded_input"),
              ElementsAre(1, 1, 1536));
}

TEST(ModelBuilderValidationTest, ConfigOverloadRejectsNonPositivePrefillRows) {
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, config, weights,
                          OwnerSpecs(config, kTinyNumOwners),
                          {.prefill_rows = 0}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(ModelBuilderValidationTest, ConfigOverloadRejectsInvalidRopeParameters) {
  Config config = SixLayerFourOwnerConfig();
  config.global_rope_proportion = 2.0f;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  ModelFactory factory;

  EXPECT_THAT(AddGemma4Signatures(factory, config, weights,
                                  OwnerSpecs(config, kTinyNumOwners),
                                  {.prefill_rows = kTinyPrefillRows}),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("RoPE")));
}

TEST(ModelBuilderValidationTest, ConfigOverloadRejectsInsufficientOwnerSpecs) {
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  ModelFactory factory;

  EXPECT_THAT(
      AddGemma4Signatures(factory, config, weights, OwnerSpecs(config, 2),
                          {.prefill_rows = kTinyPrefillRows}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid KV owner index")));
}

TEST(ModelBuilderValidationTest,
     ConfigOverloadRejectsOutOfRangeOwnerLayerIndex) {
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  std::vector<KvOwnerSpec> specs = OwnerSpecs(config, kTinyNumOwners);
  specs.push_back({kTinyNumLayers, config.head_dim, 0.03125f, 0.03125f, 1});
  ModelFactory factory;

  EXPECT_THAT(AddGemma4Signatures(factory, config, weights, specs,
                                  {.prefill_rows = kTinyPrefillRows}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid KV owner layer index")));
}

TEST(ModelBuilderValidationTest,
     ConfigOverloadSkipsUnusedMaskForSingleLayerTypeConfig) {
  Config config = SixLayerFourOwnerConfig();
  config.num_layers = 1;
  config.attention_pattern_size = 1;
  config.frac_shared_layers = 0.0f;
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  ModelFactory factory;

  ASSERT_THAT(
      AddGemma4Signatures(
          factory, config, weights, OwnerSpecs(config, /*count=*/1),
          {.prefill_rows = kTinyPrefillRows, .explicit_dynamic_shapes = true}),
      IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> flatbuffer,
                                  factory.CreateFlatbuffer());
  const absl::flat_hash_map<std::string, SignatureSnapshot> snaps =
      ParseSignatureSnapshots(flatbuffer);

  ASSERT_THAT(snaps, Contains(Key("decode")));
  EXPECT_THAT(snaps.at("decode").input_signatures,
              Contains(Key("joined_global_mask")));
  EXPECT_THAT(snaps.at("decode").input_signatures,
              Not(Contains(Key("joined_local_mask"))));
  EXPECT_THAT(snaps.at("decode").input_signatures.at("joined_global_mask"),
              ElementsAre(1, 1, 2, -1));
}

TEST(ModelBuilderTest, TinyConfigCacheSharingReusesLastLocalAndGlobalOwners) {
  EXPECT_THAT(
      examples::gemma4::GetKvCacheSharingPatterns(SixLayerFourOwnerConfig()),
      ElementsAre(0, 1, 2, 3, 2, 3));
}

TEST(ModelBuilderTest, E2BCacheSharingReusesOwners13And14) {
  // E2B has 35 layers, 15 unique owners (0..14), attention pattern period 5
  // (global layers at indices 4, 9, 14, 19, 24, 29, 34).
  EXPECT_THAT(examples::gemma4::GetKvCacheSharingPatterns(Config::E2B()),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13,
                          13, 13, 13, 14, 13, 13, 13, 13, 14, 13, 13, 13, 13,
                          14, 13, 13, 13, 13, 14));
}

TEST(ModelBuilderTest, E4BCacheSharingReusesOwners22And23) {
  // E4B has 42 layers, 24 unique owners (0..23), attention pattern period 6
  // (global layers at indices 5, 11, 17, 23, 29, 35, 41).
  EXPECT_THAT(examples::gemma4::GetKvCacheSharingPatterns(Config::E4B()),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 22, 22, 22, 22, 22,
                          23, 22, 22, 22, 22, 22, 23, 22, 22, 22, 22, 22, 23));
}

class ModelBuilderExecutionTest : public testing::TestWithParam<ShapeMode> {
 protected:
  void SetUp() override {
    Expected<Environment> env = Environment::Create({});
    ASSERT_TRUE(env.HasValue()) << env.Error().Message();
    env_.emplace(std::move(*env));
    Expected<Options> options = Options::Create();
    ASSERT_TRUE(options.HasValue()) << options.Error().Message();
    options_.emplace(std::move(*options));
    ASSERT_TRUE(options_->SetHardwareAccelerators(HwAccelerators::kCpu));
    Expected<CpuOptions&> cpu = options_->GetCpuOptions();
    ASSERT_TRUE(cpu.HasValue()) << cpu.Error().Message();
    ASSERT_TRUE(cpu->SetNumThreads(1));
    ASSERT_TRUE(cpu->SetXNNPackFlags(
        TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS));
    ASSERT_TRUE(cpu->SetHintFullyDelegatedToSingleDelegate(true));
  }

  absl::StatusOr<Runner*> NewRunner(int capacity) {
    LRT_TENSOR_ASSIGN_OR_RETURN(LoadedTensors weights, TinyWeights(config_));
    const std::vector<KvOwnerSpec> specs = OwnerSpecs(config_, kTinyNumOwners);
    const ShapeMode mode = GetParam();
    const ModelAuthoringOptions authoring{
        .prefill_rows = kTinyPrefillRows,
        .dynamic_prefill_rows = mode.dynamic_prefill_rows,
        .explicit_dynamic_shapes = mode.explicit_dynamic_shapes};
    ModelFactory factory;
    LRT_TENSOR_RETURN_IF_ERROR(
        AddGemma4Signatures(factory, config_, weights, specs, authoring));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<char> bytes,
                                factory.CreateFlatbuffer());
    models_.push_back(std::make_unique<std::vector<char>>(std::move(bytes)));
    const std::vector<char>& model = *models_.back();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        LitertDynamicRunner compiled,
        LitertDynamicRunner::Create(
            *env_,
            absl::Span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(model.data()), model.size()),
            *options_));
    LRT_TENSOR_ASSIGN_OR_RETURN(ActiveKvBank bank,
                                ActiveKvBank::Create(specs, capacity));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::unique_ptr<Runner> runner,
        Runner::Create(*env_, std::move(compiled), std::move(weights),
                       std::move(bank), config_,
                       {.alignment = kTinyAlignment,
                        .dynamic_prefill_rows = authoring.dynamic_prefill_rows,
                        .strict_shapes = authoring.explicit_dynamic_shapes}));
    runners_.push_back(std::move(runner));
    return runners_.back().get();
  }

  const Config config_ = SixLayerFourOwnerConfig();
  std::optional<Environment> env_;
  std::optional<Options> options_;
  // Model bytes and environment must outlive all compiled runners.
  std::vector<std::unique_ptr<std::vector<char>>> models_;
  std::vector<std::unique_ptr<Runner>> runners_;
};

TEST_P(ModelBuilderExecutionTest, ChunkedPrefillMatchesTokenByTokenDecode) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * chunked, NewRunner(/*capacity=*/32));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference,
                                  NewRunner(/*capacity=*/32));
  const std::vector<int32_t> tokens{2, 5, 11, 17, 23, 7, 9, 3, 31, 4, 29, 13};

  // Reuse the same compiled models through shorter/longer prompts, empty
  // history, partial chunks, and both sides of the sliding-window boundary (4).
  for (int prefix : {0, 1, 3, 4, 5, 11, 2}) {
    SCOPED_TRACE(absl::StrCat("prefix=", prefix));
    chunked->Reset();
    reference->Reset();
    ASSERT_THAT(
        chunked->Prefill(absl::MakeConstSpan(tokens).subspan(0, prefix)),
        IsOk());
    for (int i = 0; i < prefix; ++i) {
      ASSERT_THAT(reference->Decode(tokens[i]), IsOk());
    }

    EXPECT_EQ(chunked->bank().length(), prefix);
    EXPECT_THAT(
        CompareBanks(chunked->bank(), reference->bank(), kTinyNumOwners),
        IsOk());
    for (int token : {tokens[prefix], 19}) {
      SCOPED_TRACE(absl::StrCat("token=", token));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> actual,
                                      chunked->Decode(token));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> expected,
                                      reference->Decode(token));
      ASSERT_EQ(actual.size(), kTinyVocabSize);
      ASSERT_EQ(expected.size(), kTinyVocabSize);
      for (int i = 0; i < kTinyVocabSize; ++i) {
        EXPECT_TRUE(std::isfinite(actual.data()[i]))
            << "logit=" << i << " value=" << actual.data()[i];
        EXPECT_NEAR(actual.data()[i], expected.data()[i], 1e-4f)
            << "logit=" << i;
      }
      EXPECT_GT(std::abs(actual.data()[0] - actual.data()[1]), 1e-4f);
    }
    EXPECT_THAT(
        CompareBanks(chunked->bank(), reference->bank(), kTinyNumOwners),
        IsOk());
  }
}

TEST_P(ModelBuilderExecutionTest, RejectsNegativeDecodeTokenPreservingCache) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference,
                                  NewRunner(/*capacity=*/4));
  ASSERT_THAT(runner->Prefill({2, 5, 11}), IsOk());
  ASSERT_THAT(reference->Prefill({2, 5, 11}), IsOk());

  EXPECT_THAT(runner->Decode(-1), StatusIs(absl::StatusCode::kInvalidArgument,
                                           HasSubstr("vocabulary")));

  EXPECT_THAT(CompareBanks(runner->bank(), reference->bank(), kTinyNumOwners),
              IsOk());
}

TEST_P(ModelBuilderExecutionTest, RejectsOutOfRangeDecodeTokenPreservingCache) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference,
                                  NewRunner(/*capacity=*/4));
  ASSERT_THAT(runner->Prefill({2, 5, 11}), IsOk());
  ASSERT_THAT(reference->Prefill({2, 5, 11}), IsOk());

  EXPECT_THAT(
      runner->Decode(kTinyVocabSize),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("vocabulary")));

  EXPECT_THAT(CompareBanks(runner->bank(), reference->bank(), kTinyNumOwners),
              IsOk());
}

TEST_P(ModelBuilderExecutionTest,
       RejectsInvalidTokenInPrefillChunkPreservingCache) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference,
                                  NewRunner(/*capacity=*/4));
  ASSERT_THAT(runner->Prefill({2, 5}), IsOk());
  ASSERT_THAT(reference->Prefill({2, 5}), IsOk());

  EXPECT_THAT(
      runner->Prefill({7, -1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("vocabulary")));

  EXPECT_THAT(CompareBanks(runner->bank(), reference->bank(), kTinyNumOwners),
              IsOk());
}

TEST_P(ModelBuilderExecutionTest,
       RejectsDecodeExceedingCapacityPreservingCache) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/4));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * reference,
                                  NewRunner(/*capacity=*/4));
  ASSERT_THAT(runner->Prefill({2, 5, 11}), IsOk());
  ASSERT_THAT(reference->Prefill({2, 5, 11}), IsOk());
  ASSERT_THAT(runner->Decode(17), IsOk());
  ASSERT_THAT(reference->Decode(17), IsOk());

  EXPECT_THAT(runner->Decode(23), StatusIs(absl::StatusCode::kOutOfRange));

  EXPECT_THAT(CompareBanks(runner->bank(), reference->bank(), kTinyNumOwners),
              IsOk());
}

TEST_P(ModelBuilderExecutionTest, ResetAllowsFreshDecodeAfterCapacityReached) {
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/4));
  ASSERT_THAT(runner->Prefill({2, 5, 11}), IsOk());
  ASSERT_THAT(runner->Decode(17), IsOk());

  runner->Reset();

  EXPECT_EQ(runner->bank().length(), 0);
  EXPECT_THAT(runner->Decode(23), IsOk());
  EXPECT_EQ(runner->bank().length(), 1);
}

TEST_P(ModelBuilderExecutionTest,
       SignaturesExposeOnlyNeededCachesAndDynamicDimensions) {
  const ShapeMode mode = GetParam();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner * runner, NewRunner(/*capacity=*/32));

  const absl::flat_hash_map<std::string, SignatureSnapshot> snaps =
      ParseSignatureSnapshots(*models_.back());

  EXPECT_EQ(runner->prefill_rows(), kTinyPrefillRows);
  ASSERT_EQ(snaps.size(), 2);
  const SignatureSnapshot& prefill = snaps.at("prefill");
  const SignatureSnapshot& decode = snaps.at("decode");
  // 4 KV owners -> 8 KV outputs for prefill, +1 logits output = 9 for decode.
  EXPECT_EQ(prefill.output_count, 8);
  EXPECT_EQ(decode.output_count, 9);
  // Final owner (layer 3) produces KV during prefill without reading history;
  // shared layers 4 and 5 never expose past KV inputs in either signature.
  EXPECT_THAT(prefill.input_shapes,
              Not(Contains(Key("model.layers.3.self_attn.past_k"))));
  EXPECT_THAT(decode.input_shapes,
              Contains(Key("model.layers.3.self_attn.past_k")));
  EXPECT_THAT(decode.input_shapes,
              Not(Contains(Key("model.layers.4.self_attn.past_k"))));
  EXPECT_THAT(decode.input_shapes,
              Not(Contains(Key("model.layers.5.self_attn.past_k"))));
  for (const auto& [name, unused_shape] : decode.input_shapes) {
    EXPECT_THAT(name, Not(HasSubstr(".weight")));
  }

  if (mode.explicit_dynamic_shapes) {
    EXPECT_THAT(prefill.input_signatures.at("model.layers.0.self_attn.past_k"),
                ElementsAre(1, 1, -1, 16));
    EXPECT_THAT(prefill.input_signatures.at("model.layers.0.self_attn.past_v"),
                ElementsAre(1, 1, -1, 16));
    EXPECT_THAT(prefill.input_signatures.at("model.layers.0.self_attn.pad_k"),
                ElementsAre(1, 1, -1, 16));
    EXPECT_THAT(prefill.input_signatures.at("model.layers.0.self_attn.pad_v"),
                ElementsAre(1, 1, -1, 16));
    EXPECT_THAT(decode.input_signatures.at("model.layers.3.self_attn.past_k"),
                ElementsAre(1, 1, -1, 32));
    EXPECT_THAT(decode.input_signatures.at("joined_global_mask"),
                ElementsAre(1, 1, 2, -1));
    if (mode.dynamic_prefill_rows) {
      EXPECT_THAT(prefill.input_signatures.at("embedded_input"),
                  ElementsAre(1, -1, 32));
      EXPECT_THAT(prefill.input_signatures.at("positions"),
                  ElementsAre(1, 1, -1, 1));
      EXPECT_THAT(prefill.input_signatures.at("per_layer_token_embedding_0"),
                  ElementsAre(1, -1, 4));
      EXPECT_THAT(prefill.input_signatures.at("joined_local_mask"),
                  ElementsAre(1, 1, -1, -1));
      EXPECT_THAT(prefill.output_signatures.at("new_key_0"),
                  ElementsAre(1, 1, -1, 16));
    } else {
      EXPECT_THAT(prefill.input_signatures.at("embedded_input"), IsEmpty());
      EXPECT_THAT(prefill.input_signatures.at("joined_local_mask"),
                  ElementsAre(1, 1, kTinyPrefillRows, -1));
    }
  } else {
    EXPECT_THAT(prefill.input_signatures.at("embedded_input"), IsEmpty());
    EXPECT_THAT(prefill.input_signatures.at("model.layers.0.self_attn.past_k"),
                IsEmpty());
    EXPECT_THAT(decode.input_signatures.at("model.layers.3.self_attn.past_k"),
                IsEmpty());
    EXPECT_THAT(decode.input_signatures.at("joined_global_mask"), IsEmpty());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShapeModes, ModelBuilderExecutionTest,
    testing::Values(
        ShapeMode{"StaticFixed", /*dynamic_prefill_rows=*/false,
                  /*explicit_dynamic_shapes=*/false},
        ShapeMode{"StaticExplicitDynamic", /*dynamic_prefill_rows=*/false,
                  /*explicit_dynamic_shapes=*/true},
        ShapeMode{"DynamicRowsImplicit", /*dynamic_prefill_rows=*/true,
                  /*explicit_dynamic_shapes=*/false},
        ShapeMode{"DynamicRowsExplicit", /*dynamic_prefill_rows=*/true,
                  /*explicit_dynamic_shapes=*/true}),
    [](const testing::TestParamInfo<ShapeMode>& info) {
      return info.param.name;
    });

TEST(RunnerValidationTest, RejectsInvalidAlignmentInInjectedCreate) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue()) << env.Error().Message();
  Expected<Options> options = Options::Create();
  ASSERT_TRUE(options.HasValue()) << options.Error().Message();
  ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kCpu));
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  const std::vector<KvOwnerSpec> specs = OwnerSpecs(config, kTinyNumOwners);
  ModelFactory factory;
  ASSERT_THAT(AddGemma4Signatures(factory, config, weights, specs,
                                  {.prefill_rows = kTinyPrefillRows}),
              IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> bytes,
                                  factory.CreateFlatbuffer());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LitertDynamicRunner compiled,
      LitertDynamicRunner::Create(
          *env,
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()),
          *options));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(ActiveKvBank bank,
                                  ActiveKvBank::Create(specs, /*capacity=*/16));

  EXPECT_THAT(
      Runner::Create(*env, std::move(compiled), std::move(weights),
                     std::move(bank), config, {.alignment = 3}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Alignment")));
}

TEST(RunnerValidationTest, RejectsInvalidAlignmentInPathBasedCreate) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue()) << env.Error().Message();
  Expected<Options> options = Options::Create();
  ASSERT_TRUE(options.HasValue()) << options.Error().Message();

  EXPECT_THAT(
      Runner::Create(*env, *options, "/nonexistent.tflite", "/nonexistent_dir",
                     /*capacity=*/16,
                     /*alignment=*/256),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Alignment")));
}

TEST(RunnerValidationTest, RejectsMismatchedSignatureEmbedDim) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue()) << env.Error().Message();
  Expected<Options> options = Options::Create();
  ASSERT_TRUE(options.HasValue()) << options.Error().Message();
  ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kCpu));
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  const std::vector<KvOwnerSpec> specs = OwnerSpecs(config, kTinyNumOwners);
  ModelFactory factory;
  ASSERT_THAT(AddGemma4Signatures(factory, config, weights, specs,
                                  {.prefill_rows = kTinyPrefillRows}),
              IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> bytes,
                                  factory.CreateFlatbuffer());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LitertDynamicRunner compiled,
      LitertDynamicRunner::Create(
          *env,
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()),
          *options));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(ActiveKvBank bank,
                                  ActiveKvBank::Create(specs, /*capacity=*/16));
  Config mismatched = config;
  mismatched.embed_dim = 64;

  EXPECT_THAT(Runner::Create(*env, std::move(compiled), std::move(weights),
                             std::move(bank), mismatched,
                             {.alignment = kTinyAlignment}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid Gemma4 signature input shape")));
}

TEST(RunnerValidationTest,
     AbortsKvTransactionWhenStrictResizeFailsOnStaticModel) {
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue()) << env.Error().Message();
  Expected<Options> options = Options::Create();
  ASSERT_TRUE(options.HasValue()) << options.Error().Message();
  ASSERT_TRUE(options->SetHardwareAccelerators(HwAccelerators::kCpu));
  const Config config = SixLayerFourOwnerConfig();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LoadedTensors weights, TinyWeights(config));
  const std::vector<KvOwnerSpec> specs = OwnerSpecs(config, kTinyNumOwners);
  ModelFactory factory;
  // Author with explicit_dynamic_shapes = false, then run with strict_shapes =
  // true so Bind() fails after ActiveKvBank::BeginAppend has started.
  ASSERT_THAT(AddGemma4Signatures(factory, config, weights, specs,
                                  {.prefill_rows = kTinyPrefillRows,
                                   .dynamic_prefill_rows = false,
                                   .explicit_dynamic_shapes = false}),
              IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<char> bytes,
                                  factory.CreateFlatbuffer());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LitertDynamicRunner compiled,
      LitertDynamicRunner::Create(
          *env,
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()),
          *options));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(ActiveKvBank bank,
                                  ActiveKvBank::Create(specs, /*capacity=*/16));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Runner> runner,
      Runner::Create(*env, std::move(compiled), std::move(weights),
                     std::move(bank), config,
                     {.alignment = kTinyAlignment,
                      .dynamic_prefill_rows = false,
                      .strict_shapes = true}));

  EXPECT_THAT(runner->Decode(5), StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_FALSE(runner->bank().append_in_flight());
  EXPECT_EQ(runner->bank().length(), 0);
}

}  // namespace
}  // namespace litert::tensor::examples::experimental::gemma4::cpu
