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

#include "tensor/examples/gemma4/gemma4_graph.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/types/span.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/helpers/rope.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "xnnpack.h"
#ifndef LITERT_TENSOR_STANDALONE
#include "tflite/delegates/xnnpack/weight_cache.h"
#endif

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

absl::flat_hash_map<std::string, XnnTensor> CreateGemma4GraphTestWeights(
    int num_layers = 2) {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  for (int l = 0; l < num_layers; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    weights.insert(
        {absl::StrCat(prefix, ".self_attn.q_proj.weight"),
         XnnTensor({.name = "q_proj",
                    .type = Type::kFP32,
                    .shape = {8, 4},
                    .buffer = std::vector<float>{
                        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}})});

    weights.insert(
        {absl::StrCat(prefix, ".self_attn.k_proj.weight"),
         XnnTensor({.name = "k_proj",
                    .type = Type::kFP32,
                    .shape = {4, 4},
                    .buffer = std::vector<float>{
                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}})});

    weights.insert(
        {absl::StrCat(prefix, ".self_attn.v_proj.weight"),
         XnnTensor({.name = "v_proj",
                    .type = Type::kFP32,
                    .shape = {4, 4},
                    .buffer = std::vector<float>{
                        0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f}})});

    weights.insert(
        {absl::StrCat(prefix, ".self_attn.o_proj.weight"),
         XnnTensor({.name = "o_proj",
                    .type = Type::kFP32,
                    .shape = {4, 8},
                    .buffer = std::vector<float>{
                        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}})});

    weights.insert({absl::StrCat(prefix, ".self_attn.q_norm.weight"),
                    XnnTensor({.name = "q_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});

    weights.insert({absl::StrCat(prefix, ".self_attn.k_norm.weight"),
                    XnnTensor({.name = "k_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});

    weights.insert({absl::StrCat(prefix, ".input_layernorm.weight"),
                    XnnTensor({.name = "pre_attn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});
    weights.insert({absl::StrCat(prefix, ".post_attention_layernorm.weight"),
                    XnnTensor({.name = "post_attn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});
    weights.insert({absl::StrCat(prefix, ".pre_feedforward_layernorm.weight"),
                    XnnTensor({.name = "pre_ffn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});
    weights.insert({absl::StrCat(prefix, ".post_feedforward_layernorm.weight"),
                    XnnTensor({.name = "post_ffn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1})});

    weights.insert(
        {absl::StrCat(prefix, ".mlp.gate_proj.weight"),
         XnnTensor({.name = "gate_proj",
                    .type = Type::kFP32,
                    .shape = {6, 4},
                    .buffer = std::vector<float>{
                        1.0f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                        1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f}})});

    weights.insert(
        {absl::StrCat(prefix, ".mlp.up_proj.weight"),
         XnnTensor({.name = "up_proj",
                    .type = Type::kFP32,
                    .shape = {6, 4},
                    .buffer = std::vector<float>{
                        0.5f, 0.5f, 0.0f,  0.0f, 0.0f, 0.5f, 0.5f, 0.0f,
                        0.0f, 0.0f, 0.5f,  0.5f, 0.5f, 0.0f, 0.0f, 0.5f,
                        1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f}})});

    weights.insert(
        {absl::StrCat(prefix, ".mlp.down_proj.weight"),
         XnnTensor({.name = "down_proj",
                    .type = Type::kFP32,
                    .shape = {4, 6},
                    .buffer = std::vector<float>{
                        1.0f,  0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f,
                        0.0f,  0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
                        -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -0.5f}})});

    weights.insert({absl::StrCat(prefix, ".layer_scalar"),
                    XnnTensor({.name = "layer_scalar",
                               .type = Type::kFP32,
                               .shape = {1},
                               .buffer = 0.5f})});
  }

  // Model Final Norm
  weights.insert(
      {"model.norm.weight",
       XnnTensor({.name = "final_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});

  // Model Embeddings (Tied weights)
  std::vector<float> embed_table_data(40);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      embed_table_data[i * 4 + j] = i * 0.1f + j * 0.01f;
    }
  }
  weights.insert(
      {"model.embed_tokens.weight", XnnTensor({.name = "embed_tokens",
                                               .type = Type::kFP32,
                                               .shape = {10, 4},
                                               .buffer = embed_table_data})});

  return weights;
}

class Gemma4OutputHeadTest : public ::testing::TestWithParam<bool> {};

TEST_P(Gemma4OutputHeadTest, ProjectsWithCheckpointWeights) {
  Config config = Config::E4B();
  // Isolate the final normalization and output projection from transformer
  // layers so that the logits have a simple numerical reference.
  config.num_layers = 0;
  config.vocab_size = 3;
  config.embed_dim = 2;
  config.final_logit_softcap = 0.0f;

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 2}});
  absl::flat_hash_map<std::string, XnnTensor> checkpoint_weights;
  checkpoint_weights.insert(
      {"model.language_model.norm.weight",
       XnnTensor({.type = Type::kFP32,
                  .shape = {2},
                  .buffer = std::vector<float>{1.0f, 1.0f}})});
  checkpoint_weights.insert(
      {"model.language_model.embed_tokens.weight",
       XnnTensor({.type = Type::kFP32,
                  .shape = {3, 2},
                  .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                                               1.0f}})});
  if (GetParam()) {
    checkpoint_weights.insert(
        {"lm_head.weight",
         XnnTensor({.type = Type::kFP32,
                    .shape = {3, 2},
                    .buffer = std::vector<float>{2.0f, 0.0f, 0.0f, 3.0f, -1.0f,
                                                 1.0f}})});
  }

  // Apply the same mapping used to load checkpoint tensors in the example.
  for (const auto& [source_name, target_name] : GetGemma4WeightMapping(0)) {
    auto it = checkpoint_weights.find(source_name);
    if (it != checkpoint_weights.end()) {
      inputs.weights.insert({target_name, it->second});
    }
  }

  auto outputs = BuildGemma4Graph(inputs, config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({outputs.logits}));
  const std::array<float, 4> input_data = {1.0f, -1.0f, -1.0f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  // RMSNorm gives approximately [1, -1] and [-1, 1]. Project each row
  // against the selected weights, allowing for the normalization epsilon.
  const std::array<float, 6> expected_logits =
      GetParam() ? std::array<float, 6>{2.0f, -3.0f, -2.0f, -2.0f, 3.0f, 2.0f}
                 : std::array<float, 6>{1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f};
  EXPECT_THAT(runner.ReadOutputAs<float>(outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-5f), expected_logits)));
}

INSTANTIATE_TEST_SUITE_P(Gemma4GraphTest, Gemma4OutputHeadTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "ExplicitHead"
                                             : "TiedEmbeddings";
                         });

TEST(Gemma4GraphTest, ModelTest) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.weights = CreateGemma4GraphTestWeights(2);

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.071322f, 0.443513f, 0.814477f, 1.183199f, 1.548685f,
      1.909981f, 2.266176f, 2.616409f, 2.959886f, 3.295875f,
      0.066592f, 0.460456f, 0.852894f, 1.242703f, 1.628713f,
      2.009801f, 2.384902f, 2.753019f, 3.113236f, 3.464724f};

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TEST(Gemma4GraphTest, GlobalLayerGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.global_key_size = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.attention_pattern_size = 2;  // Layer 0 is local, Layer 1 is global

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.global_attention_mask.Set({.name = "global_attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_global_cos.Set(
      {.name = "rope_global_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_global_sin.Set(
      {.name = "rope_global_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.weights = CreateGemma4GraphTestWeights(2);

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> smask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, smask_data),
              IsOk());

  const std::array<float, 4> gmask_data = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.global_attention_mask, gmask_data),
              IsOk());

  const std::array<float, 8> cos_local_data = {
      0.8660254f, 0.5f, 0.8660254f, 0.5f, 0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_local_data), IsOk());

  const std::array<float, 8> sin_local_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_local_data), IsOk());

  const std::array<float, 8> cos_global_data = {1.0f, 0.0f, 1.0f, 0.0f,
                                                0.0f, 1.0f, 0.0f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_global_cos, cos_global_data), IsOk());

  const std::array<float, 8> sin_global_data = {0.0f, 1.0f, 0.0f, 1.0f,
                                                1.0f, 0.0f, 1.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_global_sin, sin_global_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.071176f, 0.446494f, 0.820556f, 1.192319f, 1.560770f,
      1.924927f, 2.283859f, 2.636688f, 2.982600f, 3.320851f,
      0.066741f, 0.460139f, 0.852115f, 1.241471f, 1.627041f,
      2.007705f, 2.382401f, 2.750137f, 3.110000f, 3.461159f};

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TEST(Gemma4GraphTest, KVCacheGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 1;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 4}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.key_caches = {XnnTensor(
      {.name = "key_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}})};
  inputs.value_caches = {XnnTensor(
      {.name = "value_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}})};
  inputs.weights = CreateGemma4GraphTestWeights(1);

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({model_outputs.logits, model_outputs.key_caches[0],
                             model_outputs.value_caches[0]}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 8> mask_data = {0.0f, 0.0f, 0.0f, -1e9f,
                                          0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  const std::array<float, 8> kc_data = {0.5f, 0.5f, 0.5f, 0.5f,
                                        1.0f, 1.0f, 1.0f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.key_caches[0], kc_data), IsOk());

  const std::array<float, 8> vc_data = {0.2f, 0.2f, 0.2f, 0.2f,
                                        0.4f, 0.4f, 0.4f, 0.4f};
  ASSERT_THAT(runner.SetInput(inputs.value_caches[0], vc_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(
                  FloatNear(1e-4f),
                  {0.072394f, 0.441714f, 0.809831f, 1.175752f, 1.538507f,
                   1.897162f, 2.250824f, 2.598654f, 2.939869f, 3.273751f,
                   0.066682f, 0.460588f, 0.853068f, 1.242918f, 1.628967f,
                   2.010093f, 2.385229f, 2.753378f, 3.113626f, 3.465142f})));

  EXPECT_THAT(
      runner.ReadOutputAs<float>(model_outputs.key_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.766109f, 0.097841f, 0.863950f, 1.630059f,
                              0.214422f, -0.909717f, 1.286534f, 1.212956f})));

  EXPECT_THAT(
      runner.ReadOutputAs<float>(model_outputs.value_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.365148f, 0.730295f, 1.095443f, 1.460591f,
                              0.758097f, 0.909716f, 1.061335f, 1.212954f})));
}

TEST(Gemma4GraphTest, SharedKVCacheGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 3;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 1.0f / 3.0f;  // Layer 2 is shared.
  config.share_local = true;

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.weights = CreateGemma4GraphTestWeights(3);

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {model_outputs.logits, model_outputs.key_caches[2]}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(
                  FloatNear(1e-4f),
                  {0.069635f, 0.444022f, 0.817165f, 1.188033f, 1.555614f,
                   1.918937f, 2.277076f, 2.629158f, 2.974375f, 3.311985f,
                   0.066323f, 0.460078f, 0.852409f, 1.242114f, 1.628025f,
                   2.009017f, 2.384028f, 2.752061f, 3.112201f, 3.463619f})));

  EXPECT_THAT(
      runner.ReadOutputAs<float>(model_outputs.key_caches[2]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.766109f, 0.097841f, 0.863950f, 1.630059f,
                              0.214422f, -0.909717f, 1.286534f, 1.212956f})));
}

TEST(Gemma4GraphTest, PerLayerInputsGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 2;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.per_layer_token_embeddings = {
      XnnTensor({.name = "per_layer_token_embedding_0",
                 .type = Type::kFP32,
                 .shape = {1, 2, 2}}),
      XnnTensor({.name = "per_layer_token_embedding_1",
                 .type = Type::kFP32,
                 .shape = {1, 2, 2}})};
  inputs.weights = CreateGemma4GraphTestWeights(2);

  inputs.weights.insert(
      {"model.layers.0.per_layer_model_projection.weight",
       XnnTensor({.name = "per_layer_model_projection_0",
                  .type = Type::kFP32,
                  .shape = {2, 4},
                  .buffer = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.1f,
                                               0.2f, 0.3f, 0.4f}})});
  inputs.weights.insert(
      {"model.layers.1.per_layer_model_projection.weight",
       XnnTensor({.name = "per_layer_model_projection_1",
                  .type = Type::kFP32,
                  .shape = {2, 4},
                  .buffer = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.1f,
                                               0.2f, 0.3f, 0.4f}})});
  inputs.weights.insert(
      {"model.per_layer_projection_norm.weight",
       XnnTensor({.name = "per_layer_projection_norm",
                  .type = Type::kFP32,
                  .shape = {2},
                  .buffer = std::vector<float>{1.0f, 1.0f}})});

  for (int l = 0; l < 2; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    inputs.weights.insert(
        {absl::StrCat(prefix, ".per_layer_input_gate.weight"),
         XnnTensor({.name = "per_layer_input_gate",
                    .type = Type::kFP32,
                    .shape = {2, 4},
                    .buffer = std::vector<float>{0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                 0.5f, 0.0f, 0.0f}})});
    inputs.weights.insert(
        {absl::StrCat(prefix, ".per_layer_projection.weight"),
         XnnTensor({.name = "per_layer_projection",
                    .type = Type::kFP32,
                    .shape = {4, 2},
                    .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.5f,
                                                 0.0f, 0.0f, 0.5f}})});
    inputs.weights.insert(
        {absl::StrCat(prefix, ".post_per_layer_input_norm.weight"),
         XnnTensor({.name = "post_per_layer_norm",
                    .type = Type::kFP32,
                    .shape = {4},
                    .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});
  }

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  const std::array<float, 4> pli_0_data = {0.1f, 0.2f, 0.3f, 0.4f};
  ASSERT_THAT(runner.SetInput(inputs.per_layer_token_embeddings[0], pli_0_data),
              IsOk());

  const std::array<float, 4> pli_1_data = {0.5f, 0.6f, 0.7f, 0.8f};
  ASSERT_THAT(runner.SetInput(inputs.per_layer_token_embeddings[1], pli_1_data),
              IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.068644f, 0.452825f, 0.835672f, 1.216068f, 1.592925f,
      1.965196f, 2.331886f, 2.692062f, 3.044862f, 3.389501f,
      0.064542f, 0.461408f, 0.856822f, 1.249556f, 1.638412f,
      2.022243f, 2.399960f, 2.770548f, 3.133072f, 3.486690f};

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-2f), expected_data)));
}

TEST(Gemma4GraphTest, NoLogitSoftcappingGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 0.0f;  // Soft capping disabled

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  inputs.weights = CreateGemma4GraphTestWeights(2);

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.071323f, 0.443804f, 0.816286f, 1.188767f, 1.561248f,
      1.933729f, 2.306210f, 2.678691f, 3.051172f, 3.423654f,
      0.066593f, 0.460782f, 0.854971f, 1.249160f, 1.643349f,
      2.037537f, 2.431726f, 2.825915f, 3.220104f, 3.614293f};

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TEST(Gemma4GraphTest, PerLayerInputsWithProjectionGraphTest) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 4;

  std::vector<int32_t> tokens = {1, 2};
  int seq_len = 2;
  int batch_size = 1;

  std::vector<float> embedded_input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                            5.0f, 6.0f, 7.0f, 8.0f};

  XnnTensor embedded_input({.name = "embedded_input",
                            .type = Type::kFP32,
                            .shape = {batch_size, seq_len, config.embed_dim}});

  XnnTensor sliding_attention_mask({.name = "sliding_attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, 2, 2}});
  XnnTensor rope_local_cos(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor rope_local_sin(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  std::vector<float> proj_w_data = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
      0.1f, 0.0f, 0.1f, 0.0f, 0.2f, 0.1f, 0.2f, 0.1f,

      0.3f, 0.3f, 0.3f, 0.3f, 0.4f, 0.4f, 0.4f, 0.4f,
      0.5f, 0.5f, 0.5f, 0.5f, 0.6f, 0.6f, 0.6f, 0.6f};

  std::vector<float> norm_w_data = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float> emb_per_layer_data(80, 0.1f);
  for (int i = 0; i < 80; ++i) emb_per_layer_data[i] = i * 0.01f;

  absl::flat_hash_map<std::string, XnnTensor> weights =
      CreateGemma4GraphTestWeights(2);
  weights.insert({"model.layers.0.per_layer_model_projection.weight",
                  XnnTensor({.name = "per_layer_model_projection_0",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = proj_w_data})});
  weights.insert({"model.layers.1.per_layer_model_projection.weight",
                  XnnTensor({.name = "per_layer_model_projection_1",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = proj_w_data})});
  weights.insert({"model.per_layer_projection_norm.weight",
                  XnnTensor({.name = "per_layer_projection_norm",
                             .type = Type::kFP32,
                             .shape = {4},
                             .buffer = norm_w_data})});

  for (int l = 0; l < 2; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    weights.insert({absl::StrCat(prefix, ".per_layer_input_gate.weight"),
                    XnnTensor({.name = "per_layer_input_gate",
                               .type = Type::kFP32,
                               .shape = {4, 4},
                               .buffer = std::vector<float>(4 * 4, 0.0f)})});
    weights.insert({absl::StrCat(prefix, ".per_layer_projection.weight"),
                    XnnTensor({.name = "per_layer_projection",
                               .type = Type::kFP32,
                               .shape = {4, 4},
                               .buffer = std::vector<float>(4 * 4, 0.0f)})});
    weights.insert({absl::StrCat(prefix, ".post_per_layer_input_norm.weight"),
                    XnnTensor({.name = "post_per_layer_input_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = std::vector<float>(4, 0.0f)})});
  }

  std::vector<std::vector<float>> layer_emb_data(
      2, std::vector<float>(2 * 4, 0.1f));

  Gemma4Inputs<XnnpackMixinTag> inputs;
  inputs.embedded_input = embedded_input;
  for (int l = 0; l < 2; ++l) {
    inputs.per_layer_token_embeddings.push_back(
        XnnTensor({.name = absl::StrCat("per_layer_token_embedding_", l),
                   .type = Type::kFP32,
                   .shape = {1, 2, 4}}));
  }
  inputs.sliding_attention_mask = sliding_attention_mask;
  inputs.rope_local_cos = rope_local_cos;
  inputs.rope_local_sin = rope_local_sin;
  inputs.weights = weights;

  Gemma4Outputs<XnnpackMixinTag> model_outputs =
      BuildGemma4Graph(inputs, config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));
  ASSERT_THAT(runner.SetInput(embedded_input, embedded_input_data), IsOk());
  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, mask_data),
              IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_sin, sin_data), IsOk());

  for (int l = 0; l < 2; ++l) {
    ASSERT_THAT(runner.SetInput(inputs.per_layer_token_embeddings[l],
                                layer_emb_data[l]),
                IsOk());
  }
  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.071181f, 0.444413f, 0.817507f, 1.190348f, 1.562821f,
      1.934811f, 2.306205f, 2.676890f, 3.046752f, 3.415681f,
      0.066562f, 0.460761f, 0.854802f, 1.248548f, 1.641863f,
      2.034613f, 2.426663f, 2.817881f, 3.208134f, 3.597293f};

  EXPECT_THAT(runner.ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-2f), expected_data)));
}

// Compare a causal four-token prefill against a two-token prefill followed by
// two invocations of one decode runtime. The second decode grows already-bound
// external buffers, exercising the cache reshape/rebind contract used by main.
TEST(Gemma4GraphTest, PrefillMatchesMultiStepCachedDecode) {
  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.frac_shared_layers = 0;
  config.final_logit_softcap = 10.0f;
  const auto weights = CreateGemma4GraphTestWeights(config.num_layers);
  auto make_inputs = [&](int seq_len, int cache_len) {
    Gemma4Inputs<XnnpackMixinTag> inputs;
    inputs.embedded_input.Set({.type = Type::kFP32, .shape = {1, seq_len, 4}});
    inputs.sliding_attention_mask.Set(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, cache_len + seq_len}});
    inputs.rope_local_cos.Set(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, 4}});
    inputs.rope_local_sin.Set(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, 4}});
    if (cache_len > 0) {
      for (int layer = 0; layer < config.num_layers; ++layer) {
        inputs.key_caches.emplace_back(
            TensorInit{.type = Type::kFP32, .shape = {1, 1, cache_len, 4}});
        inputs.value_caches.emplace_back(
            TensorInit{.type = Type::kFP32, .shape = {1, 1, cache_len, 4}});
      }
    }
    inputs.weights = weights;
    return inputs;
  };

  const std::array<float, 16> embeddings = {1, 2,  3, 4,    5,  6, 7,    8,
                                            2, -1, 3, 0.5f, -2, 1, 0.5f, 4};
  std::array<float, 16> cos, sin;
  RopeCosSin(/*start=*/0, /*seq_len=*/4, /*head_dim=*/4,
             /*rope_base=*/10000.0f, /*rope_proportion=*/1.0f,
             absl::MakeSpan(cos), absl::MakeSpan(sin));
  const std::array<float, 16> full_mask = {
      0, -1e9f, -1e9f, -1e9f, 0, 0, -1e9f, -1e9f, 0, 0, 0, -1e9f, 0, 0, 0, 0};
  // Generated independently by
  // helpers/reference/gemma4_graph.py's multi-step decode reference case.
  const std::array<float, 40> expected = {
      0.07118023f, 0.44415282f, 0.81589137f, 1.18537364f, 1.55160218f,
      1.91361499f, 2.27049538f, 2.62138022f, 2.96546808f, 3.30202547f,
      0.06653243f, 0.46050390f, 0.85304801f, 1.24296166f, 1.62907398f,
      2.01026017f, 2.38545424f, 2.75365961f, 3.11395924f, 3.46552305f,
      0.04697882f, 0.32250549f, 0.59754287f, 0.87167595f, 1.14449521f,
      1.41559899f, 1.68459597f, 1.95110697f, 2.21476733f, 2.47522896f,
      0.07168668f, 0.35620077f, 0.64013865f, 0.92304388f, 1.20446668f,
      1.48396671f, 1.76111605f, 2.03550113f, 2.30672551f, 2.57441229f};

  auto full_inputs = make_inputs(/*seq_len=*/4, /*cache_len=*/0);
  auto full_outputs = BuildGemma4Graph(full_inputs, config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto full_runner, XnnpackRunner::Create(full_outputs.GetAllHandles()));
  ASSERT_THAT(full_runner.SetInput(full_inputs.embedded_input, embeddings),
              IsOk());
  ASSERT_THAT(
      full_runner.SetInput(full_inputs.sliding_attention_mask, full_mask),
      IsOk());
  ASSERT_THAT(full_runner.SetInput(full_inputs.rope_local_cos, cos), IsOk());
  ASSERT_THAT(full_runner.SetInput(full_inputs.rope_local_sin, sin), IsOk());
  ASSERT_THAT(full_runner.Run(), IsOk());
  EXPECT_THAT(full_runner.ReadOutputAs<float>(full_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected)));

  auto prefill_inputs = make_inputs(/*seq_len=*/2, /*cache_len=*/0);
  auto prefill_outputs = BuildGemma4Graph(prefill_inputs, config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto prefill_runner,
      XnnpackRunner::Create(prefill_outputs.GetAllHandles()));
  const std::array<float, 4> prefill_mask = {0, -1e9f, 0, 0};
  const auto prefill_embeddings = absl::MakeConstSpan(embeddings).first(8);
  const auto prefill_cos = absl::MakeConstSpan(cos).first(8);
  const auto prefill_sin = absl::MakeConstSpan(sin).first(8);
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.embedded_input,
                                      prefill_embeddings),
              IsOk());
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.sliding_attention_mask,
                                      prefill_mask),
              IsOk());
  ASSERT_THAT(
      prefill_runner.SetInput(prefill_inputs.rope_local_cos, prefill_cos),
      IsOk());
  ASSERT_THAT(
      prefill_runner.SetInput(prefill_inputs.rope_local_sin, prefill_sin),
      IsOk());
  ASSERT_THAT(prefill_runner.Run(), IsOk());
  EXPECT_THAT(prefill_runner.ReadOutputAs<float>(prefill_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                                     absl::MakeConstSpan(expected).first(20))));

  std::vector<std::vector<float>> keys(config.num_layers);
  std::vector<std::vector<float>> values(config.num_layers);
  for (int layer = 0; layer < config.num_layers; ++layer) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto key,
        prefill_runner.ReadOutputAs<float>(prefill_outputs.key_caches[layer]));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto value,
                                    prefill_runner.ReadOutputAs<float>(
                                        prefill_outputs.value_caches[layer]));
    keys[layer].assign(key.begin(), key.end());
    values[layer].assign(value.begin(), value.end());
  }

  auto decode_inputs = make_inputs(/*seq_len=*/1, /*cache_len=*/2);
  auto decode_outputs = BuildGemma4Graph(decode_inputs, config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto decode_runner,
      XnnpackRunner::Create(decode_outputs.GetAllHandles()));
  const std::array<float, 4> decode_mask = {0, 0, 0, 0};
  for (int position = 2; position < 4; ++position) {
    SCOPED_TRACE(position);
    const auto mask_view = absl::MakeConstSpan(decode_mask).first(position + 1);
    const auto embedding_view =
        absl::MakeConstSpan(embeddings).subspan(position * 4, 4);
    const auto cos_view = absl::MakeConstSpan(cos).subspan(position * 4, 4);
    const auto sin_view = absl::MakeConstSpan(sin).subspan(position * 4, 4);
    ASSERT_THAT(decode_runner.ReshapeInput(decode_inputs.sliding_attention_mask,
                                           {1, 1, 1, position + 1}),
                IsOk());
    ASSERT_THAT(
        decode_runner.SetInput(decode_inputs.sliding_attention_mask, mask_view),
        IsOk());
    ASSERT_THAT(
        decode_runner.SetInput(decode_inputs.embedded_input, embedding_view),
        IsOk());
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.rope_local_cos, cos_view),
                IsOk());
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.rope_local_sin, sin_view),
                IsOk());
    for (int layer = 0; layer < config.num_layers; ++layer) {
      ASSERT_THAT(decode_runner.ReshapeInput(decode_inputs.key_caches[layer],
                                             {1, 1, position, 4}),
                  IsOk());
      ASSERT_THAT(decode_runner.ReshapeInput(decode_inputs.value_caches[layer],
                                             {1, 1, position, 4}),
                  IsOk());
      ASSERT_THAT(
          decode_runner.SetInput(decode_inputs.key_caches[layer], keys[layer]),
          IsOk());
      ASSERT_THAT(decode_runner.SetInput(decode_inputs.value_caches[layer],
                                         values[layer]),
                  IsOk());
    }
    ASSERT_THAT(decode_runner.Run(), IsOk());
    EXPECT_THAT(decode_runner.ReadOutputAs<float>(decode_outputs.logits),
                IsOkAndHolds(Pointwise(
                    FloatNear(1e-4f),
                    absl::MakeConstSpan(expected).subspan(position * 10, 10))));
    for (int layer = 0; layer < config.num_layers; ++layer) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          auto key,
          decode_runner.ReadOutputAs<float>(decode_outputs.key_caches[layer]));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto value,
                                      decode_runner.ReadOutputAs<float>(
                                          decode_outputs.value_caches[layer]));
      ASSERT_EQ(key.size(), 4);
      ASSERT_EQ(value.size(), 4);
      keys[layer].insert(keys[layer].end(), key.begin(), key.end());
      values[layer].insert(values[layer].end(), value.begin(), value.end());
    }
  }
  for (int layer = 0; layer < config.num_layers; ++layer) {
    EXPECT_THAT(full_runner.ReadOutputAs<float>(full_outputs.key_caches[layer]),
                IsOkAndHolds(Pointwise(FloatNear(1e-4f), keys[layer])));
    EXPECT_THAT(
        full_runner.ReadOutputAs<float>(full_outputs.value_caches[layer]),
        IsOkAndHolds(Pointwise(FloatNear(1e-4f), values[layer])));
  }
}

TEST(Gemma4GraphTest, NativeWeightCacheTest) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_weights_cache_t raw_cache = nullptr;
  ASSERT_EQ(xnn_create_weights_cache(&raw_cache), xnn_status_success);
  // Runners borrow the cache, so it must outlive both runtime instances.
  std::unique_ptr<xnn_weights_cache_provider,
                  decltype(&xnn_delete_weights_cache)>
      cache(raw_cache, &xnn_delete_weights_cache);

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  XnnTensor embedded_input(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor sliding_attention_mask({.name = "sliding_attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, 2, 2}});
  XnnTensor rope_local_cos(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor rope_local_sin(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  auto weights = CreateGemma4GraphTestWeights(2);

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};

  std::vector<float> run1_logits;

  // Run 1 packs the shared constant weights into the native in-memory cache.
  {
    Gemma4Inputs<XnnpackMixinTag> inputs;
    inputs.embedded_input = embedded_input;
    inputs.sliding_attention_mask = sliding_attention_mask;
    inputs.rope_local_cos = rope_local_cos;
    inputs.rope_local_sin = rope_local_sin;
    inputs.weights = weights;

    Gemma4Outputs<XnnpackMixinTag> model_outputs =
        BuildGemma4Graph(inputs, config);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));
    runner.SetWeightsCache(cache.get());
    ASSERT_THAT(runner.PrepareRuntime(), IsOk());

    // Soft finalization permits a later runtime to reuse these same weights.
    ASSERT_EQ(xnn_finalize_weights_cache(
                  cache.get(), xnn_weights_cache_finalization_kind_soft),
              xnn_status_success);

    ASSERT_THAT(runner.SetInput(embedded_input, input_data), IsOk());
    ASSERT_THAT(runner.SetInput(sliding_attention_mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_cos, cos_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_sin, sin_data), IsOk());

    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                    runner.ReadOutput(model_outputs.logits));
    LockedBufferSpan<const float> floats = std::move(result).As<const float>();
    ASSERT_EQ(floats.size(), 20);

    // Expected values calculated using helpers/reference/gemma4_graph.py, these
    // results are the same as for `ModelTest`.
    const std::array<float, 20> expected_data = {
        0.071322f, 0.443513f, 0.814477f, 1.183199f, 1.548685f,
        1.909981f, 2.266176f, 2.616409f, 2.959886f, 3.295875f,
        0.066592f, 0.460456f, 0.852894f, 1.242703f, 1.628713f,
        2.009801f, 2.384902f, 2.753019f, 3.113236f, 3.464724f};

    EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), expected_data));
    run1_logits.assign(floats.begin(), floats.end());
  }

  // Run 2 creates a fresh graph/runtime and reuses the finalized cache.
  {
    EXPECT_TRUE(xnn_weights_cache_is_finalized(cache.get()));

    Gemma4Inputs<XnnpackMixinTag> inputs;
    inputs.embedded_input = embedded_input;
    inputs.sliding_attention_mask = sliding_attention_mask;
    inputs.rope_local_cos = rope_local_cos;
    inputs.rope_local_sin = rope_local_sin;
    inputs.weights = weights;

    Gemma4Outputs<XnnpackMixinTag> model_outputs =
        BuildGemma4Graph(inputs, config);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));
    runner.SetWeightsCache(cache.get());

    ASSERT_THAT(runner.SetInput(embedded_input, input_data), IsOk());
    ASSERT_THAT(runner.SetInput(sliding_attention_mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_cos, cos_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_sin, sin_data), IsOk());

    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                    runner.ReadOutput(model_outputs.logits));
    LockedBufferSpan<const float> floats = std::move(result).As<const float>();
    ASSERT_EQ(floats.size(), 20);

    EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), run1_logits));
  }
}

#ifndef LITERT_TENSOR_STANDALONE
void MapWeightIdentifiers(
    tflite::xnnpack::MMapWeightCacheProvider& cache_provider,
    const absl::flat_hash_map<std::string, XnnTensor>& weights) {
  for (const auto& [name, tensor] : weights) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
    LockedBufferSpan<const std::byte> locked = buffer.Lock();
    uint64_t identifier = static_cast<uint64_t>(std::hash<std::string>{}(name));
    ASSERT_TRUE(cache_provider.MapBufferIdentifier(locked.data(), locked.size(),
                                                   identifier));
  }
}

TEST(Gemma4GraphTest, WeightCacheTest) {
  std::string cache_path =
      absl::StrCat(testing::TempDir(), "/gemma4_weight_cache_test.cache");
  remove(cache_path.c_str());

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  XnnTensor embedded_input(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor sliding_attention_mask({.name = "sliding_attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, 2, 2}});
  XnnTensor rope_local_cos(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor rope_local_sin(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  auto weights = CreateGemma4GraphTestWeights(2);

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};

  std::vector<float> run1_logits;

  // Run 1: Build Cache
  {
    tflite::xnnpack::MMapWeightCacheProvider cache_provider;
    ASSERT_TRUE(cache_provider.LoadOrStartBuild(cache_path.c_str()));

    MapWeightIdentifiers(cache_provider, weights);
    ASSERT_TRUE(cache_provider.StartBuildStep());

    Gemma4Inputs<XnnpackMixinTag> inputs;
    inputs.embedded_input = embedded_input;
    inputs.sliding_attention_mask = sliding_attention_mask;
    inputs.rope_local_cos = rope_local_cos;
    inputs.rope_local_sin = rope_local_sin;
    inputs.weights = weights;

    Gemma4Outputs<XnnpackMixinTag> model_outputs =
        BuildGemma4Graph(inputs, config);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));
    runner.SetWeightsCache(&cache_provider.GetCacheProvider());
    ASSERT_THAT(runner.PrepareRuntime(), IsOk());

    ASSERT_TRUE(cache_provider.StopBuildStep());
    cache_provider.StopBuild();

    ASSERT_THAT(runner.SetInput(embedded_input, input_data), IsOk());
    ASSERT_THAT(runner.SetInput(sliding_attention_mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_cos, cos_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_sin, sin_data), IsOk());

    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                    runner.ReadOutput(model_outputs.logits));
    LockedBufferSpan<const float> floats = std::move(result).As<const float>();
    ASSERT_EQ(floats.size(), 20);

    // Expected values calculated using helpers/reference/gemma4_graph.py, these
    // results are the same as for `ModelTest`.
    const std::array<float, 20> expected_data = {
        0.071322f, 0.443513f, 0.814477f, 1.183199f, 1.548685f,
        1.909981f, 2.266176f, 2.616409f, 2.959886f, 3.295875f,
        0.066592f, 0.460456f, 0.852894f, 1.242703f, 1.628713f,
        2.009801f, 2.384902f, 2.753019f, 3.113236f, 3.464724f};

    EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), expected_data));
    run1_logits.assign(floats.begin(), floats.end());
  }

  // Run 2: Load Cache from Disk
  {
    tflite::xnnpack::MMapWeightCacheProvider cache_provider;
    ASSERT_TRUE(cache_provider.LoadOrStartBuild(cache_path.c_str()));
    EXPECT_FALSE(cache_provider.CanStartBuildStep());

    MapWeightIdentifiers(cache_provider, weights);

    Gemma4Inputs<XnnpackMixinTag> inputs;
    inputs.embedded_input = embedded_input;
    inputs.sliding_attention_mask = sliding_attention_mask;
    inputs.rope_local_cos = rope_local_cos;
    inputs.rope_local_sin = rope_local_sin;
    inputs.weights = weights;

    Gemma4Outputs<XnnpackMixinTag> model_outputs =
        BuildGemma4Graph(inputs, config);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        XnnpackRunner runner, XnnpackRunner::Create({model_outputs.logits}));
    runner.SetWeightsCache(&cache_provider.GetCacheProvider());

    ASSERT_THAT(runner.SetInput(embedded_input, input_data), IsOk());
    ASSERT_THAT(runner.SetInput(sliding_attention_mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_cos, cos_data), IsOk());
    ASSERT_THAT(runner.SetInput(rope_local_sin, sin_data), IsOk());

    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                    runner.ReadOutput(model_outputs.logits));
    LockedBufferSpan<const float> floats = std::move(result).As<const float>();
    ASSERT_EQ(floats.size(), 20);

    EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), run1_logits));
  }

  remove(cache_path.c_str());
}

#endif  // LITERT_TENSOR_STANDALONE

}  // namespace
}  // namespace litert::tensor::examples::gemma4
