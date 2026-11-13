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

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/gemma4/gemma4_weights.h"
#include "tensor/examples/gemma4/test_backends.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/examples/utils/tensor_mapping.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "tflite/delegates/xnnpack/weight_cache.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::Pointwise;

absl::flat_hash_map<std::string, TensorHandle> CreateGemma4GraphTestWeights(
    int num_layers = 2) {
  absl::flat_hash_map<std::string, TensorHandle> weights;

  for (int l = 0; l < num_layers; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    weights.emplace(
        absl::StrCat(prefix, ".self_attn.q_proj.weight"),
        TensorInit{.name = "q_proj",
                   .type = Type::kFP32,
                   .shape = {8, 4},
                   .buffer = std::vector<float>{
                       1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                       0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}});

    weights.emplace(
        absl::StrCat(prefix, ".self_attn.k_proj.weight"),
        TensorInit{.name = "k_proj",
                   .type = Type::kFP32,
                   .shape = {4, 4},
                   .buffer = std::vector<float>{
                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                       0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}});

    weights.emplace(
        absl::StrCat(prefix, ".self_attn.v_proj.weight"),
        TensorInit{.name = "v_proj",
                   .type = Type::kFP32,
                   .shape = {4, 4},
                   .buffer = std::vector<float>{
                       0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f}});

    weights.emplace(
        absl::StrCat(prefix, ".self_attn.o_proj.weight"),
        TensorInit{.name = "o_proj",
                   .type = Type::kFP32,
                   .shape = {4, 8},
                   .buffer = std::vector<float>{
                       1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                       0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}});

    weights.emplace(
        absl::StrCat(prefix, ".self_attn.q_norm.weight"),
        TensorInit{
            .name = "q_norm", .type = Type::kFP32, .shape = {4}, .buffer = 1});

    weights.emplace(
        absl::StrCat(prefix, ".self_attn.k_norm.weight"),
        TensorInit{
            .name = "k_norm", .type = Type::kFP32, .shape = {4}, .buffer = 1});

    weights.emplace(absl::StrCat(prefix, ".input_layernorm.weight"),
                    TensorInit{.name = "pre_attn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1});
    weights.emplace(absl::StrCat(prefix, ".post_attention_layernorm.weight"),
                    TensorInit{.name = "post_attn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1});
    weights.emplace(absl::StrCat(prefix, ".pre_feedforward_layernorm.weight"),
                    TensorInit{.name = "pre_ffn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1});
    weights.emplace(absl::StrCat(prefix, ".post_feedforward_layernorm.weight"),
                    TensorInit{.name = "post_ffn_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = 1});

    weights.emplace(
        absl::StrCat(prefix, ".mlp.gate_proj.weight"),
        TensorInit{.name = "gate_proj",
                   .type = Type::kFP32,
                   .shape = {6, 4},
                   .buffer = std::vector<float>{
                       1.0f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                       0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                       1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f}});

    weights.emplace(
        absl::StrCat(prefix, ".mlp.up_proj.weight"),
        TensorInit{.name = "up_proj",
                   .type = Type::kFP32,
                   .shape = {6, 4},
                   .buffer = std::vector<float>{
                       0.5f, 0.5f, 0.0f,  0.0f, 0.0f, 0.5f, 0.5f, 0.0f,
                       0.0f, 0.0f, 0.5f,  0.5f, 0.5f, 0.0f, 0.0f, 0.5f,
                       1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f}});

    weights.emplace(
        absl::StrCat(prefix, ".mlp.down_proj.weight"),
        TensorInit{.name = "down_proj",
                   .type = Type::kFP32,
                   .shape = {4, 6},
                   .buffer = std::vector<float>{
                       1.0f,  0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f,
                       0.0f,  0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
                       -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -0.5f}});

    weights.emplace(absl::StrCat(prefix, ".layer_scalar"),
                    TensorInit{.name = "layer_scalar",
                               .type = Type::kFP32,
                               .shape = {1},
                               .buffer = 0.5f});
  }

  // Model Final Norm
  weights.emplace("model.norm.weight", TensorInit{.name = "final_norm",
                                                  .type = Type::kFP32,
                                                  .shape = {4},
                                                  .buffer = std::vector<float>{
                                                      1.0f, 1.0f, 1.0f, 1.0f}});

  // Model Embeddings (Tied weights)
  std::vector<float> embed_table_data(40);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      embed_table_data[i * 4 + j] = i * 0.1f + j * 0.01f;
    }
  }
  weights.emplace("model.embed_tokens.weight",
                  TensorInit{.name = "embed_tokens",
                             .type = Type::kFP32,
                             .shape = {10, 4},
                             .buffer = embed_table_data});

  return weights;
}

template <class Backend>
class Gemma4GraphTest : public ::testing::Test {};
TYPED_TEST_SUITE(Gemma4GraphTest, TestBackends, TestBackendNames);

TYPED_TEST(Gemma4GraphTest, ModelTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(2))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, GlobalLayerGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.global_key_size = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.attention_pattern_size = 2;  // Layer 0 is local, Layer 1 is global

  Gemma4Inputs<Tag> inputs;
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
  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(2))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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
      0.071182f, 0.444328f, 0.816239f, 1.185890f, 1.552284f,
      1.914455f, 2.271488f, 2.622517f, 2.966740f, 3.303423f,
      0.066505f, 0.460559f, 0.853184f, 1.243178f, 1.629367f,
      2.010628f, 2.385894f, 2.754166f, 3.114528f, 3.466150f};

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, KVCacheGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 1;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 4}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.key_caches = {Tensor(
      {.name = "key_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}})};
  inputs.value_caches = {Tensor(
      {.name = "value_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}})};
  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(1))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner runner,
      Runner::Create({model_outputs.logits, model_outputs.key_caches[0],
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

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(
                  FloatNear(1e-4f),
                  {0.072394f, 0.441714f, 0.809831f, 1.175752f, 1.538507f,
                   1.897162f, 2.250824f, 2.598654f, 2.939869f, 3.273751f,
                   0.066682f, 0.460588f, 0.853068f, 1.242918f, 1.628967f,
                   2.010093f, 2.385229f, 2.753378f, 3.113626f, 3.465142f})));

  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.key_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.766109f, 0.097841f, 0.863950f, 1.630059f,
                              0.214422f, -0.909717f, 1.286534f, 1.212956f})));

  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.value_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.365148f, 0.730295f, 1.095443f, 1.460591f,
                              0.758097f, 0.909716f, 1.061335f, 1.212954f})));
}

TYPED_TEST(Gemma4GraphTest, SharedKVCacheGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 3;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 1.0f / 3.0f;  // Layer 2 shares Layer 1's KV.
  config.share_local = true;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(3))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  // Shared layer 2 does not produce its own updated KV cache handles, and
  // `GetAllHandles()` omits the invalid entries.
  EXPECT_EQ(model_outputs.key_caches[2], TensorHandle::Invalid());
  EXPECT_EQ(model_outputs.value_caches[2], TensorHandle::Invalid());
  EXPECT_THAT(
      model_outputs.GetAllHandles(),
      ElementsAre(model_outputs.logits, model_outputs.key_caches[0],
                  model_outputs.key_caches[1], model_outputs.value_caches[0],
                  model_outputs.value_caches[1]));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner runner,
      Runner::Create({model_outputs.logits, model_outputs.key_caches[1]}));

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

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(
                  FloatNear(1e-4f),
                  {0.069418f, 0.444620f, 0.818571f, 1.190232f, 1.558587f,
                   1.922658f, 2.281512f, 2.634272f, 2.980126f, 3.318328f,
                   0.066271f, 0.460083f, 0.852470f, 1.242231f, 1.628195f,
                   2.009240f, 2.384300f, 2.752381f, 3.112565f, 3.464023f})));

  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.key_caches[1]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.828487f, 0.029702f, 0.914964f, 1.573392f,
                              0.232546f, -0.913679f, 1.293728f, 1.198907f})));
}

TYPED_TEST(Gemma4GraphTest, GrowingKVCacheDecodeGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 3;
  config.vocab_size = 10;
  config.embed_dim = 8;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 1.0f / 3.0f;  // Layer 2 shares Layer 1's KV.
  config.share_local = true;

  auto make_values = [](size_t n, float scale, float offset) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
      out[i] = scale * static_cast<float>((i * 7 + 3) % 13) + offset;
    }
    return out;
  };

  absl::flat_hash_map<std::string, TensorHandle> weight_map;
  for (int l = 0; l < config.num_layers; ++l) {
    const std::string prefix = absl::StrCat("model.layers.", l);
    const float base = 0.02f * static_cast<float>(l + 1);
    weight_map.emplace(
        absl::StrCat(prefix, ".self_attn.q_proj.weight"),
        TensorInit{.type = Type::kFP32,
                   .shape = {16, 8},
                   .buffer = make_values(128, 0.03f, -0.15f + base)});
    if (l < 2) {
      weight_map.emplace(
          absl::StrCat(prefix, ".self_attn.k_proj.weight"),
          TensorInit{.type = Type::kFP32,
                     .shape = {8, 8},
                     .buffer = make_values(64, 0.03f, -0.12f + base)});
      weight_map.emplace(
          absl::StrCat(prefix, ".self_attn.v_proj.weight"),
          TensorInit{.type = Type::kFP32,
                     .shape = {8, 8},
                     .buffer = make_values(64, 0.025f, -0.10f + base)});
      weight_map.emplace(
          absl::StrCat(prefix, ".self_attn.k_norm.weight"),
          TensorInit{.type = Type::kFP32, .shape = {4}, .buffer = 1.0f});
    }
    weight_map.emplace(
        absl::StrCat(prefix, ".self_attn.o_proj.weight"),
        TensorInit{.type = Type::kFP32,
                   .shape = {8, 16},
                   .buffer = make_values(128, 0.02f, -0.10f + base)});
    weight_map.emplace(
        absl::StrCat(prefix, ".self_attn.q_norm.weight"),
        TensorInit{.type = Type::kFP32, .shape = {4}, .buffer = 1.0f});
    weight_map.emplace(
        absl::StrCat(prefix, ".input_layernorm.weight"),
        TensorInit{.type = Type::kFP32, .shape = {8}, .buffer = 1.0f});
    weight_map.emplace(
        absl::StrCat(prefix, ".post_attention_layernorm.weight"),
        TensorInit{.type = Type::kFP32, .shape = {8}, .buffer = 1.0f});
    weight_map.emplace(
        absl::StrCat(prefix, ".pre_feedforward_layernorm.weight"),
        TensorInit{.type = Type::kFP32, .shape = {8}, .buffer = 1.0f});
    weight_map.emplace(
        absl::StrCat(prefix, ".post_feedforward_layernorm.weight"),
        TensorInit{.type = Type::kFP32, .shape = {8}, .buffer = 1.0f});
    weight_map.emplace(
        absl::StrCat(prefix, ".mlp.gate_proj.weight"),
        TensorInit{.type = Type::kFP32,
                   .shape = {6, 8},
                   .buffer = make_values(48, 0.03f, -0.12f + base)});
    weight_map.emplace(
        absl::StrCat(prefix, ".mlp.up_proj.weight"),
        TensorInit{.type = Type::kFP32,
                   .shape = {6, 8},
                   .buffer = make_values(48, 0.025f, -0.10f + base)});
    weight_map.emplace(
        absl::StrCat(prefix, ".mlp.down_proj.weight"),
        TensorInit{.type = Type::kFP32,
                   .shape = {8, 6},
                   .buffer = make_values(48, 0.02f, -0.08f + base)});
    weight_map.emplace(
        absl::StrCat(prefix, ".layer_scalar"),
        TensorInit{.type = Type::kFP32, .shape = {1}, .buffer = 0.5f});
  }
  weight_map.emplace(
      "model.norm.weight",
      TensorInit{.type = Type::kFP32, .shape = {8}, .buffer = 1.0f});
  weight_map.emplace("model.embed_tokens.weight",
                     TensorInit{.type = Type::kFP32,
                                .shape = {10, 8},
                                .buffer = make_values(80, 0.04f, -0.2f)});

  const std::vector<float> emb_all = make_values(3 * 8, 0.1f, -0.3f);
  const std::array<float, 12> cos_all = {
      1.0f,       1.0f, 1.0f, 1.0f,       0.8660254f, 0.5f,
      0.8660254f, 0.5f, 0.5f, 0.8660254f, 0.5f,       0.8660254f};
  const std::array<float, 12> sin_all = {
      0.0f, 0.0f,       0.0f,       0.0f, 0.5f,       0.8660254f,
      0.5f, 0.8660254f, 0.8660254f, 0.5f, 0.8660254f, 0.5f};
  const std::array<float, 9> causal_mask = {0.0f,  -1e9f, -1e9f, 0.0f, 0.0f,
                                            -1e9f, 0.0f,  0.0f,  0.0f};

  // 1. Run 3-token causal prefill as reference.
  Gemma4Inputs<Tag> prefill_inputs;
  prefill_inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 3, 8}});
  prefill_inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                             .type = Type::kFP32,
                                             .shape = {1, 1, 3, 3}});
  prefill_inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 3, 4}});
  prefill_inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 3, 4}});
  LazyTensorMapping prefill_weights =
      LazyTensorMapping(weight_map).Register<Gemma4WeightHooks>(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma4Outputs<Tag> prefill_outputs,
      BuildGemma4Graph(prefill_inputs, prefill_weights, config));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner prefill_runner,
      Runner::Create({prefill_outputs.logits, prefill_outputs.key_caches[0],
                      prefill_outputs.value_caches[0],
                      prefill_outputs.key_caches[1],
                      prefill_outputs.value_caches[1]}));
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.embedded_input, emb_all),
              IsOk());
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.sliding_attention_mask,
                                      causal_mask),
              IsOk());
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.rope_local_cos, cos_all),
              IsOk());
  ASSERT_THAT(prefill_runner.SetInput(prefill_inputs.rope_local_sin, sin_all),
              IsOk());
  ASSERT_THAT(prefill_runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      LockedBufferSpan<const float> prefill_logits_span,
      prefill_runner.template ReadOutputAs<float>(prefill_outputs.logits));
  const std::vector<float> prefill_logits(prefill_logits_span.begin(),
                                          prefill_logits_span.end());

  // Extract token 0's K/V ([1, 2, 1, 4]) from the 3-token prefill caches
  // ([1, 2, 3, 4]) for unshared layers 0 and 1.
  std::array<std::vector<float>, 2> cache_k;
  std::array<std::vector<float>, 2> cache_v;
  for (int l = 0; l < 2; ++l) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> pk,
                                    prefill_runner.template ReadOutputAs<float>(
                                        prefill_outputs.key_caches[l]));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> pv,
                                    prefill_runner.template ReadOutputAs<float>(
                                        prefill_outputs.value_caches[l]));
    for (int h = 0; h < 2; ++h) {
      cache_k[l].insert(cache_k[l].end(), pk.begin() + h * 12,
                        pk.begin() + h * 12 + 4);
      cache_v[l].insert(cache_v[l].end(), pv.begin() + h * 12,
                        pv.begin() + h * 12 + 4);
    }
  }

  // 2. Build a single decode runner once (`cache_len = 1`) and reshape its KV
  // caches across two sequential decode steps (`cache_len` = 1, then 2).
  Gemma4Inputs<Tag> decode_inputs;
  decode_inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 1, 8}});
  decode_inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                            .type = Type::kFP32,
                                            .shape = {1, 1, 1, 2}});
  decode_inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 1, 4}});
  decode_inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 1, 4}});
  for (int l = 0; l < config.num_layers; ++l) {
    decode_inputs.key_caches.push_back(Tensor({.name = absl::StrCat("kc_", l),
                                               .type = Type::kFP32,
                                               .shape = {1, 2, 1, 4}}));
    decode_inputs.value_caches.push_back(Tensor({.name = absl::StrCat("vc_", l),
                                                 .type = Type::kFP32,
                                                 .shape = {1, 2, 1, 4}}));
  }
  LazyTensorMapping decode_weights =
      LazyTensorMapping(weight_map).Register<Gemma4WeightHooks>(config);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma4Outputs<Tag> decode_outputs,
      BuildGemma4Graph(decode_inputs, decode_weights, config));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner decode_runner,
      Runner::Create({decode_outputs.logits, decode_outputs.key_caches[0],
                      decode_outputs.value_caches[0],
                      decode_outputs.key_caches[1],
                      decode_outputs.value_caches[1]}));

  for (int step = 1; step <= 2; ++step) {
    const int32_t cache_len = step;
    const std::array<int32_t, 4> cache_shape = {1, 2, cache_len, 4};
    const std::array<int32_t, 4> mask_shape = {1, 1, 1, cache_len + 1};
    ASSERT_THAT(decode_runner.ReshapeInput(decode_inputs.sliding_attention_mask,
                                           mask_shape),
                IsOk());
    const std::vector<float> mask_zeros(cache_len + 1, 0.0f);
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.sliding_attention_mask,
                                       mask_zeros),
                IsOk());
    for (int l = 0; l < 2; ++l) {
      ASSERT_THAT(
          decode_runner.ReshapeInput(decode_inputs.key_caches[l], cache_shape),
          IsOk());
      ASSERT_THAT(decode_runner.ReshapeInput(decode_inputs.value_caches[l],
                                             cache_shape),
                  IsOk());
      ASSERT_THAT(
          decode_runner.SetInput(decode_inputs.key_caches[l], cache_k[l]),
          IsOk());
      ASSERT_THAT(
          decode_runner.SetInput(decode_inputs.value_caches[l], cache_v[l]),
          IsOk());
    }
    const std::vector<float> step_emb(emb_all.begin() + step * 8,
                                      emb_all.begin() + (step + 1) * 8);
    const std::vector<float> step_cos(cos_all.begin() + step * 4,
                                      cos_all.begin() + (step + 1) * 4);
    const std::vector<float> step_sin(sin_all.begin() + step * 4,
                                      sin_all.begin() + (step + 1) * 4);
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.embedded_input, step_emb),
                IsOk());
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.rope_local_cos, step_cos),
                IsOk());
    ASSERT_THAT(decode_runner.SetInput(decode_inputs.rope_local_sin, step_sin),
                IsOk());
    ASSERT_THAT(decode_runner.Run(), IsOk());

    const std::vector<float> expected_step_logits(
        prefill_logits.begin() + step * 10,
        prefill_logits.begin() + (step + 1) * 10);
    EXPECT_THAT(
        decode_runner.template ReadOutputAs<float>(decode_outputs.logits),
        IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_step_logits)));

    // Append new K/V per head (`[1, 2, cache_len, 4]` -> `[1, 2, cache_len+1,
    // 4]`) for the next decode step.
    for (int l = 0; l < 2; ++l) {
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          LockedBufferSpan<const float> nk,
          decode_runner.template ReadOutputAs<float>(
              decode_outputs.key_caches[l]));
      LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
          LockedBufferSpan<const float> nv,
          decode_runner.template ReadOutputAs<float>(
              decode_outputs.value_caches[l]));
      std::vector<float> next_k;
      std::vector<float> next_v;
      for (int h = 0; h < 2; ++h) {
        next_k.insert(next_k.end(), cache_k[l].begin() + h * cache_len * 4,
                      cache_k[l].begin() + (h + 1) * cache_len * 4);
        next_k.insert(next_k.end(), nk.begin() + h * 4,
                      nk.begin() + (h + 1) * 4);
        next_v.insert(next_v.end(), cache_v[l].begin() + h * cache_len * 4,
                      cache_v[l].begin() + (h + 1) * cache_len * 4);
        next_v.insert(next_v.end(), nv.begin() + h * 4,
                      nv.begin() + (h + 1) * 4);
      }
      cache_k[l] = std::move(next_k);
      cache_v[l] = std::move(next_v);
    }
  }
}

TYPED_TEST(Gemma4GraphTest, PerLayerInputsGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

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

  Gemma4Inputs<Tag> inputs;
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
      Tensor({.name = "per_layer_token_embedding_0",
              .type = Type::kFP32,
              .shape = {1, 2, 2}}),
      Tensor({.name = "per_layer_token_embedding_1",
              .type = Type::kFP32,
              .shape = {1, 2, 2}})};
  absl::flat_hash_map<std::string, TensorHandle> weight_map =
      CreateGemma4GraphTestWeights(2);

  weight_map.emplace(
      "model.layers.0.per_layer_model_projection.weight",
      TensorInit{.name = "per_layer_model_projection_0",
                 .type = Type::kFP32,
                 .shape = {2, 4},
                 .buffer = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.1f,
                                              0.2f, 0.3f, 0.4f}});
  weight_map.emplace(
      "model.layers.1.per_layer_model_projection.weight",
      TensorInit{.name = "per_layer_model_projection_1",
                 .type = Type::kFP32,
                 .shape = {2, 4},
                 .buffer = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.1f,
                                              0.2f, 0.3f, 0.4f}});
  weight_map.emplace("model.per_layer_projection_norm.weight",
                     TensorInit{.name = "per_layer_projection_norm",
                                .type = Type::kFP32,
                                .shape = {2},
                                .buffer = std::vector<float>{1.0f, 1.0f}});

  for (int l = 0; l < 2; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    weight_map.emplace(
        absl::StrCat(prefix, ".per_layer_input_gate.weight"),
        TensorInit{.name = "per_layer_input_gate",
                   .type = Type::kFP32,
                   .shape = {2, 4},
                   .buffer = std::vector<float>{0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                0.5f, 0.0f, 0.0f}});
    weight_map.emplace(
        absl::StrCat(prefix, ".per_layer_projection.weight"),
        TensorInit{.name = "per_layer_projection",
                   .type = Type::kFP32,
                   .shape = {4, 2},
                   .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.5f,
                                                0.0f, 0.0f, 0.5f}});
    weight_map.emplace(
        absl::StrCat(prefix, ".post_per_layer_input_norm.weight"),
        TensorInit{.name = "post_per_layer_norm",
                   .type = Type::kFP32,
                   .shape = {4},
                   .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}});
  }
  LazyTensorMapping weights =
      LazyTensorMapping(std::move(weight_map)).Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-2f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, NoLogitSoftcappingGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 0.0f;  // Soft capping disabled

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(2))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, PerLayerInputsWithProjectionGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

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

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set({.name = "embedded_input",
                             .type = Type::kFP32,
                             .shape = {batch_size, seq_len, config.embed_dim}});

  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  std::vector<float> proj_w_data = {
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
      0.1f, 0.0f, 0.1f, 0.0f, 0.2f, 0.1f, 0.2f, 0.1f,

      0.3f, 0.3f, 0.3f, 0.3f, 0.4f, 0.4f, 0.4f, 0.4f,
      0.5f, 0.5f, 0.5f, 0.5f, 0.6f, 0.6f, 0.6f, 0.6f};

  std::vector<float> norm_w_data = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float> emb_per_layer_data(80, 0.1f);
  for (int i = 0; i < 80; ++i) emb_per_layer_data[i] = i * 0.01f;

  absl::flat_hash_map<std::string, TensorHandle> weights =
      CreateGemma4GraphTestWeights(2);
  weights.emplace("model.layers.0.per_layer_model_projection.weight",
                  TensorInit{.name = "per_layer_model_projection_0",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = proj_w_data});
  weights.emplace("model.layers.1.per_layer_model_projection.weight",
                  TensorInit{.name = "per_layer_model_projection_1",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = proj_w_data});
  weights.emplace("model.per_layer_projection_norm.weight",
                  TensorInit{.name = "per_layer_projection_norm",
                             .type = Type::kFP32,
                             .shape = {4},
                             .buffer = norm_w_data});

  for (int l = 0; l < 2; ++l) {
    std::string prefix = absl::StrCat("model.layers.", l);
    weights.emplace(absl::StrCat(prefix, ".per_layer_input_gate.weight"),
                    TensorInit{.name = "per_layer_input_gate",
                               .type = Type::kFP32,
                               .shape = {4, 4},
                               .buffer = std::vector<float>(4 * 4, 0.0f)});
    weights.emplace(absl::StrCat(prefix, ".per_layer_projection.weight"),
                    TensorInit{.name = "per_layer_projection",
                               .type = Type::kFP32,
                               .shape = {4, 4},
                               .buffer = std::vector<float>(4 * 4, 0.0f)});
    weights.emplace(absl::StrCat(prefix, ".post_per_layer_input_norm.weight"),
                    TensorInit{.name = "post_per_layer_input_norm",
                               .type = Type::kFP32,
                               .shape = {4},
                               .buffer = std::vector<float>(4, 0.0f)});
  }

  for (int l = 0; l < 2; ++l) {
    inputs.per_layer_token_embeddings.emplace_back(
        TensorInit{.name = absl::StrCat("per_layer_token_embedding_", l),
                   .type = Type::kFP32,
                   .shape = {1, 2, 4}});
  }
  LazyTensorMapping weight_mapping =
      LazyTensorMapping(weights).Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Gemma4Outputs<Tag> model_outputs,
      BuildGemma4Graph(inputs, weight_mapping, config));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, embedded_input_data),
              IsOk());
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
    ASSERT_THAT(runner.SetInputAsCopy(inputs.per_layer_token_embeddings[l],
                                      std::vector<float>(2 * 4, 0.1f)),
                IsOk());
  }
  ASSERT_THAT(runner.Run(), IsOk());

  // Expected values calculated using helpers/reference/gemma4_graph.py:
  const std::array<float, 20> expected_data = {
      0.071181f, 0.444413f, 0.817507f, 1.190348f, 1.562821f,
      1.934811f, 2.306205f, 2.676890f, 3.046752f, 3.415681f,
      0.066562f, 0.460761f, 0.854802f, 1.248548f, 1.641863f,
      2.034613f, 2.426663f, 2.817881f, 3.208134f, 3.597293f};

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-2f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, EmptyKVCacheGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 0.0f;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  for (int l = 0; l < config.num_layers; ++l) {
    inputs.key_caches.push_back(Tensor({.name = absl::StrCat("key_cache_", l),
                                        .type = Type::kFP32,
                                        .shape = {1, 1, 0, 4}}));
    inputs.value_caches.push_back(
        Tensor({.name = absl::StrCat("value_cache_", l),
                .type = Type::kFP32,
                .shape = {1, 1, 0, 4}}));
  }

  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(2))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner runner,
      Runner::Create({model_outputs.logits, model_outputs.key_caches[0],
                      model_outputs.value_caches[0],
                      model_outputs.key_caches[1],
                      model_outputs.value_caches[1]}));

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
  const std::array<float, 20> expected_logits = {
      0.071180f, 0.444153f, 0.815891f, 1.185374f, 1.551602f,
      1.913615f, 2.270495f, 2.621380f, 2.965468f, 3.302026f,
      0.066561f, 0.460472f, 0.852956f, 1.242811f, 1.628866f,
      2.009996f, 2.385138f, 2.753294f, 3.113547f, 3.465068f};
  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_logits)));

  const Shape expected_cache_shape = {1, 1, 2, 4};
  EXPECT_EQ(model_outputs.key_caches[0].GetShape(), expected_cache_shape);
  EXPECT_EQ(model_outputs.value_caches[0].GetShape(), expected_cache_shape);
  EXPECT_EQ(model_outputs.key_caches[1].GetShape(), expected_cache_shape);
  EXPECT_EQ(model_outputs.value_caches[1].GetShape(), expected_cache_shape);

  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.key_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.766109f, 0.097841f, 0.863950f, 1.630059f,
                              0.214422f, -0.909717f, 1.286535f, 1.212956f})));
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.value_caches[0]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.365148f, 0.730295f, 1.095443f, 1.460591f,
                              0.758097f, 0.909716f, 1.061335f, 1.212954f})));
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.key_caches[1]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.828487f, 0.029702f, 0.914964f, 1.573392f,
                              0.232546f, -0.913679f, 1.293728f, 1.198907f})));
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(model_outputs.value_caches[1]),
      IsOkAndHolds(Pointwise(FloatNear(1e-4f),
                             {0.378138f, 0.760972f, 1.174971f, 1.377447f,
                              0.750368f, 0.913678f, 1.079237f, 1.198905f})));
}

TYPED_TEST(Gemma4GraphTest, GlobalLayerDistinctKeySizeGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 2;
  config.global_key_size = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 0.0f;
  config.attention_pattern_size = 2;  // Layer 0 is local, Layer 1 is global.

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.global_attention_mask.Set({.name = "global_attention_mask",
                                    .type = Type::kFP32,
                                    .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  inputs.rope_global_cos.Set(
      {.name = "rope_global_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_global_sin.Set(
      {.name = "rope_global_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, TensorHandle> weight_map =
      CreateGemma4GraphTestWeights(2);
  weight_map.insert_or_assign(
      "model.layers.0.self_attn.q_proj.weight",
      TensorHandle({.name = "q_proj_0",
                    .type = Type::kFP32,
                    .shape = {4, 4},
                    .buffer = std::vector<float>{
                        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}}));
  weight_map.insert_or_assign(
      "model.layers.0.self_attn.k_proj.weight",
      TensorHandle({.name = "k_proj_0",
                    .type = Type::kFP32,
                    .shape = {2, 4},
                    .buffer = std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                                 0.0f, 0.0f, 1.0f}}));
  weight_map.insert_or_assign(
      "model.layers.0.self_attn.v_proj.weight",
      TensorHandle({.name = "v_proj_0",
                    .type = Type::kFP32,
                    .shape = {2, 4},
                    .buffer = std::vector<float>{0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                 0.5f, 0.0f, 0.0f}}));
  weight_map.insert_or_assign(
      "model.layers.0.self_attn.o_proj.weight",
      TensorHandle({.name = "o_proj_0",
                    .type = Type::kFP32,
                    .shape = {4, 4},
                    .buffer = std::vector<float>{
                        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}}));
  weight_map.insert_or_assign("model.layers.0.self_attn.q_norm.weight",
                              TensorHandle({.name = "q_norm_0",
                                            .type = Type::kFP32,
                                            .shape = {2},
                                            .buffer = 1}));
  weight_map.insert_or_assign("model.layers.0.self_attn.k_norm.weight",
                              TensorHandle({.name = "k_norm_0",
                                            .type = Type::kFP32,
                                            .shape = {2},
                                            .buffer = 1}));

  LazyTensorMapping weights =
      LazyTensorMapping(std::move(weight_map)).Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(inputs.embedded_input, input_data), IsOk());

  const std::array<float, 4> smask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.sliding_attention_mask, smask_data),
              IsOk());

  const std::array<float, 4> gmask_data = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(inputs.global_attention_mask, gmask_data),
              IsOk());

  const std::array<float, 4> cos_local_data = {0.8660254f, 0.8660254f,
                                               0.7071068f, 0.7071068f};
  ASSERT_THAT(runner.SetInput(inputs.rope_local_cos, cos_local_data), IsOk());

  const std::array<float, 4> sin_local_data = {0.5f, 0.5f, 0.7071068f,
                                               0.7071068f};
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
      0.071526f, 0.446340f, 0.819901f, 1.191172f, 1.559142f,
      1.922835f, 2.281321f, 2.633727f, 2.979241f, 3.317121f,
      0.066537f, 0.460666f, 0.853366f, 1.243432f, 1.629693f,
      2.011021f, 2.386351f, 2.754684f, 3.115103f, 3.466776f};

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, DisabledPostNormsGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = false;
  config.use_post_ffw_norm = false;
  config.final_logit_softcap = 10.0f;
  config.frac_shared_layers = 0.0f;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, TensorHandle> weight_map =
      CreateGemma4GraphTestWeights(2);
  for (int l = 0; l < config.num_layers; ++l) {
    const std::string prefix = absl::StrCat("model.layers.", l);
    weight_map.erase(absl::StrCat(prefix, ".post_attention_layernorm.weight"));
    weight_map.erase(
        absl::StrCat(prefix, ".post_feedforward_layernorm.weight"));
  }
  LazyTensorMapping weights =
      LazyTensorMapping(std::move(weight_map)).Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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
      0.069731f, 0.448385f, 0.825755f, 1.200771f, 1.572392f,
      1.939613f, 2.301477f, 2.657086f, 3.005609f, 3.346286f,
      0.065993f, 0.460743f, 0.854058f, 1.244730f, 1.631580f,
      2.013478f, 2.389350f, 2.758197f, 3.119095f, 3.471212f};

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

TYPED_TEST(Gemma4GraphTest, AttnLogitsSoftCapGraphTest) {
  using Tag = typename TypeParam::Tag;
  using Runner = typename TypeParam::Runner;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;
  config.attn_logits_soft_cap = 1.0f;
  config.frac_shared_layers = 0.0f;

  Gemma4Inputs<Tag> inputs;
  inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                     .type = Type::kFP32,
                                     .shape = {1, 1, 2, 2}});
  inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  LazyTensorMapping weights = LazyTensorMapping(CreateGemma4GraphTestWeights(2))
                                  .Register<Gemma4WeightHooks>();

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Gemma4Outputs<Tag> model_outputs,
                                  BuildGemma4Graph(inputs, weights, config));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({model_outputs.logits}));

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
      0.071180f, 0.444153f, 0.815891f, 1.185374f, 1.551602f,
      1.913615f, 2.270495f, 2.621380f, 2.965468f, 3.302026f,
      0.066696f, 0.460179f, 0.852238f, 1.241676f, 1.627325f,
      2.008066f, 2.382836f, 2.750641f, 3.110567f, 3.461786f};

  EXPECT_THAT(runner.template ReadOutputAs<float>(model_outputs.logits),
              IsOkAndHolds(Pointwise(FloatNear(1e-4f), expected_data)));
}

void MapWeightIdentifiers(
    tflite::xnnpack::MMapWeightCacheProvider& cache_provider,
    const absl::flat_hash_map<std::string, TensorHandle>& weights) {
  for (const auto& [name, tensor] : weights) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Buffer & buffer, tensor.GetBuffer());
    LockedBufferSpan<const std::byte> locked = buffer.Lock();
    uint64_t identifier = static_cast<uint64_t>(std::hash<std::string>{}(name));
    ASSERT_TRUE(cache_provider.MapBufferIdentifier(locked.data(), locked.size(),
                                                   identifier));
  }
}

// The weight cache is an XNNPACK-only feature, so this test is not
// parametrized over the backends.
TEST(Gemma4GraphWeightCacheTest, WeightCacheTest) {
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
  config.per_layer_input_dim = 0;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.final_logit_softcap = 10.0f;

  Tensor embedded_input(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  Tensor sliding_attention_mask({.name = "sliding_attention_mask",
                                 .type = Type::kFP32,
                                 .shape = {1, 1, 2, 2}});
  Tensor rope_local_cos(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  Tensor rope_local_sin(
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
    LazyTensorMapping weight_mapping =
        LazyTensorMapping(weights).Register<Gemma4WeightHooks>();

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        Gemma4Outputs<XnnpackMixinTag> model_outputs,
        BuildGemma4Graph(inputs, weight_mapping, config));

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
    LazyTensorMapping weight_mapping =
        LazyTensorMapping(weights).Register<Gemma4WeightHooks>();

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        Gemma4Outputs<XnnpackMixinTag> model_outputs,
        BuildGemma4Graph(inputs, weight_mapping, config));

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

// E2B: 35 layers, pattern 5, 20 shared layers -> 0..14 unshared;
// shared local layers reuse 13; shared global layers reuse 14.
TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsE2B) {
  const std::vector<int> e2b = GetKvCacheSharingPatterns(Config::E2B());
  ASSERT_EQ(e2b.size(), 35u);
  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(e2b[i], i) << "layer " << i;
  }
  for (int i = 15; i < 35; ++i) {
    const int expected = ((i + 1) % 5 == 0) ? 14 : 13;
    EXPECT_EQ(e2b[i], expected) << "layer " << i;
  }
}

// E4B: 42 layers, pattern 6, 18 shared layers -> 0..23 unshared;
// shared local layers reuse 22; shared global layers reuse 23.
TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsE4B) {
  const std::vector<int> e4b = GetKvCacheSharingPatterns(Config::E4B());
  ASSERT_EQ(e4b.size(), 42u);
  for (int i = 0; i < 24; ++i) {
    EXPECT_EQ(e4b[i], i) << "layer " << i;
  }
  for (int i = 24; i < 42; ++i) {
    const int expected = ((i + 1) % 6 == 0) ? 23 : 22;
    EXPECT_EQ(e4b[i], expected) << "layer " << i;
  }
}

// Float32 rounding: `35 - (18.0f / 35.0f) * 35` evaluates to `16.999998f` in
// float32; rounding `frac_shared_layers * num_layers` must yield 17 unshared
// layers (0..16), not 16.
TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsFloat32Rounding) {
  Config config = Config::E2B();
  config.num_layers = 35;
  config.frac_shared_layers = 18.0f / 35.0f;
  const std::vector<int> patterns = GetKvCacheSharingPatterns(config);
  ASSERT_EQ(patterns.size(), 35u);
  EXPECT_EQ(patterns[16], 16);
  EXPECT_EQ(patterns[17], 16);  // Layer 17 (local) shares Layer 16 (last L).
  EXPECT_EQ(patterns[19], 14);  // Layer 19 (global) shares Layer 14 (last G).
}

// Non-multiple `num_unshared_layers`: 8 layers, pattern 4 (`[L, L, L, G, L,
// L, L, G]`), 3 shared layers -> unshared `[0:L, 1:L, 2:L, 3:G, 4:L]`.
// Shared local layers (5, 6) must reuse 4 (not 3, which is global), and
// shared global layer (7) must reuse 3 (not 4, which is local).
TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsNonMultipleUnsharedLayers) {
  Config config = Config::E4B();
  config.num_layers = 8;
  config.attention_pattern_size = 4;
  config.frac_shared_layers = 3.0f / 8.0f;
  config.share_local = true;
  config.share_global = true;
  EXPECT_THAT(GetKvCacheSharingPatterns(config),
              ElementsAre(0, 1, 2, 3, 4, 4, 4, 3));
}

// Fallback when no unshared layer of matching `LayerType` exists:
// 4 layers, pattern 4 (`[L, L, L, G]`), 2 shared layers -> unshared `[0:L,
// 1:L]`. Layer 3 is global with no unshared global layer before it, so it
// must stay unshared (`patterns[3] == 3`).
TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsFallback) {
  Config config = Config::E4B();
  config.num_layers = 4;
  config.attention_pattern_size = 4;
  config.frac_shared_layers = 2.0f / 4.0f;
  config.share_local = true;
  config.share_global = true;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre(0, 1, 1, 3));
}

TEST(Gemma4GraphConfigTest, KvCacheSharingPatternsEdgeCases) {
  Config config = Config::E4B();

  // Non-positive `num_layers` returns an empty pattern.
  config.num_layers = 0;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre());
  config.num_layers = -2;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre());

  // `share_local = false` / `share_global = false` disables sharing per type.
  config.num_layers = 4;
  config.attention_pattern_size = 2;  // [L, G, L, G]
  config.frac_shared_layers = 0.5f;   // layers 2, 3 are in shared range
  config.share_local = false;
  config.share_global = true;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre(0, 1, 2, 1));

  config.share_local = true;
  config.share_global = false;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre(0, 1, 0, 3));

  // Out-of-range `frac_shared_layers` is clamped to [0, num_layers].
  config.share_local = true;
  config.share_global = true;
  config.frac_shared_layers = -0.5f;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre(0, 1, 2, 3));

  config.frac_shared_layers = 1.5f;
  EXPECT_THAT(GetKvCacheSharingPatterns(config), ElementsAre(0, 1, 2, 3));
}

TYPED_TEST(Gemma4GraphTest, InputAndWeightValidationErrors) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;

  Config config = Config::E4B();
  config.num_layers = 2;
  config.vocab_size = 10;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.head_dim = 4;
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.per_layer_input_dim = 0;

  Gemma4Inputs<Tag> valid_inputs;
  valid_inputs.embedded_input.Set(
      {.name = "embedded_input", .type = Type::kFP32, .shape = {1, 2, 4}});
  valid_inputs.sliding_attention_mask.Set({.name = "sliding_attention_mask",
                                           .type = Type::kFP32,
                                           .shape = {1, 1, 2, 2}});
  valid_inputs.rope_local_cos.Set(
      {.name = "rope_local_cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  valid_inputs.rope_local_sin.Set(
      {.name = "rope_local_sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  // Invalid or rank-2 `embedded_input` fails cleanly instead of indexing
  // `GetShape()[1]` out of bounds.
  {
    Gemma4Inputs<Tag> bad_inputs = valid_inputs;
    bad_inputs.embedded_input = Tensor(TensorHandle::Invalid());
    LazyTensorMapping weights =
        LazyTensorMapping(CreateGemma4GraphTestWeights(2))
            .Register<Gemma4WeightHooks>();
    EXPECT_THAT(BuildGemma4Graph(bad_inputs, weights, config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Gemma4Inputs<Tag> rank1_inputs = valid_inputs;
    rank1_inputs.embedded_input = Tensor({.type = Type::kFP32, .shape = {4}});
    LazyTensorMapping weights =
        LazyTensorMapping(CreateGemma4GraphTestWeights(2))
            .Register<Gemma4WeightHooks>();
    EXPECT_THAT(BuildGemma4Graph(rank1_inputs, weights, config),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("rank 3 embedded_input")));
  }

  // When `per_layer_input_dim > 0`, missing or error per-layer token embeddings
  // fail loudly instead of silently skipping PLE.
  {
    Config ple_config = config;
    ple_config.per_layer_input_dim = 2;
    LazyTensorMapping weights =
        LazyTensorMapping(CreateGemma4GraphTestWeights(2))
            .Register<Gemma4WeightHooks>();
    EXPECT_THAT(
        BuildGemma4Graph(valid_inputs, weights, ple_config),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("requires per_layer_token_embeddings of size")));

    Gemma4Inputs<Tag> err_ple_inputs = valid_inputs;
    err_ple_inputs.per_layer_token_embeddings = {
        Tensor(absl::InternalError("ple embedding failure")),
        Tensor({.type = Type::kFP32, .shape = {1, 2, 2}})};
    EXPECT_THAT(BuildGemma4Graph(err_ple_inputs, weights, ple_config),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("ple embedding failure")));
  }

  // Bad `model.norm.weight` or `lm_head.weight` shape fails during
  // `BuildGemma4Graph` rather than returning `OkStatus()` with an error tensor.
  {
    auto weight_map = CreateGemma4GraphTestWeights(2);
    weight_map.insert_or_assign(
        "model.norm.weight",
        TensorHandle({.type = Type::kFP32, .shape = {8}, .buffer = 1.0f}));
    LazyTensorMapping weights =
        LazyTensorMapping(std::move(weight_map)).Register<Gemma4WeightHooks>();
    EXPECT_THAT(BuildGemma4Graph(valid_inputs, weights, config),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("model.norm.weight to have shape")));
  }
  {
    auto weight_map = CreateGemma4GraphTestWeights(2);
    weight_map.insert_or_assign(
        "model.embed_tokens.weight",
        TensorHandle({.type = Type::kFP32, .shape = {10, 8}, .buffer = 0.1f}));
    LazyTensorMapping weights =
        LazyTensorMapping(std::move(weight_map)).Register<Gemma4WeightHooks>();
    EXPECT_THAT(BuildGemma4Graph(valid_inputs, weights, config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
