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

#include "tensor/examples/gemma4/helpers/transformer.h"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

absl::flat_hash_map<std::string, XnnTensor> CreateDefaultWeights() {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  weights.insert(
      {"model.layers.0.self_attn.q_proj.weight",
       XnnTensor({.name = "q_proj",
                  .type = Type::kFP32,
                  .shape = {8, 4},
                  .buffer = std::vector<float>{
                      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}})});

  weights.insert(
      {"model.layers.0.self_attn.k_proj.weight",
       XnnTensor({.name = "k_proj",
                  .type = Type::kFP32,
                  .shape = {4, 4},
                  .buffer = std::vector<float>{
                      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}})});

  weights.insert(
      {"model.layers.0.self_attn.v_proj.weight",
       XnnTensor({.name = "v_proj",
                  .type = Type::kFP32,
                  .shape = {4, 4},
                  .buffer = std::vector<float>{
                      0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f}})});

  weights.insert(
      {"model.layers.0.self_attn.o_proj.weight",
       XnnTensor({.name = "o_proj",
                  .type = Type::kFP32,
                  .shape = {4, 8},
                  .buffer = std::vector<float>{
                      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f}})});

  weights.insert(
      {"model.layers.0.self_attn.q_norm.weight",
       XnnTensor({.name = "q_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});

  weights.insert(
      {"model.layers.0.self_attn.k_norm.weight",
       XnnTensor({.name = "k_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});

  weights.insert(
      {"model.layers.0.input_layernorm.weight",
       XnnTensor({.name = "pre_attn_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});
  weights.insert(
      {"model.layers.0.post_attention_layernorm.weight",
       XnnTensor({.name = "post_attn_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});
  weights.insert(
      {"model.layers.0.pre_feedforward_layernorm.weight",
       XnnTensor({.name = "pre_ffn_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});
  weights.insert(
      {"model.layers.0.post_feedforward_layernorm.weight",
       XnnTensor({.name = "post_ffn_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});

  weights.insert(
      {"model.layers.0.mlp.gate_proj.weight",
       XnnTensor({.name = "gate_proj",
                  .type = Type::kFP32,
                  .shape = {6, 4},
                  .buffer = std::vector<float>{
                      1.0f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                      1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f}})});

  weights.insert(
      {"model.layers.0.mlp.up_proj.weight",
       XnnTensor({.name = "up_proj",
                  .type = Type::kFP32,
                  .shape = {6, 4},
                  .buffer = std::vector<float>{
                      0.5f, 0.5f, 0.0f,  0.0f, 0.0f, 0.5f, 0.5f, 0.0f,
                      0.0f, 0.0f, 0.5f,  0.5f, 0.5f, 0.0f, 0.0f, 0.5f,
                      1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f}})});

  weights.insert(
      {"model.layers.0.mlp.down_proj.weight",
       XnnTensor({.name = "down_proj",
                  .type = Type::kFP32,
                  .shape = {4, 6},
                  .buffer = std::vector<float>{
                      1.0f,  0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f,
                      0.0f,  0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
                      -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -0.5f}})});

  weights.insert({"model.layers.0.layer_scalar",
                  XnnTensor({.name = "layer_scalar",
                             .type = Type::kFP32,
                             .shape = {1},
                             .buffer = std::vector<float>{0.5f}})});

  return weights;
}

TEST(Gemma4GraphTest, TransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.914647f, 1.843373f, 2.878430f, 3.077854f, 3.197403f,
                         3.938960f, 4.694313f, 5.088164f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.766109f, 0.097841f, 0.863950f, 1.630059f, 0.214422f,
                         -0.909717f, 1.286534f, 1.212956f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.365148f, 0.730295f, 1.095443f, 1.460591f, 0.758097f,
                         0.909716f, 1.061335f, 1.212954f}));
}

TEST(Gemma4GraphTest, DisabledPostNormsTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = false;
  config.use_post_ffw_norm = false;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({layer_out.output}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(result).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {1.1730204f, 2.2576695f, 3.4221691f, 3.4042250f,
                         3.5445033f, 4.3602568f, 5.1829415f, 5.4708247f}));
}

TEST(Gemma4GraphTest, PerLayerInputTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.per_layer_input_dim = 2;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor per_layer_input(
      {.name = "per_layer_input", .type = Type::kFP32, .shape = {1, 2, 2}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  weights.insert(
      {"model.layers.0.per_layer_input_gate.weight",
       XnnTensor({.name = "per_layer_input_gate",
                  .type = Type::kFP32,
                  .shape = {2, 4},
                  .buffer = std::vector<float>{0.5f, 0.0f, 0.0f, 0.0f, 0.0f,
                                               0.5f, 0.0f, 0.0f}})});

  weights.insert(
      {"model.layers.0.per_layer_projection.weight",
       XnnTensor({.name = "per_layer_projection",
                  .type = Type::kFP32,
                  .shape = {4, 2},
                  .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.5f,
                                               0.0f, 0.0f, 0.5f}})});

  weights.insert(
      {"model.layers.0.post_per_layer_input_norm.weight",
       XnnTensor({.name = "post_per_layer_norm",
                  .type = Type::kFP32,
                  .shape = {4},
                  .buffer = std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}})});

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({layer_out.output}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  const std::array<float, 4> pli_data = {0.1f, 0.2f, 0.3f, 0.4f};
  ASSERT_THAT(runner.SetInput(per_layer_input, pli_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(result).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {1.0986485f, 2.7186582f, 2.9704310f, 3.5154968f,
                         3.6623252f, 4.7030583f, 4.9267740f, 5.4702132f}));
}

TEST(Gemma4GraphTest, KVCacheTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor key_cache(
      {.name = "key_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor value_cache(
      {.name = "value_cache", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 8> mask_data = {0.0f, 0.0f, 0.0f, -1e9f,
                                          0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  const std::array<float, 8> kc_data = {0.5f, 0.5f, 0.5f, 0.5f,
                                        1.0f, 1.0f, 1.0f, 1.0f};
  ASSERT_THAT(runner.SetInput(key_cache, kc_data), IsOk());

  const std::array<float, 8> vc_data = {0.2f, 0.2f, 0.2f, 0.2f,
                                        0.4f, 0.4f, 0.4f, 0.4f};
  ASSERT_THAT(runner.SetInput(value_cache, vc_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.9396022f, 1.8487435f, 2.8596824f, 3.0836855f,
                         3.1999975f, 3.9395518f, 4.6928586f, 5.0873050f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.766109f, 0.097841f, 0.863950f, 1.630059f, 0.214422f,
                         -0.909717f, 1.286534f, 1.212956f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.365148f, 0.730295f, 1.095443f, 1.460591f, 0.758097f,
                         0.909716f, 1.061335f, 1.212954f}));
}

TEST(Gemma4GraphTest, SharedKVTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor shared_key(
      {.name = "shared_key", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor shared_value(
      {.name = "shared_value", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  const std::array<float, 8> sk_data = {0.3f, 0.3f, 0.3f, 0.3f,
                                        0.6f, 0.6f, 0.6f, 0.6f};
  ASSERT_THAT(runner.SetInput(shared_key, sk_data), IsOk());

  const std::array<float, 8> sv_data = {0.1f, 0.1f, 0.1f, 0.1f,
                                        0.8f, 0.8f, 0.8f, 0.8f};
  ASSERT_THAT(runner.SetInput(shared_value, sv_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {1.1655178f, 1.8674172f, 2.6372028f, 3.1569301f,
                         3.3158966f, 3.9430903f, 4.5857431f, 5.1006519f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3f, 0.3f, 0.3f, 0.3f, 0.6f, 0.6f, 0.6f, 0.6f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.1f, 0.1f, 0.1f, 0.1f, 0.8f, 0.8f, 0.8f, 0.8f}));
}

TEST(Gemma4GraphTest, SoftCappingTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.attn_logits_soft_cap = 1.0f;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.914647f, 1.843373f, 2.878430f, 3.077854f, 3.149959f,
                         3.935589f, 4.734564f, 5.078285f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.766109f, 0.097841f, 0.863950f, 1.630059f, 0.214422f,
                         -0.909717f, 1.286534f, 1.212956f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.365148f, 0.730295f, 1.095443f, 1.460591f, 0.758097f,
                         0.909716f, 1.061335f, 1.212954f}));
}

TEST(Gemma4GraphTest, GlobalLayerTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.global_key_size = 4;
  config.embed_dim = 4;
  config.hidden_dim = 6;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;
  config.attention_pattern_size = 1;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.914647f, 1.843373f, 2.878430f, 3.077854f, 3.197403f,
                         3.938960f, 4.694313f, 5.088164f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.766109f, 0.097841f, 0.863950f, 1.630059f, 0.214422f,
                         -0.909717f, 1.286534f, 1.212956f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.365148f, 0.730295f, 1.095443f, 1.460591f, 0.758097f,
                         0.909716f, 1.061335f, 1.212954f}));
}

absl::flat_hash_map<std::string, XnnTensor> CreateGqaTransformerWeights() {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  std::vector<float> q_proj_buf(128, 0.0f);
  for (int row = 0; row < 16; ++row) {
    q_proj_buf[row * 8 + (row % 4)] = 1.0f;
    q_proj_buf[row * 8 + (row % 4) + 4] = 1.0f;
  }
  weights.insert({"model.layers.0.self_attn.q_proj.weight",
                  XnnTensor({.name = "q_proj",
                             .type = Type::kFP32,
                             .shape = {16, 8},
                             .buffer = q_proj_buf})});

  std::vector<float> k_proj_buf(64, 0.0f);
  for (int row = 0; row < 8; ++row) {
    k_proj_buf[row * 8 + (row % 4)] = 1.0f;
    k_proj_buf[row * 8 + (row % 4) + 4] = 1.0f;
  }
  weights.insert({"model.layers.0.self_attn.k_proj.weight",
                  XnnTensor({.name = "k_proj",
                             .type = Type::kFP32,
                             .shape = {8, 8},
                             .buffer = k_proj_buf})});

  std::vector<float> v_proj_buf(64, 0.0f);
  for (int row = 0; row < 8; ++row) {
    v_proj_buf[row * 8 + (row % 4)] = 0.5f;
    v_proj_buf[row * 8 + (row % 4) + 4] = 0.5f;
  }
  weights.insert({"model.layers.0.self_attn.v_proj.weight",
                  XnnTensor({.name = "v_proj",
                             .type = Type::kFP32,
                             .shape = {8, 8},
                             .buffer = v_proj_buf})});

  std::vector<float> o_proj_buf(128, 0.0f);
  for (int row = 0; row < 8; ++row) {
    for (int k = 0; k < 4; ++k) {
      o_proj_buf[row * 16 + (row % 4) + k * 4] = 0.5f;
    }
  }
  weights.insert({"model.layers.0.self_attn.o_proj.weight",
                  XnnTensor({.name = "o_proj",
                             .type = Type::kFP32,
                             .shape = {8, 16},
                             .buffer = o_proj_buf})});

  weights.insert({"model.layers.0.self_attn.q_norm.weight",
                  XnnTensor({.name = "q_norm",
                             .type = Type::kFP32,
                             .shape = {4},
                             .buffer = std::vector<float>(4, 1.0f)})});
  weights.insert({"model.layers.0.self_attn.k_norm.weight",
                  XnnTensor({.name = "k_norm",
                             .type = Type::kFP32,
                             .shape = {4},
                             .buffer = std::vector<float>(4, 1.0f)})});
  weights.insert({"model.layers.0.input_layernorm.weight",
                  XnnTensor({.name = "pre_attn_norm",
                             .type = Type::kFP32,
                             .shape = {8},
                             .buffer = std::vector<float>(8, 1.0f)})});
  weights.insert({"model.layers.0.post_attention_layernorm.weight",
                  XnnTensor({.name = "post_attn_norm",
                             .type = Type::kFP32,
                             .shape = {8},
                             .buffer = std::vector<float>(8, 1.0f)})});
  weights.insert({"model.layers.0.pre_feedforward_layernorm.weight",
                  XnnTensor({.name = "pre_ffn_norm",
                             .type = Type::kFP32,
                             .shape = {8},
                             .buffer = std::vector<float>(8, 1.0f)})});
  weights.insert({"model.layers.0.post_feedforward_layernorm.weight",
                  XnnTensor({.name = "post_ffn_norm",
                             .type = Type::kFP32,
                             .shape = {8},
                             .buffer = std::vector<float>(8, 1.0f)})});

  std::vector<float> gate_buf(96, 0.0f);
  for (int row = 0; row < 12; ++row) {
    gate_buf[row * 8 + (row % 4)] = 1.0f;
    gate_buf[row * 8 + (row % 4) + 4] = 1.0f;
  }
  weights.insert(
      {"model.layers.0.mlp.gate_proj.weight", XnnTensor({.name = "gate_proj",
                                                         .type = Type::kFP32,
                                                         .shape = {12, 8},
                                                         .buffer = gate_buf})});

  std::vector<float> up_buf(96, 0.0f);
  for (int row = 0; row < 12; ++row) {
    up_buf[row * 8 + (row % 4)] = 0.5f;
    up_buf[row * 8 + (row % 4) + 4] = 0.5f;
  }
  weights.insert(
      {"model.layers.0.mlp.up_proj.weight", XnnTensor({.name = "up_proj",
                                                       .type = Type::kFP32,
                                                       .shape = {12, 8},
                                                       .buffer = up_buf})});

  std::vector<float> down_buf(96, 0.0f);
  for (int row = 0; row < 8; ++row) {
    for (int k = 0; k < 3; ++k) {
      down_buf[row * 12 + (row % 4) + k * 4] = 0.5f;
    }
  }
  weights.insert(
      {"model.layers.0.mlp.down_proj.weight", XnnTensor({.name = "down_proj",
                                                         .type = Type::kFP32,
                                                         .shape = {8, 12},
                                                         .buffer = down_buf})});

  weights.insert({"model.layers.0.layer_scalar",
                  XnnTensor({.name = "layer_scalar",
                             .type = Type::kFP32,
                             .shape = {1},
                             .buffer = std::vector<float>{0.5f}})});

  return weights;
}

TEST(Gemma4GraphTest, MultiKvHeadsGqaTransformerLayerTest) {
  Config config = Config::E4B();
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.head_dim = 4;
  config.embed_dim = 8;
  config.hidden_dim = 12;
  config.use_post_attn_norm = true;
  config.use_post_ffw_norm = true;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 8}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights =
      CreateGqaTransformerWeights();

  XnnTensor key_cache;
  XnnTensor value_cache;
  XnnTensor per_layer_input;
  XnnTensor shared_key;
  XnnTensor shared_value;

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  TransformerLayerOutput<XnnpackMixinTag> layer_out = TransformerLayer(
      input, attention_mask, cos, sin, key_cache, value_cache, per_layer_input,
      shared_key, shared_value, config, weights, 0, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {layer_out.output, layer_out.key_cache, layer_out.value_cache}));

  const std::array<float, 16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f,
                                            3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                            5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 8> cos_data = {0.8660254f, 0.5f, 0.8660254f, 0.5f,
                                         0.7071068f, 0.0f, 0.7071068f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());

  const std::array<float, 8> sin_data = {
      0.5f, 0.8660254f, 0.5f, 0.8660254f, 0.7071068f, 1.0f, 0.7071068f, 1.0f};
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/transformer.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(layer_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.723690f, 1.564007f, 2.523176f, 3.586281f, 0.723690f,
                         1.564007f, 2.523176f, 3.586281f, 3.031578f, 3.790843f,
                         4.576719f, 5.387666f, 3.031578f, 3.790843f, 4.576719f,
                         5.387666f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(layer_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.231495f, -0.899763f, 1.131257f, 1.362752f,
                         -0.214423f, -1.212957f, 1.286535f, 0.909718f,
                         -0.231495f, -0.899763f, 1.131257f, 1.362752f,
                         -0.214423f, -1.212957f, 1.286535f, 0.909718f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(layer_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.365148f, 0.730297f, 1.095445f, 1.460593f, 0.758098f,
                         0.909717f, 1.061337f, 1.212956f, 0.365148f, 0.730297f,
                         1.095445f, 1.460593f, 0.758098f, 0.909717f, 1.061337f,
                         1.212956f}));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
