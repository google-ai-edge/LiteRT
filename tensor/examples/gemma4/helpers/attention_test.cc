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

#include "tensor/examples/gemma4/helpers/attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
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

template <class RetManual = void, class F, class T,
          class Ret = std::conditional_t<
              std::is_same_v<RetManual, void>,
              decltype(std::declval<F>()(std::declval<T>())), RetManual>>
std::vector<Ret> vector_from(std::vector<T> v, F&& func) {
  std::vector<Ret> res(v.size());
  std::transform(v.begin(), v.end(), res.begin(), std::forward<F>(func));
  return res;
}

std::vector<float> GetAttentionAngles() {
  return {
      M_PI / 6.0f, M_PI / 3.0f, M_PI / 6.0f, M_PI / 3.0f,
      M_PI / 4.0f, M_PI / 2.0f, M_PI / 4.0f, M_PI / 2.0f,
  };
}

std::vector<float> GenerateWeights(size_t m, size_t n, float scale = 0.01f,
                                   float start = 0.1f) {
  std::vector<float> matrix(m * n);
  for (size_t i = 0; i < m * n; ++i) {
    matrix[i] = start + static_cast<float>(i) * scale;
  }
  return matrix;
}

absl::flat_hash_map<std::string, XnnTensor> CreateDefaultWeights(
    absl::string_view prefix = "attn") {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  weights.insert({absl::StrCat(prefix, ".q_proj.weight"),
                  XnnTensor({.name = "q_proj",
                             .type = Type::kFP32,
                             .shape = {8, 4},
                             .buffer = GenerateWeights(8, 4, 0.01f, 0.1f)})});

  weights.insert({absl::StrCat(prefix, ".k_proj.weight"),
                  XnnTensor({.name = "k_proj",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = GenerateWeights(4, 4, 0.02f, 0.05f)})});

  weights.insert({absl::StrCat(prefix, ".v_proj.weight"),
                  XnnTensor({.name = "v_proj",
                             .type = Type::kFP32,
                             .shape = {4, 4},
                             .buffer = GenerateWeights(4, 4, 0.015f, 0.02f)})});

  weights.insert({absl::StrCat(prefix, ".o_proj.weight"),
                  XnnTensor({.name = "o_proj",
                             .type = Type::kFP32,
                             .shape = {4, 8},
                             .buffer = GenerateWeights(4, 8, 0.01f, 0.05f)})});

  weights.insert(
      {absl::StrCat(prefix, ".q_norm.weight"), XnnTensor({.name = "q_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  weights.insert(
      {absl::StrCat(prefix, ".k_norm.weight"), XnnTensor({.name = "k_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  return weights;
}

absl::flat_hash_map<std::string, XnnTensor> CreateGqaWeights(
    absl::string_view prefix = "attn") {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  weights.insert(
      {absl::StrCat(prefix, ".q_proj.weight"),
       XnnTensor({.name = "q_proj",
                  .type = Type::kFP32,
                  .shape = {16, 8},
                  .buffer = GenerateWeights(16, 8, 0.005f, 0.01f)})});

  weights.insert({absl::StrCat(prefix, ".k_proj.weight"),
                  XnnTensor({.name = "k_proj",
                             .type = Type::kFP32,
                             .shape = {8, 8},
                             .buffer = GenerateWeights(8, 8, 0.01f, 0.02f)})});

  weights.insert({absl::StrCat(prefix, ".v_proj.weight"),
                  XnnTensor({.name = "v_proj",
                             .type = Type::kFP32,
                             .shape = {8, 8},
                             .buffer = GenerateWeights(8, 8, 0.008f, 0.01f)})});

  weights.insert(
      {absl::StrCat(prefix, ".o_proj.weight"),
       XnnTensor({.name = "o_proj",
                  .type = Type::kFP32,
                  .shape = {8, 16},
                  .buffer = GenerateWeights(8, 16, 0.005f, 0.01f)})});

  weights.insert(
      {absl::StrCat(prefix, ".q_norm.weight"), XnnTensor({.name = "q_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  weights.insert(
      {absl::StrCat(prefix, ".k_norm.weight"), XnnTensor({.name = "k_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  return weights;
}

absl::flat_hash_map<std::string, XnnTensor> CreateMhaWeights(
    absl::string_view prefix = "attn") {
  absl::flat_hash_map<std::string, XnnTensor> weights;

  weights.insert({absl::StrCat(prefix, ".q_proj.weight"),
                  XnnTensor({.name = "q_proj",
                             .type = Type::kFP32,
                             .shape = {8, 4},
                             .buffer = GenerateWeights(8, 4, 0.01f, 0.1f)})});

  weights.insert({absl::StrCat(prefix, ".k_proj.weight"),
                  XnnTensor({.name = "k_proj",
                             .type = Type::kFP32,
                             .shape = {8, 4},
                             .buffer = GenerateWeights(8, 4, 0.02f, 0.05f)})});

  weights.insert({absl::StrCat(prefix, ".v_proj.weight"),
                  XnnTensor({.name = "v_proj",
                             .type = Type::kFP32,
                             .shape = {8, 4},
                             .buffer = GenerateWeights(8, 4, 0.015f, 0.02f)})});

  weights.insert({absl::StrCat(prefix, ".o_proj.weight"),
                  XnnTensor({.name = "o_proj",
                             .type = Type::kFP32,
                             .shape = {4, 8},
                             .buffer = GenerateWeights(4, 8, 0.01f, 0.05f)})});

  weights.insert(
      {absl::StrCat(prefix, ".q_norm.weight"), XnnTensor({.name = "q_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  weights.insert(
      {absl::StrCat(prefix, ".k_norm.weight"), XnnTensor({.name = "k_norm",
                                                          .type = Type::kFP32,
                                                          .shape = {4},
                                                          .buffer = 1.0f})});

  return weights;
}

TEST(Gemma4GraphTest, SingleKVHeadAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();
  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6518863f, 1.2290505f, 1.8062148f, 2.3833790f,
                         0.6504126f, 1.2256507f, 1.8008889f, 2.3761272f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

// Grouped-Query Attention: specifically testing the per-head Slice -> Tile ->
// Concatenation pipeline that duplicates KV heads to match query heads.
TEST(Gemma4GraphTest, MultiKvHeadsGqaAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.head_dim = 4;
  config.embed_dim = 8;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 8}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateGqaWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();
  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 16> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f,
                                            3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                            5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.7551928f, 1.9530578f, 3.1509228f, 4.3487878f,
                         5.5466528f, 6.7445178f, 7.9423828f, 9.1402483f,
                         0.7549210f, 1.9515085f, 3.1480958f, 4.3446836f,
                         5.5412712f, 6.7378583f, 7.9344454f, 9.1310329f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2887521f, -0.9443282f, 1.0971556f, 1.3494868f,
                         -0.5706881f, -1.4977449f, 0.9767548f, 0.6906699f,
                         0.1163326f, -0.6042792f, 1.2947545f, 1.3946054f,
                         -0.2240744f, -1.2218513f, 1.2798100f, 0.9049622f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.2701872f, 0.6819012f, 1.0936151f, 1.5053290f,
                         0.2579717f, 0.6755445f, 1.0931174f, 1.5106902f,
                         0.7441726f, 0.9039948f, 1.0638171f, 1.2236395f,
                         0.7425159f, 0.9033106f, 1.0641053f, 1.2248999f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2887521f, -0.9443282f, 1.0971556f, 1.3494868f,
                         -0.5706881f, -1.4977449f, 0.9767548f, 0.6906699f,
                         0.1163326f, -0.6042792f, 1.2947545f, 1.3946054f,
                         -0.2240744f, -1.2218513f, 1.2798100f, 0.9049622f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.2701872f, 0.6819012f, 1.0936151f, 1.5053290f,
                         0.2579717f, 0.6755445f, 1.0931174f, 1.5106902f,
                         0.7441726f, 0.9039948f, 1.0638171f, 1.2236395f,
                         0.7425159f, 0.9033106f, 1.0641053f, 1.2248999f}));
}

// Multi-Head Attention (MHA): testing standard attention where num_heads ==
// num_kv_heads (GQA tiling bypassed).
TEST(Gemma4GraphTest, MultiHeadAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 2;
  config.head_dim = 4;
  config.embed_dim = 4;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateMhaWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();
  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6751769f, 1.2788864f, 1.8825960f, 2.4863057f,
                         0.6746137f, 1.2772517f, 1.8798898f, 2.4825280f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f,
                         0.1303651f, -0.5914791f, 1.2997992f, 1.3941592f,
                         -0.2128929f, -1.2115418f, 1.2875929f, 0.9104656f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f,
                         0.7515375f, 0.9070280f, 1.0625186f, 1.2180090f,
                         0.7483901f, 0.9057335f, 1.0630770f, 1.2204205f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f,
                         0.1303651f, -0.5914791f, 1.2997992f, 1.3941592f,
                         -0.2128929f, -1.2115418f, 1.2875929f, 0.9104656f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f,
                         0.7515375f, 0.9070280f, 1.0625186f, 1.2180090f,
                         0.7483901f, 0.9057335f, 1.0630770f, 1.2204205f}));
}

// Checks logits soft-capping behavior when attn_logits_soft_cap is specified,
TEST(Gemma4GraphTest, SoftCappingAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  config.attn_logits_soft_cap = 1.0f;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();
  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6518863f, 1.2290505f, 1.8062148f, 2.3833790f,
                         0.6504511f, 1.2257649f, 1.8010787f, 2.3763924f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

// Verifies Attention layer configured as a Global Layer (is_global = true).
// `global_key_size` should be used for the key dimension instead of `head_dim`.
TEST(Gemma4GraphTest, GlobalLayerAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 2;
  config.global_key_size = 4;
  config.embed_dim = 4;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();
  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  // is_global=true uses global_key_size (4) instead of head_dim (2)
  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/true, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6518863f, 1.2290505f, 1.8062148f, 2.3833790f,
                         0.6504126f, 1.2256507f, 1.8008889f, 2.3761272f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

// Checks pre-populated KV tensors
TEST(Gemma4GraphTest, KVCacheAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;

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

  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 8> mask_data = {0.0f, 0.0f, 0.0f, -1e9f,
                                          0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  const std::array<float, 8> kc_data = {0.5f, 0.5f, 0.5f, 0.5f,
                                        1.0f, 1.0f, 1.0f, 1.0f};
  ASSERT_THAT(runner.SetInput(key_cache, kc_data), IsOk());

  const std::array<float, 8> vc_data = {0.2f, 0.2f, 0.2f, 0.2f,
                                        0.4f, 0.4f, 0.4f, 0.4f};
  ASSERT_THAT(runner.SetInput(value_cache, vc_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.5692459f, 1.0782182f, 1.5871906f, 2.0961628f,
                         0.6344495f, 1.1964998f, 1.7585504f, 2.3206010f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f,
                         -0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.2f, 0.2f, 0.2f, 0.2f, 0.4f, 0.4f, 0.4f, 0.4f,
                         0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

// Checks when provided KV cache has sequence length 0 (empty cache fallback).
TEST(Gemma4GraphTest, EmptyKVCacheAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  XnnTensor key_cache(
      {.name = "key_cache", .type = Type::kFP32, .shape = {1, 1, 0, 4}});
  XnnTensor value_cache(
      {.name = "value_cache", .type = Type::kFP32, .shape = {1, 1, 0, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6518863f, 1.2290505f, 1.8062148f, 2.3833790f,
                         0.6504126f, 1.2256507f, 1.8008889f, 2.3761272f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

// Grouped-Query Attention with dynamic pre-existing KV cache history.
// Verifies that dynamic sequence slicing (-1) in GQA correctly handles
// expanding KV cache sequence lengths without static dimension errors.
TEST(Gemma4GraphTest, MultiKvHeadsGqaDynamicKVCacheAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 4;
  config.num_kv_heads = 2;
  config.head_dim = 4;
  config.embed_dim = 8;

  // 1 token decode step
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 8}});
  // History of 3 cached tokens + 1 new token = 4 sequence length for attention
  // mask
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 1, 4}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 1, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 1, 4}});

  // Cached history of 3 tokens for 2 KV heads, head_dim 4 -> 1*2*3*4 = 24
  // floats
  XnnTensor key_cache(
      {.name = "key_cache", .type = Type::kFP32, .shape = {1, 2, 3, 4}});
  XnnTensor value_cache(
      {.name = "value_cache", .type = Type::kFP32, .shape = {1, 2, 3, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateGqaWeights();

  XnnTensor shared_key = XnnTensor::Invalid();
  XnnTensor shared_value = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create(
          {attn_out.output, attn_out.key_cache, attn_out.value_cache}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::array<float, 4> cos_data = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::array<float, 4> sin_data = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  std::vector<float> kc_data(24, 0.5f);
  std::vector<float> vc_data(24, 0.2f);
  ASSERT_THAT(runner.SetInput(key_cache, kc_data), IsOk());
  ASSERT_THAT(runner.SetInput(value_cache, vc_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.5797290f, 1.5080730f, 2.4364171f, 3.3647606f,
                         4.2931042f, 5.2214484f, 6.1497927f, 7.0781364f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.2985111f, 0.6965259f, 1.0945408f, 1.4925557f,
                         0.7481243f, 0.9056241f, 1.0631239f, 1.2206237f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.2701873f, 0.6819012f, 1.0936151f, 1.5053290f,
                         0.7441726f, 0.9039949f, 1.0638173f, 1.2236395f}));
}

// Checks when shared_key and shared_value tensors are provided, which should
// bypass standard projection and cache update logic.
TEST(Gemma4GraphTest, SharedKVAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;

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

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  const std::array<float, 8> sk_data = {0.3f, 0.3f, 0.3f, 0.3f,
                                        0.6f, 0.6f, 0.6f, 0.6f};
  ASSERT_THAT(runner.SetInput(shared_key, sk_data), IsOk());

  const std::array<float, 8> sv_data = {0.1f, 0.1f, 0.1f, 0.1f,
                                        0.8f, 0.8f, 0.8f, 0.8f};
  ASSERT_THAT(runner.SetInput(shared_value, sv_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/attention.py`.
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.0680000f, 0.1320000f, 0.1960000f, 0.2600000f,
                         0.3324648f, 0.6436169f, 0.9547691f, 1.2659214f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(
      std::move(res_kc).As<const float>(),
      Pointwise(FloatNear(1e-4f), {0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(
      std::move(res_vc).As<const float>(),
      Pointwise(FloatNear(1e-4f), {0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(
      std::move(res_ka).As<const float>(),
      Pointwise(FloatNear(1e-4f), {0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(
      std::move(res_va).As<const float>(),
      Pointwise(FloatNear(1e-4f), {0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8}));
}

// Checks fallback behavior when shared KV shapes mismatch.
TEST(Gemma4GraphTest, MismatchedSharedKVAttentionTest) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 2, 4}});
  XnnTensor attention_mask(
      {.name = "attention_mask", .type = Type::kFP32, .shape = {1, 1, 2, 2}});
  XnnTensor cos({.name = "cos", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor sin({.name = "sin", .type = Type::kFP32, .shape = {1, 1, 2, 4}});

  XnnTensor shared_key(
      {.name = "shared_key", .type = Type::kFP32, .shape = {1, 1, 2, 4}});
  XnnTensor shared_value(
      {.name = "shared_value", .type = Type::kFP32, .shape = {1, 1, 3, 4}});

  absl::flat_hash_map<std::string, XnnTensor> weights = CreateDefaultWeights();

  XnnTensor key_cache = XnnTensor::Invalid();
  XnnTensor value_cache = XnnTensor::Invalid();

  XnnTensor eps_tensor({
      .type = Type::kFP32,
      .shape = {1},
      .buffer = config.rms_norm_eps,
  });
  AttentionOutput<XnnpackMixinTag> attn_out = Attention(
      input, attention_mask, cos, sin, key_cache, value_cache, shared_key,
      shared_value, config, weights, "attn", /*is_global=*/false, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      XnnpackRunner runner,
      XnnpackRunner::Create({attn_out.output, attn_out.key_cache,
                             attn_out.value_cache, attn_out.key_for_attn,
                             attn_out.value_for_attn}));

  const std::array<float, 8> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  const std::array<float, 4> mask_data = {0.0f, -1e9f, 0.0f, 0.0f};
  ASSERT_THAT(runner.SetInput(attention_mask, mask_data), IsOk());

  const std::vector<float> angles = GetAttentionAngles();
  const std::vector<float> cos_data = vector_from(angles, cosf);
  const std::vector<float> sin_data = vector_from(angles, sinf);
  ASSERT_THAT(runner.SetInput(cos, cos_data), IsOk());
  ASSERT_THAT(runner.SetInput(sin, sin_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_out,
                                  runner.ReadOutput(attn_out.output));
  EXPECT_THAT(std::move(res_out).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.6518863f, 1.2290505f, 1.8062148f, 2.3833790f,
                         0.6504126f, 1.2256507f, 1.8008889f, 2.3761272f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_kc,
                                  runner.ReadOutput(attn_out.key_cache));
  EXPECT_THAT(std::move(res_kc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_vc,
                                  runner.ReadOutput(attn_out.value_cache));
  EXPECT_THAT(std::move(res_vc).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_ka,
                                  runner.ReadOutput(attn_out.key_for_attn));
  EXPECT_THAT(std::move(res_ka).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {-0.2061636f, -0.8796699f, 1.1456800f, 1.3678794f,
                         -0.5082401f, -1.4547977f, 1.0409147f, 0.7360378f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> res_va,
                                  runner.ReadOutput(attn_out.value_for_attn));
  EXPECT_THAT(std::move(res_va).As<const float>(),
              Pointwise(FloatNear(1e-4f),
                        {0.3220783f, 0.7085721f, 1.0950661f, 1.4815600f,
                         0.3003760f, 0.6974834f, 1.0945907f, 1.4916979f}));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
