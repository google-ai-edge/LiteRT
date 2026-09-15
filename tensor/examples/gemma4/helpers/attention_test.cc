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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"          // from @com_google_absl
#include "absl/strings/string_view.h"      // from @com_google_absl
#include "absl/types/span.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/gemma4_config.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "xnnpack.h"  // from @XNNPACK

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

TEST(Gemma4GraphTest, SingleKvHeadSupportsConsistentArithmetic) {
  Config config = Config::E4B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 4;
  config.embed_dim = 4;
  const auto weights = CreateDefaultWeights();

  // Cover one-token and multi-token prefill, then decode with cached history.
  for (const auto& [seq_len, cache_len] :
       {std::pair{1, 0}, std::pair{3, 0}, std::pair{1, 2}}) {
    SCOPED_TRACE(::testing::Message()
                 << "seq_len=" << seq_len << ", cache_len=" << cache_len);
    XnnTensor input({.type = Type::kFP32, .shape = {1, seq_len, 4}});
    XnnTensor mask(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, seq_len + cache_len}});
    XnnTensor cos(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, 4}, .buffer = 1.0f});
    XnnTensor sin(
        {.type = Type::kFP32, .shape = {1, 1, seq_len, 4}, .buffer = 0.0f});
    XnnTensor key_cache = XnnTensor::Invalid();
    XnnTensor value_cache = XnnTensor::Invalid();
    if (cache_len > 0) {
      key_cache =
          XnnTensor({.type = Type::kFP32, .shape = {1, 1, cache_len, 4}});
      value_cache =
          XnnTensor({.type = Type::kFP32, .shape = {1, 1, cache_len, 4}});
    }
    XnnTensor eps(
        {.type = Type::kFP32, .shape = {1}, .buffer = config.rms_norm_eps});
    const XnnTensor no_shared_kv = TensorHandle::Invalid();
    auto attention = Attention(input, mask, cos, sin, key_cache, value_cache,
                               no_shared_kv, no_shared_kv, config, weights,
                               "attn", /*is_global=*/false, eps);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto graph, BuildXnnpackGraph({attention.output, attention.key_cache,
                                       attention.value_cache}));

    // This mode preserves an explicit Tile's broadcast, which fails runtime
    // creation. Implicit BatchMatMul broadcasting must work without that
    // optimizer rewrite, as well as in the default runner tests above.
    xnn_runtime_t raw_runtime = nullptr;
    const auto status = xnn_create_runtime_v3(
        graph->GetSubgraph(), /*weights_cache=*/nullptr, /*threadpool=*/nullptr,
        XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC, &raw_runtime);
    XnnpackRunner::RuntimePtr runtime(raw_runtime);
    ASSERT_EQ(status, xnn_status_success);
    EXPECT_EQ(xnn_reshape_runtime(runtime.get()), xnn_status_success);
  }
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

// Identity Q/O projections expose both query heads separately. K selects the
// first input pair and V the second; identity RoPE leaves only causal attention
// and RMS normalization in the expected calculation below.
AttentionOutput<XnnpackMixinTag> BuildSingleKvBroadcastAttention(
    const XnnTensor& input, const XnnTensor& mask,
    const XnnTensor& key_cache = XnnTensor::Invalid(),
    const XnnTensor& value_cache = XnnTensor::Invalid()) {
  Config config = Config::E2B();
  config.num_heads = 2;
  config.num_kv_heads = 1;
  config.head_dim = 2;
  config.embed_dim = 4;
  const std::vector<float> identity = {1, 0, 0, 0, 0, 1, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 1};
  absl::flat_hash_map<std::string, XnnTensor> weights;
  for (const char* projection : {"q_proj", "o_proj"}) {
    weights.emplace(
        absl::StrCat("attn.", projection, ".weight"),
        XnnTensor({.type = Type::kFP32, .shape = {4, 4}, .buffer = identity}));
  }
  weights.emplace(
      "attn.k_proj.weight",
      XnnTensor({.type = Type::kFP32,
                 .shape = {2, 4},
                 .buffer = std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0}}));
  weights.emplace(
      "attn.v_proj.weight",
      XnnTensor({.type = Type::kFP32,
                 .shape = {2, 4},
                 .buffer = std::vector<float>{0, 0, 1, 0, 0, 0, 0, 1}}));
  for (const char* norm : {"q_norm", "k_norm"}) {
    weights.emplace(
        absl::StrCat("attn.", norm, ".weight"),
        XnnTensor({.type = Type::kFP32, .shape = {2}, .buffer = 1.0f}));
  }
  const int sequence_length = input.GetShape()[1];
  XnnTensor cos({.type = Type::kFP32,
                 .shape = {1, 1, sequence_length, 2},
                 .buffer = 1.0f});
  XnnTensor sin({.type = Type::kFP32,
                 .shape = {1, 1, sequence_length, 2},
                 .buffer = 0.0f});
  XnnTensor eps(
      {.type = Type::kFP32, .shape = {1}, .buffer = config.rms_norm_eps});
  const XnnTensor no_shared_kv(TensorHandle::Invalid());
  return Attention(input, mask, cos, sin, key_cache, value_cache, no_shared_kv,
                   no_shared_kv, config, weights, "attn", /*is_global=*/false,
                   eps);
}

using AttentionToken = std::array<float, 4>;
const std::array<AttentionToken, 4> kBroadcastTokens = {
    AttentionToken{1, 0, 0, 1}, AttentionToken{0, 1, 1, 0},
    AttentionToken{1, 1, 1, -1}, AttentionToken{-1, 1, 1, 1}};

std::array<double, 2> NormalizeAttentionPair(const AttentionToken& token,
                                             int offset) {
  const double x = token[offset];
  const double y = token[offset + 1];
  const double rms = std::sqrt((x * x + y * y) / 2 + 1.0e-6f);
  return {x / rms, y / rms};
}

// Independent scalar reference: each query head attends to the same past K/V
// sequence. Compute in double without tensor operators, tiling or broadcasting.
std::vector<float> SingleKvCausalReference(
    absl::Span<const AttentionToken> tokens, size_t first_query = 0) {
  std::vector<float> expected;
  for (size_t query = first_query; query < tokens.size(); ++query) {
    for (int head = 0; head < 2; ++head) {
      const auto q = NormalizeAttentionPair(tokens[query], 2 * head);
      std::vector<double> scores(query + 1);
      for (size_t past = 0; past <= query; ++past) {
        const auto k = NormalizeAttentionPair(tokens[past], 0);
        scores[past] = q[0] * k[0] + q[1] * k[1];
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double denominator = 0;
      std::array<double, 2> numerator = {0, 0};
      for (size_t past = 0; past <= query; ++past) {
        const double probability = std::exp(scores[past] - maximum);
        const auto v = NormalizeAttentionPair(tokens[past], 2);
        denominator += probability;
        for (int channel = 0; channel < 2; ++channel) {
          numerator[channel] += probability * v[channel];
        }
      }
      for (double value : numerator) expected.push_back(value / denominator);
    }
  }
  return expected;
}

std::vector<float> SingleKvCacheReference(
    absl::Span<const AttentionToken> tokens, int offset) {
  std::vector<float> expected;
  for (const auto& token : tokens) {
    const auto pair = NormalizeAttentionPair(token, offset);
    expected.insert(expected.end(), pair.begin(), pair.end());
  }
  return expected;
}

class SingleKvBroadcastAttentionTest
    : public ::testing::TestWithParam<uint32_t> {};

TEST_P(SingleKvBroadcastAttentionTest, PrefillMatchesScalarReference) {
  for (int sequence_length : {1, 3}) {
    SCOPED_TRACE(sequence_length);
    XnnTensor input({.type = Type::kFP32, .shape = {1, sequence_length, 4}});
    XnnTensor mask({.type = Type::kFP32,
                    .shape = {1, 1, sequence_length, sequence_length}});
    auto attention = BuildSingleKvBroadcastAttention(input, mask);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto runner,
        XnnpackRunner::Create(
            {attention.output, attention.key_cache, attention.value_cache},
            GetParam()));
    const absl::Span<const AttentionToken> tokens(kBroadcastTokens.data(),
                                                  sequence_length);
    std::vector<float> input_data;
    for (const auto& token : tokens) {
      input_data.insert(input_data.end(), token.begin(), token.end());
    }
    std::vector<float> mask_data(sequence_length * sequence_length, 0);
    for (int query = 0; query < sequence_length; ++query) {
      for (int future = query + 1; future < sequence_length; ++future) {
        mask_data[query * sequence_length + future] =
            -std::numeric_limits<float>::infinity();
      }
    }
    ASSERT_THAT(runner.SetInput(input, input_data), IsOk());
    ASSERT_THAT(runner.SetInput(mask, mask_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());

    const auto expected = SingleKvCausalReference(tokens);
    if (sequence_length > 1) {
      // The second token must not receive the same context in both heads.
      ASSERT_GT(std::abs(expected[4] - expected[6]), 0.5f);
    }
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto output, runner.ReadOutputAs<float>(attention.output));
    EXPECT_THAT(output, Pointwise(FloatNear(2.0e-6f), expected));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto keys, runner.ReadOutputAs<float>(attention.key_cache));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto values, runner.ReadOutputAs<float>(attention.value_cache));
    EXPECT_THAT(
        keys, Pointwise(FloatNear(2.0e-6f), SingleKvCacheReference(tokens, 0)));
    EXPECT_THAT(values, Pointwise(FloatNear(2.0e-6f),
                                  SingleKvCacheReference(tokens, 2)));
  }
}

TEST_P(SingleKvBroadcastAttentionTest, ReusedDecodeRunnerGrowsKvHistory) {
  XnnTensor first_input({.type = Type::kFP32, .shape = {1, 1, 4}});
  XnnTensor first_mask({.type = Type::kFP32, .shape = {1, 1, 1, 1}});
  auto prefill = BuildSingleKvBroadcastAttention(first_input, first_mask);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto prefill_runner,
      XnnpackRunner::Create(
          {prefill.output, prefill.key_cache, prefill.value_cache},
          GetParam()));
  const std::array<float, 1> zero_mask = {0};
  ASSERT_THAT(prefill_runner.SetInput(first_input, kBroadcastTokens[0]),
              IsOk());
  ASSERT_THAT(prefill_runner.SetInput(first_mask, zero_mask), IsOk());
  ASSERT_THAT(prefill_runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto initial_keys, prefill_runner.ReadOutputAs<float>(prefill.key_cache));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto initial_values,
      prefill_runner.ReadOutputAs<float>(prefill.value_cache));
  std::vector<float> key_data(initial_keys.begin(), initial_keys.end());
  std::vector<float> value_data(initial_values.begin(), initial_values.end());

  XnnTensor input({.type = Type::kFP32, .shape = {1, 1, 4}});
  XnnTensor mask({.type = Type::kFP32, .shape = {1, 1, 1, 2}});
  XnnTensor key_cache({.type = Type::kFP32, .shape = {1, 1, 1, 2}});
  XnnTensor value_cache({.type = Type::kFP32, .shape = {1, 1, 1, 2}});
  auto attention =
      BuildSingleKvBroadcastAttention(input, mask, key_cache, value_cache);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner,
      XnnpackRunner::Create(
          {attention.output, attention.key_cache, attention.value_cache,
           attention.key_for_attn, attention.value_for_attn},
          GetParam()));
  for (int query = 1; query < kBroadcastTokens.size(); ++query) {
    SCOPED_TRACE(query);
    ASSERT_THAT(runner.ReshapeInput(key_cache, {1, 1, query, 2}), IsOk());
    ASSERT_THAT(runner.ReshapeInput(value_cache, {1, 1, query, 2}), IsOk());
    ASSERT_THAT(runner.ReshapeInput(mask, {1, 1, 1, query + 1}), IsOk());
    std::vector<float> mask_data(query + 1, 0);
    ASSERT_THAT(runner.SetInput(input, kBroadcastTokens[query]), IsOk());
    ASSERT_THAT(runner.SetInput(mask, mask_data), IsOk());
    ASSERT_THAT(runner.SetInput(key_cache, key_data), IsOk());
    ASSERT_THAT(runner.SetInput(value_cache, value_data), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());

    const absl::Span<const AttentionToken> tokens(kBroadcastTokens.data(),
                                                  query + 1);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto output, runner.ReadOutputAs<float>(attention.output));
    EXPECT_THAT(output, Pointwise(FloatNear(2.0e-6f),
                                  SingleKvCausalReference(tokens, query)));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto keys, runner.ReadOutputAs<float>(attention.key_for_attn));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto values, runner.ReadOutputAs<float>(attention.value_for_attn));
    EXPECT_THAT(
        keys, Pointwise(FloatNear(2.0e-6f), SingleKvCacheReference(tokens, 0)));
    EXPECT_THAT(values, Pointwise(FloatNear(2.0e-6f),
                                  SingleKvCacheReference(tokens, 2)));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto new_keys, runner.ReadOutputAs<float>(attention.key_cache));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        auto new_values, runner.ReadOutputAs<float>(attention.value_cache));
    ASSERT_EQ(new_keys.size(), 2);
    ASSERT_EQ(new_values.size(), 2);
    key_data.insert(key_data.end(), new_keys.begin(), new_keys.end());
    value_data.insert(value_data.end(), new_values.begin(), new_values.end());
  }
}

INSTANTIATE_TEST_SUITE_P(
    RuntimeFlags, SingleKvBroadcastAttentionTest,
    ::testing::Values(uint32_t{0},
                      uint32_t{XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC}),
    [](const ::testing::TestParamInfo<uint32_t>& info) {
      return info.param == 0 ? "Default" : "ConsistentArithmetic";
    });

}  // namespace
}  // namespace litert::tensor::examples::gemma4
