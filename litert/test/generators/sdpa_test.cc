// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/test/generators/sdpa.h"

#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"
#include "tflite/types/half.h"

namespace litert::testing {
namespace {

class SdpaTest : public RngTest {};

TEST_F(SdpaTest, BuildGraphMha) {
  using SdpaGen = Sdpa<float, std::false_type, std::false_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 2;
  params.num_kv_heads = 2;
  params.seq_q = 4;
  params.seq_k = 4;
  params.head_dim = 16;
  params.scale = 0.25f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));
  // StableHLOComposite creates main graph + decomposed fallback subgraph.
  EXPECT_EQ(gen->Graph().NumSubgraphs(), 2);
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(sg.Inputs().size(), 3);  // Q, K, V
  EXPECT_EQ(sg.Outputs().size(), 1);
}

TEST_F(SdpaTest, BuildGraphWithMaskAndSoftcap) {
  using SdpaGen = Sdpa<float, std::true_type, std::true_type>;
  SdpaGen::Params params;
  params.batch = 2;
  params.num_q_heads = 4;
  params.num_kv_heads = 2;  // GQA (4:2)
  params.seq_q = 8;
  params.seq_k = 8;
  params.head_dim = 32;
  params.scale = 0.125f;
  params.softcap_val = 30.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));
  EXPECT_EQ(gen->Graph().NumSubgraphs(), 2);
  const auto& sg = gen->Graph().Subgraph(0);
  EXPECT_EQ(sg.Inputs().size(), 4);  // Q, K, V, Mask
  EXPECT_EQ(sg.Outputs().size(), 1);
}

TEST_F(SdpaTest, ReferenceSimpleIdentityAttention) {
  // 1 batch, 1 head, 1 query, 1 key, 2 head_dim
  // Q = [1, 0], K = [1, 0], V = [5, 10]
  // QK^T = 1.0 * scale(1.0) = 1.0
  // softmax([1.0]) = [1.0]
  // Output = 1.0 * V = [5, 10]
  using SdpaGen = Sdpa<float, std::false_type, std::false_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 1;
  params.num_kv_heads = 1;
  params.seq_q = 1;
  params.seq_k = 1;
  params.head_dim = 2;
  params.scale = 1.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q, SimpleBuffer::Create<float>({1, 1, 1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k, SimpleBuffer::Create<float>({1, 1, 1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v, SimpleBuffer::Create<float>({1, 1, 1, 2}));

  q.Span<float>()[0] = 1.0f;
  q.Span<float>()[1] = 0.0f;
  k.Span<float>()[0] = 1.0f;
  k.Span<float>()[1] = 0.0f;
  v.Span<float>()[0] = 5.0f;
  v.Span<float>()[1] = 10.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(q));
  inputs.push_back(std::move(k));
  inputs.push_back(std::move(v));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out, SimpleBuffer::Create<float>({1, 1, 1, 2}));
  VarBuffers outputs;
  outputs.push_back(std::move(out));

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  EXPECT_NEAR(out_span[0], 5.0f, 1e-4);
  EXPECT_NEAR(out_span[1], 10.0f, 1e-4);
}

TEST_F(SdpaTest, ReferenceGqaBroadcast) {
  // GQA: 2 Q heads, 1 KV head (MQA style).
  // Q head 0 and Q head 1 both attend to the single KV head.
  // Q = [[1, 0], [0, 1]] (shape: [1, 1, 2, 2])
  // K = [1, 0] (shape: [1, 1, 1, 2])
  // V = [3, 7] (shape: [1, 1, 1, 2])
  // Both heads produce output = [3, 7]
  using SdpaGen = Sdpa<float, std::false_type, std::false_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 2;
  params.num_kv_heads = 1;
  params.seq_q = 1;
  params.seq_k = 1;
  params.head_dim = 2;
  params.scale = 1.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q, SimpleBuffer::Create<float>({1, 1, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k, SimpleBuffer::Create<float>({1, 1, 1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v, SimpleBuffer::Create<float>({1, 1, 1, 2}));

  // Q: head 0 = [1, 0], head 1 = [0, 1]
  q.Span<float>()[0] = 1.0f;
  q.Span<float>()[1] = 0.0f;
  q.Span<float>()[2] = 0.0f;
  q.Span<float>()[3] = 1.0f;

  k.Span<float>()[0] = 1.0f;
  k.Span<float>()[1] = 0.0f;

  v.Span<float>()[0] = 3.0f;
  v.Span<float>()[1] = 7.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(q));
  inputs.push_back(std::move(k));
  inputs.push_back(std::move(v));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out, SimpleBuffer::Create<float>({1, 1, 2, 2}));
  VarBuffers outputs;
  outputs.push_back(std::move(out));

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  // Head 0 output
  EXPECT_NEAR(out_span[0], 3.0f, 1e-4);
  EXPECT_NEAR(out_span[1], 7.0f, 1e-4);
  // Head 1 output
  EXPECT_NEAR(out_span[2], 3.0f, 1e-4);
  EXPECT_NEAR(out_span[3], 7.0f, 1e-4);
}

TEST_F(SdpaTest, ReferenceCausalMask) {
  // 1 batch, 1 head, 2 query tokens, 2 key tokens, 1 head_dim
  // Q = [[1], [1]]
  // K = [[1], [1]]
  // V = [[10], [20]]
  // Mask = [[0, -10000], [0, 0]] (Token 0 cannot see Token 1)
  // For Token 0: scores = [1, 1 - 10000] -> probs = [1.0, 0.0] -> Out = 10
  // For Token 1: scores = [1, 1] -> probs = [0.5, 0.5] ->
  //              Out = 0.5 * 10 + 0.5 * 20 = 15
  using SdpaGen = Sdpa<float, std::true_type, std::false_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 1;
  params.num_kv_heads = 1;
  params.seq_q = 2;
  params.seq_k = 2;
  params.head_dim = 1;
  params.scale = 1.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q, SimpleBuffer::Create<float>({1, 2, 1, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k, SimpleBuffer::Create<float>({1, 2, 1, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v, SimpleBuffer::Create<float>({1, 2, 1, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto mask, SimpleBuffer::Create<float>({1, 1, 2, 2}));

  q.Span<float>()[0] = 1.0f;
  q.Span<float>()[1] = 1.0f;

  k.Span<float>()[0] = 1.0f;
  k.Span<float>()[1] = 1.0f;

  v.Span<float>()[0] = 10.0f;
  v.Span<float>()[1] = 20.0f;

  mask.Span<float>()[0] = 0.0f;
  mask.Span<float>()[1] = -10000.0f;
  mask.Span<float>()[2] = 0.0f;
  mask.Span<float>()[3] = 0.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(q));
  inputs.push_back(std::move(k));
  inputs.push_back(std::move(v));
  inputs.push_back(std::move(mask));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out, SimpleBuffer::Create<float>({1, 2, 1, 1}));
  VarBuffers outputs;
  outputs.push_back(std::move(out));

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  EXPECT_NEAR(out_span[0], 10.0f, 1e-4);
  EXPECT_NEAR(out_span[1], 15.0f, 1e-4);
}

TEST_F(SdpaTest, ReferenceSoftcap) {
  // Soft-capping: score' = cap * tanh(score / cap)
  // Let Q = [10], K = [10], scale = 1.0 -> dot = 100.0
  // cap = 2.0 -> scaled score = 2.0 * tanh(100.0 / 2.0) ~= 2.0
  using SdpaGen = Sdpa<float, std::false_type, std::true_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 1;
  params.num_kv_heads = 1;
  params.seq_q = 1;
  params.seq_k = 1;
  params.head_dim = 1;
  params.scale = 1.0f;
  params.softcap_val = 2.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q, SimpleBuffer::Create<float>({1, 1, 1, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k, SimpleBuffer::Create<float>({1, 1, 1, 1}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v, SimpleBuffer::Create<float>({1, 1, 1, 1}));

  q.Span<float>()[0] = 10.0f;
  k.Span<float>()[0] = 10.0f;
  v.Span<float>()[0] = 42.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(q));
  inputs.push_back(std::move(k));
  inputs.push_back(std::move(v));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out, SimpleBuffer::Create<float>({1, 1, 1, 1}));
  VarBuffers outputs;
  outputs.push_back(std::move(out));

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  EXPECT_NEAR(out_span[0], 42.0f, 1e-4);
}

TEST_F(SdpaTest, ReferenceFp16Half) {
  using SdpaGen = Sdpa<tflite::half, std::false_type, std::false_type>;
  SdpaGen::Params params;
  params.batch = 1;
  params.num_q_heads = 1;
  params.num_kv_heads = 1;
  params.seq_q = 1;
  params.seq_k = 1;
  params.head_dim = 2;
  params.scale = 1.0f;

  LITERT_ASSERT_OK_AND_ASSIGN(auto gen, SdpaGen::Create(params));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q, SimpleBuffer::Create<tflite::half>({1, 1, 1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k, SimpleBuffer::Create<tflite::half>({1, 1, 1, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v, SimpleBuffer::Create<tflite::half>({1, 1, 1, 2}));

  q.Span<tflite::half>()[0] = tflite::half(1.0f);
  q.Span<tflite::half>()[1] = tflite::half(0.0f);
  k.Span<tflite::half>()[0] = tflite::half(1.0f);
  k.Span<tflite::half>()[1] = tflite::half(0.0f);
  v.Span<tflite::half>()[0] = tflite::half(8.0f);
  v.Span<tflite::half>()[1] = tflite::half(16.0f);

  VarBuffers inputs;
  inputs.push_back(std::move(q));
  inputs.push_back(std::move(k));
  inputs.push_back(std::move(v));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out, SimpleBuffer::Create<tflite::half>({1, 1, 1, 2}));
  VarBuffers outputs;
  outputs.push_back(std::move(out));

  LITERT_ASSERT_OK(gen->Reference(inputs, outputs));

  auto out_span = outputs[0].Span<tflite::half>();
  EXPECT_NEAR(static_cast<float>(out_span[0]), 8.0f, 1e-2);
  EXPECT_NEAR(static_cast<float>(out_span[1]), 16.0f, 1e-2);
}

}  // namespace
}  // namespace litert::testing
