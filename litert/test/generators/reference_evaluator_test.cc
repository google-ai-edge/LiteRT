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

#include "litert/test/generators/reference_evaluator.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::testing {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(ReferenceEvaluatorTest, RunCompositeArithmetic) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf in1 = litert::tensor::Create(
      "in1", litert::tensor::ApiType<float>::value, {2, 3});
  TensorTf in2 = litert::tensor::Create(
      "in2", litert::tensor::ApiType<float>::value, {2, 3});

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_composite"},
      [](auto x, auto y) {
        auto added = litert::tensor::Add(x, y);
        return litert::tensor::Mul(added, x);
      },
      in1, in2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b1, SimpleBuffer::Create<float>({2, 3}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b2, SimpleBuffer::Create<float>({2, 3}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out, SimpleBuffer::Create<float>({2, 3}));

  auto s1 = b1.Span<float>();
  auto s2 = b2.Span<float>();
  for (size_t i = 0; i < 6; ++i) {
    s1[i] = static_cast<float>(i + 1);
    s2[i] = 2.0f;
  }

  VarBuffers inputs;
  inputs.push_back(std::move(b1));
  inputs.push_back(std::move(b2));

  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(
      ReferenceEvaluator::EvaluateCompositeReference(*model, inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  // out = (x + 2) * x
  // For x = 1: (1 + 2) * 1 = 3
  // For x = 2: (2 + 2) * 2 = 8
  // For x = 3: (3 + 2) * 3 = 15
  // For x = 4: (4 + 2) * 4 = 24
  // For x = 5: (5 + 2) * 5 = 35
  // For x = 6: (6 + 2) * 6 = 48
  std::vector<float> expected = {3.0f, 8.0f, 15.0f, 24.0f, 35.0f, 48.0f};
  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-5f), expected));
}

TEST(ReferenceEvaluatorTest, RunCompositeBatchMatmulAndSoftmax) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf q = litert::tensor::Create(
      "q", litert::tensor::ApiType<float>::value, {1, 1, 2, 2});
  TensorTf k = litert::tensor::Create(
      "k", litert::tensor::ApiType<float>::value, {1, 1, 2, 2});

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_attention"},
      [](auto q_in, auto k_in) {
        auto qk = litert::tensor::BatchMatMul(q_in, k_in, /*adj_x=*/false,
                                              /*adj_y=*/true);
        return litert::tensor::Softmax(qk, /*beta=*/1.0f);
      },
      q, k);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b_q,
                              SimpleBuffer::Create<float>({1, 1, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_k,
                              SimpleBuffer::Create<float>({1, 1, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out,
                              SimpleBuffer::Create<float>({1, 1, 2, 2}));

  // Q = [[1, 0], [0, 1]]
  // K = [[1, 0], [0, 1]]
  auto sq = b_q.Span<float>();
  sq[0] = 1.0f;
  sq[1] = 0.0f;
  sq[2] = 0.0f;
  sq[3] = 1.0f;

  auto sk = b_k.Span<float>();
  sk[0] = 1.0f;
  sk[1] = 0.0f;
  sk[2] = 0.0f;
  sk[3] = 1.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(b_q));
  inputs.push_back(std::move(b_k));

  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(
      ReferenceEvaluator::EvaluateCompositeReference(*model, inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  // Q * K^T = [[1, 0], [0, 1]]
  // Softmax on each row:
  // row 0: exp(1)/(exp(1)+exp(0)), exp(0)/(exp(1)+exp(0)) = [0.731058,
  // 0.268941] row 1: exp(0)/(exp(1)+exp(0)), exp(1)/(exp(1)+exp(0)) =
  // [0.268941, 0.731058]
  float e1 = std::exp(1.0f);
  float e0 = 1.0f;
  float p1 = e1 / (e1 + e0);
  float p0 = e0 / (e1 + e0);

  std::vector<float> expected = {p1, p0, p0, p1};
  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-5f), expected));
}

TEST(ReferenceEvaluatorTest, CustomOpRegistration) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf in1 =
      litert::tensor::Create("in1", litert::tensor::ApiType<float>::value, {2});
  TensorTf in2 =
      litert::tensor::Create("in2", litert::tensor::ApiType<float>::value, {2});

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "custom_composite"},
      [](auto x, auto y) { return litert::tensor::Add(x, y); }, in1, in2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b1, SimpleBuffer::Create<float>({2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b2, SimpleBuffer::Create<float>({2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out, SimpleBuffer::Create<float>({2}));

  b1.Span<float>()[0] = 10.0f;
  b1.Span<float>()[1] = 20.0f;
  b2.Span<float>()[0] = 3.0f;
  b2.Span<float>()[1] = 4.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(b1));
  inputs.push_back(std::move(b2));
  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  ReferenceEvaluator custom_evaluator;
  // Override Add with custom scaled addition: (a + b) * 2.0f
  custom_evaluator.RegisterOp(
      kLiteRtOpCodeTflAdd,
      [](const LiteRtOpT& op, const ReferenceEvaluator::TensorEnv& env,
         ReferenceEvaluator::TensorData& out) -> Expected<void> {
        const auto& in1 = env.at(op.Inputs()[0]);
        const auto& in2 = env.at(op.Inputs()[1]);
        for (size_t i = 0; i < out.f32_data.size(); ++i) {
          out.f32_data[i] = (in1.f32_data[i] + in2.f32_data[i]) * 2.0f;
        }
        return {};
      });

  LITERT_ASSERT_OK(custom_evaluator.EvaluateComposite(*model, inputs, outputs));

  EXPECT_EQ(outputs[0].Span<float>()[0], 26.0f);
  EXPECT_EQ(outputs[0].Span<float>()[1], 48.0f);
}

}  // namespace
}  // namespace litert::testing
