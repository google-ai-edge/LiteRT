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
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
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

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

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

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

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

TEST(ReferenceEvaluatorTest, RunCompositeSwiglu) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf in = litert::tensor::Create(
      "gate_up", litert::tensor::ApiType<float>::value, {1, 1, 4});

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_swiglu"},
      [](auto x) {
        auto gate = litert::tensor::Slice(x, {0, 0, 0}, {1, 1, 2});
        auto up = litert::tensor::Slice(x, {0, 0, 2}, {1, 1, 2});
        auto silu_gate =
            litert::tensor::Mul(gate, litert::tensor::Logistic(gate));
        return litert::tensor::Mul(silu_gate, up);
      },
      in);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b_in,
                              SimpleBuffer::Create<float>({1, 1, 4}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out,
                              SimpleBuffer::Create<float>({1, 1, 2}));

  auto in_span = b_in.Span<float>();
  in_span[0] = 0.0f;
  in_span[1] = 2.0f;
  in_span[2] = 3.0f;
  in_span[3] = 0.5f;

  VarBuffers inputs;
  inputs.push_back(std::move(b_in));
  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

  auto out_span = outputs[0].Span<float>();
  float expected_1 = (2.0f / (1.0f + std::exp(-2.0f))) * 0.5f;
  std::vector<float> expected = {0.0f, expected_1};
  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-5f), expected));
}

TEST(ReferenceEvaluatorTest, RunCompositeSelectV2WithBoolMask) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf scores = litert::tensor::Create(
      "scores", litert::tensor::ApiType<float>::value, {1, 2, 2, 2});
  TensorTf mask =
      litert::tensor::Create("mask", litert::tensor::Type::kBOOL, {1, 1, 2, 2});

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_bool_select"},
      [](auto s, auto m) {
        TensorTf neg_inf = litert::tensor::Create(
            "neg_inf", litert::tensor::ApiType<float>::value,
            /*shape=*/{1, 1, 1, 1},
            litert::tensor::OwningCpuBuffer::CopyAs(
                litert::tensor::ApiType<float>::value,
                std::vector<float>{-10000.0f}));
        return litert::tensor::SelectV2(m, s, neg_inf);
      },
      scores, mask);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b_scores,
                              SimpleBuffer::Create<float>({1, 2, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_mask,
                              SimpleBuffer::Create<bool>({1, 1, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out,
                              SimpleBuffer::Create<float>({1, 2, 2, 2}));

  auto scores_span = b_scores.Span<float>();
  for (size_t i = 0; i < 8; ++i) {
    scores_span[i] = static_cast<float>(i + 1);
  }
  auto mask_span = b_mask.Span<bool>();
  mask_span[0] = true;
  mask_span[1] = false;
  mask_span[2] = false;
  mask_span[3] = true;

  VarBuffers inputs;
  inputs.push_back(std::move(b_scores));
  inputs.push_back(std::move(b_mask));
  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

  std::vector<float> expected = {1.0f, -10000.0f, -10000.0f, 4.0f,
                                 5.0f, -10000.0f, -10000.0f, 8.0f};
  EXPECT_THAT(outputs[0].Span<float>(), Pointwise(FloatNear(1e-5f), expected));
}

TEST(ReferenceEvaluatorTest, RunCompositeWithConstantWeightInput) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf x = litert::tensor::Create(
      "x", litert::tensor::ApiType<float>::value, {1, 2, 2});
  TensorTf scale = litert::tensor::Create(
      "scale", litert::tensor::ApiType<float>::value, {2},
      litert::tensor::OwningCpuBuffer::CopyAs(
          litert::tensor::ApiType<float>::value,
          std::vector<float>{2.0f, 3.0f}));

  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_const_scale"},
      [](auto x_in, auto scale_in) {
        return litert::tensor::Mul(x_in, scale_in);
      },
      x, scale);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b_x, SimpleBuffer::Create<float>({1, 2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out,
                              SimpleBuffer::Create<float>({1, 2, 2}));

  auto x_span = b_x.Span<float>();
  x_span[0] = 1.0f;
  x_span[1] = 2.0f;
  x_span[2] = 3.0f;
  x_span[3] = 4.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(b_x));
  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

  std::vector<float> expected = {2.0f, 6.0f, 6.0f, 12.0f};
  EXPECT_THAT(outputs[0].Span<float>(), Pointwise(FloatNear(1e-5f), expected));
}

TEST(ReferenceEvaluatorTest, RunChainedComposites) {
  using TensorTf = litert::tensor::Tensor<litert::tensor::TfLiteMixinTag>;

  TensorTf x =
      litert::tensor::Create("x", litert::tensor::ApiType<float>::value, {2});
  TensorTf y =
      litert::tensor::Create("y", litert::tensor::ApiType<float>::value, {2});
  TensorTf z =
      litert::tensor::Create("z", litert::tensor::ApiType<float>::value, {2});

  // First composite produces two intermediate tensors: (x + y, x - y).
  auto [sum_xy, diff_xy] = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_stage1"},
      [](auto x_in, auto y_in) {
        return std::make_tuple(litert::tensor::Add(x_in, y_in),
                               litert::tensor::Sub(x_in, y_in));
      },
      x, y);

  // Second composite consumes both outputs of stage1 plus z:
  // (sum_xy * diff_xy) + z.
  TensorTf out = litert::tensor::StableHLOComposite(
      litert::tensor::StableHLOCompositeOptions{.name = "test_stage2"},
      [](auto a_in, auto b_in, auto z_in) {
        return litert::tensor::Add(litert::tensor::Mul(a_in, b_in), z_in);
      },
      sum_xy, diff_xy, z);

  LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                              litert::testing::SaveTensorGraph({out}));

  LITERT_ASSERT_OK_AND_ASSIGN(auto b_x, SimpleBuffer::Create<float>({2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_y, SimpleBuffer::Create<float>({2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_z, SimpleBuffer::Create<float>({2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto b_out, SimpleBuffer::Create<float>({2}));

  b_x.Span<float>()[0] = 5.0f;
  b_x.Span<float>()[1] = 4.0f;
  b_y.Span<float>()[0] = 3.0f;
  b_y.Span<float>()[1] = 1.0f;
  b_z.Span<float>()[0] = 2.0f;
  b_z.Span<float>()[1] = 10.0f;

  VarBuffers inputs;
  inputs.push_back(std::move(b_x));
  inputs.push_back(std::move(b_y));
  inputs.push_back(std::move(b_z));
  VarBuffers outputs;
  outputs.push_back(std::move(b_out));

  LITERT_ASSERT_OK(ReferenceEvaluator::Evaluate(*model, inputs, outputs));

  // Element 0: (5 + 3) * (5 - 3) + 2 = 16 + 2 = 18
  // Element 1: (4 + 1) * (4 - 1) + 10 = 15 + 10 = 25
  std::vector<float> expected = {18.0f, 25.0f};
  EXPECT_THAT(outputs[0].Span<float>(), Pointwise(FloatNear(1e-5f), expected));
}

}  // namespace
}  // namespace litert::testing
