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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_NNPACK_COMMON_RUNNER_TEST_SUITE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_NNPACK_COMMON_RUNNER_TEST_SUITE_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {

template <typename FixtureTraits>
class NnpackRunnerTest : public ::testing::Test {
 public:
  using Traits = FixtureTraits;
  using Tag = typename FixtureTraits::Tag;
  using Runner = typename FixtureTraits::Runner;
  using TensorType = Tensor<Tag>;
};

TYPED_TEST_SUITE_P(NnpackRunnerTest);

TYPED_TEST_P(NnpackRunnerTest, SetInputRejectsNonExternalTensors) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType lhs({.name = "lhs",
                  .type = Type::kFP32,
                  .shape = {2},
                  .buffer = std::vector<float>{1.f, 2.f}});
  TensorType rhs({.name = "rhs",
                  .type = Type::kFP32,
                  .shape = {2},
                  .buffer = std::vector<float>{3.f, 4.f}});
  TensorType output = Add(lhs, rhs);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, Runner::Create({output}));
  const std::array<float, 2> data = {0.f, 0.f};
  absl::Span<const std::byte> bytes(
      reinterpret_cast<const std::byte*>(data.data()), sizeof(data));
  absl::Status status = runner.SetInput(lhs, bytes);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConstantAdd) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a(TensorInit{.name = "a",
                          .type = Type::kFP32,
                          .shape = {2},
                          .buffer = std::vector<float>{1.f, 2.f}});
  TensorType b(TensorInit{.name = "b",
                          .type = Type::kFP32,
                          .shape = {2},
                          .buffer = std::vector<float>{3.f, 4.f}});
  TensorType c = Add(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 4.f);
  EXPECT_FLOAT_EQ(values[1], 6.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesRuntimeInputAdd) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType runtime_input(
      {.name = "input", .type = Type::kFP32, .shape = {2}});
  TensorType bias({.name = "bias",
                   .type = Type::kFP32,
                   .shape = {2},
                   .buffer = std::vector<float>{0.5f, 0.5f}});
  TensorType sum = Add(runtime_input, bias);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, Runner::Create({sum}));
  const std::array<float, 2> host = {10.f, 20.f};
  ASSERT_THAT(
      runner.SetInput(
          runtime_input,
          absl::Span<const std::byte>(
              reinterpret_cast<const std::byte*>(host.data()), sizeof(host))),
      IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto result, runner.ReadOutput(sum));
  auto floats = std::move(result).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 10.5f);
  EXPECT_FLOAT_EQ(values[1], 20.5f);
}

TYPED_TEST_P(NnpackRunnerTest, MoveConstructorTransfersRuntime) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{1.f, 2.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{3.f, 4.f}});
  TensorType c = Add(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());

  Runner moved_runner = std::move(runner);
  ASSERT_THAT(moved_runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, moved_runner.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 4.f);
  EXPECT_FLOAT_EQ(values[1], 6.f);
}

TYPED_TEST_P(NnpackRunnerTest, MoveAssignmentTransfersRuntime) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{1.f, 2.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{3.f, 4.f}});
  TensorType c = Add(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner_a, Runner::Create({c}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner_b, Runner::Create({c}));

  ASSERT_THAT(runner_a.Run(), IsOk());
  runner_b = std::move(runner_a);

  ASSERT_THAT(runner_b.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner_b.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 4.f);
  EXPECT_FLOAT_EQ(values[1], 6.f);
}

TYPED_TEST_P(NnpackRunnerTest, ConstantsAreNotBoundAsExternals) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;
  using BackendTraits = typename Runner::BackendTraits;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{1.f, 2.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{3.f, 4.f}});
  TensorType c = Add(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());

  // Only the external output should be bound; constants are internal.
  int externals = 0;
  for (const auto& value : runner.graph().values()) {
    if (value.flags & BackendTraits::kFlagExternalOutput) {
      externals++;
    }
    // Constant inputs should not be external.
    EXPECT_FALSE(value.flags & BackendTraits::kFlagExternalInput);
  }
  EXPECT_EQ(externals, 1);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConstantMul) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{2.f, -1.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{4.f, 0.5f}});
  TensorType c = Mul(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 8.f);
  EXPECT_FLOAT_EQ(values[1], -0.5f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConstantSub) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{5.f, 1.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{3.f, 4.f}});
  TensorType c = Sub(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 2.f);
  EXPECT_FLOAT_EQ(values[1], -3.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConstantDiv) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{8.f, -6.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {2},
                .buffer = std::vector<float>{2.f, 3.f}});
  TensorType c = Div(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({c}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(c));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 4.f);
  EXPECT_FLOAT_EQ(values[1], -2.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesMaximumAndMinimum) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a({.name = "a",
                .type = Type::kFP32,
                .shape = {3},
                .buffer = std::vector<float>{-1.f, 2.f, 5.f}});
  TensorType b({.name = "b",
                .type = Type::kFP32,
                .shape = {3},
                .buffer = std::vector<float>{0.f, 4.f, 1.f}});
  TensorType max_out = Maximum(a, b);
  TensorType min_out = Minimum(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({max_out, min_out}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto max_bytes, runner.ReadOutput(max_out));
  auto max_vals = std::move(max_bytes).template As<const float>();
  ASSERT_EQ(max_vals.size(), 3);
  EXPECT_FLOAT_EQ(max_vals.data()[0], 0.f);
  EXPECT_FLOAT_EQ(max_vals.data()[1], 4.f);
  EXPECT_FLOAT_EQ(max_vals.data()[2], 5.f);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto min_bytes, runner.ReadOutput(min_out));
  auto min_vals = std::move(min_bytes).template As<const float>();
  ASSERT_EQ(min_vals.size(), 3);
  EXPECT_FLOAT_EQ(min_vals.data()[0], -1.f);
  EXPECT_FLOAT_EQ(min_vals.data()[1], 2.f);
  EXPECT_FLOAT_EQ(min_vals.data()[2], 1.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesPow) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType base({.name = "base",
                   .type = Type::kFP32,
                   .shape = {2},
                   .buffer = std::vector<float>{2.f, 9.f}});
  TensorType exp({.name = "exp",
                  .type = Type::kFP32,
                  .shape = {2},
                  .buffer = std::vector<float>{3.f, 0.5f}});
  TensorType out = Pow(base, exp);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({out}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(out));
  auto vals = std::move(bytes).template As<const float>();
  ASSERT_EQ(vals.size(), 2);
  EXPECT_FLOAT_EQ(vals.data()[0], 8.f);
  EXPECT_FLOAT_EQ(vals.data()[1], 3.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesAbs) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {3},
                    .buffer = std::vector<float>{-3.f, 0.f, 5.f}});
  TensorType output = Abs(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 3);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 3.f);
  EXPECT_FLOAT_EQ(values[1], 0.f);
  EXPECT_FLOAT_EQ(values[2], 5.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSquare) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {3},
                    .buffer = std::vector<float>{-3.f, 2.f, 0.5f}});
  TensorType output = Square(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 3);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 9.f);
  EXPECT_FLOAT_EQ(values[1], 4.f);
  EXPECT_FLOAT_EQ(values[2], 0.25f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesRsqrt) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{4.f, 9.f}});
  TensorType output = Rsqrt(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 0.5f);
  EXPECT_FLOAT_EQ(values[1], 1.f / 3.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSqrt) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{4.f, 2.25f}});
  TensorType output = Sqrt(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 2.f);
  EXPECT_FLOAT_EQ(values[1], 1.5f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesNeg) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {3},
                    .buffer = std::vector<float>{-2.f, 0.f, 7.f}});
  TensorType output = Neg(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 3);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 2.f);
  EXPECT_FLOAT_EQ(values[1], -0.f);
  EXPECT_FLOAT_EQ(values[2], -7.f);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesTanh) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{0.f, 1.f}});
  TensorType output = Tanh(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_NEAR(values[0], 0.f, 1e-6);
  EXPECT_NEAR(values[1], 0.7615942f, 1e-6);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSigmoid) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{0.f, 1.f}});
  TensorType output = Logistic(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_NEAR(values[0], 0.5f, 1e-6);
  EXPECT_NEAR(values[1], 1.f / (1.f + std::exp(-1.f)), 1e-6);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesCos) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2},
       .buffer = std::vector<float>{0.f, static_cast<float>(M_PI)}});
  TensorType output = Cos(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_NEAR(values[0], 1.f, 1e-6);
  EXPECT_NEAR(values[1], -1.f, 1e-6);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSin) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2},
       .buffer = std::vector<float>{0.f, static_cast<float>(M_PI / 2)}});
  TensorType output = Sin(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_NEAR(values[0], 0.f, 1e-6);
  EXPECT_NEAR(values[1], 1.f, 1e-6);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesGelu) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1},
                    .buffer = std::vector<float>{1.f}});
  TensorType output = Gelu(input, /*approximate=*/false);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 1);
  const float value = floats.data()[0];
  // Expected exact GELU(1) ≈ 0.8413447
  EXPECT_NEAR(value, 0.8413447f, 1e-5);
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSoftmax) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{0.f, 0.f}});
  TensorType output = Softmax(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 2);
  const float* values = floats.data();
  EXPECT_NEAR(values[0], 0.5f, 1e-5);
  EXPECT_NEAR(values[1], 0.5f, 1e-5);
}

TYPED_TEST_P(NnpackRunnerTest, SoftmaxRejectsBetaNotEqualToOne) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {2},
                    .buffer = std::vector<float>{0.f, 0.f}});
  TensorType output = Softmax(input, /*beta=*/2.0f);

  auto runner = Runner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConv2D) {
  if constexpr (!TestFixture::Traits::kSupportsConv2D) {
    GTEST_SKIP() << "Conv2D not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 5, 5, 1},
                      .buffer = std::vector<float>{
                          1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,
                          10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f,
                          19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f}});
    TensorType filter({.name = "filter",
                       .type = Type::kFP32,
                       .shape = {1, 2, 2, 1},
                       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});
    TensorType bias({.name = "bias",
                     .type = Type::kFP32,
                     .shape = {1},
                     .buffer = std::vector<float>{1.f}});
    TensorType output =
        Conv2D(input, filter, bias, /*stride_h=*/2, /*stride_w=*/2,
               /*padding=*/kPaddingValid, /*dilation_h_factor=*/2,
               /*dilation_w_factor=*/2, /*activation=*/kActNone);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 4);
    EXPECT_FLOAT_EQ(floats.data()[0], 93.f);
    EXPECT_FLOAT_EQ(floats.data()[1], 113.f);
    EXPECT_FLOAT_EQ(floats.data()[2], 193.f);
    EXPECT_FLOAT_EQ(floats.data()[3], 213.f);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ComputesDepthwiseConv2D) {
  if constexpr (!TestFixture::Traits::kSupportsDepthwiseConv2D) {
    GTEST_SKIP() << "DepthwiseConv2D not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 5, 5, 1},
                      .buffer = std::vector<float>{
                          1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,
                          10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f,
                          19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f}});
    TensorType filter({.name = "filter",
                       .type = Type::kFP32,
                       .shape = {1, 2, 2, 1},
                       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});
    TensorType bias({.name = "bias",
                     .type = Type::kFP32,
                     .shape = {1},
                     .buffer = std::vector<float>{1.f}});
    TensorType output =
        DepthwiseConv2D(input, filter, bias, /*stride_h=*/2, /*stride_w=*/2,
                        /*padding=*/kPaddingValid, /*dilation_h_factor=*/2,
                        /*dilation_w_factor=*/2, /*depth_multiplier=*/1,
                        /*activation=*/kActNone);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 4);
    EXPECT_FLOAT_EQ(floats.data()[0], 93.f);
    EXPECT_FLOAT_EQ(floats.data()[1], 113.f);
    EXPECT_FLOAT_EQ(floats.data()[2], 193.f);
    EXPECT_FLOAT_EQ(floats.data()[3], 213.f);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ComputesFullyConnected) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType weights(
      {.name = "weights",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType bias({.name = "bias",
                   .type = Type::kFP32,
                   .shape = {2},
                   .buffer = std::vector<float>{0.5f, 1.5f}});
  TensorType output = FullyConnected(input, weights, bias);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, Runner::Create({output}));
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
  auto floats = std::move(bytes).template As<const float>();
  ASSERT_EQ(floats.size(), 4);
  const float* values = floats.data();
  EXPECT_FLOAT_EQ(values[0], 14.5f);
  EXPECT_FLOAT_EQ(values[1], 33.5f);
  EXPECT_FLOAT_EQ(values[2], 32.5f);
  EXPECT_FLOAT_EQ(values[3], 78.5f);
}

TYPED_TEST_P(NnpackRunnerTest, BatchMatMulSupportsTransposeFlags) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType a1({.name = "a1",
                 .type = Type::kFP32,
                 .shape = {1, 2, 3},
                 .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType b1({.name = "b1",
                 .type = Type::kFP32,
                 .shape = {1, 4, 3},
                 .buffer = std::vector<float>{1.f, 0.f, 1.f, 0.f, 1.f, 1.f, 1.f,
                                              1.f, 0.f, 2.f, 1.f, -1.f}});
  TensorType out1 = BatchMatMul(a1, b1, /*adj_x=*/false, /*adj_y=*/true);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner1, Runner::Create({out1}));
  ASSERT_THAT(runner1.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out1_bytes, runner1.ReadOutput(out1));
  auto out1_vals = std::move(out1_bytes).template As<const float>();
  EXPECT_THAT(out1_vals, ::testing::ElementsAreArray(
                             {4.f, 5.f, 3.f, 1.f, 10.f, 11.f, 9.f, 7.f}));

  TensorType a2({.name = "a2",
                 .type = Type::kFP32,
                 .shape = {1, 2, 3},
                 .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType b2({.name = "b2",
                 .type = Type::kFP32,
                 .shape = {1, 3, 2},
                 .buffer = std::vector<float>{1.f, 0.f, 1.f, 1.f, 2.f, 1.f}});
  TensorType out2 = BatchMatMul(a2, b2);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({out2}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto out2_bytes, runner.ReadOutput(out2));
  auto out2_vals = std::move(out2_bytes).template As<const float>();
  EXPECT_THAT(out2_vals, ::testing::ElementsAreArray({9.f, 5.f, 21.f, 11.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesTranspose) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});

  TensorType transposed = Transpose(input, {1, 0});
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({transposed}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto t_bytes, runner.ReadOutput(transposed));
  auto t_vals = std::move(t_bytes).template As<const float>();
  EXPECT_THAT(t_vals,
              ::testing::ElementsAreArray({1.f, 4.f, 2.f, 5.f, 3.f, 6.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesMeanKeepDimsAndSqueeze) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});

  TensorType mean_keep = Mean(input, /*axes=*/{1}, /*keep_dims=*/true);
  TensorType mean_squeeze = Mean(input, /*axes=*/{1}, /*keep_dims=*/false);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({mean_keep, mean_squeeze}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto mk_bytes, runner.ReadOutput(mean_keep));
  auto mk_vals = std::move(mk_bytes).template As<const float>();
  EXPECT_THAT(mk_vals, ::testing::ElementsAreArray({2.f, 5.f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto ms_bytes,
                                  runner.ReadOutput(mean_squeeze));
  auto ms_vals = std::move(ms_bytes).template As<const float>();
  EXPECT_THAT(ms_vals, ::testing::ElementsAreArray({2.f, 5.f}));
}

TYPED_TEST_P(NnpackRunnerTest, TransposeAndMeanCombined) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input(
      {.name = "input",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});

  TensorType transposed = Transpose(input, {1, 0});
  TensorType mean_on_transposed =
      Mean(transposed, /*axes=*/{1}, /*keep_dims=*/false);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      Runner runner, Runner::Create({transposed, mean_on_transposed}));

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto t_bytes, runner.ReadOutput(transposed));
  auto t_vals = std::move(t_bytes).template As<const float>();
  EXPECT_THAT(t_vals,
              ::testing::ElementsAreArray({1.f, 4.f, 2.f, 5.f, 3.f, 6.f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto m_bytes,
                                  runner.ReadOutput(mean_on_transposed));
  auto m_vals = std::move(m_bytes).template As<const float>();
  EXPECT_THAT(m_vals, ::testing::ElementsAreArray({2.5f, 3.5f, 4.5f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesSlice) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType base({.name = "base",
                   .type = Type::kFP32,
                   .shape = {2, 3},
                   .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType slice = Slice(base, /*begin=*/{0, 1}, /*size=*/{2, 2});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({slice}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto slice_bytes, runner.ReadOutput(slice));
  auto slice_vals = std::move(slice_bytes).template As<const float>();
  EXPECT_THAT(slice_vals, ::testing::ElementsAreArray({2.f, 3.f, 5.f, 6.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesConcatenation) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType left({.name = "left",
                   .type = Type::kFP32,
                   .shape = {2, 2},
                   .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});
  TensorType right({.name = "right",
                    .type = Type::kFP32,
                    .shape = {2, 2},
                    .buffer = std::vector<float>{10.f, 20.f, 30.f, 40.f}});

  TensorType concatenated =
      Concatenation({left, right}, /*axis=*/1, FusedActivation::kActNone);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({concatenated}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto concat_bytes,
                                  runner.ReadOutput(concatenated));
  auto concat_vals = std::move(concat_bytes).template As<const float>();
  EXPECT_THAT(concat_vals, ::testing::ElementsAreArray(
                               {1.f, 2.f, 10.f, 20.f, 3.f, 4.f, 30.f, 40.f}));
}

TYPED_TEST_P(NnpackRunnerTest, SliceAndConcatenationCombined) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType base({.name = "base",
                   .type = Type::kFP32,
                   .shape = {2, 3},
                   .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}});
  TensorType slice = Slice(base, /*begin=*/{0, 1}, /*size=*/{2, 2});

  TensorType other({.name = "other",
                    .type = Type::kFP32,
                    .shape = {2, 2},
                    .buffer = std::vector<float>{10.f, 20.f, 30.f, 40.f}});

  TensorType concatenated =
      Concatenation({slice, other}, /*axis=*/1, FusedActivation::kActNone);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({slice, concatenated}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto slice_bytes, runner.ReadOutput(slice));
  auto slice_vals = std::move(slice_bytes).template As<const float>();
  EXPECT_THAT(slice_vals, ::testing::ElementsAreArray({2.f, 3.f, 5.f, 6.f}));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto concat_bytes,
                                  runner.ReadOutput(concatenated));
  auto concat_vals = std::move(concat_bytes).template As<const float>();
  EXPECT_THAT(concat_vals, ::testing::ElementsAreArray(
                               {2.f, 3.f, 10.f, 20.f, 5.f, 6.f, 30.f, 40.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesReshape) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 4},
                    .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  TensorType reshaped = Reshape(input, /*new_shape=*/{2, 2});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({reshaped}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto reshape_bytes,
                                  runner.ReadOutput(reshaped));
  auto reshape_vals = std::move(reshape_bytes).template As<const float>();
  EXPECT_THAT(reshape_vals, ::testing::ElementsAreArray({1.f, 2.f, 3.f, 4.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ComputesTile) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 2},
                    .buffer = std::vector<float>{1.f, 2.f}});

  TensorType tiled = Tile(input, /*multiples=*/{3, 1});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({tiled}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto tile_bytes, runner.ReadOutput(tiled));
  auto tile_vals = std::move(tile_bytes).template As<const float>();
  EXPECT_THAT(tile_vals,
              ::testing::ElementsAreArray({1.f, 2.f, 1.f, 2.f, 1.f, 2.f}));
}

TYPED_TEST_P(NnpackRunnerTest, TilAndReshape) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 2},
                    .buffer = std::vector<float>{1.f, 2.f}});

  TensorType tiled = Tile(input, /*multiples=*/{3, 1});
  TensorType reshaped = Reshape(tiled, /*new_shape=*/{2, 3});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner,
                                  Runner::Create({tiled, reshaped}));
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto reshape_bytes,
                                  runner.ReadOutput(reshaped));
  auto reshape_vals = std::move(reshape_bytes).template As<const float>();
  EXPECT_THAT(reshape_vals,
              ::testing::ElementsAreArray({1.f, 2.f, 1.f, 2.f, 1.f, 2.f}));
}

TYPED_TEST_P(NnpackRunnerTest, ResizeBilinearAlignCorners) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 2x2
    // 1 2
    // 3 4
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Resize to 4x4 with align_corners=true.
    // This should preserve corner values exactly.
    TensorType output = ResizeBilinear(input, {4, 4}, /*align_corners=*/true);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 16);

    // Top-left should remain 1.0
    EXPECT_NEAR(floats.data()[0], 1.f, 1e-5);
    // Top-right should remain 2.0
    EXPECT_NEAR(floats.data()[3], 2.f, 1e-5);
    // Bottom-left should remain 3.0
    EXPECT_NEAR(floats.data()[12], 3.f, 1e-5);
    // Bottom-right should remain 4.0
    EXPECT_NEAR(floats.data()[15], 4.f, 1e-5);

    // Midpoint check (row 0):
    // spacing is (2-1)/(4-1) = 1/3.
    // indices: 0, 0.33, 0.66, 1.0
    // val = 1 + index * (2-1)
    // [0] = 1.0
    // [1] = 1.333
    // [2] = 1.666
    // [3] = 2.0
    EXPECT_NEAR(floats.data()[1], 1.333333f, 1e-4);
    EXPECT_NEAR(floats.data()[2], 1.666667f, 1e-4);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeBilinearHalfPixelCenters) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 2x2
    // 1 2
    // 3 4
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Resize to 4x4 with half_pixel_centers=true.
    TensorType output = ResizeBilinear(input, {4, 4}, /*align_corners=*/false,
                                       /*half_pixel_centers=*/true);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 16);

    // Scale = 2/4 = 0.5.
    // in_coord = (out + 0.5) * scale - 0.5
    // out=0: (0.5)*0.5 - 0.5 = -0.25 -> clamped to 0 -> val 1.0
    // out=1: (1.5)*0.5 - 0.5 = 0.25  -> lerp(1, 2, 0.25) = 1.25
    // out=2: (2.5)*0.5 - 0.5 = 0.75  -> lerp(1, 2, 0.75) = 1.75
    // out=3: (3.5)*0.5 - 0.5 = 1.25  -> clamped to 1 -> val 2.0

    // Row 0 checks:
    EXPECT_NEAR(floats.data()[0], 1.0f, 1e-5);
    EXPECT_NEAR(floats.data()[1], 1.25f, 1e-5);
    EXPECT_NEAR(floats.data()[2], 1.75f, 1e-5);
    EXPECT_NEAR(floats.data()[3], 2.0f, 1e-5);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeBilinearLegacy) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 2x2
    // 10 20
    // 30 40
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{10.f, 20.f, 30.f, 40.f}});

    // Resize to 4x4 with defaults (align_corners=false,
    // half_pixel_centers=false).
    TensorType output = ResizeBilinear(input, {4, 4});

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();

    // Scale = 2/4 = 0.5.
    // in_coord = out * scale
    // out=0: 0.0 -> index 0 -> 10
    // out=1: 0.5 -> lerp(10, 20, 0.5) -> 15
    // out=2: 1.0 -> index 1 -> 20
    // out=3: 1.5 -> index 1.5 -> clamped to 1? or extrapolated?
    // Usually legacy TF behavior for index > max is clamp to boundary.
    // So 1.5 clamps to 1 -> 20.

    EXPECT_NEAR(floats.data()[0], 10.f, 1e-5);
    EXPECT_NEAR(floats.data()[1], 15.f, 1e-5);
    EXPECT_NEAR(floats.data()[2], 20.f, 1e-5);
    EXPECT_NEAR(floats.data()[3], 20.f, 1e-5);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeBilinearDownsample) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 1x4x4x1
    // All 1s to verify basic shape/pipeline, using simple values.
    std::vector<float> input_data(16, 1.0f);
    input_data[0] = 10.0f;  // Top-left
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 4, 4, 1},
                      .buffer = input_data});

    // Downsample to 2x2.
    TensorType output = ResizeBilinear(input, {2, 2}, /*align_corners=*/false,
                                       /*half_pixel_centers=*/true);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 4);

    // Scale = 4/2 = 2.
    // in = (out + 0.5)*2 - 0.5
    // out=0: 0.5 -> index 0.5.
    // Sampling at (0.5, 0.5).
    // Top-left 2x2 block is [[10, 1], [1, 1]].
    // Bilinear interpolation at center gives average: (10+1+1+1)/4 = 3.25.
    EXPECT_NEAR(floats.data()[0], 3.25f, 1e-5);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeBilinearBatchAndChannels) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 2 images, 2x2 size, 2 channels.
    // Batch 0:
    //  [1, 2]   [3, 4]
    //  [5, 6]   [7, 8]
    // Batch 1: All 10s.
    std::vector<float> data = {1,  2,  3,  4,  5,  6,  7,  8,
                               10, 10, 10, 10, 10, 10, 10, 10};
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {2, 2, 2, 2},  // BHWC
                      .buffer = data});

    // Resize to 4x4.
    TensorType output = ResizeBilinear(input, {4, 4});

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 2 * 4 * 4 * 2);  // 64 elements

    // Check Batch 0, Channel 0, Top-Left (should be 1)
    EXPECT_NEAR(floats.data()[0], 1.f, 1e-5);
    // Check Batch 1 (offset 32), should all be 10.
    EXPECT_NEAR(floats.data()[32], 10.f, 1e-5);
    EXPECT_NEAR(floats.data()[63], 10.f, 1e-5);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborIntegerScale) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // Input: 2x2
    // 1 2
    // 3 4
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Resize to 4x4 (2x scale).
    TensorType output = ResizeNearestNeighbor(input, {4, 4});

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    ASSERT_EQ(floats.size(), 16);

    // Expected:
    // 1 1 2 2
    // 1 1 2 2
    // 3 3 4 4
    // 3 3 4 4

    // Row 0
    EXPECT_NEAR(floats.data()[0], 1.f, 1e-5);
    EXPECT_NEAR(floats.data()[1], 1.f, 1e-5);
    EXPECT_NEAR(floats.data()[2], 2.f, 1e-5);
    EXPECT_NEAR(floats.data()[3], 2.f, 1e-5);

    // Row 1
    EXPECT_NEAR(floats.data()[4], 1.f, 1e-5);
    EXPECT_NEAR(floats.data()[5], 1.f, 1e-5);
    EXPECT_NEAR(floats.data()[6], 2.f, 1e-5);
    EXPECT_NEAR(floats.data()[7], 2.f, 1e-5);

    // Row 2
    EXPECT_NEAR(floats.data()[8], 3.f, 1e-5);
    EXPECT_NEAR(floats.data()[9], 3.f, 1e-5);
    EXPECT_NEAR(floats.data()[10], 4.f, 1e-5);
    EXPECT_NEAR(floats.data()[11], 4.f, 1e-5);

    // Row 3
    EXPECT_NEAR(floats.data()[12], 3.f, 1e-5);
    EXPECT_NEAR(floats.data()[13], 3.f, 1e-5);
    EXPECT_NEAR(floats.data()[14], 4.f, 1e-5);
    EXPECT_NEAR(floats.data()[15], 4.f, 1e-5);
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborRejectsNonInteger) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  // Resize to 3x3 (1.5x scale) - unsupported.
  TensorType output = ResizeNearestNeighbor(input, {3, 3});

  auto runner = Runner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()) ||
              absl::IsInvalidArgument(runner.status()));
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborRejectsAlignCorners) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  TensorType output =
      ResizeNearestNeighbor(input, {4, 4}, /*align_corners=*/true);

  auto runner = Runner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()) ||
              absl::IsInvalidArgument(runner.status()));
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborRejectsHalfPixel) {
  using TensorType = typename TestFixture::TensorType;
  using Runner = typename TestFixture::Runner;

  TensorType input({.name = "input",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

  TensorType output =
      ResizeNearestNeighbor(input, {4, 4}, /*align_corners=*/false,
                            /*half_pixel_centers=*/true);

  auto runner = Runner::Create({output});
  EXPECT_TRUE(absl::IsUnimplemented(runner.status()) ||
              absl::IsInvalidArgument(runner.status()));
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborIdentityScale) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});
    TensorType output = ResizeNearestNeighbor(input, {2, 2});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();
    EXPECT_THAT(floats, ::testing::ElementsAreArray({1.f, 2.f, 3.f, 4.f}));
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborAnisotropicScale) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // 1x2 -> 2x6 (H*2, W*3)
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 1, 2, 1},
                      .buffer = std::vector<float>{10.f, 20.f}});
    TensorType output = ResizeNearestNeighbor(input, {2, 6});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();

    // Row 0: [10, 10, 10, 20, 20, 20]
    // Row 1: [10, 10, 10, 20, 20, 20]
    std::vector<float> expected = {10.f, 10.f, 10.f, 20.f, 20.f, 20.f,
                                   10.f, 10.f, 10.f, 20.f, 20.f, 20.f};
    EXPECT_THAT(floats, ::testing::ElementsAreArray(expected));
  }
}

TYPED_TEST_P(NnpackRunnerTest, ResizeNearestNeighborBatchAndChannels) {
  if constexpr (!TestFixture::Traits::kSupportsResize) {
    GTEST_SKIP() << "Resize not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    // 2 batches, 2 channels. 1x1 spatial.
    // B0: [1, 2]
    // B1: [3, 4]
    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {2, 1, 1, 2},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Resize to 2x2.
    TensorType output = ResizeNearestNeighbor(input, {2, 2});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();

    // Output: 2x2x2x2
    // B0 should be all [1, 2] blocks.
    // B1 should be all [3, 4] blocks.
    // 4 pixels per batch. 8 floats per batch.
    ASSERT_EQ(floats.size(), 16);

    // Check B0
    for (int i = 0; i < 8; i += 2) {
      EXPECT_EQ(floats.data()[i], 1.f);
      EXPECT_EQ(floats.data()[i + 1], 2.f);
    }
    // Check B1
    for (int i = 8; i < 16; i += 2) {
      EXPECT_EQ(floats.data()[i], 3.f);
      EXPECT_EQ(floats.data()[i + 1], 4.f);
    }
  }
}

TYPED_TEST_P(NnpackRunnerTest, ComputesTransposeConv2D) {
  if constexpr (!TestFixture::Traits::kSupportsTransposeConv2D) {
    GTEST_SKIP() << "TransposeConv2D not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Filter: [O, H, W, I] = [1, 2, 2, 1]
    // All 1s.
    TensorType filter({.name = "filter",
                       .type = Type::kFP32,
                       .shape = {1, 2, 2, 1},
                       .buffer = std::vector<float>{1.f, 1.f, 1.f, 1.f}});

    TensorType bias({.name = "bias",
                     .type = Type::kFP32,
                     .shape = {1},
                     .buffer = std::vector<float>{0.f}});

    // Output shape: {1, 4, 4, 1}
    // Stride 2, Valid padding (no crop).
    // 2x2 input upscaled to 4x4.
    TensorType output =
        TransposeConv(filter, input, bias, {1, 4, 4, 1}, kPaddingValid,
                      /*stride_h=*/2, /*stride_w=*/2);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();

    // Expected:
    // 1 1 2 2
    // 1 1 2 2
    // 3 3 4 4
    // 3 3 4 4
    std::vector<float> expected = {1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f,
                                   3.f, 3.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f};
    EXPECT_THAT(floats, ::testing::ElementsAreArray(expected));
  }
}

TYPED_TEST_P(NnpackRunnerTest, ComputesTransposeConv2DSame) {
  if constexpr (!TestFixture::Traits::kSupportsTransposeConv2D) {
    GTEST_SKIP() << "TransposeConv2D not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});

    // Filter 3x3 all 1s.
    TensorType filter({.name = "filter",
                       .type = Type::kFP32,
                       .shape = {1, 3, 3, 1},
                       .buffer = std::vector<float>(9, 1.f)});

    TensorType bias({.name = "bias",
                     .type = Type::kFP32,
                     .shape = {1},
                     .buffer = std::vector<float>{0.f}});

    // SAME padding usually implies Output = Input * Stride = 4x4.
    TensorType output =
        TransposeConv(filter, input, bias, {1, 4, 4, 1}, kPaddingSame,
                      /*stride_h=*/2, /*stride_w=*/2);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto bytes, runner.ReadOutput(output));
    auto floats = std::move(bytes).template As<const float>();

    // Expected:
    // 1 1 3 2
    // 1 1 3 2
    // 4 4 10 6
    // 3 3 7 4
    std::vector<float> expected = {1.f, 1.f, 3.f,  2.f, 1.f, 1.f, 3.f, 2.f,
                                   4.f, 4.f, 10.f, 6.f, 3.f, 3.f, 7.f, 4.f};
    EXPECT_THAT(floats, ::testing::ElementsAreArray(expected));
  }
}

TYPED_TEST_P(NnpackRunnerTest, TransposeConvRejectsInvalidOutputShape) {
  if constexpr (!TestFixture::Traits::kSupportsTransposeConv2D) {
    GTEST_SKIP() << "TransposeConv2D not supported for this backend";
  } else {
    using TensorType = typename TestFixture::TensorType;
    using Runner = typename TestFixture::Runner;

    TensorType input({.name = "input",
                      .type = Type::kFP32,
                      .shape = {1, 2, 2, 1},
                      .buffer = std::vector<float>{1.f, 2.f, 3.f, 4.f}});
    TensorType filter({.name = "filter",
                       .type = Type::kFP32,
                       .shape = {1, 1, 1, 1},
                       .buffer = std::vector<float>{1.f}});
    TensorType bias({.name = "bias",
                     .type = Type::kFP32,
                     .shape = {1},
                     .buffer = std::vector<float>{0.f}});

    // Stride 2, Filter 1. Base output = (2-1)*2 + 1 = 3.
    // Requesting 5 means adjustment 2.
    // Adjustment 2 >= Stride 2 -> Error.
    TensorType output =
        TransposeConv(filter, input, bias, {1, 5, 5, 1}, kPaddingValid,
                      /*stride_h=*/2, /*stride_w=*/2);

    auto runner = Runner::Create({output});
    EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

REGISTER_TYPED_TEST_SUITE_P(
    NnpackRunnerTest, SetInputRejectsNonExternalTensors, ComputesConstantAdd,
    ComputesRuntimeInputAdd, MoveConstructorTransfersRuntime,
    MoveAssignmentTransfersRuntime, ConstantsAreNotBoundAsExternals,
    ComputesConstantMul, ComputesConstantSub, ComputesConstantDiv,
    ComputesMaximumAndMinimum, ComputesPow, ComputesAbs, ComputesSquare,
    ComputesRsqrt, ComputesSqrt, ComputesNeg, ComputesTanh, ComputesSigmoid,
    ComputesCos, ComputesSin, ComputesGelu, ComputesSoftmax,
    SoftmaxRejectsBetaNotEqualToOne, ComputesConv2D, ComputesDepthwiseConv2D,
    ComputesFullyConnected, BatchMatMulSupportsTransposeFlags,
    ComputesTranspose, ComputesMeanKeepDimsAndSqueeze, TransposeAndMeanCombined,
    ComputesSlice, ComputesConcatenation, SliceAndConcatenationCombined,
    ComputesReshape, ComputesTile, TilAndReshape, ResizeBilinearAlignCorners,
    ResizeBilinearHalfPixelCenters, ResizeBilinearLegacy,
    ResizeBilinearDownsample, ResizeBilinearBatchAndChannels,
    ResizeNearestNeighborIntegerScale, ResizeNearestNeighborRejectsNonInteger,
    ResizeNearestNeighborRejectsAlignCorners,
    ResizeNearestNeighborRejectsHalfPixel, ResizeNearestNeighborIdentityScale,
    ResizeNearestNeighborAnisotropicScale,
    ResizeNearestNeighborBatchAndChannels, ComputesTransposeConv2D,
    ComputesTransposeConv2DSame, TransposeConvRejectsInvalidOutputShape);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_NNPACK_COMMON_RUNNER_TEST_SUITE_H_
