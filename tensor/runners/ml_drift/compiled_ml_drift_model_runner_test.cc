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

#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/compile.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using Runner = CompiledMlDriftModelRunner<MlDriftClBackend>;
using TensorMLD = Tensor<MlDriftMixinTag>;

TEST(CompiledMlDriftModelRunnerTest, BasicExecution) {
  auto lambda = [](TensorMLD x) {
    auto y = litert::tensor::Add(x, x);
    y.SetName("out");
    return y;
  };

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2}});
  auto compiled_runner = compile<Runner>(lambda, x);

  std::vector<float> data_x = {1.0f, 2.0f};
  TensorMLD x_input(
      {.name = "x", .type = Type::kFP32, .shape = {1, 2}, .buffer = data_x});

  auto y = compiled_runner(x_input);

  EXPECT_THAT(y.GetShape(), testing::ElementsAre(1, 2));
}

// Constant FP32 FullyConnected weights must be packed into the kernel's
// weights layout; with the generic linear-constant path the kernel reads a
// truncated, un-rearranged buffer and only the first output channels are
// non-zero. The sizes are deliberately not multiples of 4.
TEST(CompiledMlDriftModelRunnerTest, ConstantFp32FullyConnectedWeights) {
  constexpr int kBatch = 3;
  constexpr int kIn = 7;
  constexpr int kOut = 10;
  std::vector<float> weights(kOut * kIn);
  for (int i = 0; i < weights.size(); ++i)
    weights[i] = 0.01f * (i % 13) - 0.05f;
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) input[i] = 0.1f * (i % 5) - 0.2f;

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kFP32,
               .shape = {kOut, kIn},
               .buffer = weights});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] += input[b * kIn + i] * weights[o * kIn + i];
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-5f), expected));
}

TEST(CompiledMlDriftModelRunnerTest, ConstantInt8FullyConnectedWeights) {
  constexpr int kBatch = 2;
  constexpr int kIn = 8;
  constexpr int kOut = 6;
  std::vector<int8_t> qweights(kOut * kIn);
  for (int i = 0; i < qweights.size(); ++i) {
    qweights[i] = static_cast<int8_t>((i % 11) - 5);
  }
  std::vector<float> scales(kOut);
  for (int o = 0; o < kOut; ++o) {
    scales[o] = 0.05f * static_cast<float>(o + 1);
  }
  std::vector<float> input(kBatch * kIn);
  for (int i = 0; i < input.size(); ++i) {
    input[i] = 0.1f * static_cast<float>((i % 7) - 3);
  }

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kI8,
               .shape = {kOut, kIn},
               .buffer = qweights,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            (static_cast<float>(qweights[o * kIn + i]) * scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

TEST(CompiledMlDriftModelRunnerTest, ConstantInt2FullyConnectedWeights) {
  constexpr int kBatch = 2;
  constexpr int kIn = 8;
  constexpr int kOut = 4;
  std::shared_ptr<OwningCpuBuffer> w_buf =
      OwningCpuBuffer::Allocate<Type::kI2>(kOut * kIn);
  int2_t* packed = w_buf->Span<int2_t>().data();
  std::vector<int> unpacked(kOut * kIn);
  for (int idx = 0; idx < (kOut * kIn) / 4; ++idx) {
    const int v0 = (idx % 3) - 1;
    const int v1 = ((idx + 1) % 3) - 1;
    const int v2 = ((idx + 2) % 3) - 1;
    const int v3 = -1;
    packed[idx].a = v0;
    packed[idx].b = v1;
    packed[idx].c = v2;
    packed[idx].d = v3;
    unpacked[4 * idx + 0] = v0;
    unpacked[4 * idx + 1] = v1;
    unpacked[4 * idx + 2] = v2;
    unpacked[4 * idx + 3] = v3;
  }
  std::vector<float> scales(kOut, 0.25f);
  std::vector<float> input(kBatch * kIn, 0.5f);

  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, kBatch, kIn}});
  TensorMLD w({.name = "w",
               .type = Type::kI2,
               .shape = {kOut, kIn},
               .buffer = w_buf,
               .quantization = std::make_shared<PerChannelAffineQuantization>(
                   scales, std::vector<int64_t>(kOut, 0),
                   /*quantized_dimension=*/0)});
  TensorMLD y = FullyConnected(x, w);
  y.SetName("y");

  Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
  ASSERT_THAT(runner.BuildModel(), IsOk());
  TensorMLD fed({.name = "x",
                 .type = Type::kFP32,
                 .shape = {1, kBatch, kIn},
                 .buffer = input});
  ASSERT_THAT(runner.SetInput("x", fed), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::vector<float> actual,
                                  runner.GetFloatOutput("y"));

  std::vector<float> expected(kBatch * kOut, 0.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOut; ++o) {
      for (int i = 0; i < kIn; ++i) {
        expected[b * kOut + o] +=
            input[b * kIn + i] *
            (static_cast<float>(unpacked[o * kIn + i]) * scales[o]);
      }
    }
  }
  EXPECT_THAT(actual, testing::Pointwise(testing::FloatNear(1e-4f), expected));
}

TEST(CompiledMlDriftModelRunnerTest,
     UnsupportedFp16AndBf16ConstantFullyConnectedWeightsReturnUnimplemented) {
  TensorMLD x({.name = "x", .type = Type::kFP32, .shape = {1, 2, 4}});
  {
    TensorMLD w_fp16({.name = "w_fp16",
                      .type = Type::kFP16,
                      .shape = {4, 4},
                      .buffer = std::vector<fp16_t>(16, fp16_t(1.0f))});
    TensorMLD y = FullyConnected(x, w_fp16);
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    EXPECT_EQ(runner.BuildModel().code(), absl::StatusCode::kUnimplemented);
  }
  {
    TensorMLD w_bf16({.name = "w_bf16",
                      .type = Type::kBF16,
                      .shape = {4, 4},
                      .buffer = std::vector<bf16_t>(16, bf16_t(1.0f))});
    TensorMLD y = FullyConnected(x, w_bf16);
    y.SetName("y");
    Runner runner({x}, {y}, ml_drift::CalculationsPrecision::kF32);
    EXPECT_EQ(runner.BuildModel().code(), absl::StatusCode::kUnimplemented);
  }
}

}  // namespace
}  // namespace litert::tensor
