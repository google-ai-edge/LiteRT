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

#include "tensor/backends/ml_drift/trace_ml_drift_runner.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/buffer.h"
#include "tensor/internal/compile.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

using TensorMLD = Tensor<MlDriftMixinTag>;

TEST(TraceMlDriftRunnerTest, BasicRun) {
  TensorMLD input1({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  input1.SetName("input1");
  TensorMLD input2({.type = Type::kFP32, .shape = {1, 2, 2, 1}});
  input2.SetName("input2");

  TensorMLD weight({.type = Type::kFP32,
                    .shape = {1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>({2.0f})});
  weight.SetName("weight");

  TensorMLD added = Add(input1, input2);
  TensorMLD output = Mul(added, weight);
  output.SetName("output");

  std::vector<TensorHandle> flat_inputs = {input1, input2};
  std::vector<TensorHandle> flat_outputs = {output};

  TraceMlDriftRunner runner(flat_inputs, flat_outputs,
                            ml_drift::CalculationsPrecision::kF32);

  ASSERT_OK(runner.BuildModel());

  std::vector<float> in1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> in2 = {5.0f, 6.0f, 7.0f, 8.0f};

  ASSERT_OK(runner.SetInput("input1", in1));
  ASSERT_OK(runner.SetInput("input2", in2));

  ASSERT_OK(runner.Run());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto result, runner.GetFloatOutput("output"));
  const std::vector<float> expected = {12.0f, 16.0f, 20.0f, 24.0f};
  EXPECT_THAT(result, Pointwise(FloatNear(1e-5), expected));
}

class DummyWeightLoader {
 public:
  void AddWeight(const std::string& name, TensorMLD tensor) {
    weights_[name] = std::move(tensor);
  }

  TensorMLD GetWeight(const std::string& name) const {
    return weights_.at(name);
  }

 private:
  absl::flat_hash_map<std::string, TensorMLD> weights_;
};

struct ModelConfig {
  int num_layers;
};

TEST(TraceMlDriftRunnerTest, LinkWithCompiledRunner) {
  // 1. Instantiate the loader and populate it ahead of time.
  DummyWeightLoader loader;
  loader.AddWeight("my_weight",
                   TensorMLD({
                       .name = "my_weight",
                       .type = Type::kFP32,
                       .shape = {1},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({3.0f}),
                   }));

  ModelConfig config = {.num_layers = 3};

  // 2. The graph lambda captures the loader by reference, and the config by
  // value. The lambda arguments (x, y) are strictly for runtime tensors.
  auto model_func = [&loader, config](auto x, auto y) {
    auto current = Add(x, y);

    // The captured config dictates the structure of the traced graph.
    for (int i = 0; i < config.num_layers; ++i) {
      // Load the weight by name from the loader during tracing.
      current = Mul(current, loader.GetWeight("my_weight"));
    }
    current.SetName("output");
    return current;
  };

  TensorMLD input1(
      {.name = "input1", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  TensorMLD input2(
      {.name = "input2", .type = Type::kFP32, .shape = {1, 2, 2, 1}});

  auto compiled_runner =
      compile<TraceMlDriftRunner>(model_func, input1, input2);

  // Verifying the returned traced tensor matches expected graph structural
  // outputs.

  std::vector<float> in1 = {1.0f, 2.0f, 3.0f, 4.0f};
  input1.SetBuffer(OwningCpuBuffer::Copy<Type::kFP32>(in1));
  std::vector<float> in2 = {5.0f, 6.0f, 7.0f, 8.0f};
  input2.SetBuffer(OwningCpuBuffer::Copy<Type::kFP32>(in2));

  TensorMLD output_tensor = compiled_runner(input1, input2);
  EXPECT_EQ(output_tensor.GetName(), "output");
  EXPECT_EQ(output_tensor.GetShape().size(), 4);

  auto buffer = output_tensor.GetBuffer();
  ASSERT_OK(buffer);
  auto locked = buffer->Lock();
  auto span = std::move(locked).As<const float>();
  int product = 1;
  for (int d : output_tensor.GetShape()) product *= d;
  std::vector<float> result(span.begin(), span.begin() + product);

  // (x + y) * weight^3
  // 1+5=6 -> 6 * 3^3 = 6 * 27 = 162
  // 2+6=8 -> 8 * 27 = 216
  // 3+7=10 -> 10 * 27 = 270
  // 4+8=12 -> 12 * 27 = 324
  const std::vector<float> expected = {162.0f, 216.0f, 270.0f, 324.0f};
  EXPECT_THAT(result, Pointwise(FloatNear(1e-5), expected));
}

}  // namespace
}  // namespace litert::tensor
