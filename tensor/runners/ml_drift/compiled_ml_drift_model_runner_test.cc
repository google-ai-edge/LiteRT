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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/internal/compile.h"
#include "tensor/datatypes.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/tensor.h"

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

}  // namespace
}  // namespace litert::tensor
