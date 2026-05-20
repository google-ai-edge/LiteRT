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

#include "tensor/internal/compile.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor {
namespace {

using ::testing::Eq;

struct TestModel {
  TensorHandle w;
  TensorHandle operator()(TensorHandle x) const { return x; }
};

TensorHandle FreeFunc(TensorHandle x) { return x; }

TEST(CompileTest, LambdaTraitsDeduction) {
  auto lambda = [](TensorHandle a, TensorHandle b) { return a; };
  using Traits = internal::LambdaTraits<decltype(lambda)>;
  EXPECT_THAT(Traits::arg_count, Eq(2));

  using TraitsStruct = internal::LambdaTraits<TestModel>;
  EXPECT_THAT(TraitsStruct::arg_count, Eq(1));

  using TraitsFree = internal::LambdaTraits<decltype(FreeFunc)>;
  EXPECT_THAT(TraitsFree::arg_count, Eq(1));
}

struct ModelInputs {
  TensorHandle a;
  TensorHandle b;
  absl::flat_hash_map<std::string, TensorHandle*> tensors() {
    return {{"a", &a}, {"b", &b}};
  }
};

struct ModelOutputs {
  TensorHandle result;
  absl::flat_hash_map<std::string, TensorHandle*> tensors() {
    return {{"result", &result}};
  }
};

TEST(CompileTest, CompileWithStructs) {
  auto lambda = [](ModelInputs inputs) {
    ModelOutputs out;
    out.result = inputs.a;
    return out;
  };

  TensorInit init_a;
  init_a.name = "arg_a";
  init_a.type = Type::kFP32;
  init_a.shape = {1, 2};
  TensorHandle t_a(std::move(init_a));

  TensorInit init_b;
  init_b.name = "arg_b";
  init_b.type = Type::kI32;
  init_b.shape = {3, 4};
  TensorHandle t_b(std::move(init_b));

  ModelInputs inputs = {t_a, t_b};

  class DummyBackendRunner {
   public:
    DummyBackendRunner(std::vector<TensorHandle> inputs,
                       std::vector<TensorHandle> outputs) {}
    absl::Status BuildModel() { return absl::OkStatus(); }
    absl::Status SetInput(const std::string& name, TensorHandle& arg) {
      return absl::OkStatus();
    }
    absl::Status Run() { return absl::OkStatus(); }
    absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(
        const std::string& name) {
      return absl::UnimplementedError("");
    }
  };

  auto compiled_runner = compile<DummyBackendRunner>(lambda, inputs);
  auto result = compiled_runner(inputs);

  EXPECT_THAT(result.result.GetShape().size(), Eq(2));
  EXPECT_THAT(result.result.GetShape()[0], Eq(1));
  EXPECT_THAT(result.result.GetShape()[1], Eq(2));
  EXPECT_THAT(result.result.GetType(), Eq(Type::kFP32));
  EXPECT_THAT(result.result.GetName(), Eq("arg_a"));
}

TEST(CompileTest, TraceFunction) {
  auto lambda = [](auto a, auto b) { return a; };

  TensorInit init_a;
  init_a.name = "arg_a";
  init_a.type = Type::kFP32;
  init_a.shape = {1, 2};
  TensorHandle t_a(std::move(init_a));

  TensorInit init_b;
  init_b.name = "arg_b";
  init_b.type = Type::kI32;
  init_b.shape = {3, 4};
  TensorHandle t_b(std::move(init_b));

  class DummyBackendRunner {
   public:
    DummyBackendRunner(std::vector<TensorHandle> inputs,
                       std::vector<TensorHandle> outputs) {}
    absl::Status BuildModel() { return absl::OkStatus(); }
    absl::Status SetInput(const std::string& name, TensorHandle& arg) {
      return absl::OkStatus();
    }
    absl::Status Run() { return absl::OkStatus(); }
    absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(
        const std::string& name) {
      return absl::UnimplementedError("");
    }
  };

  auto compiled_runner = compile<DummyBackendRunner>(lambda, t_a, t_b);
  auto result = compiled_runner(t_a, t_b);
  EXPECT_THAT(result.GetShape().size(), Eq(2));
  EXPECT_THAT(result.GetShape()[0], Eq(1));
  EXPECT_THAT(result.GetShape()[1], Eq(2));
  EXPECT_THAT(result.GetType(), Eq(Type::kFP32));
  EXPECT_THAT(result.GetName(), Eq("arg_a"));
}

}  // namespace
}  // namespace litert::tensor
