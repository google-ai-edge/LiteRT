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

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/test_backends.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"  // IWYU pragma: keep
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/mixin.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::Pointwise;

template <class Backend>
class RmsNormTest : public ::testing::Test {};
TYPED_TEST_SUITE(RmsNormTest, TestBackends, TestBackendNames);

TYPED_TEST(RmsNormTest, MatchesReference) {
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  Tensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});

  Tensor scale({.name = "scale",
                .type = Type::kFP32,
                .shape = {4},
                .buffer = std::vector<float>{1.0f, 1.3f, 0.9f, 1.5f}});

  Tensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
  Tensor output = RmsNorm(input, scale, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.949386f, 0.985901f, 2.190890f})));
}

TYPED_TEST(RmsNormTest, WithoutScale) {
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  Tensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});

  Tensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
  Tensor output = RmsNorm(input, Tensor(TensorHandle::Invalid()), eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.730297f, 1.095445f, 1.460593f})));
}

TYPED_TEST(RmsNormTest, WithAttributeEpsilon) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  // This test manually constructs the RmsNormOperation to verify that the
  // backend correctly handles the case where epsilon is set as an operation
  // attribute (`op->epsilon`) rather than as an input tensor. The `RmsNorm`
  // helper function always passes epsilon as an input tensor.
  auto op = std::make_shared<graph::RmsNormOperation>();
  RegisterMixins<Tag>(op);

  Tensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
  Tensor scale({.name = "scale",
                .type = Type::kFP32,
                .shape = {4},
                .buffer = std::vector<float>{1.0f, 1.3f, 0.9f, 1.5f}});
  AddInputs(op, input, scale);
  op->epsilon = 1e-6f;

  TensorHandle output = AddOutput(op, source_location::current());
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info,
                                  graph::GetInfo(output.GetRaw()));
  output_info.shape = {1, 1, 4};
  output_info.type = Type::kFP32;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(Runner runner, Runner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.template ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.949386f, 0.985901f, 2.190890f})));
}

TYPED_TEST(RmsNormTest, RejectsInvalidInputCounts) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;
  using Operation = typename TypeParam::Operation;

  // Test with 1 input (too few)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<Tag>(op);
    using RmsNormMixin = graph::OpMixin<graph::RmsNormOperation, Tag>;
    static_assert(std::is_base_of_v<Operation, RmsNormMixin>);

    Tensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
    AddInputs(op, input);

    TensorHandle output = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info,
                                    graph::GetInfo(output.GetRaw()));
    output_info.shape = {1, 1, 4};
    output_info.type = Type::kFP32;

    EXPECT_THAT(Runner::Create({output}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 2 or 3 inputs")));
  }

  // Test with 4 inputs (too many)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<Tag>(op);

    Tensor input1({.name = "input1", .type = Type::kFP32, .shape = {1, 1, 4}});
    Tensor input2({.name = "input2", .type = Type::kFP32, .shape = {1, 1, 4}});
    Tensor input3({.name = "input3", .type = Type::kFP32, .shape = {1, 1, 4}});
    Tensor input4({.name = "input4", .type = Type::kFP32, .shape = {1, 1, 4}});
    AddInputs(op, input1, input2, input3, input4);

    TensorHandle output = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info,
                                    graph::GetInfo(output.GetRaw()));
    output_info.shape = {1, 1, 4};
    output_info.type = Type::kFP32;

    EXPECT_THAT(Runner::Create({output}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 2 or 3 inputs")));
  }
}

TYPED_TEST(RmsNormTest, RejectsInvalidOutputCounts) {
  using Tag = typename TypeParam::Tag;
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  // Test with 2 outputs (too many)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<Tag>(op);

    Tensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
    Tensor scale({.name = "scale", .type = Type::kFP32, .shape = {4}});
    AddInputs(op, input, scale);

    // Add first output
    TensorHandle output1 = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info1,
                                    graph::GetInfo(output1.GetRaw()));
    output_info1.shape = {1, 1, 4};
    output_info1.type = Type::kFP32;

    // Add second output
    TensorHandle output2 = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info2,
                                    graph::GetInfo(output2.GetRaw()));
    output_info2.shape = {1, 1, 4};
    output_info2.type = Type::kFP32;

    EXPECT_THAT(Runner::Create({output1}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 1 output, got 2")));
  }
}

TYPED_TEST(RmsNormTest, RejectsScalarInput) {
  using Tensor = typename TypeParam::Tensor;
  using Runner = typename TypeParam::Runner;

  // Test with scalar input (empty shape)
  {
    Tensor input({.name = "input", .type = Type::kFP32, .shape = {}});
    Tensor scale({.name = "scale", .type = Type::kFP32, .shape = {4}});
    Tensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
    Tensor output = RmsNorm(input, scale, eps_tensor);

    EXPECT_THAT(
        Runner::Create({output}),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("RmsNorm input tensor must have at least 1 dimension")));
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
