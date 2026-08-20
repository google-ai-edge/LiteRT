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
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/examples/ops/transformer/transformer_ops_graph.h"
#include "tensor/examples/ops/transformer/transformer_ops_xnnpack.h"
#include "tensor/internal/arithmetic_helpers.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"
#include "tensor/utils/source_location.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

TEST(Gemma4GraphTest, RmsNormTest) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});

  XnnTensor scale({.name = "scale",
                   .type = Type::kFP32,
                   .shape = {4},
                   .buffer = std::vector<float>{1.0f, 1.3f, 0.9f, 1.5f}});

  XnnTensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
  XnnTensor output = RmsNorm(input, scale, eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.949386f, 0.985901f, 2.190890f})));
}

TEST(Gemma4GraphTest, RmsNormNoScaleTest) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});

  XnnTensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
  XnnTensor output =
      RmsNorm(input, XnnTensor(TensorHandle::Invalid()), eps_tensor);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.730297f, 1.095445f, 1.460593f})));
}

TEST(Gemma4GraphTest, RmsNormWithAttributeEpsilonTest) {
  // This test manually constructs the RmsNormOperation to verify that the
  // backend correctly handles the case where epsilon is set as an operation
  // attribute (`op->epsilon`) rather than as an input tensor. The `RmsNorm`
  // helper function always passes epsilon as an input tensor.
  auto op = std::make_shared<graph::RmsNormOperation>();
  RegisterMixins<XnnpackMixinTag>(op);

  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
  XnnTensor scale({.name = "scale",
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

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 4> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  // Expected data computed using the script in `./reference/rmsnorm.py`.
  EXPECT_THAT(
      runner.ReadOutputAs<float>(output),
      IsOkAndHolds(Pointwise(FloatNear(1e-5f),
                             {0.365148f, 0.949386f, 0.985901f, 2.190890f})));
}

TEST(Gemma4GraphTest, RmsNormInvalidInputsTest) {
  // Test with 1 input (too few)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<XnnpackMixinTag>(op);
    using XnnpackRmsNormMixin =
        graph::OpMixin<graph::RmsNormOperation, XnnpackMixinTag>;
    static_assert(std::is_base_of_v<XnnpackOperation, XnnpackRmsNormMixin>);

    XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
    AddInputs(op, input);

    TensorHandle output = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info,
                                    graph::GetInfo(output.GetRaw()));
    output_info.shape = {1, 1, 4};
    output_info.type = Type::kFP32;

    EXPECT_THAT(XnnpackRunner::Create({output}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 2 or 3 inputs")));
  }

  // Test with 4 inputs (too many)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<XnnpackMixinTag>(op);

    XnnTensor input1(
        {.name = "input1", .type = Type::kFP32, .shape = {1, 1, 4}});
    XnnTensor input2(
        {.name = "input2", .type = Type::kFP32, .shape = {1, 1, 4}});
    XnnTensor input3(
        {.name = "input3", .type = Type::kFP32, .shape = {1, 1, 4}});
    XnnTensor input4(
        {.name = "input4", .type = Type::kFP32, .shape = {1, 1, 4}});
    AddInputs(op, input1, input2, input3, input4);

    TensorHandle output = AddOutput(op, source_location::current());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph::TensorInformation & output_info,
                                    graph::GetInfo(output.GetRaw()));
    output_info.shape = {1, 1, 4};
    output_info.type = Type::kFP32;

    EXPECT_THAT(XnnpackRunner::Create({output}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 2 or 3 inputs")));
  }
}

TEST(Gemma4GraphTest, RmsNormInvalidOutputsTest) {
  // Test with 2 outputs (too many)
  {
    auto op = std::make_shared<graph::RmsNormOperation>();
    RegisterMixins<XnnpackMixinTag>(op);

    XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 4}});
    XnnTensor scale({.name = "scale", .type = Type::kFP32, .shape = {4}});
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

    EXPECT_THAT(XnnpackRunner::Create({output1}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("RmsNorm expects 1 output, got 2")));
  }
}

TEST(Gemma4GraphTest, RmsNormScalarInputTest) {
  // Test with scalar input (empty shape)
  {
    XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {}});
    XnnTensor scale({.name = "scale", .type = Type::kFP32, .shape = {4}});
    XnnTensor eps_tensor({.type = Type::kFP32, .shape = {1}, .buffer = 1e-6f});
    XnnTensor output = RmsNorm(input, scale, eps_tensor);

    EXPECT_THAT(
        XnnpackRunner::Create({output}),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("RmsNorm input tensor must have at least 1 dimension")));
  }
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
