/* Copyright 2025 Google LLC.

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

#include "tensor/runners/xnnpack/runner.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"

namespace litert::tensor {

struct XnnpackTestTraits {
  using Tag = XnnpackMixinTag;
  using Runner = XnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = XNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  static constexpr bool kSupportsConv2D = true;
  static constexpr bool kSupportsDepthwiseConv2D = true;
  static constexpr bool kSupportsTransposeConv2D = true;
  static constexpr bool kSupportsResize = true;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Xnnpack, NnpackRunnerTest, XnnpackTestTraits);

namespace {

using XnnTensor = Tensor<XnnpackMixinTag>;

constexpr uint32_t kRuntimeFlags =
    XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC | XNN_FLAG_BASIC_PROFILING;

void ExpectConfiguredRun(XnnpackRunner& runner, const XnnTensor& output,
                        const std::array<float, 3>& input_values) {
  ASSERT_THAT(runner.Run(), IsOk());
  EXPECT_EQ(runner.runtime_flags(), kRuntimeFlags);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto values,
                                runner.ReadOutputAs<float>(output));
  ASSERT_EQ(values.size(), input_values.size());
  for (size_t i = 0; i < input_values.size(); ++i) {
    EXPECT_FLOAT_EQ(values.data()[i], 1.0f / std::sqrt(input_values[i]));
  }

  // Profiling succeeds only if the configured flags reached the actual runtime.
  size_t operator_count = 0;
  size_t required_size = 0;
  ASSERT_EQ(xnn_get_runtime_profiling_info(
                runner.runtime(), xnn_profile_info_num_operators,
                sizeof(operator_count), &operator_count, &required_size),
            xnn_status_success);
  EXPECT_GT(operator_count, 0);
}

TEST(XnnpackRunnerTest, RuntimeFlagsApplyToLazyCreationAndRepeatedRuns) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, XnnpackRunner::Create({output}, kRuntimeFlags));
  EXPECT_EQ(runner.runtime(), nullptr);

  std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, data), IsOk());
  ExpectConfiguredRun(runner, output, data);
  xnn_runtime_t runtime = runner.runtime();

  data = {9.0f, 16.0f, 25.0f};
  ExpectConfiguredRun(runner, output, data);
  EXPECT_EQ(runner.runtime(), runtime);
}

TEST(XnnpackRunnerTest, MoveBeforePreparationPreservesRuntimeFlags) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      auto runner, XnnpackRunner::Create({output}, kRuntimeFlags));
  const std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(runner.SetInput(input, data), IsOk());

  XnnpackRunner moved(std::move(runner));
  EXPECT_EQ(moved.runtime(), nullptr);
  ExpectConfiguredRun(moved, output, data);
}

TEST(XnnpackRunnerTest, MoveAssignmentReplacesPreparedRuntimeAndFlags) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor output = Rsqrt(input);
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));
  XnnpackRunner source(std::move(graph), kRuntimeFlags);
  const std::array<float, 3> data = {0.9646427035331726f, 1.0f, 4.0f};
  ASSERT_THAT(source.SetInput(input, data), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto destination,
                                XnnpackRunner::Create({output}));
  ASSERT_THAT(destination.SetInput(input, data), IsOk());
  ASSERT_THAT(destination.Run(), IsOk());
  EXPECT_EQ(destination.runtime_flags(), 0);
  size_t operator_count = 0;
  size_t required_size = 0;
  EXPECT_EQ(xnn_get_runtime_profiling_info(
                destination.runtime(), xnn_profile_info_num_operators,
                sizeof(operator_count), &operator_count, &required_size),
            xnn_status_invalid_state);

  destination = std::move(source);
  EXPECT_EQ(destination.runtime(), nullptr);
  ExpectConfiguredRun(destination, output, data);
}

}  // namespace

}  // namespace litert::tensor
