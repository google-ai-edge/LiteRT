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

#include "tensor/examples/gemma4/helpers/feed_forward_network.h"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor::examples::gemma4 {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;
using XnnTensor = Tensor<XnnpackMixinTag>;

TEST(Gemma4GraphTest, FeedForwardNetworkTest) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {1, 1, 2}});

  XnnTensor gate_proj(
      {.name = "gate_proj",
       .type = Type::kFP32,
       .shape = {3, 2},
       .buffer = std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f}});
  XnnTensor up_proj(
      {.name = "up_proj",
       .type = Type::kFP32,
       .shape = {3, 2},
       .buffer = std::vector<float>{0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.5f}});
  XnnTensor down_proj(
      {.name = "down_proj",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.0f, 0.0f, 0.5f, 0.0f, 1.0f, -0.5f}});

  XnnTensor output = FeedForwardNetwork(input, gate_proj, up_proj, down_proj);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({output}));

  const std::array<float, 2> input_data = {1.0f, 2.0f};
  ASSERT_THAT(runner.SetInput(input, input_data), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const std::byte> result,
                                  runner.ReadOutput(output));
  LockedBufferSpan<const float> floats = std::move(result).As<const float>();
  ASSERT_EQ(floats.size(), 2);

  // Expected data computed using the script in
  // `./reference/feed_forward_network.py`.
  const std::array<float, 2> expected_data = {1.182384f, 2.034002f};

  EXPECT_THAT(floats, Pointwise(FloatNear(1e-5f), expected_data));
}

}  // namespace
}  // namespace litert::tensor::examples::gemma4
