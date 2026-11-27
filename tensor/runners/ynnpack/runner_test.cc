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

#include "tensor/runners/ynnpack/runner.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"
#include "tensor/tensor.h"

namespace litert::tensor {

struct YnnpackTestTraits {
  using Tag = YnnpackMixinTag;
  using Runner = YnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = YNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      YNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  // YNNPACK has no convolution, pooling or resize primitives. These would have
  // to be composed out of stencil copies and dots, which isn't implemented.
  static constexpr bool kSupportsConv2D = false;
  static constexpr bool kSupportsDepthwiseConv2D = false;
  static constexpr bool kSupportsTransposeConv2D = false;
  static constexpr bool kSupportsResize = false;
  static constexpr bool kSupportsSoftmaxBeta = true;
  static constexpr bool kSupportsSqueezeOfReshapedInput = true;
  static constexpr bool kRejectsInvalidSqueeze = true;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NnpackRunnerTest, YnnpackTestTraits);

namespace {

using YnnTensor = Tensor<YnnpackMixinTag>;

TEST(YnnpackRunnerTest, RejectsConstantWithTooSmallBuffer) {
  YnnTensor x({.name = "x",
               .type = Type::kFP32,
               .shape = {4},
               .buffer = std::vector<float>{1.f, 2.f}});
  YnnTensor output = Add(x, x);
  EXPECT_EQ(YnnpackRunner::Create({output}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(YnnpackRunnerTest, SharesNonOwningThreadpoolAcrossRunners) {
  YnnpackScheduler scheduler(/*num_background_threads=*/2);
  ynn_threadpool_t raw_pool = nullptr;
  ASSERT_EQ(ynn_create_threadpool(YnnpackScheduler::Vtable(), &scheduler,
                                  /*flags=*/0, &raw_pool),
            ynn_status_success);
  YnnpackRunner::ThreadpoolPtr pool(raw_pool);

  const std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  YnnTensor w(
      {.name = "w", .type = Type::kFP32, .shape = {2, 3}, .buffer = weights});
  YnnTensor x({.name = "x", .type = Type::kFP32, .shape = {1, 3}});
  YnnTensor y = FullyConnected(x, w);

  auto r1 = YnnpackRunner::Create({y});
  ASSERT_TRUE(r1.ok());
  ASSERT_TRUE(r1->SetThreadpool(pool.get()).ok());
  EXPECT_EQ(r1->threadpool(), pool.get());

  auto r2 = YnnpackRunner::Create({y});
  ASSERT_TRUE(r2.ok());
  ASSERT_TRUE(r2->SetThreadpool(pool.get()).ok());
  EXPECT_EQ(r2->threadpool(), pool.get());

  const std::vector<float> in1 = {1.0f, 0.0f, 2.0f};
  ASSERT_TRUE(r1->SetInput(x, in1).ok());
  ASSERT_TRUE(r1->Run().ok());
  auto out1 = r1->ReadOutputAs<float>(y);
  ASSERT_TRUE(out1.ok());
  EXPECT_FLOAT_EQ(out1->data()[0], 7.0f);
  EXPECT_FLOAT_EQ(out1->data()[1], 16.0f);

  const std::vector<float> in2 = {0.5f, 1.0f, -1.0f};
  ASSERT_TRUE(r2->SetInput(x, in2).ok());
  ASSERT_TRUE(r2->Run().ok());
  auto out2 = r2->ReadOutputAs<float>(y);
  ASSERT_TRUE(out2.ok());
  EXPECT_FLOAT_EQ(out2->data()[0], -0.5f);
  EXPECT_FLOAT_EQ(out2->data()[1], 1.0f);
  EXPECT_FALSE(r2->SetThreadpool(pool.get()).ok());
}

}  // namespace
}  // namespace litert::tensor
