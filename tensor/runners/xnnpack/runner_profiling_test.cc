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

// Covers the `LRT_XNN_PROFILE=1` path of `XnnpackRunner`. The BUILD target sets
// the variable in the test environment because the runner reads it once per
// process.

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/scoped_mock_log.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::HasSubstr;

TEST(XnnpackRunnerProfilingTest, LogsProfileSummaryOnlyOnFirstRun) {
  Tensor<XnnpackMixinTag> w(
      {.name = "w",
       .type = Type::kFP32,
       .shape = {2, 3},
       .buffer = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
  Tensor<XnnpackMixinTag> x(
      {.name = "x", .type = Type::kFP32, .shape = {1, 3}});
  Tensor<XnnpackMixinTag> y = FullyConnected(x, w);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(XnnpackRunner runner,
                                  XnnpackRunner::Create({y}));

  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(_, _, _)).Times(AnyNumber());
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kInfo, _, HasSubstr("XNNPACK profile (")))
      .Times(1);
  log.StartCapturingLogs();

  const std::vector<float> input = {0.5f, 1.0f, -1.0f};
  for (int run = 0; run < 3; ++run) {
    ASSERT_OK(runner.SetInput(x, input));
    ASSERT_OK(runner.Run());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(LockedBufferSpan<const float> output,
                                    runner.ReadOutputAs<float>(y));
    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.data()[0], -0.5f);
    EXPECT_FLOAT_EQ(output.data()[1], 1.0f);
  }
}

}  // namespace
}  // namespace litert::tensor
