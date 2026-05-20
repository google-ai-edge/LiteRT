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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/scoped_mock_log.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert::tensor {

namespace {

using ::testing::_;
using ::testing::HasSubstr;

TEST(OpDebuggerTest, LogsOnOpCreation) {
  if (!graph::OpDebugger::Enabled()) {
    GTEST_SKIP() << "Define LITERT_OP_DEBUGGER_ENABLED to run this test.";
  }
  absl::ScopedMockLog log;

  {
    testing::InSequence s;
    // Expect logging in topological order (creation order).
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                         HasSubstr("Add(0:input1:FP32[1,1], "
                                   "0:input2:FP32[1,1]) -> (0::FP32[1,1])")))
        .Times(1);
    EXPECT_CALL(
        log,
        Log(absl::LogSeverity::kInfo, _,
            HasSubstr("Add(0:a:FP32[1,1], 0::FP32[1,1]) -> (0::FP32[1,1])")))
        .Times(1);
  }
  log.StartCapturingLogs();

  {
    // Build a graph with two ops: op1 -> op2.
    // op2 depends on op1.
    Tensor<> input1({.name = "input1", .type = Type::kFP32, .shape = {1, 1}});
    Tensor<> input2({.name = "input2", .type = Type::kFP32, .shape = {1, 1}});

    // op1: output1 = Add(input1, input2)
    // This should trigger the first log message immediately.
    auto output1 = Add(input1, input2);

    // op2: output2 = Add(a, output1)
    // We reuse input1 as 'a' for the second add to make it simple.
    Tensor<> a({.name = "a", .type = Type::kFP32, .shape = {1, 1}});
    // This should trigger the second log message immediately.
    auto output2 = Add(a, output1);
  }
}

TEST(OpDebuggerTest, LogsDoubleForMultiOutput) {
  if (!graph::OpDebugger::Enabled()) {
    GTEST_SKIP() << "Define LITERT_OP_DEBUGGER_ENABLED to run this test.";
  }
  absl::ScopedMockLog log;

  {
    testing::InSequence s;
    // We expect TopK to be logged.
    // Issue: Currently it might be logged twice because SetProducer is called
    // twice. valid check: we expect it ONCE.
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, HasSubstr("TopK")))
        .Times(1);
  }
  log.StartCapturingLogs();

  {
    Tensor<> input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
    // TopK with k=1.
    auto outputs = TopK(input, 1);

    // Verify that the second output also has the producer set correctly.
    // This confirms that setting the producer on one tensor's group
    // (outputs[0]) sets it for all tensors in that group (including
    // outputs[1]). Using graph::GetProducer and GetRaw() to access the internal
    // tensor info.
    auto producer = graph::GetProducer(outputs[1].GetRaw());
    EXPECT_TRUE(producer.ok());
    EXPECT_EQ((*producer)->GetName(), "TopK");
  }
}

}  // namespace
}  // namespace litert::tensor
