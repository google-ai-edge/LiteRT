// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "litert/runtime/profiler.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/api/profiler.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/subgraph.h"

namespace litert {
namespace {

TEST(LiteRtProfilerSummarizerTest, GetProfileSummary) {
  LiteRtProfilerT profiler;
  profiler.StartProfiling();

  // Add some events. Node index 0.
  auto handle = profiler.BeginEvent(
      "Op1", tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT, 0, 0);
  profiler.EndEvent(handle);

  profiler.StopProfiling();

  tflite::Interpreter interpreter;
  // One subgraph is created by default.
  interpreter.subgraph(0)->SetName("Main");

  // Add tensors (0: input, 1: output)
  interpreter.AddTensors(2);
  interpreter.SetInputs({0});
  interpreter.SetOutputs({1});

  // Add a node at index 0 matching the event
  TfLiteRegistration reg = {};
  reg.builtin_code = kTfLiteBuiltinCustom;
  reg.custom_name = "MyCustomOp";
  reg.version = 1;

  interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);

  std::string summary = profiler.GetProfileSummary(interpreter);
  // Summary should contain "Op1" (tag)
  EXPECT_TRUE(absl::StrContains(summary, "Op1"));

  // Test accumulation
  profiler.Reset();
  profiler.StartProfiling();
  auto handle2 = profiler.BeginEvent(
      "Op2", tflite::Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT, 0, 0);
  profiler.EndEvent(handle2);
  profiler.StopProfiling();

  EXPECT_EQ(profiler.GetProfiledEvents().size(), 1);
  summary = profiler.GetProfileSummary(interpreter);
  EXPECT_TRUE(absl::StrContains(summary, "Op2"));
}

}  // namespace
}  // namespace litert
