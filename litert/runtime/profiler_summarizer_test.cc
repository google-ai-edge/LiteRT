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
#include "litert/runtime/profiler_summarizer.h"

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
#include "tflite/profiling/profile_buffer.h"
#include "tflite/profiling/profile_summarizer.h"

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
  // Summary should contain "Op1" (tag is used as primary ID now to match
  // TFLite)
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

TEST(LiteRtProfilerSummarizerTest, CompareWithTfliteSummarizer) {
  tflite::Interpreter interpreter;
  interpreter.subgraph(0)->SetName("Main");
  interpreter.AddTensors(2);
  interpreter.SetInputs({0});
  interpreter.SetOutputs({1});

  TfLiteRegistration reg = {};
  reg.builtin_code = kTfLiteBuiltinCustom;
  reg.custom_name = "TestOp";
  reg.version = 1;
  interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);

  std::vector<tflite::profiling::ProfileEvent> events;
  tflite::profiling::ProfileEvent event;
  event.tag = "Op1";
  event.event_type = tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT;
  event.begin_timestamp_us = 1000;
  event.elapsed_time = 500;
  event.event_metadata = 0;        // node index
  event.extra_event_metadata = 0;  // subgraph index
  events.push_back(event);

  std::vector<const tflite::profiling::ProfileEvent*> event_ptrs;
  for (const auto& e : events) {
    event_ptrs.push_back(&e);
  }

  // LiteRt Summarizer
  litert::profiling::LiteRtProfileSummarizer litert_summarizer;
  litert_summarizer.ProcessProfiles(event_ptrs, interpreter);

  const auto& stats = litert_summarizer.GetStats();
  ASSERT_EQ(stats.size(), 1);
  EXPECT_EQ(stats.at("Op1").count, 1);
  EXPECT_EQ(stats.at("Op1").total_time_us, 500);

  // TFLite Summarizer
  tflite::profiling::ProfileSummarizer tflite_summarizer;
  tflite_summarizer.ProcessProfiles(event_ptrs, interpreter);
  std::string tflite_output = tflite_summarizer.GetOutputString();

  // Basic consistency check: TFLite output should contain "Op1" and "0.500"
  // (ms)
  EXPECT_TRUE(absl::StrContains(tflite_output, "Op1"));
  EXPECT_TRUE(absl::StrContains(tflite_output, "0.500"));
}

}  // namespace
}  // namespace litert
