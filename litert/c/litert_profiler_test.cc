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

#include "litert/c/litert_profiler.h"

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler_event.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace {
LiteRtProfiler profiler;

TEST(LiteRtProfilerTest, CreateAndDestroy) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // SetUp and TearDown handle the basic creation and destruction.
  // This test just verifies that the profiler handle is not null after SetUp.
  EXPECT_NE(profiler, nullptr);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, GetInitialNumEvents) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  int num_events = -1;
  EXPECT_EQ(LiteRtGetNumProfilerEvents(profiler, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(num_events, 0);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, StartAndStopProfiling) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Just test that the calls succeed.
  EXPECT_EQ(LiteRtStartProfiler(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtStopProfiler(profiler), kLiteRtStatusOk);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, ResetProfiler) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // A simple test to ensure the Reset call succeeds. A more complex test
  // would involve adding events, resetting, and then checking if the count is
  // 0.
  EXPECT_EQ(LiteRtStartProfiler(profiler), kLiteRtStatusOk);
  // In a real scenario, events would be added here.
  EXPECT_EQ(LiteRtStopProfiler(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtResetProfiler(profiler), kLiteRtStatusOk);

  int num_events = -1;
  EXPECT_EQ(LiteRtGetNumProfilerEvents(profiler, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(num_events, 0);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, SetEventSource) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // This test just verifies that the call succeeds.
  EXPECT_EQ(LiteRtSetProfilerCurrentEventSource(profiler,
                                                ProfiledEventSource::LITERT),
            kLiteRtStatusOk);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, GetEventsWhenEmpty) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  int num_events = 1;  // Set to a non-zero value initially
  // We expect num_events to be updated to 0.
  ProfiledEventData events[1];  // Dummy buffer
  EXPECT_EQ(LiteRtGetProfilerEvents(profiler, num_events, events),
            kLiteRtStatusOk);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, GetProfileSummary) {
  auto path = litert::testing::GetTestFilePath(kModelFileName);

  LiteRtEnvironment environment;
  LiteRtEnvOption options = {};
  ASSERT_EQ(LiteRtCreateEnvironment(/*num_options=*/0, &options, &environment),
            kLiteRtStatusOk);

  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(environment, path.c_str(), &model),
            kLiteRtStatusOk);

  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LrtRuntimeOptions* runtime_options = nullptr;
  ASSERT_EQ(LrtCreateRuntimeOptions(&runtime_options), kLiteRtStatusOk);
  ASSERT_EQ(LrtSetRuntimeOptionsEnableProfiling(runtime_options, true),
            kLiteRtStatusOk);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options, &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_runtime_options = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_runtime_options),
            kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtAddOpaqueOptions(jit_compilation_options, opaque_runtime_options),
      kLiteRtStatusOk);
  LrtDestroyRuntimeOptions(runtime_options);

  LiteRtCompiledModel compiled_model;
  ASSERT_EQ(LiteRtCreateCompiledModel(environment, model,
                                      jit_compilation_options, &compiled_model),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtStartProfiler(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtStopProfiler(profiler), kLiteRtStatusOk);

  const char* summary = nullptr;
  EXPECT_EQ(LiteRtGetProfileSummary(profiler, compiled_model, &summary),
            kLiteRtStatusOk);
  ASSERT_NE(summary, nullptr);
  EXPECT_GT(std::strlen(summary), 0);

  // Verify deallocation runs safely without crashing or memory corruption.
  std::free(const_cast<char*>(summary));

  // Verify null arguments error handling.
  const char* dummy_summary = nullptr;
  EXPECT_NE(LiteRtGetProfileSummary(nullptr, compiled_model, &dummy_summary),
            kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetProfileSummary(profiler, nullptr, &dummy_summary),
            kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetProfileSummary(profiler, compiled_model, nullptr),
            kLiteRtStatusOk);

  LiteRtDestroyProfiler(profiler);
  LiteRtDestroyCompiledModel(compiled_model);
  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(environment);
}

// --- Error Handling Tests ---

TEST(LiteRtProfilerErrorTest, CreateWithNullProfiler) {
  EXPECT_NE(LiteRtCreateProfiler(10, nullptr), kLiteRtStatusOk);
}

TEST(LiteRtProfilerErrorTest, CreateWithNonPositiveSize) {
  EXPECT_NE(LiteRtCreateProfiler(0, &profiler), kLiteRtStatusOk);
  EXPECT_NE(LiteRtCreateProfiler(-1, &profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerErrorTest, PassNullToFunctions) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Test that all functions handle a null profiler handle gracefully.
  int num_events = 0;
  ProfiledEventData events[1];

  EXPECT_NE(LiteRtStartProfiler(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtStopProfiler(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtResetProfiler(nullptr), kLiteRtStatusOk);
  EXPECT_NE(
      LiteRtSetProfilerCurrentEventSource(nullptr, ProfiledEventSource::LITERT),
      kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetNumProfilerEvents(nullptr, &num_events), kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetProfilerEvents(nullptr, num_events, events),
            kLiteRtStatusOk);
  LiteRtDestroyProfiler(profiler);
}

TEST(LiteRtProfilerTest, PassNullToOutputPointers) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Test that functions with output pointers handle null correctly.
  ProfiledEventData events[1];
  int num_events = 1;

  EXPECT_NE(LiteRtGetNumProfilerEvents(profiler, nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetProfilerEvents(profiler, num_events, nullptr),
            kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetProfilerEvents(profiler, -1, events),
            kLiteRtStatusOk);
  LiteRtDestroyProfiler(profiler);
}
}  // namespace
