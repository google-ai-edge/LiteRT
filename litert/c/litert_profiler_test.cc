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

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_event.h"

namespace {
LiteRtProfiler profiler;

TEST(LiteRtProfilerTest, CreateAndDestroy) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // SetUp and TearDown handle the basic creation and destruction.
  // This test just verifies that the profiler handle is not null after SetUp.
  EXPECT_NE(profiler, nullptr);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, GetInitialNumEvents) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  int num_events = -1;
  EXPECT_EQ(LiteRtGetNumEvents(profiler, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(num_events, 0);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, StartAndStopProfiling) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Just test that the calls succeed.
  EXPECT_EQ(LiteRtStartProfiling(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtStopProfiling(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, ResetProfiler) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // A simple test to ensure the Reset call succeeds. A more complex test
  // would involve adding events, resetting, and then checking if the count is
  // 0.
  EXPECT_EQ(LiteRtStartProfiling(profiler), kLiteRtStatusOk);
  // In a real scenario, events would be added here.
  EXPECT_EQ(LiteRtStopProfiling(profiler), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtResetProfiler(profiler), kLiteRtStatusOk);

  int num_events = -1;
  EXPECT_EQ(LiteRtGetNumEvents(profiler, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(num_events, 0);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, SetEventSource) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // This test just verifies that the call succeeds.
  EXPECT_EQ(LiteRtSetCurrentEventSource(profiler, ProfiledEventSource::LITERT),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, GetEventsWhenEmpty) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  int num_events = 1;  // Set to a non-zero value initially
  // We expect num_events to be updated to 0.
  ProfiledEventData events[1];  // Dummy buffer
  EXPECT_EQ(LiteRtGetEvents(profiler, events, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(num_events, 0);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

// --- Error Handling Tests ---

TEST(LiteRtProfilerErrorTest, CreateWithNullProfiler) {
  EXPECT_NE(LiteRtCreateProfiler(10, nullptr), kLiteRtStatusOk);
}

TEST(LiteRtProfilerErrorTest, PassNullToFunctions) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Test that all functions handle a null profiler handle gracefully.
  int num_events;
  ProfiledEventData events[1];

  EXPECT_NE(LiteRtDestroyProfiler(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtStartProfiling(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtStopProfiling(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtResetProfiler(nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtSetCurrentEventSource(nullptr, ProfiledEventSource::LITERT),
            kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetNumEvents(nullptr, &num_events), kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetEvents(nullptr, events, &num_events), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}

TEST(LiteRtProfilerTest, PassNullToOutputPointers) {
  EXPECT_EQ(LiteRtCreateProfiler(10, &profiler), kLiteRtStatusOk);
  // Test that functions with output pointers handle null correctly.
  ProfiledEventData events[1];
  int num_events = 1;

  EXPECT_NE(LiteRtGetNumEvents(profiler, nullptr), kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetEvents(profiler, nullptr, &num_events), kLiteRtStatusOk);
  EXPECT_NE(LiteRtGetEvents(profiler, events, nullptr), kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDestroyProfiler(profiler), kLiteRtStatusOk);
}
}  // namespace
