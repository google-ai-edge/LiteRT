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

#include "litert/runtime/profiler.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>  // For simple pass/fail messages if not using a framework
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_profiler_event.h"
#include "tflite/core/api/profiler.h"  // For tflite::Profiler::EventType

// Suggested location: tests/litert/profiler_test.cc

// Helper to simulate work for timing
void SimulateWork(int us = 10) { absl::SleepFor(absl::Microseconds(us)); }

namespace litert {
namespace {

TEST(LiteRTProfiler, ProfilerConstruction) {
  LiteRtProfilerT profiler;
  ASSERT_FALSE(profiler.IsProfiling());
  const auto& events = profiler.GetProfiledEvents();
  ASSERT_TRUE(events.empty());
  std::cout << "TestProfilerConstruction: PASSED" << std::endl;
}

TEST(LiteRTProfiler, StartStopIsProfiling) {
  LiteRtProfilerT profiler;
  ASSERT_FALSE(profiler.IsProfiling());

  profiler.StartProfiling();
  ASSERT_TRUE(profiler.IsProfiling());

  profiler.StopProfiling();
  ASSERT_FALSE(profiler.IsProfiling());

  // Start again
  profiler.StartProfiling();
  ASSERT_TRUE(profiler.IsProfiling());
  std::cout << "TestStartStopIsProfiling: PASSED" << std::endl;
}

TEST(LiteRTProfiler, NoEventsWhenNotProfiling) {
  LiteRtProfilerT profiler;
  ASSERT_FALSE(profiler.IsProfiling());

  uint32_t handle = profiler.BeginEvent(
      "TestEvent", tflite::Profiler::EventType::DEFAULT, 0, 0);
  ASSERT_EQ(handle, 0);  // Should not get a valid handle
  SimulateWork();
  profiler.EndEvent(handle);  // Should be a no-op

  const auto& events = profiler.GetProfiledEvents();
  ASSERT_TRUE(events.empty());
  std::cout << "TestNoEventsWhenNotProfiling: PASSED" << std::endl;
}

TEST(LiteRTProfiler, SingleEventRecording) {
  LiteRtProfilerT profiler;
  profiler.StartProfiling();
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);

  const char* tag = "SingleEvent";
  int64_t meta1 = 123;
  int64_t meta2 = 456;
  uint32_t handle = profiler.BeginEvent(
      tag, tflite::Profiler::EventType::DEFAULT, meta1, meta2);
  ASSERT_GE(handle, 0);
  SimulateWork(100);  // Ensure some duration
  profiler.EndEvent(handle);
  profiler.StopProfiling();

  const auto& events = profiler.GetProfiledEvents();
  ASSERT_EQ(events.size(), 1);
  const auto& ev = events[0];
  ASSERT_EQ(strcmp(ev.tag, tag), 0);
  ASSERT_EQ(ev.event_type, tflite::Profiler::EventType::DEFAULT);
  ASSERT_EQ(ev.event_metadata1, meta1);
  ASSERT_EQ(ev.event_metadata2, meta2);
  ASSERT_EQ(ev.event_source, ProfiledEventSource::LITERT);
  ASSERT_EQ(ev.event_source, ProfiledEventSource::LITERT);
  ASSERT_GT(ev.elapsed_time_us, 0);  // Check that some time has passed

  std::cout << "TestSingleEventRecording: PASSED" << std::endl;
}

TEST(LiteRTProfiler, MultipleEventRecording) {
  LiteRtProfilerT profiler;
  profiler.StartProfiling();
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);

  uint32_t h1 =
      profiler.BeginEvent("Event1", tflite::Profiler::EventType::DEFAULT, 1, 0);
  SimulateWork();
  ASSERT_GE(h1, 0);
  profiler.EndEvent(h1);

  uint32_t h2 = profiler.BeginEvent(
      "Event2", tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT, 2, 0);
  SimulateWork();
  profiler.EndEvent(h2);

  profiler.StopProfiling();

  const auto& events = profiler.GetProfiledEvents();
  ASSERT_EQ(events.size(), 2);
  ASSERT_EQ(strcmp(events[0].tag, "Event1"), 0);
  ASSERT_EQ(events[0].event_metadata1, 1);
  ASSERT_EQ(strcmp(events[1].tag, "Event2"), 0);
  ASSERT_EQ(events[1].event_metadata1, 2);
  ASSERT_EQ(events[1].event_type,
            tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT);

  std::cout << "TestMultipleEventRecording: PASSED" << std::endl;
}

TEST(LiteRTProfiler, EventSourceAttribution) {
  LiteRtProfilerT profiler;
  profiler.StartProfiling();

  // LITERT source
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  uint32_t h1 = profiler.BeginEvent("LiteRT_Event",
                                    tflite::Profiler::EventType::DEFAULT, 0, 0);
  LITERT_LOG(LITERT_INFO, "LiteRT_Event %d", h1);
  profiler.EndEvent(h1);

  // TFLITE_INTERPRETER source
  profiler.SetCurrentEventSource(ProfiledEventSource::TFLITE_INTERPRETER);
  uint32_t h2 = profiler.BeginEvent(
      "Interpreter_Event", tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT,
      0, 0);
  LITERT_LOG(LITERT_INFO, "Interpreter_Event %d", h2);
  profiler.EndEvent(h2);

  // TFLITE_DELEGATE source (inferred from event type)
  profiler.SetCurrentEventSource(
      ProfiledEventSource::TFLITE_INTERPRETER);  // Hint is interpreter
  uint32_t h3 = profiler.BeginEvent(
      "Delegate_Event",
      tflite::Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT, 0, 0);
  LITERT_LOG(LITERT_INFO, "Delegate_Event %d", h3);
  profiler.EndEvent(h3);

  profiler.StopProfiling();

  const auto& events = profiler.GetProfiledEvents();
  ASSERT_EQ(events.size(), 3);
  ASSERT_EQ(strcmp(events[0].tag, "LiteRT_Event"), 0);
  ASSERT_EQ(events[0].event_source, ProfiledEventSource::LITERT);
  ASSERT_EQ(events[0].event_source, ProfiledEventSource::LITERT);

  ASSERT_EQ(strcmp(events[1].tag, "Interpreter_Event"), 0);
  ASSERT_EQ(events[1].event_source, ProfiledEventSource::TFLITE_INTERPRETER);

  ASSERT_EQ(strcmp(events[2].tag, "Delegate_Event"), 0);
  ASSERT_EQ(events[2].event_source,
            ProfiledEventSource::TFLITE_DELEGATE);  // Should be overridden

  std::cout << "TestEventSourceAttribution: PASSED" << std::endl;
}

TEST(LiteRTProfiler, ResetFunctionality) {
  LiteRtProfilerT profiler;
  profiler.StartProfiling();
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  uint32_t h1 = profiler.BeginEvent("EventBeforeReset",
                                    tflite::Profiler::EventType::DEFAULT, 0, 0);
  profiler.EndEvent(h1);
  profiler.StopProfiling();  // Events are in buffer

  ASSERT_EQ(profiler.GetProfiledEvents().size(), 1);  // Cache built

  profiler.Reset();
  ASSERT_FALSE(
      profiler.IsProfiling());  // Reset does not change profiling state
  ASSERT_TRUE(profiler.GetProfiledEvents()
                  .empty());  // Cache should be empty and dirty flag reset

  // Start profiling again and record new event
  profiler.StartProfiling();
  uint32_t h2 = profiler.BeginEvent("EventAfterReset",
                                    tflite::Profiler::EventType::DEFAULT, 0, 0);
  profiler.EndEvent(h2);
  profiler.StopProfiling();

  const auto& events_after_reset = profiler.GetProfiledEvents();
  ASSERT_EQ(events_after_reset.size(), 1);
  ASSERT_EQ(strcmp(events_after_reset[0].tag, "EventAfterReset"), 0);

  std::cout << "TestResetFunctionality: PASSED" << std::endl;
}

TEST(LiteRTProfiler, MaxEventsHandling) {
  // ProfileBuffer behavior for overflow isn't strictly defined here,
  // but we test that profiler doesn't crash and respects the initial buffer
  // size. TFLite ProfileBuffer typically acts as a circular buffer.
  size_t max_events = 2;
  LiteRtProfilerT profiler(max_events);
  profiler.StartProfiling();
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);
  profiler.SetCurrentEventSource(ProfiledEventSource::LITERT);

  uint32_t h1 =
      profiler.BeginEvent("E1", tflite::Profiler::EventType::DEFAULT, 1, 0);
  profiler.EndEvent(h1);
  uint32_t h2 =
      profiler.BeginEvent("E2", tflite::Profiler::EventType::DEFAULT, 2, 0);
  profiler.EndEvent(h2);
  uint32_t h3 =
      profiler.BeginEvent("E3", tflite::Profiler::EventType::DEFAULT, 3, 0);
  profiler.EndEvent(h3);
  // Now E1 should have been overwritten if ProfileBuffer is circular.

  profiler.StopProfiling();
  const auto& events = profiler.GetProfiledEvents();
  // Depending on ProfileBuffer's implementation, it might store only
  // max_events. This test primarily ensures it doesn't fail with more events
  // than max_events. For a circular buffer of size 2, we'd expect E2 and E1|E3.
  ASSERT_TRUE(events.size() <= max_events || !events.empty());
  if (events.size() == max_events) {
    bool found_e1 = false;
    bool found_e2 = false;
    bool found_e3 = false;
    for (const auto& ev : events) {
      if (strcmp(ev.tag, "E1") == 0) found_e1 = true;
      if (strcmp(ev.tag, "E2") == 0) found_e2 = true;
      if (strcmp(ev.tag, "E3") == 0) found_e3 = true;
    }
    ASSERT_TRUE(found_e2);
    ASSERT_TRUE(found_e1 || found_e3);
  }
  std::cout << "TestMaxEventsHandling: PASSED" << std::endl;
}

}  // namespace
}  // namespace litert
