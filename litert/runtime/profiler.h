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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_profiler_event.h"
#include "tflite/core/api/profiler.h"
#include "tflite/profiling/profile_buffer.h"

class LiteRtProfilerT : public tflite::Profiler {
 public:
  // Constructor: max_num_events for the internal buffer.
  explicit LiteRtProfilerT(size_t max_num_events = 1024 * 10);
  ~LiteRtProfilerT() override;

  // --- tflite::Profiler API Implementation ---
  // These methods will be called by TFLite internals when this profiler is
  // registered. They will also be used by LiteRT's ScopedProfile
  // macros/helpers.
  // tag is copied and owned by the profiler, caller does not need to keep
  // the string alive.
  // event_metadata1 and event_metadata2 are used to pass additional
  // information about the event. For example, the TFLite op index for
  // OPERATOR_INVOKE_EVENT is set for event_metadata1 and the subgraph index for
  // SUBGRAPH_INVOKE_EVENT is set for event_metadata2.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  void EndEvent(uint32_t event_handle) override;

  // tag is copied and owned by the profiler, caller does not need to keep
  // the string alive.
  // `metric` field has different intreptation based on `event_type`.
  // e.g. it means elapsed time for [DELEGATE_]OPERATOR_INVOKE_EVENT types,
  // and interprets as source and status code for TELEMETRY_[DELEGATE_]EVENT
  // event types.
  void AddEvent(const char* tag, EventType event_type, uint64_t metric,
                int64_t event_metadata1, int64_t event_metadata2) override;

  // Enables profiling. Events will start being recorded.
  void StartProfiling();

  // Disables profiling. No new events will be recorded.
  // This might also trigger processing of buffered events if not done
  // continuously.
  void StopProfiling();

  // Clears all previously collected profiling data.
  void Reset();

  // Returns true if the profiler is currently enabled and recording events.
  bool IsProfiling() const;

  // Returns the number of events currently in the buffer.
  size_t GetNumEvents() const;

  // Retrieves the collected profile events.
  // This might involve transforming data from ProfileBuffer into
  // ProfiledEventData.
  std::vector<ProfiledEventData> GetProfiledEvents() const;

  // Allows LiteRT to hint the source of the next set of events,
  // particularly useful before calling into TFLite interpreter.
  void SetCurrentEventSource(ProfiledEventSource source);

  std::string GetProfiledEventsString() const {
    std::string result;
    for (const auto& event : GetProfiledEvents()) {
      absl::StrAppend(
          &result, "tag:", event.tag, " type:", event.event_type,
          " source:", event.event_source,
          " start time:", event.start_timestamp_us,
          " elapsed time:", event.elapsed_time_us, " begin mem usage:",
          absl::StrFormat("%d", event.begin_mem_usage.total_allocated_bytes),
          " end mem usage:",
          absl::StrFormat("%d", event.end_mem_usage.total_allocated_bytes),
          " meta1:", event.event_metadata1, " meta2:", event.event_metadata2,
          "\n");
    }
    return result;
  }

 private:
  // Collection to own unique copies of tag strings
  std::set<std::string> owned_tags_set_;

  // Internal buffer to store raw event data.
  // tflite::profiling::ProfileBuffer is well-suited for this as it handles
  // event handle management, timing, and can store events from multiple
  // threads.
  std::unique_ptr<tflite::profiling::ProfileBuffer> profile_buffer_;

  bool profiling_enabled_ = false;
  ProfiledEventSource current_event_source_ = ProfiledEventSource::LITERT;
  // Map of event handle to event source. This is used to track the source of
  // the events that are currently active.
  std::map<uint32_t, ProfiledEventSource> active_event_sources_map_;
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
