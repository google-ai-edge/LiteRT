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

#include "tflite/core/api/profiler.h"
#include "tflite/profiling/memory_info.h"
#include "tflite/profiling/profile_buffer.h"

namespace litert {

// Structure to hold processed profile event data for retrieval by the user.
// This can be an extension of what ProfileBuffer might offer internally.

struct ProfiledEventData {
  const char* tag;  // Event description (e.g., "Conv2D", "LiteRT::Invoke")
  tflite::Profiler::EventType
      event_type;               // Type of event (e.g., OPERATOR_INVOKE_EVENT)
  uint64_t start_timestamp_us;  // Absolute start timestamp in microseconds
  uint64_t elapsed_time_us;     // Duration of the event in microseconds
  tflite::profiling::memory::MemoryUsage
      begin_mem_usage;  // Memory usage at the start of the event
  tflite::profiling::memory::MemoryUsage
      end_mem_usage;            // Memory usage at the end of the event
  int64_t event_metadata1;      // First metadata field (e.g., TFLite op index)
  int64_t
      event_metadata2;  // Second metadata field (e.g., TFLite subgraph index)
  enum class Source {
    LITERT,              // Event originated from LiteRT instrumentation
    TFLITE_INTERPRETER,  // Event from TFLite interpreter core
    TFLITE_DELEGATE      // Event from a TFLite delegate
  } event_source;
  // uint32_t thread_id; // Optional: if thread-specific profiling is needed
};

class LiteRtProfilerT : public tflite::Profiler {
 public:
  // Constructor: max_num_events for the internal buffer.
  explicit LiteRtProfilerT(size_t max_num_events = 1024 * 10);
  ~LiteRtProfilerT() override;

  // --- tflite::Profiler API Implementation ---
  // These methods will be called by TFLite internals when this profiler is
  // registered. They will also be used by LiteRT's ScopedProfile
  // macros/helpers.
  // event_metadata1 and event_metadata2 are used to pass additional
  // information about the event. For example, the TFLite op index for
  // OPERATOR_INVOKE_EVENT is set for event_metadata1 and the subgraph index for
  // SUBGRAPH_INVOKE_EVENT is set for event_metadata2.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  void EndEvent(uint32_t event_handle) override;

  // Optional: Override if custom logic is needed for events added with
  // duration. void AddEvent(const char* tag, EventType event_type, uint64_t
  // elapsed_time, int64_t event_metadata1, int64_t event_metadata2) override;

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
  void SetCurrentEventSource(ProfiledEventData::Source source);

 private:
  // Collection to own unique copies of tag strings
  std::set<std::string> owned_tags_set_;

  // Internal buffer to store raw event data.
  // tflite::profiling::ProfileBuffer is well-suited for this as it handles
  // event handle management, timing, and can store events from multiple
  // threads.
  std::unique_ptr<tflite::profiling::ProfileBuffer> profile_buffer_;

  bool profiling_enabled_ = false;
  ProfiledEventData::Source current_event_source_ =
      ProfiledEventData::Source::LITERT;
  // Map of event handle to event source. This is used to track the source of
  // the events that are currently active.
  std::map<uint32_t, ProfiledEventData::Source> active_event_sources_map_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_PROFILER_H_
