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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_profiler_event.h"
#include "tflite/profiling/profile_buffer.h"  // IWYU pragma: keep

LiteRtProfilerT::LiteRtProfilerT(size_t max_num_events)
    : profiling_enabled_(false),
      current_event_source_(ProfiledEventSource::LITERT) {
  // Initialize the profile buffer with the TFLite metadata version
  // and the maximum number of events.
  profile_buffer_ = std::make_unique<tflite::profiling::ProfileBuffer>(
      max_num_events, false);
}

LiteRtProfilerT::~LiteRtProfilerT() = default;

uint32_t LiteRtProfilerT::BeginEvent(const char* tag, EventType event_type,
                              int64_t event_metadata1,
                              int64_t event_metadata2) {
  if (!profiling_enabled_ || !profile_buffer_) {
    return 0;  // Return an invalid handle
  }

  // 1. Convert input tag to std::string and insert into the set to get a
  // unique, owned string.
  std::string s_tag(tag);
  auto [it, inserted] = owned_tags_set_.insert(std::move(s_tag));
  const char* owned_tag_ptr = it->c_str();

  // Delegate to the owned ProfileBuffer to record the event and get a handle
  uint32_t event_handle = profile_buffer_->BeginEvent(
      owned_tag_ptr, event_type, event_metadata1, event_metadata2);

  if (event_handle != 0) {
    // Determine the effective source for this specific event
    ProfiledEventSource effective_source = this->current_event_source_;
    if (event_type == EventType::DELEGATE_OPERATOR_INVOKE_EVENT ||
        event_type == EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT) {
      effective_source = ProfiledEventSource::TFLITE_DELEGATE;
    }
    // Store our determined source associated with the handle
    active_event_sources_map_[event_handle] = effective_source;
  }
  return event_handle;
}

void LiteRtProfilerT::EndEvent(uint32_t event_handle) {
  if (!profiling_enabled_ || !profile_buffer_ || event_handle < 0) {
    return;
  }
  profile_buffer_->EndEvent(event_handle);
  // The source associated with event_handle in active_event_sources_map_
  // will be used when GetProfiledEvents is called.
  // We don't remove from active_event_sources_map_ here; it's cleared on Reset.
}

void LiteRtProfilerT::AddEvent(const char* tag, EventType event_type,
                               uint64_t metric, int64_t event_metadata1,
                               int64_t event_metadata2) {
  if (!profiling_enabled_ || !profile_buffer_) {
    return;
  }
  // 1. Convert input tag to std::string and insert into the set to get a
  // unique, owned string.
  std::string s_tag(tag);
  auto [it, inserted] = owned_tags_set_.insert(std::move(s_tag));
  const char* owned_tag_ptr = it->c_str();
  profile_buffer_->AddEvent(owned_tag_ptr, event_type, metric, event_metadata1,
                            event_metadata2);
}

void LiteRtProfilerT::StartProfiling() {
  if (!profile_buffer_) {
    // Or handle error: Profiler not properly initialized
    return;
  }
  // Reset previous data if starting a new session without explicit reset
  profile_buffer_->Reset();
  active_event_sources_map_.clear();
  current_event_source_ = ProfiledEventSource::LITERT;

  profiling_enabled_ = true;
  profile_buffer_->SetEnabled(profiling_enabled_);
}

void LiteRtProfilerT::StopProfiling() {
  profiling_enabled_ = false;
  profile_buffer_->SetEnabled(profiling_enabled_);
  // Events already in the buffer are preserved until Reset() or
  // GetProfiledEvents().
}

bool LiteRtProfilerT::IsProfiling() const {
  return profiling_enabled_;
}

void LiteRtProfilerT::Reset() {
  if (profile_buffer_) {
    profile_buffer_->Reset();
  }
  active_event_sources_map_.clear();
  current_event_source_ = ProfiledEventSource::LITERT;  // Reset hint
  // litert_profiling_enabled_ remains as is, Reset just clears data.
}

std::vector<ProfiledEventData> LiteRtProfilerT::GetProfiledEvents() const {
  std::vector<ProfiledEventData> result_events;

if (!profile_buffer_) {
    return result_events;
  }

  // Get the raw events from the owned ProfileBuffer.
  // These events are typically TFLite's internal representation.
  for (int i = 0; i < profile_buffer_->Size(); ++i) {
    ProfiledEventData ev_data;
    auto tf_event = profile_buffer_->At(i);
    ev_data.tag = tf_event->tag.c_str();
    ev_data.event_type = tf_event->event_type;
    ev_data.start_timestamp_us = tf_event->begin_timestamp_us;
    ev_data.elapsed_time_us = tf_event->elapsed_time;
    ev_data.event_metadata1 = tf_event->event_metadata;
    ev_data.event_metadata2 = tf_event->extra_event_metadata;
    ev_data.begin_mem_usage = tf_event->begin_mem_usage;
    ev_data.end_mem_usage = tf_event->end_mem_usage;

    // Look up our attributed source
    auto it = active_event_sources_map_.find(i);
    if (it != active_event_sources_map_.end()) {
      ev_data.event_source = it->second;
    } else {
      // This case could happen if an event was somehow incompletely recorded
      // or if handles are reused in a way not perfectly aligned with map
      // clearing. A robust profiler might log this. For now, fallback.
      ev_data.event_source = ProfiledEventSource::LITERT;  // Default fallback
    }
    result_events.push_back(ev_data);
  }

  // GetProfiledEvents is non-consuming, Reset() is the explicit way to clear
  // data.
  return result_events;
}

void LiteRtProfilerT::SetCurrentEventSource(ProfiledEventSource source_hint) {
  this->current_event_source_ = source_hint;
}

size_t LiteRtProfilerT::GetNumEvents() const {
  return profile_buffer_->Size();
}
