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

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_event.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/profiler.h"

extern "C" {
LiteRtStatus LiteRtCreateProfiler(int size, LiteRtProfiler* profiler) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  *profiler =
      reinterpret_cast<LiteRtProfiler>(new litert::LiteRtProfilerT(size));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyProfiler(LiteRtProfiler profiler) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  delete reinterpret_cast<litert::LiteRtProfilerT*>(profiler);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStartProfiling(LiteRtProfiler profiler) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->StartProfiling();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStopProfiling(LiteRtProfiler profiler) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->StopProfiling();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtResetProfiler(LiteRtProfiler profiler) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->Reset();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCurrentEventSource(LiteRtProfiler profiler,
                                         ProfiledEventSource event_source) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->SetCurrentEventSource(
      event_source);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumEvents(LiteRtProfiler profiler, int* num_events) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  LITERT_RETURN_IF_ERROR(num_events,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "num_events is null.";
  *num_events =
      reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->GetNumEvents();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetEvents(LiteRtProfiler profiler, ProfiledEventData* events,
                             int* num_events) {
  LITERT_RETURN_IF_ERROR(profiler,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "profiler is null.";
  LITERT_RETURN_IF_ERROR(events, litert::ErrorStatusBuilder::InvalidArgument())
      << "events is null.";
  LITERT_RETURN_IF_ERROR(num_events,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "num_events is null.";
  auto internal_events =
      reinterpret_cast<litert::LiteRtProfilerT*>(profiler)->GetProfiledEvents();
  *num_events = internal_events.size();
  for (int i = 0; i < *num_events; ++i) {
    events[i] = internal_events[i];
  }
  return kLiteRtStatusOk;
}
}  // extern "C"
