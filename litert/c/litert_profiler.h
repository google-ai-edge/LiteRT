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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_event.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a compilation option object.
LiteRtStatus LiteRtCreateProfiler(int size, LiteRtProfiler* profiler);

// Destroys a compilation option object.
void LiteRtDestroyProfiler(LiteRtProfiler profiler);

// Starts profiling.
LiteRtStatus LiteRtStartProfiler(LiteRtProfiler profiler);

// Stops profiling.
LiteRtStatus LiteRtStopProfiler(LiteRtProfiler profiler);

// Resets the profiler.
LiteRtStatus LiteRtResetProfiler(LiteRtProfiler profiler);

// Sets the current event source.
LiteRtStatus LiteRtSetProfilerCurrentEventSource(
    LiteRtProfiler profiler, ProfiledEventSource event_source);

// Gets the number of events.
LiteRtStatus LiteRtGetNumProfilerEvents(LiteRtProfiler profiler,
                                        int* num_events);

// Get events. The events are copied to the provided buffer, caller is
// responsible to allocate the buffer and provide the size of the buffer
// (num_events).
LiteRtStatus LiteRtGetProfilerEvents(LiteRtProfiler profiler, int num_events,
                                     ProfiledEventData* events);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_H_
