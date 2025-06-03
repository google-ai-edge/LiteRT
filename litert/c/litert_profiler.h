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
LiteRtStatus LiteRtDestroyProfiler(LiteRtProfiler profiler);

// Starts profiling.
LiteRtStatus LiteRtStartProfiling(LiteRtProfiler profiler);

// Stops profiling.
LiteRtStatus LiteRtStopProfiling(LiteRtProfiler profiler);

// Resets the profiler.
LiteRtStatus LiteRtResetProfiler(LiteRtProfiler profiler);

// Sets the current event source.
LiteRtStatus LiteRtSetCurrentEventSource(LiteRtProfiler profiler,
                                         ProfiledEventSource event_source);

// Gets the number of events.
LiteRtStatus LiteRtGetNumEvents(LiteRtProfiler profiler, int* num_events);

// Get events.
LiteRtStatus LiteRtGetEvents(LiteRtProfiler profiler, ProfiledEventData* events,
                             int* num_events);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_H_
