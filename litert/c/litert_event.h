// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_EVENT_H_
#define ODML_LITERT_LITERT_C_LITERT_EVENT_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declaration of OpenCL event to avoid including OpenCL headers.
typedef struct _cl_event* cl_event;
typedef void* EGLSyncKHR;

// Creates a LiteRtEvent from a sync fence fd.
// If owns_fd is true, the LiteRtEvent will own the fd and close it when
// destroyed.
//
// Caller owns the returned LiteRtEvent. The owner is responsible for
// calling LiteRtDestroyEvent() to release the object.
LiteRtStatus LiteRtCreateEventFromSyncFenceFd(LiteRtEnvironment env,
                                              int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event);

// Another LiteRtEvent creation API for OpenCL event.
//
// Caller owns the returned LiteRtEvent (but not the cl_event). The owner is
// responsible for calling LiteRtDestroyEvent() to release the object (but not
// the underlying cl_event).
LiteRtStatus LiteRtCreateEventFromOpenClEvent(LiteRtEnvironment env,
                                              cl_event cl_event,
                                              LiteRtEvent* event);

// Another LiteRtEvent creation API for OpenCL event.
//
// Caller owns the returned LiteRtEvent. The owner is responsible for
// calling LiteRtDestroyEvent() to release the object.
LiteRtStatus LiteRtCreateEventFromEglSyncFence(LiteRtEnvironment env,
                                               EGLSyncKHR egl_sync,
                                               LiteRtEvent* event);

// Creates a LiteRtEvent with the given type.
//
// Caller owns the returned LiteRtEvent. The owner is responsible for
// calling LiteRtDestroyEvent() to release the object.
LiteRtStatus LiteRtCreateManagedEvent(LiteRtEnvironment env,
                                      LiteRtEventType type, LiteRtEvent* event);

// Sets a custom event to the LiteRtEvent. Event type must be
// LiteRtEventTypeCustom.
LiteRtStatus LiteRtSetCustomEvent(LiteRtEvent event,
                                  LiteRtCustomEvent custom_event);

LiteRtStatus LiteRtGetEventEventType(LiteRtEvent event, LiteRtEventType* type);

LiteRtStatus LiteRtGetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd);

LiteRtStatus LiteRtGetEventOpenClEvent(LiteRtEvent event, cl_event* cl_event);

LiteRtStatus LiteRtGetEventEglSync(LiteRtEvent event, EGLSyncKHR* egl_sync);

// Returns the native event of the custom event if supported.
LiteRtStatus LiteRtGetEventCustomNativeEvent(LiteRtEvent event, void** native);

// Pass -1 for timeout_in_ms for indefinite wait.
LiteRtStatus LiteRtWaitEvent(LiteRtEvent event, int64_t timeout_in_ms);

// Signals the event to notify the waiters.
LiteRtStatus LiteRtSignalEvent(LiteRtEvent event);

// Returns true if the event is signaled.
LiteRtStatus LiteRtIsEventSignaled(LiteRtEvent event, bool* is_signaled);

// Returns a dup of the event's sync fence fd.
LiteRtStatus LiteRtDupFdEvent(LiteRtEvent event, int* dup_fd);

// Destroys an owned LiteRtEvent object.
void LiteRtDestroyEvent(LiteRtEvent event);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_EVENT_H_
