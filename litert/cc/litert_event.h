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

#ifndef ODML_LITERT_LITERT_CC_LITERT_EVENT_H_
#define ODML_LITERT_LITERT_CC_LITERT_EVENT_H_

#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

extern "C" {
// Forward declaration of OpenCL event to avoid including OpenCL headers.
typedef struct _cl_event* cl_event;
typedef void* EGLSyncKHR;
}

namespace litert {

class Event : public internal::Handle<LiteRtEvent, LiteRtDestroyEvent> {
 public:
  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit Event(LiteRtEvent event, OwnHandle owned)
      : internal::Handle<LiteRtEvent, LiteRtDestroyEvent>(event, owned) {}

  // Creates an Event object with the given `sync_fence_fd`.
  //
  // Warning: This is an old API that does not take LiteRtEnvironment. It is
  // provided for backward compatibility. New code should use the other
  // CreateFromSyncFenceFd() API.
  static Expected<Event> CreateFromSyncFenceFd(int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtCreateEventFromSyncFenceFd(
        /*env=*/nullptr, sync_fence_fd, owns_fd, &event));
    return Event(event, OwnHandle::kYes);
  }

  // Creates an Event object with the given `sync_fence_fd`.
  static Expected<Event> CreateFromSyncFenceFd(LiteRtEnvironment env,
                                               int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromSyncFenceFd(env, sync_fence_fd, owns_fd, &event));
    return Event(event, OwnHandle::kYes);
  }

  // Creates an Event object with the given `cl_event`.
  static Expected<Event> CreateFromOpenClEvent(LiteRtEnvironment env,
                                               cl_event cl_event) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromOpenClEvent(env, cl_event, &event));
    return Event(event, OwnHandle::kYes);
  }

  // Creates an Event object with the given `egl_sync`.
  // Note: This function assumes that all GL operations have been already added
  // to the GPU command queue.
  static Expected<Event> CreateFromEglSyncFence(LiteRtEnvironment env,
                                                EGLSyncKHR egl_sync) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromEglSyncFence(env, egl_sync, &event));
    return Event(event, OwnHandle::kYes);
  }

  // Creates a managed event of the given `type`. Currently only
  // LiteRtEventTypeOpenCl is supported.
  static Expected<Event> CreateManaged(LiteRtEnvironment env,
                                       LiteRtEventType type) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedEvent(env, type, &event));
    return Event(event, OwnHandle::kYes);
  }

  Expected<int> GetSyncFenceFd() {
    int fd;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventSyncFenceFd(Get(), &fd));
    return fd;
  }

  // Returns the underlying OpenCL event if the event type is OpenCL.
  Expected<cl_event> GetOpenClEvent() {
    cl_event cl_event;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventOpenClEvent(Get(), &cl_event));
    return cl_event;
  }

  Expected<EGLSyncKHR> GetEglSync() {
    EGLSyncKHR egl_sync;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventEglSync(Get(), &egl_sync));
    return egl_sync;
  }

  // Pass -1 for timeout_in_ms for indefinite wait.
  Expected<void> Wait(int64_t timeout_in_ms = -1) {
    LITERT_RETURN_IF_ERROR(LiteRtWaitEvent(Get(), timeout_in_ms));
    return {};
  }

  // Signals the event.
  // Note: This is only supported for OpenCL events.
  Expected<void> Signal() {
    LITERT_RETURN_IF_ERROR(LiteRtSignalEvent(Get()));
    return {};
  }

  // Returns true if the event is signaled.
  // Note: This is only supported for sync fence events.
  Expected<bool> IsSignaled() {
    bool is_signaled;
    LITERT_RETURN_IF_ERROR(LiteRtIsEventSignaled(Get(), &is_signaled));
    return is_signaled;
  }

  // Returns a dup of the event's sync fence fd.
  // Note: This is only supported for sync fence events.
  Expected<int> DupFd() {
    int dup_fd;
    LITERT_RETURN_IF_ERROR(LiteRtDupFdEvent(Get(), &dup_fd));
    return dup_fd;
  }

  // Returns the underlying event type.
  LiteRtEventType Type() const {
    LiteRtEventType type;
    LiteRtGetEventEventType(Get(), &type);
    return type;
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_EVENT_H_
