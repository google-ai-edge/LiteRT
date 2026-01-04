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
/// @brief Forward declaration of OpenCL event to avoid including OpenCL
/// headers.
typedef struct _cl_event* cl_event;
typedef void* EGLSyncKHR;
}

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT events, used for synchronization.
class Event : public internal::Handle<LiteRtEvent, LiteRtDestroyEvent> {
 public:
  /// @brief Creates an `Event` object from a sync fence file descriptor.
  /// @warning This is a legacy API that does not take a `LiteRtEnvironment`.
  /// New code should use the overload that accepts an environment.
  static Expected<Event> CreateFromSyncFenceFd(int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtCreateEventFromSyncFenceFd(
        /*env=*/nullptr, sync_fence_fd, owns_fd, &event));
    return Event(event, OwnHandle::kYes);
  }

  /// @brief Creates an `Event` object from a sync fence file descriptor.
  static Expected<Event> CreateFromSyncFenceFd(LiteRtEnvironment env,
                                               int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromSyncFenceFd(env, sync_fence_fd, owns_fd, &event));
    return Event(event, OwnHandle::kYes);
  }

  /// @brief Creates an `Event` object from an OpenCL event.
  static Expected<Event> CreateFromOpenClEvent(LiteRtEnvironment env,
                                               cl_event cl_event) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromOpenClEvent(env, cl_event, &event));
    return Event(event, OwnHandle::kYes);
  }

  /// @brief Creates an `Event` object from an EGL sync fence.
  /// @note This function assumes that all GL operations have already been
  /// added to the GPU command queue.
  static Expected<Event> CreateFromEglSyncFence(LiteRtEnvironment env,
                                                EGLSyncKHR egl_sync) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromEglSyncFence(env, egl_sync, &event));
    return Event(event, OwnHandle::kYes);
  }

  /// @brief Creates a managed event of a given type.
  ///
  /// Currently, only `LiteRtEventTypeOpenCl` is supported.
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

  /// @brief Returns the underlying OpenCL event if the event type is OpenCL.
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

  Expected<void*> GetCustomNativeEvent() {
    void* native = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventCustomNativeEvent(Get(), &native));
    return native;
  }

  /// @brief Waits for the event to be signaled.
  /// @param timeout_in_ms The timeout in milliseconds. A value of -1 indicates
  /// an indefinite wait.
  Expected<void> Wait(int64_t timeout_in_ms = -1) {
    LITERT_RETURN_IF_ERROR(LiteRtWaitEvent(Get(), timeout_in_ms));
    return {};
  }

  /// @brief Signals the event.
  /// @note This is only supported for OpenCL events.
  Expected<void> Signal() {
    LITERT_RETURN_IF_ERROR(LiteRtSignalEvent(Get()));
    return {};
  }

  /// @brief Returns `true` if the event is signaled.
  /// @note This is only supported for sync fence events.
  Expected<bool> IsSignaled() {
    bool is_signaled;
    LITERT_RETURN_IF_ERROR(LiteRtIsEventSignaled(Get(), &is_signaled));
    return is_signaled;
  }

  /// @brief Returns a duplicate of the event's sync fence file descriptor.
  /// @note This is only supported for sync fence events.
  Expected<int> DupFd() {
    int dup_fd;
    LITERT_RETURN_IF_ERROR(LiteRtDupFdEvent(Get(), &dup_fd));
    return dup_fd;
  }

  /// @brief Returns the underlying event type.
  LiteRtEventType Type() const {
    LiteRtEventType type;
    LiteRtGetEventEventType(Get(), &type);
    return type;
  }

  /// @internal
  /// @brief Wraps a `LiteRtEvent` C object in an `Event` C++ object.
  /// @warning This is for internal use only.
  static Event WrapCObject(LiteRtEvent event, OwnHandle owned) {
    return Event(event, owned);
  }

 private:
  /// @param owned Indicates if the created `TensorBufferRequirements` object
  /// should take ownership of the provided `requirements` handle.
  explicit Event(LiteRtEvent event, OwnHandle owned)
      : internal::Handle<LiteRtEvent, LiteRtDestroyEvent>(event, owned) {}
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_EVENT_H_
