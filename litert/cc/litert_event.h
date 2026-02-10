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
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_opencl_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace litert {

/// @brief Defines the C++ wrapper for LiteRT events, used for synchronization.
class Event : public internal::BaseHandle<LiteRtEvent> {
 public:
  // LINT.IfChange(event_type)
  enum class Type {
    kUnknown = LiteRtEventTypeUnknown,
    kSyncFenceFd = LiteRtEventTypeSyncFenceFd,
    kOpenCl = LiteRtEventTypeOpenCl,
    kEglSyncFence = LiteRtEventTypeEglSyncFence,
    kEglNativeSyncFence = LiteRtEventTypeEglNativeSyncFence,
    kCustom = LiteRtEventTypeCustom,
  };
  // LINT.ThenChange(../c/litert_event_type.h:event_type)

  /// @brief Creates an `Event` object from a sync fence file descriptor.
  static Expected<Event> CreateFromSyncFenceFd(const Environment& env,
                                               int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
      auto env_holder = env.GetHolder();
    LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateEventFromSyncFenceFd(
        env.Get(), sync_fence_fd, owns_fd, &event));
    return Event(env_holder, event, OwnHandle::kYes);
  }

  /// @brief Creates an `Event` object from an OpenCL event.
  static Expected<Event> CreateFromOpenClEvent(const Environment& env,
                                               LiteRtClEvent cl_event) {
    LiteRtEvent event;
    auto env_holder = env.GetHolder();
    LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateEventFromOpenClEvent(
        env_holder.handle, cl_event, &event));
    return Event(env_holder, event, OwnHandle::kYes);
  }

  /// @brief Creates an `Event` object from an EGL sync fence.
  /// @note This function assumes that all GL operations have already been
  /// added to the GPU command queue.
  static Expected<Event> CreateFromEglSyncFence(const Environment& env,
                                                LiteRtEglSyncKhr egl_sync) {
    LiteRtEvent event;
    auto env_holder = env.GetHolder();
    LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateEventFromEglSyncFence(
        env_holder.handle, egl_sync, &event));
    return Event(env_holder, event, OwnHandle::kYes);
  }

  /// @brief Creates a managed event of a given type.
  ///
  /// Currently, only `LiteRtEventTypeOpenCl` is supported.
  static Expected<Event> CreateManaged(const Environment& env, Type type) {
    LiteRtEvent event;
    auto env_holder = env.GetHolder();
    LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateManagedEvent(
        env_holder.handle, static_cast<LiteRtEventType>(type), &event));
    return Event(env_holder, event, OwnHandle::kYes);
  }

  Expected<int> GetSyncFenceFd() {
    int fd;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetEventSyncFenceFd(Get(), &fd));
    return fd;
  }

  /// @brief Returns the underlying OpenCL event if the event type is OpenCL.
  Expected<LiteRtClEvent> GetOpenClEvent() {
    LiteRtClEvent cl_event;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetEventOpenClEvent(Get(), &cl_event));
    return cl_event;
  }

  Expected<LiteRtEglSyncKhr> GetEglSync() {
    LiteRtEglSyncKhr egl_sync;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetEventEglSync(Get(), &egl_sync));
    return egl_sync;
  }

  Expected<void*> GetCustomNativeEvent() {
    void* native = nullptr;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetEventCustomNativeEvent(Get(), &native));
    return native;
  }

  /// @brief Waits for the event to be signaled.
  /// @param timeout_in_ms The timeout in milliseconds. A value of -1 indicates
  /// an indefinite wait.
  Expected<void> Wait(int64_t timeout_in_ms = -1) {
    LITERT_RETURN_IF_ERROR(env_.runtime->WaitEvent(Get(), timeout_in_ms));
    return {};
  }

  /// @brief Signals the event.
  /// @note This is only supported for OpenCL events.
  Expected<void> Signal() {
    LITERT_RETURN_IF_ERROR(env_.runtime->SignalEvent(Get()));
    return {};
  }

  /// @brief Returns `true` if the event is signaled.
  /// @note This is only supported for sync fence events.
  Expected<bool> IsSignaled() {
    bool is_signaled;
    LITERT_RETURN_IF_ERROR(env_.runtime->IsEventSignaled(Get(), &is_signaled));
    return is_signaled;
  }

  /// @brief Returns a duplicate of the event's sync fence file descriptor.
  /// @note This is only supported for sync fence events.
  Expected<int> DupFd() {
    int dup_fd;
    LITERT_RETURN_IF_ERROR(env_.runtime->DupFdEvent(Get(), &dup_fd));
    return dup_fd;
  }

  /// @brief Returns the underlying event type.
  Type Type() const {
    LiteRtEventType type;
    env_.runtime->GetEventEventType(Get(), &type);
    return static_cast<enum Type>(type);
  }

  /// @internal
  /// @brief Wraps a `LiteRtEvent` C object in an `Event` C++ object.
  /// @warning This is for internal use only.
  static Event WrapCObject(const internal::EnvironmentHolder& env,
                           LiteRtEvent event, OwnHandle owned) {
    return Event(env, event, owned);
  }

 private:
  /// @param owned Indicates if the created `TensorBufferRequirements` object
  /// should take ownership of the provided `requirements` handle.
  explicit Event(const internal::EnvironmentHolder& env, LiteRtEvent event,
                 OwnHandle owned)
      : internal::BaseHandle<LiteRtEvent>(
            event,
            [runtime = env.runtime](LiteRtEvent event) {
              runtime->DestroyEvent(event);
            },
            owned),
        env_(env) {}

  internal::EnvironmentHolder env_;
};

}  // namespace litert

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_CC_LITERT_EVENT_H_
