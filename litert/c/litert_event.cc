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

#include "litert/c/litert_event.h"

#include <fcntl.h>

#include <cstdint>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/event.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateEventFromSyncFenceFd(LiteRtEnvironment env,
                                              int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  *event = new LiteRtEventT{.env = env,
                            .type = LiteRtEventTypeSyncFenceFd,
                            .fd = sync_fence_fd,
                            .owns_fd = owns_fd};
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

LiteRtStatus LiteRtCreateEventFromOpenClEvent(LiteRtEnvironment env,
                                              cl_event cl_event,
                                              LiteRtEvent* event) {
#if LITERT_HAS_OPENCL_SUPPORT
  *event = new LiteRtEventT{
      .env = env,
      .type = LiteRtEventTypeOpenCl,
      .opencl_event = cl_event,
  };
  cl_int res = tflite::gpu::cl::clRetainEvent(cl_event);
  if (res != CL_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to retain OpenCL event: %d", res);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

LiteRtStatus LiteRtGetEventEventType(LiteRtEvent event, LiteRtEventType* type) {
  *type = event->type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  if (event->type == LiteRtEventTypeSyncFenceFd) {
    *sync_fence_fd = event->fd;
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGetEventOpenClEvent(LiteRtEvent event, cl_event* cl_event) {
#if LITERT_HAS_OPENCL_SUPPORT
  if (event->type == LiteRtEventTypeOpenCl) {
    *cl_event = event->opencl_event;
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGetEventEglSync(LiteRtEvent event, EGLSyncKHR* egl_sync) {
#if LITERT_HAS_OPENGL_SUPPORT
  if (event->type == LiteRtEventTypeEglSyncFence ||
      event->type == LiteRtEventTypeEglNativeSyncFence) {
    *egl_sync = event->egl_sync;
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtCreateEventFromEglSyncFence(LiteRtEnvironment env,
                                               EGLSyncKHR egl_sync,
                                               LiteRtEvent* event) {
#if LITERT_HAS_OPENGL_SUPPORT
  LITERT_ASSIGN_OR_RETURN(LiteRtEventType type,
                          GetEventTypeFromEglSync(env, egl_sync));
  *event = new LiteRtEventT{
      .env = env,
      .type = type,
      .egl_sync = egl_sync,
  };
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

LiteRtStatus LiteRtGetEventCustomNativeEvent(LiteRtEvent event, void** native) {
#if LITERT_HAS_CUSTOM_EVENT_SUPPORT
  if (event->type == LiteRtEventTypeCustom && event->custom_event != nullptr &&
      event->custom_event->GetNative != nullptr) {
    *native = event->custom_event->GetNative(event->custom_event);
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtCreateManagedEvent(LiteRtEnvironment env,
                                      LiteRtEventType type,
                                      LiteRtEvent* event) {
  LITERT_ASSIGN_OR_RETURN(LiteRtEventT * event_res,
                          LiteRtEventT::CreateManaged(env, type));
  *event = event_res;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCustomEvent(LiteRtEvent event,
                                  LiteRtCustomEvent custom_event) {
#if LITERT_HAS_CUSTOM_EVENT_SUPPORT
  if (event->type == LiteRtEventTypeCustom) {
    if (event->custom_event != nullptr &&
        event->custom_event->Release != nullptr) {
      event->custom_event->Release(event->custom_event);
    }
    event->custom_event = custom_event;
    if (custom_event && custom_event->Retain != nullptr) {
      custom_event->Retain(custom_event);
    }
    return kLiteRtStatusOk;
  }
#endif  // LITERT_HAS_CUSTOM_EVENT_SUPPORT
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtWaitEvent(LiteRtEvent event, int64_t timeout_in_ms) {
  LITERT_RETURN_IF_ERROR(event->Wait(timeout_in_ms));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSignalEvent(LiteRtEvent event) {
  LITERT_RETURN_IF_ERROR(event->Signal());
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtIsEventSignaled(LiteRtEvent event, bool* is_signaled) {
  LITERT_ASSIGN_OR_RETURN(auto is_signaled_res, event->IsSignaled());
  *is_signaled = is_signaled_res;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDupFdEvent(LiteRtEvent event, int* dup_fd) {
  LITERT_ASSIGN_OR_RETURN(auto dup_fd_res, event->DupFd());
  *dup_fd = dup_fd_res;
  return kLiteRtStatusOk;
}

void LiteRtDestroyEvent(LiteRtEvent event) { delete event; }

#ifdef __cplusplus
}  // extern "C"
#endif
