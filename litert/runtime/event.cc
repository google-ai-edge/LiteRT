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

#include "litert/runtime/event.h"

#include <fcntl.h>

#include <cerrno>
#include <cstdint>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_gl_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/gpu_environment.h"

#if LITERT_HAS_SYNC_FENCE_SUPPORT
#include <poll.h>
#include <unistd.h>
#endif  // LITERT_HAS_SYNC_FENCE_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

using litert::Error;
using litert::Expected;

Expected<void> LiteRtEventT::Wait(int64_t timeout_in_ms) {
  if (type == LiteRtEventTypeSyncFenceFd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
    pollfd fds = {
        .fd = fd,
        .events = POLLIN,
    };

    int ret;
    do {
      ret = poll(&fds, 1, timeout_in_ms);
      if (ret == 1) {
        break;
      }
      if (ret == 0) {
        return Error(kLiteRtStatusErrorTimeoutExpired, "Timeout expired");
      }
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

    if (ret < 0) {
      return Error(kLiteRtStatusErrorRuntimeFailure, "Error waiting for fence");
    }

    return {};

#else
    return Error(kLiteRtStatusErrorUnsupported,
                 "LiteRtEventWait not implemented for this platform");
#endif
  }
  if (type == LiteRtEventTypeOpenCl) {
#if LITERT_HAS_OPENCL_SUPPORT
    cl_int res = tflite::gpu::cl::clWaitForEvents(/*num_events=*/1,
                                                  /*event_list=*/&opencl_event);
    if (res != CL_SUCCESS) {
      return Error(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat("clWaitForEvents fails with error code %d", res));
    }
    return {};
#else
    return Error(kLiteRtStatusErrorUnsupported,
                 "LiteRtEventWait not implemented for this platform");
#endif
  }
  return Error(kLiteRtStatusErrorInvalidArgument, "Invalid event type");
}

#if LITERT_HAS_SYNC_FENCE_SUPPORT
namespace {
bool IsFdValid(int fd) { return fcntl(fd, F_GETFD) != -1 || errno != EBADF; }
}  // namespace
#endif

LiteRtEventT::~LiteRtEventT() {
  if (type == LiteRtEventTypeSyncFenceFd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
    if (owns_fd && IsFdValid(fd)) {
      close(fd);
    }
#endif
  } else if (type == LiteRtEventTypeEglSyncFence ||
             type == LiteRtEventTypeEglNativeSyncFence) {
#if LITERT_HAS_OPENGL_SUPPORT

    auto gpu_env = env->GetGpuEnvironment();
    EGLDisplay display = (*gpu_env)->GetEglDisplay();
    if (display == EGL_NO_DISPLAY) {
      LITERT_LOG(LITERT_ERROR,
                 "Cannot destroy EGL sync: EGL display is EGL_NO_DISPLAY");
      return;
    }

    static auto* egl_destroy_sync_khr =
        reinterpret_cast<decltype(&eglDestroySyncKHR)>(
            eglGetProcAddress("eglDestroySyncKHR"));
    if (egl_destroy_sync_khr == nullptr) {
      LITERT_LOG(LITERT_ERROR,
                 "Cannot destroy EGL sync: Failed to load eglDestroySyncKHR");
      return;
    }

    EGLBoolean destroy_success = egl_destroy_sync_khr(display, egl_sync);
    if (destroy_success == EGL_FALSE) {
      LITERT_LOG(LITERT_ERROR,
                 "EGL sync destroy failed: eglDestroySyncKHR failed");
    }
#endif  // LITERT_HAS_OPENGL_SUPPORT
  } else if (type == LiteRtEventTypeOpenCl) {
#if LITERT_HAS_OPENCL_SUPPORT
    tflite::gpu::cl::clReleaseEvent(opencl_event);
#endif  // LITERT_HAS_OPENCL_SUPPORT
  }
}

Expected<int> LiteRtEventT::GetSyncFenceFd() {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  if (type == LiteRtEventTypeSyncFenceFd) {
    return fd;
  }
  if (type == LiteRtEventTypeEglNativeSyncFence) {
    return litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Querying of sync fence fd(EGL_SYNC_NATIVE_FENCE_FD_ANDROID) is not "
        "allowed by EGL. Call DupFd() instead.");
  }
  return litert::Unexpected(
      kLiteRtStatusErrorInvalidArgument,
      absl::StrFormat("GetSyncFenceFd is not supported for this event type: %d",
                      static_cast<int>(type)));
#else
  return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "Sync fence is not supported on this platform");
#endif
}

Expected<void> LiteRtEventT::Signal() {
#if LITERT_HAS_OPENCL_SUPPORT
  if (type == LiteRtEventTypeOpenCl) {
    cl_int res =
        tflite::gpu::cl::clSetUserEventStatus(opencl_event, CL_COMPLETE);
    if (res != CL_SUCCESS) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   absl::StrFormat(
                       "clSetUserEventStatus fails with error code %d", res));
    }
  }
#endif
  return Error(kLiteRtStatusErrorInvalidArgument,
               "The event signal is not supported");
}

Expected<LiteRtEventT*> LiteRtEventT::CreateManaged(LiteRtEnvironment env,
                                                    LiteRtEventType type) {
  if (type == LiteRtEventTypeOpenCl) {
#if LITERT_HAS_OPENCL_SUPPORT
    LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment());
    cl_int res;
    cl_event user_event = tflite::gpu::cl::clCreateUserEvent(
        gpu_env->GetContext()->context(), &res);
    if (res != CL_SUCCESS) {
      return Error(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat("clCreateUserEvent fails with error code %d", res));
    }
    return new LiteRtEventT{
        .env = env,
        .type = LiteRtEventTypeOpenCl,
        .opencl_event = user_event,
    };
#else
    return Error(
        kLiteRtStatusErrorUnsupported,
        "Creating managed OpenCL event is not supported on this platform");
#endif
  }
  if (type == LiteRtEventTypeEglSyncFence) {
#if LITERT_HAS_OPENGL_SUPPORT
    LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment());
    EGLDisplay display = gpu_env->GetEglDisplay();
    LITERT_RETURN_IF_ERROR(display != EGL_NO_DISPLAY,
                           litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                              "Failed to get EGL display"));
    static auto* egl_create_sync_khr =
        reinterpret_cast<decltype(&eglCreateSyncKHR)>(
            eglGetProcAddress("eglCreateSyncKHR"));
    LITERT_RETURN_IF_ERROR(
        egl_create_sync_khr != nullptr,
        litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to load EGL function: eglCreateSyncKHR"));

    EGLSyncKHR egl_sync =
        egl_create_sync_khr(display, EGL_SYNC_FENCE_KHR, nullptr);
    LITERT_RETURN_IF_ERROR(egl_sync != EGL_NO_SYNC_KHR,
                           litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                              "Failed to create EGLSyncKHR"));
    return new LiteRtEventT{
        .env = env,
        .type = LiteRtEventTypeEglSyncFence,
        .egl_sync = egl_sync,
    };

#else
    return Error(kLiteRtStatusErrorUnsupported,
                 "Creating managed EGLSyncFence event is not supported on this "
                 "platform");
#endif  // LITERT_HAS_OPENGL_SUPPORT
  }
  if (type == LiteRtEventTypeEglNativeSyncFence) {
#if LITERT_HAS_OPENGL_SUPPORT
    LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment());
    EGLDisplay display = gpu_env->GetEglDisplay();
    LITERT_RETURN_IF_ERROR(display != EGL_NO_DISPLAY,
                           litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                              "Failed to get EGL display"));
    static auto* egl_create_sync_khr =
        reinterpret_cast<decltype(&eglCreateSyncKHR)>(
            eglGetProcAddress("eglCreateSyncKHR"));
    LITERT_RETURN_IF_ERROR(
        egl_create_sync_khr != nullptr,
        litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to load EGL function: eglCreateSyncKHR"));

    EGLSyncKHR egl_sync =
        egl_create_sync_khr(display, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
    LITERT_RETURN_IF_ERROR(egl_sync != EGL_NO_SYNC_KHR,
                           litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                              "Failed to create EGLSyncKHR"));

    return new LiteRtEventT{
        .env = env,
        .type = LiteRtEventTypeEglNativeSyncFence,
        .egl_sync = egl_sync,
    };
#else
    return Error(kLiteRtStatusErrorUnsupported,
                 "Creating managed EGLNativeSyncFence event is not supported "
                 "on this platform");
#endif  // LITERT_HAS_OPENGL_SUPPORT
  }

  return Error(kLiteRtStatusErrorInvalidArgument,
               absl::StrFormat("CreateManaged doesn't support type %d", type));
}

Expected<bool> LiteRtEventT::IsSignaled() const {
  if (type != LiteRtEventTypeSyncFenceFd) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "IsSignaled is not supported for this event type");
  }
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  LITERT_RETURN_IF_ERROR(fd >= 0) << "Invalid fd";

  pollfd fds = {
      .fd = fd,
      .events = POLLIN,
  };

  int ret;
  do {
    ret = poll(&fds, 1, /*timeout_in_ms=*/0);
    if (ret == 1) {
      LITERT_RETURN_IF_ERROR((fds.revents & POLLERR) == 0) << "POLLERR error";
      LITERT_RETURN_IF_ERROR((fds.revents & POLLNVAL) == 0) << "POLLNVAL error";
      return true;
    }
    if (ret == 0) {
      return false;
    }
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  return Error(kLiteRtStatusErrorRuntimeFailure,
               absl::StrFormat("Failed to check if fd %d is signaled", fd));
#else
  return Error(kLiteRtStatusErrorUnsupported,
               "LiteRT does not have sync fence support enabled.");
#endif
}

Expected<int> LiteRtEventT::DupFd() const {
  if (type == LiteRtEventTypeSyncFenceFd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
    int dup_fd = dup(fd);
    LITERT_RETURN_IF_ERROR(dup_fd >= 0) << "Failed to dup fd " << fd;
    return dup_fd;
#else
  return Error(kLiteRtStatusErrorUnsupported,
               "LiteRT does not have sync fence support enabled.");

#endif  // LITERT_HAS_SYNC_FENCE_SUPPORT
  }
  if (type == LiteRtEventTypeEglNativeSyncFence) {
#if LITERT_HAS_OPENGL_SUPPORT
    LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment());
    EGLDisplay display = gpu_env->GetEglDisplay();
    static auto* egl_dup_native_fence_fd_android =
        reinterpret_cast<decltype(&eglDupNativeFenceFDANDROID)>(
            eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    LITERT_RETURN_IF_ERROR(
        egl_dup_native_fence_fd_android != nullptr,
        litert::Unexpected(
            kLiteRtStatusErrorRuntimeFailure,
            "Failed to load EGL function: eglDupNativeFenceFDANDROID"));
    int egl_sync_fd = egl_dup_native_fence_fd_android(display, egl_sync);
    LITERT_RETURN_IF_ERROR(
        egl_sync_fd != -1,
        litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                           "Failed to dup EGL native sync fence."));
    return egl_sync_fd;
#else
    return Error(kLiteRtStatusErrorUnsupported,
                 "LiteRT does not have EGL native fence support enabled.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
  }
  return Error(
      kLiteRtStatusErrorInvalidArgument,
      absl::StrFormat("DupFd is not supported for this event type: %d", type));
}

Expected<LiteRtEventType> GetEventTypeFromEglSync(LiteRtEnvironment env,
                                                  EGLSyncKHR egl_sync) {
#if LITERT_HAS_OPENGL_SUPPORT
  LITERT_RETURN_IF_ERROR(egl_sync != EGL_NO_SYNC_KHR);
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, env->GetGpuEnvironment());
  EGLDisplay display = gpu_env->GetEglDisplay();
  LITERT_RETURN_IF_ERROR(display != EGL_NO_DISPLAY);
  static auto* egl_get_sync_attrib_khr =
      reinterpret_cast<decltype(&eglGetSyncAttribKHR)>(
          eglGetProcAddress("eglGetSyncAttribKHR"));
  LITERT_RETURN_IF_ERROR(
      egl_get_sync_attrib_khr != nullptr,
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to load EGL function: eglGetSyncAttribKHR"));
  EGLint sync_type;
  const EGLBoolean success =
      egl_get_sync_attrib_khr(display, egl_sync, EGL_SYNC_TYPE_KHR, &sync_type);
  LITERT_RETURN_IF_ERROR(
      success,
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "eglGetSyncAttribKHR: Failed to get EGL sync type"));
  if (sync_type == EGL_SYNC_FENCE_KHR) {
    return LiteRtEventTypeEglSyncFence;
  }
  if (sync_type == EGL_SYNC_NATIVE_FENCE_ANDROID) {
    return LiteRtEventTypeEglNativeSyncFence;
  }
  return litert::Unexpected(
      kLiteRtStatusErrorInvalidArgument,
      absl::StrFormat("EGL sync type %d is not supported", sync_type));
#else
  return Error(kLiteRtStatusErrorUnsupported,
               "LiteRT does not have OpenGL support enabled.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}
