#include "litert/kotlin/src/main/jni/litert_event_jni.h"

#include <jni.h>

#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_logging.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromSyncFenceFd(
    JNIEnv* env, jclass clazz, jint sync_fence_fd, jboolean owns_fd) {
  LiteRtEvent event = nullptr;
  LiteRtStatus status = LiteRtCreateEventFromSyncFenceFd(
      static_cast<int>(sync_fence_fd), (owns_fd == JNI_TRUE), &event);
  if (status != kLiteRtStatusOk || !event) {
    LITERT_LOG(LITERT_ERROR, "Failed to create event from sync fence fd.");
    return 0;
  }
  return reinterpret_cast<jlong>(event);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromOpenClEvent(
    JNIEnv* env, jclass clazz, jlong cl_event_handle) {
#if defined(LITERT_HAS_OPENCL_SUPPORT) || defined(__ANDROID__)
  LiteRtEvent event = nullptr;
  // Convert the handle to an OpenCL event
  cl_event c_ev = reinterpret_cast<cl_event>(cl_event_handle);

  LiteRtStatus status = LiteRtCreateEventFromOpenClEvent(c_ev, &event);
  if (status != kLiteRtStatusOk || !event) {
    LITERT_LOG(LITERT_ERROR, "Failed to create event from OpenCL event.");
    return 0;
  }
  return reinterpret_cast<jlong>(event);
#else
  LITERT_LOG(LITERT_ERROR, "OpenCL not supported in this build.");
  return 0;
#endif
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateManaged(JNIEnv* env,
                                                         jclass clazz,
                                                         jint event_type) {
  LiteRtEvent event = nullptr;
  // Convert Java int to LiteRtEventType enum
  auto ctype = static_cast<LiteRtEventType>(event_type);

  LiteRtStatus status = LiteRtCreateManagedEvent(ctype, &event);
  if (status != kLiteRtStatusOk || !event) {
    LITERT_LOG(LITERT_ERROR, "Failed to create managed event with type=%d.",
               event_type);
    return 0;
  }
  return reinterpret_cast<jlong>(event);
}

JNIEXPORT jint JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetSyncFenceFd(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle) {
  if (!event_handle) return -1;
  int fd_out = -1;
  LiteRtStatus status = LiteRtGetEventSyncFenceFd(
      reinterpret_cast<LiteRtEvent>(event_handle), &fd_out);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING,
               "GetSyncFenceFd failed or not a sync fence event.");
  }
  return fd_out;
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetOpenClEvent(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle) {
#if defined(LITERT_HAS_OPENCL_SUPPORT) || defined(__ANDROID__)
  if (!event_handle) return -1;
  cl_event ce = nullptr;
  LiteRtStatus status = LiteRtGetEventOpenClEvent(
      reinterpret_cast<LiteRtEvent>(event_handle), &ce);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING, "GetOpenClEvent failed or not an OpenCL event.");
    return -1;
  }
  return reinterpret_cast<jlong>(ce);
#else
  return -1;
#endif
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeWait(
    JNIEnv* env, jclass clazz, jlong event_handle, jlong timeout_ms) {
  if (!event_handle) return;
  LiteRtStatus status =
      LiteRtEventWait(reinterpret_cast<LiteRtEvent>(event_handle),
                      static_cast<int64_t>(timeout_ms));
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "EventWait failed, handle=%p, timeout=%lld ms",
               (void*)event_handle, (int64_t)timeout_ms);
  }
}

JNIEXPORT jint JNICALL Java_com_google_ai_edge_litert_Event_nativeGetType(
    JNIEnv* env, jclass clazz, jlong event_handle) {
  if (!event_handle) return -1;
  LiteRtEventType t = LiteRtEventTypeUnknown;
  LiteRtStatus status =
      LiteRtGetEventEventType(reinterpret_cast<LiteRtEvent>(event_handle), &t);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "GetEventType failed: handle=%p",
               (void*)event_handle);
    return -1;
  }
  return static_cast<jint>(t);
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong event_handle) {
  if (!event_handle) return;
  LiteRtDestroyEvent(reinterpret_cast<LiteRtEvent>(event_handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif
