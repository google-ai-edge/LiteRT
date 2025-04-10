#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a LiteRtEvent from a Linux sync fence file descriptor.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromSyncFenceFd(
    JNIEnv* env, jclass clazz, jint sync_fence_fd, jboolean owns_fd);

// Creates a LiteRtEvent from an OpenCL event.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateFromOpenClEvent(
    JNIEnv* env, jclass clazz, jlong cl_event_handle);

// Creates a managed LiteRtEvent with the specified event type.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeCreateManaged(JNIEnv* env,
                                                         jclass clazz,
                                                         jint event_type);

// Extracts the sync fence file descriptor from a LiteRtEvent.
// Returns -1 if the event is not a sync fence event.
JNIEXPORT jint JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetSyncFenceFd(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle);

// Extracts the OpenCL event from a LiteRtEvent.
// Returns -1 if the event is not an OpenCL event.
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Event_nativeGetOpenClEvent(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong event_handle);

// Waits for the event to complete with the specified timeout in milliseconds.
// A timeout of -1 means wait indefinitely.
JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeWait(
    JNIEnv* env, jclass clazz, jlong event_handle, jlong timeout_ms);

// Returns the event type as defined in LiteRtEventType.
JNIEXPORT jint JNICALL Java_com_google_ai_edge_litert_Event_nativeGetType(
    JNIEnv* env, jclass clazz, jlong event_handle);

// Destroys the LiteRtEvent and releases associated resources.
JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Event_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong event_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_EVENT_JNI_H_
