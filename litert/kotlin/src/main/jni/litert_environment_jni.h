#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_ENVIRONMENT_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_ENVIRONMENT_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The client needs to keep the environment object alive for the duration of
// the entire inference.
JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Environment_nativeCreate(
    JNIEnv* env, jclass clazz, jintArray tags, jobjectArray values);

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_Environment_nativeGetAvailableAccelerators(
    JNIEnv* env, jclass clazz, jlong handle);

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Environment_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_ENVIRONMENT_JNI_H_
