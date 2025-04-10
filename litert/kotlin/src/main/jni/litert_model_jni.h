#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadAsset(
    JNIEnv* env, jclass clazz, jobject asset_manager, jstring asset_name);

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadFile(
    JNIEnv* env, jclass clazz, jstring file_path);

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Model_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_JNI_H_
