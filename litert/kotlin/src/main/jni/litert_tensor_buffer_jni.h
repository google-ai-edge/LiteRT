#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jintArray input);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle,
                                                             jfloatArray input);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt8(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jbyteArray input);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteBoolean(
    JNIEnv* env, jclass clazz, jlong handle, jbooleanArray input);

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle);

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle);

JNIEXPORT jbyteArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt8(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle);

JNIEXPORT jbooleanArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadBoolean(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_
