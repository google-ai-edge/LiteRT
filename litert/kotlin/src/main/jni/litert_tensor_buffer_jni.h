// Copyright 2025 Google LLC.
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

#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_TENSOR_BUFFER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TensorBuffer
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

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteLong(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jlongArray input);

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

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadLong(JNIEnv* env,
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
