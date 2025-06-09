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

#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreate(
    JNIEnv* env, jclass clazz, jlong env_handle, jlong model_handle,
    jintArray accelerators, jintArray cpu_options_keys,
    jobjectArray cpu_options_values, jintArray gpu_options_keys,
    jobjectArray gpu_options_values);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name);

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetInputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name);

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name);

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetOutputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index);

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature);

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_CompiledModel_nativeRun(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jlongArray input_buffers, jlongArray output_buffers);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunBySignatureWithMap(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jobjectArray input_keys, jlongArray input_buffers,
    jobjectArray output_keys, jlongArray output_buffers);

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroy(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_COMPILED_MODEL_JNI_H_
