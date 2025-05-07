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

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_Model_nativeGetInputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name,
    jstring signature);

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_Model_nativeGetOutputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name,
    jstring signature);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_JNI_H_
