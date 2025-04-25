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

#include "litert/kotlin/src/main/jni/litert_model_jni.h"

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <cstdint>

#include "litert/cc/litert_buffer_ref.h"
#endif  // __ANDROID__

#include <jni.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {
using litert::jni::ThrowLiteRtException;
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __ANDROID__
JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadAsset(
    JNIEnv* env, jclass clazz, jobject asset_manager, jstring asset_name) {
  auto am = AAssetManager_fromJava(env, asset_manager);
  AUTO_CLEANUP_JNI_STRING(env, asset_name);
  auto g_model_asset =
      AAssetManager_open(am, asset_name_str, AASSET_MODE_BUFFER);

  auto buffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(g_model_asset)),
      AAsset_getLength(g_model_asset));
  AAsset_close(g_model_asset);

  LiteRtModel model = nullptr;
  auto status =
      LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from asset.");
    ThrowLiteRtException(env, status, "Failed to create model from asset.");
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}
#endif  // __ANDROID__

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadFile(
    JNIEnv* env, jclass clazz, jstring file_path) {
  AUTO_CLEANUP_JNI_STRING(env, file_path);
  LiteRtModel model = nullptr;
  auto status = LiteRtCreateModelFromFile(file_path_str, &model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from file.");
    ThrowLiteRtException(env, status, "Failed to create model from file.");
    return 0;
  }
  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Model_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  LiteRtDestroyModel(reinterpret_cast<LiteRtModel>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
