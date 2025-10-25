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

#include "litert/kotlin/src/main/jni/litert_tensor_buffer_jni.h"

#include <jni.h>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {
using litert::jni::ThrowLiteRtException;
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TensorBuffer
JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jintArray input) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto write_result = tensor_buffer.Write<jint>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray input) {
  AUTO_CLEANUP_JNI_FLOAT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto write_result = tensor_buffer.Write<jfloat>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt8(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jbyteArray input) {
  AUTO_CLEANUP_JNI_BYTE_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto write_result = tensor_buffer.Write<jbyte>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteBoolean(
    JNIEnv* env, jclass clazz, jlong handle, jbooleanArray input) {
  AUTO_CLEANUP_JNI_BOOLEAN_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto write_result = tensor_buffer.Write<jboolean>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteLong(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jlongArray input) {
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);

  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto write_result = tensor_buffer.Write<jlong>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements");
    ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                         "Failed to get tensor num elements.");
    return nullptr;
  }
  auto lock_and_addr = litert::TensorBufferScopedLock::Create<const int>(
      tensor_buffer, litert::TensorBuffer::LockMode::kRead);
  jintArray result = env->NewIntArray(*num_elements);
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetIntArrayRegion(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                         "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
      tensor_buffer, litert::TensorBuffer::LockMode::kRead);
  jfloatArray result = env->NewFloatArray(*num_elements);
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetFloatArrayRegion(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt8(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                         "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr = litert::TensorBufferScopedLock::Create<const jbyte>(
      tensor_buffer, litert::TensorBuffer::LockMode::kRead);
  jbyteArray result = env->NewByteArray(*num_elements);
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetByteArrayRegion(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

JNIEXPORT jbooleanArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadBoolean(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                         "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr = litert::TensorBufferScopedLock::Create<const jboolean>(
      tensor_buffer, litert::TensorBuffer::LockMode::kRead);
  jbooleanArray result = env->NewBooleanArray(*num_elements);
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetBooleanArrayRegion(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadLong(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  auto tb = reinterpret_cast<LiteRtTensorBuffer>(handle);
  auto tensor_buffer =
      litert::TensorBuffer::WrapCObject(tb, litert::OwnHandle::kNo);
  auto tensor_type = tensor_buffer.TensorType();
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         "Failed to get tensor type.");
    return nullptr;
  }
  auto num_elements = tensor_type->Layout().NumElements();
  if (!num_elements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor num elements.");
    ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                         "Failed to get tensor num elements.");
    return nullptr;
  }

  auto lock_and_addr = litert::TensorBufferScopedLock::Create<const jlong>(
      tensor_buffer, litert::TensorBuffer::LockMode::kRead);
  jlongArray result = env->NewLongArray(*num_elements);
  // Copy the data from the locked tensor buffer to the JVM array.
  env->SetLongArrayRegion(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  LiteRtDestroyTensorBuffer(reinterpret_cast<LiteRtTensorBuffer>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
