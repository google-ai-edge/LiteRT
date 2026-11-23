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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {
using ::litert::TensorBuffer;
using ::litert::TensorBufferScopedLock;
using ::litert::jni::ThrowLiteRtException;

jlong CopyDirectBuffer(JNIEnv* env, jlong handle, jobject buffer,
                       TensorBuffer::LockMode mode) {
  if (handle == 0 || buffer == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "A tensor handle and direct ByteBuffer are required.");
    return 0;
  }
  const jlong capacity = env->GetDirectBufferCapacity(buffer);
  void* address = env->GetDirectBufferAddress(buffer);
  if (capacity < 0 || address == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "The buffer must be a direct ByteBuffer.");
    return 0;
  }
  if (mode == TensorBuffer::LockMode::kRead) {
    jclass buffer_class = env->GetObjectClass(buffer);
    if (buffer_class == nullptr) return 0;
    jmethodID is_read_only =
        env->GetMethodID(buffer_class, "isReadOnly", "()Z");
    if (is_read_only == nullptr) return 0;
    const jboolean read_only = env->CallBooleanMethod(buffer, is_read_only);
    if (env->ExceptionCheck()) return 0;
    if (read_only) {
      ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                           "The destination must be writable.");
      return 0;
    }
  }
  auto* tensor_buffer = reinterpret_cast<TensorBuffer*>(handle);
  auto packed_size = tensor_buffer->PackedSize();
  if (!packed_size) {
    ThrowLiteRtException(env, packed_size.Error().Status(),
                         packed_size.Error().Message());
    return 0;
  }
  if (*packed_size > static_cast<size_t>(capacity)) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "The buffer has insufficient remaining bytes.");
    return 0;
  }
  auto lock_and_addr =
      TensorBufferScopedLock::Create<uint8_t>(*tensor_buffer, mode);
  if (!lock_and_addr) {
    ThrowLiteRtException(env, lock_and_addr.Error().Status(),
                         lock_and_addr.Error().Message());
    return 0;
  }
  if (mode == TensorBuffer::LockMode::kWrite) {
    std::memmove(lock_and_addr->second, address, *packed_size);
  } else {
    std::memmove(address, lock_and_addr->second, *packed_size);
  }
  return static_cast<jlong>(*packed_size);
}

template <typename T>
void WriteImp(JNIEnv* env, jlong handle, absl::Span<const T> input_span) {
  auto* tensor_buffer = reinterpret_cast<TensorBuffer*>(handle);
  auto write_result = tensor_buffer->Write<T>(input_span);
  if (!write_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to write tensor buffer: %s",
               write_result.Error().Message().c_str());
    ThrowLiteRtException(env, write_result.Error().Status(),
                         write_result.Error().Message());
  }
}

template <typename T, typename JArray>
JArray ReadImp(JNIEnv* env, jlong handle, JArray (JNIEnv::*new_array)(jsize),
               void (JNIEnv::*set_array_region)(JArray, jsize, jsize,
                                                const T*)) {
  auto* tensor_buffer = reinterpret_cast<TensorBuffer*>(handle);
  auto tensor_type = tensor_buffer->TensorType();
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
  auto lock_and_addr = TensorBufferScopedLock::Create<const T>(
      *tensor_buffer, TensorBuffer::LockMode::kRead);
  if (!lock_and_addr) {
    LITERT_LOG(LITERT_ERROR, "Unable to lock the tensor buffer: %s",
               lock_and_addr.Error().Message().c_str());
    ThrowLiteRtException(env, lock_and_addr.Error().Status(),
                         lock_and_addr.Error().Message());
    return nullptr;
  }

  JArray result = (env->*new_array)(*num_elements);
  if (result == nullptr) return nullptr;
  // Copy the data from the locked tensor buffer to the JVM array.
  (env->*set_array_region)(result, 0, *num_elements, lock_and_addr->second);
  return result;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TensorBuffer
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFrom(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jobject source) {
  return CopyDirectBuffer(env, handle, source, TensorBuffer::LockMode::kWrite);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInto(
    JNIEnv* env, jclass clazz, jlong handle, jobject destination) {
  return CopyDirectBuffer(env, handle, destination,
                          TensorBuffer::LockMode::kRead);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jintArray input) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<int>(env, handle, input_span);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt16(
    JNIEnv* env, jclass clazz, jlong handle, jshortArray input) {
  AUTO_CLEANUP_JNI_SHORT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<jshort>(env, handle, input_span);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteFloat(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray input) {
  AUTO_CLEANUP_JNI_FLOAT_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<jfloat>(env, handle, input_span);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteInt8(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jbyteArray input) {
  AUTO_CLEANUP_JNI_BYTE_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<jbyte>(env, handle, input_span);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteBoolean(
    JNIEnv* env, jclass clazz, jlong handle, jbooleanArray input) {
  AUTO_CLEANUP_JNI_BOOLEAN_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<jboolean>(env, handle, input_span);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeWriteLong(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jlongArray input) {
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input);
  auto num_elements = env->GetArrayLength(input);
  auto input_span = absl::MakeConstSpan(input_array, num_elements);
  WriteImp<jlong>(env, handle, input_span);
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  return ReadImp<jint, jintArray>(env, handle, &JNIEnv::NewIntArray,
                                  &JNIEnv::SetIntArrayRegion);
}

JNIEXPORT jshortArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt16(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
  return ReadImp<jshort, jshortArray>(env, handle, &JNIEnv::NewShortArray,
                                      &JNIEnv::SetShortArrayRegion);
}

JNIEXPORT jfloatArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadFloat(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
  return ReadImp<jfloat, jfloatArray>(env, handle, &JNIEnv::NewFloatArray,
                                      &JNIEnv::SetFloatArrayRegion);
}

JNIEXPORT jbyteArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadInt8(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  return ReadImp<jbyte, jbyteArray>(env, handle, &JNIEnv::NewByteArray,
                                    &JNIEnv::SetByteArrayRegion);
}

JNIEXPORT jbooleanArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadBoolean(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle) {
  return ReadImp<jboolean, jbooleanArray>(env, handle, &JNIEnv::NewBooleanArray,
                                          &JNIEnv::SetBooleanArrayRegion);
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeReadLong(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  return ReadImp<jlong, jlongArray>(env, handle, &JNIEnv::NewLongArray,
                                    &JNIEnv::SetLongArrayRegion);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_TensorBuffer_nativeDestroy(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  delete reinterpret_cast<TensorBuffer*>(handle);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
