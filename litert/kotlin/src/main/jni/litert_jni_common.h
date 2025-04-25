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

#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_

#include <jni.h>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"

namespace litert {
namespace jni {

// Values here must match the values in the Kotlin enum
// com.google.ai.edge.litert.Accelerator.
constexpr int kAccelatorNone = 0;
constexpr int kAccelatorCpu = 1;
constexpr int kAccelatorGpu = 2;
constexpr int kAccelatorNpu = 3;

// Throws a LiteRtException in Kotlin.
inline void ThrowLiteRtException(JNIEnv* env, LiteRtStatus status,
                                 absl::string_view message) {
  auto ex_class = env->FindClass("com/google/ai/edge/litert/LiteRtException");
  ABSL_CHECK(ex_class != nullptr) << "Failed to find LiteRtException class";

  auto constructor =
      env->GetMethodID(ex_class, "<init>", "(ILjava/lang/String;)V");
  if (constructor == nullptr) {
    env->DeleteLocalRef(ex_class);
    ABSL_CHECK(false) << "Failed to get LiteRtException constructor";
  }

  auto message_jstr = env->NewStringUTF(message.data());
  if (message_jstr == nullptr) {
    env->DeleteLocalRef(ex_class);
    ABSL_CHECK(false) << "Failed to create message string";
  }

  auto ex_obj = env->NewObject(ex_class, constructor, status, message_jstr);
  env->DeleteLocalRef(message_jstr);
  env->DeleteLocalRef(ex_class);

  ABSL_CHECK(ex_obj != nullptr) << "Failed to create LiteRtException object";
  env->Throw((jthrowable)ex_obj);
}

}  // namespace jni
}  // namespace litert

// A macro to access a JNI string and automatically release after use.
#define AUTO_CLEANUP_JNI_STRING(env, jstr)                               \
  auto jstr##_str =                                                      \
      jstr == nullptr ? nullptr : env->GetStringUTFChars(jstr, nullptr); \
  auto jstr##_cleanup = absl::MakeCleanup([&]() {                        \
    if (jstr != nullptr) {                                               \
      env->ReleaseStringUTFChars(jstr, jstr##_str);                      \
    }                                                                    \
  });

// A macro to help automatically release a JNI string array after use.
#define AUTO_CLEANUP_JNI_STRING_ARRAY(env, jarray)                             \
  auto jarray##_size = env->GetArrayLength(jarray);                            \
  std::vector<const char*> jarray##_vector;                                    \
  jarray##_vector.reserve(jarray##_size);                                      \
  for (int i = 0; i < jarray##_size; ++i) {                                    \
    auto jstr = static_cast<jstring>(env->GetObjectArrayElement(jarray, i));   \
    jarray##_vector.push_back(env->GetStringUTFChars(jstr, nullptr));          \
  }                                                                            \
  auto jarray##_cleanup = absl::MakeCleanup([&]() {                            \
    for (int i = 0; i < jarray##_size; ++i) {                                  \
      auto jstr = static_cast<jstring>(env->GetObjectArrayElement(jarray, i)); \
      env->ReleaseStringUTFChars(jstr, jarray##_vector[i]);                    \
    }                                                                          \
  });

// A macro to help automatically release a JNI primitive array after use.
#define AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, type)               \
  auto jarray##_array = env->Get##type##ArrayElements(jarray, nullptr);   \
  auto jarray##_cleanup = absl::MakeCleanup([&]() {                       \
    env->Release##type##ArrayElements(jarray, jarray##_array, JNI_ABORT); \
  });

// A macro to help automatically release a JNI long array after use.
#define AUTO_CLEANUP_JNI_LONG_ARRAY(env, jarray) \
  AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, Long)

// A macro to help automatically release a JNI int array after use.
#define AUTO_CLEANUP_JNI_INT_ARRAY(env, jarray) \
  AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, Int)

// A macro to help automatically release a JNI float array after use.
#define AUTO_CLEANUP_JNI_FLOAT_ARRAY(env, jarray) \
  AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, Float)

// A macro to help automatically release a JNI byte array after use.
#define AUTO_CLEANUP_JNI_BYTE_ARRAY(env, jarray) \
  AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, Byte)

// A macro to help automatically release a JNI boolean array after use.
#define AUTO_CLEANUP_JNI_BOOLEAN_ARRAY(env, jarray) \
  AUTO_CLEANUP_JNI_PRIMITIVE_ARRAY(env, jarray, Boolean)

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_
