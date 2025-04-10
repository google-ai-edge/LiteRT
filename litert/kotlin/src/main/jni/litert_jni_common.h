#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_JNI_COMMON_H_

#include "absl/cleanup/cleanup.h"  // from @com_google_absl  // IWYU pragma: keep

namespace litert {
namespace jni {

// Values here must match the values in the Kotlin enum
// com.google.ai.edge.litert.Accelerator.
constexpr int kAccelatorNone = 0;
constexpr int kAccelatorCpu = 1;
constexpr int kAccelatorGpu = 2;
constexpr int kAccelatorNpu = 3;

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
