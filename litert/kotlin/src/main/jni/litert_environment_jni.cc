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

#include "litert/kotlin/src/main/jni/litert_environment_jni.h"

#include <jni.h>

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {

using ::litert::Environment;
using ::litert::jni::ThrowLiteRtException;

struct JniEnvironmentOptionsBuilder {
  // LiteRtVariant stores string options as non-owning `const char*`, and
  // EnvironmentOptions only copies them in Environment::Create. Own the bytes
  // here until then. std::deque keeps element addresses (and thus c_str())
  // stable across emplace_back, unlike std::vector with SSO strings.
  std::deque<std::string> strings;
  std::vector<litert::EnvironmentOptions::Option> options;
};

// Converts a litert::HwAccelerators to the value used in the Kotlin enum.
int ToJniAccelerator(litert::HwAccelerators accelerator) {
  switch (accelerator) {
    case litert::HwAccelerators::kCpu:
      return litert::jni::kAccelatorCpu;
    case litert::HwAccelerators::kGpu:
      return litert::jni::kAccelatorGpu;
    case litert::HwAccelerators::kNpu:
      return litert::jni::kAccelatorNpu;
    default:
      if (accelerator != litert::HwAccelerators::kNone) {
        LITERT_LOG(LITERT_ERROR, "Unknown accelerator: %d.",
                   static_cast<int>(accelerator));
      }
      return litert::jni::kAccelatorNone;
  }
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_Environment_nativeCreateOptions(JNIEnv* env,
                                                               jclass clazz) {
  return reinterpret_cast<jlong>(new JniEnvironmentOptionsBuilder());
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Environment_nativeDestroyOptions(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  delete reinterpret_cast<JniEnvironmentOptionsBuilder*>(handle);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Environment_nativeAddStringOption(
    JNIEnv* env, jclass clazz, jlong handle, jint tag, jstring value) {
  auto* builder = reinterpret_cast<JniEnvironmentOptionsBuilder*>(handle);
  LITERT_INTERNAL_CHECK(builder != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, value);
  const std::string& stored =
      builder->strings.emplace_back(value_str != nullptr ? value_str : "");
  builder->options.push_back(litert::EnvironmentOptions::Option{
      static_cast<litert::EnvironmentOptions::Tag>(tag),
      litert::LiteRtVariant(stored.c_str())});
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_Environment_nativeAddPointerOption(
    JNIEnv* env, jclass clazz, jlong handle, jint tag, jlong pointer) {
  auto* builder = reinterpret_cast<JniEnvironmentOptionsBuilder*>(handle);
  LITERT_INTERNAL_CHECK(builder != nullptr);
  // An intermediate static_cast to std::uintptr_t is used before the
  // reinterpret_cast to const void* to avoid size-mismatch compilation
  // warnings/errors on 32-bit platforms (where pointers are 32-bit but jlong
  // is 64-bit).
  const void* ptr =
      reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pointer));
  builder->options.push_back(litert::EnvironmentOptions::Option{
      static_cast<litert::EnvironmentOptions::Tag>(tag),
      litert::LiteRtVariant(ptr)});
}

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Environment_nativeCreate(
    JNIEnv* env, jclass clazz, jlong options_handle) {
  const auto* builder =
      reinterpret_cast<const JniEnvironmentOptionsBuilder*>(options_handle);
  LITERT_INTERNAL_CHECK(builder != nullptr);

  auto litert_env = Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(builder->options)));
  if (!litert_env) {
    LITERT_LOG(LITERT_ERROR, "Failed to create environment: %s.",
               litert_env.Error().Message().c_str());
    ThrowLiteRtException(env, litert_env.Error().Status(),
                         litert_env.Error().Message());
    return 0;
  }

  auto* litert_env_ptr = new Environment(std::move(*litert_env));
  return reinterpret_cast<jlong>(litert_env_ptr);
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_Environment_nativeGetAvailableAccelerators(
    JNIEnv* env, jclass clazz, jlong handle) {
  auto litert_env = reinterpret_cast<Environment*>(handle);
  LITERT_INTERNAL_CHECK(litert_env != nullptr);

  auto accelerators_res = litert_env->GetAvailableAccelerators();
  if (!accelerators_res) {
    LITERT_LOG(LITERT_ERROR, "Failed to get available accelerators: %s.",
               accelerators_res.Error().Message().c_str());
    ThrowLiteRtException(env, accelerators_res.Error().Status(),
                         accelerators_res.Error().Message());
    return nullptr;
  }

  const auto& accelerators = *accelerators_res;
  std::vector<jint> jni_accelerators;
  jni_accelerators.reserve(accelerators.size());
  for (const auto& accelerator : accelerators) {
    jni_accelerators.push_back(ToJniAccelerator(accelerator));
  }

  jsize num_accelerators = static_cast<jsize>(jni_accelerators.size());
  jintArray result = env->NewIntArray(num_accelerators);
  if (result != nullptr) {
    env->SetIntArrayRegion(result, 0, num_accelerators,
                           jni_accelerators.data());
  }
  return result;
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Environment_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  delete reinterpret_cast<Environment*>(handle);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
