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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {

using ::litert::Environment;
using ::litert::jni::ThrowLiteRtException;

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

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Environment_nativeCreate(
    JNIEnv* env, jclass clazz, jintArray tags, jobjectArray values) {
  ABSL_CHECK_EQ(env->GetArrayLength(tags), env->GetArrayLength(values))
      << "Number of tags and values do not match.";

  auto num_tags = env->GetArrayLength(tags);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, values);
  std::vector<litert::EnvironmentOptions::Option> options;
  if (num_tags > 0) {
    options.reserve(num_tags);

    AUTO_CLEANUP_JNI_INT_ARRAY(env, tags);
    for (int i = 0; i < num_tags; ++i) {
      auto value = values_vector[i];
      auto tag = static_cast<litert::EnvironmentOptions::Tag>(tags_array[i]);
      if (tag == litert::EnvironmentOptions::Tag::kSystemRuntimeHandle ||
          tag == litert::EnvironmentOptions::Tag::kSystemGpuAcceleratorHandle) {
        int64_t handle;
        ABSL_CHECK(absl::SimpleAtoi(value, &handle))
            << "Failed to parse system handle: " << value;
        options.push_back(litert::EnvironmentOptions::Option{
            // An intermediate static_cast to std::uintptr_t is used before the
            // reinterpret_cast to const void* to avoid size-mismatch
            // compilation warnings/errors on 32-bit platforms (where pointers
            // are 32-bit but int64_t is 64-bit).
            tag, litert::LiteRtVariant(reinterpret_cast<const void*>(
                     static_cast<std::uintptr_t>(handle)))});
      } else {
        options.push_back(litert::EnvironmentOptions::Option{
            tag, litert::LiteRtVariant(value)});
      }
    }
  }

  auto litert_env = Environment::Create(
      litert::EnvironmentOptions(absl::MakeConstSpan(options)));
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
  ABSL_CHECK(litert_env != nullptr);

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
