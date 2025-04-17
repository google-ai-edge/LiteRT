#include "litert/kotlin/src/main/jni/litert_environment_jni.h"

#include <jni.h>

#include <any>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_accelerator.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_environment.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {

using litert::Environment;
using litert::jni::ThrowLiteRtException;

// Converts a LiteRtHwAccelerators to the value used in the Kotlin enum.
int ToJniAccelerator(LiteRtHwAcceleratorSet accelerator) {
  switch (accelerator) {
    case kLiteRtHwAcceleratorCpu:
      return litert::jni::kAccelatorCpu;
    case kLiteRtHwAcceleratorGpu:
      return litert::jni::kAccelatorGpu;
    case kLiteRtHwAcceleratorNpu:
      return litert::jni::kAccelatorNpu;
    default:
      if (accelerator != kLiteRtHwAcceleratorNone) {
        LITERT_LOG(LITERT_ERROR, "Unknown accelerator: %d.", accelerator);
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
  std::vector<Environment::Option> options;
  if (num_tags > 0) {
    options.reserve(num_tags);

    AUTO_CLEANUP_JNI_INT_ARRAY(env, tags);
    for (int i = 0; i < num_tags; ++i) {
      auto value = values_vector[i];
      auto option = Environment::Option{
          static_cast<Environment::OptionTag>(tags_array[i]), std::any(value)};
      options.push_back(option);
    }
  }

  auto litert_env = Environment::Create(absl::MakeConstSpan(options));
  if (!litert_env) {
    LITERT_LOG(LITERT_ERROR, "Failed to create environment: %s.",
               litert_env.Error().Message().c_str());
    ThrowLiteRtException(env, litert_env.Error().Status(),
                         litert_env.Error().Message());
    return 0;
  }
  return reinterpret_cast<jlong>(litert_env->Release());
}

JNIEXPORT jintArray JNICALL
Java_com_google_ai_edge_litert_Environment_nativeGetAvailableAccelerators(
    JNIEnv* env, jclass clazz, jlong handle) {
  auto litert_env = reinterpret_cast<LiteRtEnvironment>(handle);

  LiteRtParamIndex size;
  auto status = LiteRtGetNumAccelerators(litert_env, &size);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to get number of accelerators.");
    ThrowLiteRtException(env, status, "Failed to get number of accelerators.");
    return nullptr;
  }

  std::vector<jint> accelerators;
  accelerators.reserve(size);
  for (LiteRtParamIndex i = 0; i < size; ++i) {
    LiteRtAccelerator accelerator;
    status = LiteRtGetAccelerator(litert_env, i, &accelerator);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to get accelerator.");
      continue;
    }
    LiteRtHwAcceleratorSet hardware;
    status = LiteRtGetAcceleratorHardwareSupport(accelerator, &hardware);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to get accelerator supported hardware.");
      continue;
    }
    accelerators.push_back(ToJniAccelerator(hardware));
  }

  jintArray result = env->NewIntArray(accelerators.size());
  env->SetIntArrayRegion(result, 0, accelerators.size(), accelerators.data());
  return result;
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Environment_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  LiteRtDestroyEnvironment(reinterpret_cast<LiteRtEnvironment>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
