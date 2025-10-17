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

#include "litert/kotlin/src/main/jni/litert_compiled_model_jni.h"

#include <jni.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "litert/kotlin/src/main/jni/litert_model_wrapper.h"

namespace {

using ::litert::jni::ModelWrapper;
using ::litert::jni::ThrowLiteRtException;

// Keys for CPU options, the values should match the ones in Kotlin.
enum CpuOptionsKey {
  kNumThreads = 0,
  kXnnPackFlags = 1,
  kXnnPackWeightCachePath = 2,
};

// Keys for GPU options, the values should match the ones in Kotlin.
enum GpuOptionsKey {
  kConstantTensorSharing = 0,
  kInfiniteFloatCapping = 1,
  kBenchmarkModel = 2,
  kAllowSrcQuantizedFcConvOps = 3,
  kPrecision = 4,
  kBufferStorageType = 5,
};

// Precision for GPU options, the values should match the ones in Kotlin.
enum Precision {
  kPrecisionDefault = 0,
  kPrecisionFp16 = 1,
  kPrecisionFp32 = 2,
};

// Converts the precision string to LiteRtDelegatePrecision.
LiteRtDelegatePrecision ToLiteRtDelegatePrecision(const char* precision_str) {
  auto precision = std::stoi(precision_str);
  switch (precision) {
    case kPrecisionFp16:
      return kLiteRtDelegatePrecisionFp16;
    case kPrecisionFp32:
      return kLiteRtDelegatePrecisionFp32;
    default:
      return kLiteRtDelegatePrecisionDefault;
  }
}

// Buffer storage type for GPU options, the values should match the ones in
// Kotlin.
enum BufferStorageType {
  kBufferStorageTypeDefault = 0,
  kBufferStorageTypeBuffer = 1,
  kBufferStorageTypeTexture2D = 2,
};

// Converts the buffer storage type string to LiteRtDelegateBufferStorageType.
LiteRtDelegateBufferStorageType ToLiteRtDelegateBufferStorageType(
    const char* buffer_storage_type_str) {
  auto type = std::stoi(buffer_storage_type_str);
  switch (type) {
    case kBufferStorageTypeBuffer:
      return kLiteRtDelegateBufferStorageTypeBuffer;
    case kBufferStorageTypeTexture2D:
      return kLiteRtDelegateBufferStorageTypeTexture2D;
    default:
      return kLiteRtDelegateBufferStorageTypeDefault;
  }
}

// Creates a CompiledModel from the given handles.
// The handles are not owned by the returned CompiledModel.
litert::CompiledModel CreateCompileModel(jlong compiled_model_handle,
                                         jlong model_handle) {
  // Extract the actual model from the wrapper
  auto* wrapper = reinterpret_cast<ModelWrapper*>(model_handle);
  ABSL_CHECK(wrapper != nullptr);
  ABSL_CHECK(wrapper->model != nullptr);
  auto c_compiled_model =
      reinterpret_cast<LiteRtCompiledModel>(compiled_model_handle);
  ABSL_CHECK(c_compiled_model != nullptr);
  return litert::CompiledModel(wrapper->model, c_compiled_model,
                               litert::OwnHandle::kNo);
}

// Creates a LiteRtOpaqueOptions from the given cpu options.
// The number of given options must be greater than 0.
LiteRtStatus CreateCpuOptions(JNIEnv* env, LiteRtOpaqueOptions* options,
                              jintArray cpu_options_keys,
                              jobjectArray cpu_options_values) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, cpu_options_keys);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, cpu_options_values);
  auto cpu_options_keys_size = env->GetArrayLength(cpu_options_keys);
  ABSL_CHECK(cpu_options_keys_size == cpu_options_values_size);

  auto status = LiteRtCreateCpuOptions(options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create CPU options.");
    return status;
  }
  LiteRtCpuOptions cpu_options = nullptr;
  status = LiteRtFindCpuOptions(*options, &cpu_options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to find CPU options.");
    return status;
  }
  for (int i = 0; i < cpu_options_keys_size; ++i) {
    if (cpu_options_keys_array[i] == CpuOptionsKey::kNumThreads) {
      status = LiteRtSetCpuOptionsNumThread(
          cpu_options, std::stoi(cpu_options_values_vector[i]));
      if (status != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR, "Failed to set CPU options num_threads.");
        return status;
      }
    } else if (cpu_options_keys_array[i] == CpuOptionsKey::kXnnPackFlags) {
      status = LiteRtSetCpuOptionsXNNPackFlags(
          cpu_options, std::stoi(cpu_options_values_vector[i]));
      if (status != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR, "Failed to set CPU options xnnpack_flags.");
        return status;
      }
    } else if (cpu_options_keys_array[i] ==
               CpuOptionsKey::kXnnPackWeightCachePath) {
      status = LiteRtSetCpuOptionsXnnPackWeightCachePath(
          cpu_options, cpu_options_values_vector[i]);
      if (status != kLiteRtStatusOk) {
        LITERT_LOG(LITERT_ERROR,
                   "Failed to set CPU options xnnpack_weight_cache_path.");
        return status;
      }
    } else {
      LITERT_LOG(LITERT_ERROR, "Unknown CPU options key: %d.",
                 cpu_options_keys_array[i]);
    }
  }
  return kLiteRtStatusOk;
}

// Creates a LiteRtOpaqueOptions from the given gpu options.
// The number of given options must be greater than 0.
LiteRtStatus CreateGpuOptions(JNIEnv* env, LiteRtOpaqueOptions* gpu_options,
                              jintArray gpu_options_keys,
                              jobjectArray gpu_options_values) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, gpu_options_keys);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, gpu_options_values);
  auto gpu_options_keys_size = env->GetArrayLength(gpu_options_keys);
  ABSL_CHECK(gpu_options_keys_size == gpu_options_values_size);

  LiteRtStatus status = LiteRtCreateGpuOptions(gpu_options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create GPU options.");
    return status;
  }
  for (int i = 0; i < gpu_options_keys_size; ++i) {
    switch (gpu_options_keys_array[i]) {
      case GpuOptionsKey::kConstantTensorSharing:
        status = LiteRtSetGpuOptionsConstantTensorSharing(
            *gpu_options, strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR,
                     "Failed to set GPU options constantTensorSharing.");
          return status;
        }
        break;
      case GpuOptionsKey::kInfiniteFloatCapping:
        status = LiteRtSetGpuOptionsInfiniteFloatCapping(
            *gpu_options, strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR,
                     "Failed to set GPU options infiniteFloatCapping.");
          return status;
        }
        break;
      case GpuOptionsKey::kBenchmarkModel:
        status = LiteRtSetGpuOptionsBenchmarkMode(
            *gpu_options, strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR, "Failed to set GPU options benchmarkModel.");
          return status;
        }
        break;
      case GpuOptionsKey::kAllowSrcQuantizedFcConvOps:
        status =
            LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
                *gpu_options,
                strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR,
                     "Failed to set GPU options allowSrcQuantizedFcConvOps.");
          return status;
        }
        break;
      case GpuOptionsKey::kPrecision:
        status = LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
            *gpu_options,
            ToLiteRtDelegatePrecision(gpu_options_values_vector[i]));
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR, "Failed to set GPU options precision.");
          return status;
        }
        break;
      case GpuOptionsKey::kBufferStorageType:
        status = LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
            *gpu_options,
            ToLiteRtDelegateBufferStorageType(gpu_options_values_vector[i]));
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_ERROR,
                     "Failed to set GPU options bufferStorageType.");
          return status;
        }
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unknown GPU options key: %d.",
                   gpu_options_keys_array[i]);
        return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

// Creates a Java TensorBufferRequirements object from the given C++ object.
jobject CreateJavaTensorBufferRequirements(
    JNIEnv* env, const litert::TensorBufferRequirements& requirements) {
  // Get the TensorBufferRequirements class and constructor.
  jclass requirements_class =
      env->FindClass("com/google/ai/edge/litert/TensorBufferRequirements");
  if (requirements_class == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to find TensorBufferRequirements class.");
    return nullptr;
  }
  jmethodID constructor =
      env->GetMethodID(requirements_class, "<init>", "([II[I)V");
  if (constructor == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get TensorBufferRequirements constructor.");
    return nullptr;
  }

  // Convert supported types to int array.
  auto supported_types = requirements.SupportedTypes();
  if (!supported_types) {
    LITERT_LOG(LITERT_ERROR, "Failed to get supported types: %s",
               supported_types.Error().Message().c_str());
    ThrowLiteRtException(env, supported_types.Error().Status(),
                         supported_types.Error().Message());
    return nullptr;
  }
  jintArray supported_types_array = env->NewIntArray(supported_types->size());
  if (supported_types_array == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to allocate int array.");
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to allocate int array.");
    return nullptr;
  }
  env->SetIntArrayRegion(
      supported_types_array, 0, supported_types->size(),
      reinterpret_cast<const jint*>(supported_types->data()));

  // Convert strides to int array.
  auto strides = requirements.Strides();
  if (!strides) {
    LITERT_LOG(LITERT_ERROR, "Failed to get strides: %s",
               strides.Error().Message().c_str());
    ThrowLiteRtException(env, strides.Error().Status(),
                         strides.Error().Message());
    return nullptr;
  }
  jintArray strides_array = env->NewIntArray(strides->size());
  if (strides_array == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to allocate int array.");
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create strides array.");
    return nullptr;
  }
  env->SetIntArrayRegion(strides_array, 0, strides->size(),
                         reinterpret_cast<const jint*>(strides->data()));

  auto buffer_size = requirements.BufferSize();
  if (!buffer_size) {
    LITERT_LOG(LITERT_ERROR, "Failed to get buffer size: %s",
               buffer_size.Error().Message().c_str());
    ThrowLiteRtException(env, buffer_size.Error().Status(),
                         buffer_size.Error().Message());
    return nullptr;
  }
  // Create and return the Java object.
  jobject java_object =
      env->NewObject(requirements_class, constructor, supported_types_array,
                     *buffer_size, strides_array);

  env->DeleteLocalRef(requirements_class);
  env->DeleteLocalRef(supported_types_array);
  env->DeleteLocalRef(strides_array);

  if (java_object == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create TensorBufferRequirements object.");
    return nullptr;
  }
  return java_object;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreate(
    JNIEnv* env, jclass clazz, jlong env_handle, jlong model_handle,
    jintArray accelerators, jintArray cpu_options_keys,
    jobjectArray cpu_options_values, jintArray gpu_options_keys,
    jobjectArray gpu_options_values) {
  int accelerators_size = env->GetArrayLength(accelerators);
  AUTO_CLEANUP_JNI_INT_ARRAY(env, accelerators);
  LiteRtHwAcceleratorSet acceleratorSet = kLiteRtHwAcceleratorNone;
  for (int i = 0; i < accelerators_size; ++i) {
    switch (accelerators_array[i]) {
      case litert::jni::kAccelatorNone:
        break;
      case litert::jni::kAccelatorCpu:
        acceleratorSet |= kLiteRtHwAcceleratorCpu;
        break;
      case litert::jni::kAccelatorGpu:
        acceleratorSet |= kLiteRtHwAcceleratorGpu;
        break;
      case litert::jni::kAccelatorNpu:
        acceleratorSet |= kLiteRtHwAcceleratorNpu;
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported accelerator: %d.",
                   accelerators_array[i]);
    }
  }

  LiteRtOptions compilation_options = nullptr;
  auto status = LiteRtCreateOptions(&compilation_options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options.");
    ThrowLiteRtException(env, status, "Failed to create compilation options.");
    return 0;
  }
  status =
      LiteRtSetOptionsHardwareAccelerators(compilation_options, acceleratorSet);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to set hardware accelerators.");
    ThrowLiteRtException(env, status, "Failed to set hardware accelerators.");
    return 0;
  }

  if (env->GetArrayLength(cpu_options_keys) > 0) {
    LiteRtOpaqueOptions options = nullptr;
    status =
        CreateCpuOptions(env, &options, cpu_options_keys, cpu_options_values);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to create CPU options.");
      ThrowLiteRtException(env, status, "Failed to create CPU options.");
      return 0;
    }
    status = LiteRtAddOpaqueOptions(compilation_options, options);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to add CPU options.");
      ThrowLiteRtException(env, status, "Failed to add CPU options.");
      return 0;
    }
  }

  if (env->GetArrayLength(gpu_options_keys) > 0) {
    LiteRtOpaqueOptions options = nullptr;
    status =
        CreateGpuOptions(env, &options, gpu_options_keys, gpu_options_values);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to create GPU options.");
      ThrowLiteRtException(env, status, "Failed to create GPU options.");
      return 0;
    }
    status = LiteRtAddOpaqueOptions(compilation_options, options);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to add GPU options.");
      ThrowLiteRtException(env, status, "Failed to add GPU options.");
      return 0;
    }
  }

  auto litert_env = reinterpret_cast<LiteRtEnvironment>(env_handle);
  ABSL_CHECK(litert_env != nullptr);
  // Extract the actual model from the wrapper
  auto* wrapper = reinterpret_cast<ModelWrapper*>(model_handle);
  ABSL_CHECK(wrapper != nullptr);
  ABSL_CHECK(wrapper->model != nullptr);
  auto model = wrapper->model;

  LiteRtCompiledModel compiled_model = nullptr;
  status = LiteRtCreateCompiledModel(
      litert_env, model, std::move(compilation_options), &compiled_model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compiled model.");
    ThrowLiteRtException(env, status, "Failed to create compiled model.");
    return 0;
  }
  return reinterpret_cast<jlong>(compiled_model);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  auto tensor_buffer =
      signature_str == nullptr
          ? compiled_model.CreateInputBuffer(input_name_str)
          : compiled_model.CreateInputBuffer(signature_str, input_name_str);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffer: %s",
               tensor_buffer.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffer.Error().Status(),
                         tensor_buffer.Error().Message());
    return 0;
  }
  return reinterpret_cast<jlong>(std::move(tensor_buffer->Release()));
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetInputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring input_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  auto requirements =
      signature_str == nullptr
          ? compiled_model.GetInputBufferRequirements(input_name_str)
          : compiled_model.GetInputBufferRequirements(signature_str,
                                                      input_name_str);
  if (!requirements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get input buffer requirements: %s",
               requirements.Error().Message().c_str());
    ThrowLiteRtException(env, requirements.Error().Status(),
                         requirements.Error().Message());
    return nullptr;
  }
  return CreateJavaTensorBufferRequirements(env, *requirements);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  auto tensor_buffer =
      signature_str == nullptr
          ? compiled_model.CreateOutputBuffer(output_name_str)
          : compiled_model.CreateOutputBuffer(signature_str, output_name_str);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffer: %s",
               tensor_buffer.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffer.Error().Status(),
                         tensor_buffer.Error().Message());
    return 0;
  }
  return reinterpret_cast<jlong>(std::move(tensor_buffer->Release()));
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetOutputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jstring output_name) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  auto requirements =
      signature_str == nullptr
          ? compiled_model.GetOutputBufferRequirements(output_name_str)
          : compiled_model.GetOutputBufferRequirements(signature_str,
                                                       output_name_str);
  if (!requirements) {
    LITERT_LOG(LITERT_ERROR, "Failed to get outpput buffer requirements: %s",
               requirements.Error().Message().c_str());
    ThrowLiteRtException(env, requirements.Error().Status(),
                         requirements.Error().Message());
    return nullptr;
  }
  return CreateJavaTensorBufferRequirements(env, *requirements);
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto tensor_buffers = compiled_model.CreateInputBuffers(signature_index);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffers: %s",
               tensor_buffers.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffers.Error().Status(),
                         tensor_buffers.Error().Message());
    return nullptr;
  }
  std::vector<jlong> input_tensor_buffers;
  input_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    input_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          input_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  auto tensor_buffers = compiled_model.CreateInputBuffers(signature_str);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffers: %s",
               tensor_buffers.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffers.Error().Status(),
                         tensor_buffers.Error().Message());
    return nullptr;
  }
  std::vector<jlong> input_tensor_buffers;
  input_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    input_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          input_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto tensor_buffers = compiled_model.CreateOutputBuffers(signature_index);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffers: %s",
               tensor_buffers.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffers.Error().Status(),
                         tensor_buffers.Error().Message());
    return nullptr;
  }
  std::vector<jlong> output_tensor_buffers;
  output_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    output_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          output_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  auto tensor_buffers = compiled_model.CreateOutputBuffers(signature_str);
  if (!tensor_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffers: %s",
               tensor_buffers.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffers.Error().Status(),
                         tensor_buffers.Error().Message());
    return nullptr;
  }
  std::vector<jlong> output_tensor_buffers;
  output_tensor_buffers.reserve(tensor_buffers->size());
  for (int i = 0; i < tensor_buffers->size(); ++i) {
    output_tensor_buffers.push_back(
        reinterpret_cast<jlong>(std::move(tensor_buffers->at(i).Release())));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          output_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_CompiledModel_nativeRun(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto num_inputs = env->GetArrayLength(input_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto litert_tensor_buffer =
        reinterpret_cast<LiteRtTensorBuffer>(input_buffers_array[i]);
    input_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, litert::OwnHandle::kNo));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto litert_tensor_buffer =
        reinterpret_cast<LiteRtTensorBuffer>(output_buffers_array[i]);
    output_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, litert::OwnHandle::kNo));
  }
  auto result = compiled_model.Run(signature_index, input_buffer_vector,
                                   output_buffer_vector);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "Failed to run model: %s",
               result.Error().Message().c_str());
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jlongArray input_buffers, jlongArray output_buffers) {
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);

  auto num_inputs = env->GetArrayLength(input_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto litert_tensor_buffer =
        reinterpret_cast<LiteRtTensorBuffer>(input_buffers_array[i]);
    input_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, litert::OwnHandle::kNo));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto litert_tensor_buffer =
        reinterpret_cast<LiteRtTensorBuffer>(output_buffers_array[i]);
    output_buffer_vector.push_back(
        litert::TensorBuffer(litert_tensor_buffer, litert::OwnHandle::kNo));
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  auto result = compiled_model.Run(signature_str, input_buffer_vector,
                                   output_buffer_vector);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "Failed to run model: %s",
               result.Error().Message().c_str());
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeRunBySignatureWithMap(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jlong model_handle,
    jstring signature, jobjectArray input_keys, jlongArray input_buffers,
    jobjectArray output_keys, jlongArray output_buffers) {
  ABSL_CHECK_EQ(env->GetArrayLength(input_keys),
                env->GetArrayLength(input_buffers))
      << "Number of input keys and buffers do not match.";
  ABSL_CHECK_EQ(env->GetArrayLength(output_keys),
                env->GetArrayLength(output_buffers))
      << "Number of output keys and buffers do not match.";

  AUTO_CLEANUP_JNI_STRING_ARRAY(env, input_keys);
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> input_buffer_map;
  input_buffer_map.reserve(input_keys_size);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  for (int i = 0; i < input_keys_size; ++i) {
    auto key = input_keys_vector[i];
    auto buffer = reinterpret_cast<LiteRtTensorBuffer>(input_buffers_array[i]);
    input_buffer_map[key] =
        litert::TensorBuffer(buffer, litert::OwnHandle::kNo);
  }

  AUTO_CLEANUP_JNI_STRING_ARRAY(env, output_keys);
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer>
      output_buffer_map;
  output_buffer_map.reserve(output_keys_size);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  for (int i = 0; i < output_keys_size; ++i) {
    auto key = output_keys_vector[i];
    auto buffer = reinterpret_cast<LiteRtTensorBuffer>(output_buffers_array[i]);
    output_buffer_map[key] =
        litert::TensorBuffer(buffer, litert::OwnHandle::kNo);
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  auto compiled_model = CreateCompileModel(compiled_model_handle, model_handle);
  auto result = signature_str == nullptr
                    ? compiled_model.Run(input_buffer_map, output_buffer_map)
                    : compiled_model.Run(signature_str, input_buffer_map,
                                         output_buffer_map);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "Failed to run model: %s",
               result.Error().Message().c_str());
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroy(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  LiteRtDestroyCompiledModel(reinterpret_cast<LiteRtCompiledModel>(handle));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
