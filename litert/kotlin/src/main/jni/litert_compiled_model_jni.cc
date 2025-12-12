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

#include <cstdint>

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "litert/cc/litert_buffer_ref.h"
#endif  // __ANDROID__

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "litert/kotlin/src/main/jni/litert_model_wrapper.h"

namespace {

using ::litert::CompiledModel;
using ::litert::CpuOptions;
using ::litert::ElementType;
using ::litert::Environment;
using ::litert::Expected;
using ::litert::GpuOptions;
using ::litert::Layout;
using ::litert::Options;
using ::litert::OwnHandle;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::Unexpected;
using ::litert::jni::CompiledModelWrapper;
using ::litert::jni::ThrowLiteRtException;
using ::litert::qualcomm::QualcommOptions;

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
  kAllowSrcQuantizedFcConvOps = 2,
  kPrecision = 3,
  kBufferStorageType = 4,
  kPreferTextureWeights = 5,
  kSerializationDir = 6,
  kModelCacheKey = 7,
  kSerializeProgramCache = 8,
  kSerializeExternalTensors = 9,
  kExternalTensorsMode = 10,
  kExternalTensorPattern = 11,
  kBackend = 12,
  kPriority = 13,
  kNumStepsOfCommandBufferPreparations = 14,
};

// Keys for Qualcomm options, the values should match the ones in Kotlin.
enum QualcommOptionsKey {
  kLogLevel = 0,
  kUseHtpPreference = 1,
  kUseQint16AsQuint16 = 2,
  kEnableWeightSharing = 3,
  kDumpTensorIds = 4,
  kUseConvHmx = 5,
  kUseFoldRelu = 6,
  kHtpPerformanceMode = 7,
  kProfiling = 8,
  kIrJsonDir = 9,
  kDlcDir = 10,
  kVtcmSize = 11,
  kNumHvxThreads = 12,
  kOptimizationLevel = 13,
};

// Precision for GPU options, the values should match the ones in Kotlin.
enum Precision {
  kPrecisionDefault = 0,
  kPrecisionFp16 = 1,
  kPrecisionFp32 = 2,
};

// Converts the precision string to LiteRtDelegatePrecision.
GpuOptions::Precision ToGpuOptionsPrecision(const char* precision_str) {
  auto precision = std::stoi(precision_str);
  switch (precision) {
    case kPrecisionFp16:
      return GpuOptions::Precision::kFp16;
    case kPrecisionFp32:
      return GpuOptions::Precision::kFp32;
    default:
      return GpuOptions::Precision::kDefault;
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
GpuOptions::BufferStorageType ToGpuOptionsBufferStorageType(
    const char* buffer_storage_type_str) {
  auto type = std::stoi(buffer_storage_type_str);
  switch (type) {
    case kBufferStorageTypeBuffer:
      return GpuOptions::BufferStorageType::kBuffer;
    case kBufferStorageTypeTexture2D:
      return GpuOptions::BufferStorageType::kTexture2D;
    default:
      return GpuOptions::BufferStorageType::kDefault;
  }
}

// Backend for GPU options, the values should match the ones in Kotlin.
enum Backend {
  kBackendAutomatic = 0,
  kBackendOpenCl = 1,
  kBackendWebGpu = 2,
};

// Converts the backend string to GpuOptions::Backend.
GpuOptions::Backend ToGpuOptionsBackend(const char* backend_str) {
  auto backend = std::stoi(backend_str);
  switch (backend) {
    case kBackendOpenCl:
      return GpuOptions::Backend::kOpenCl;
    case kBackendWebGpu:
      return GpuOptions::Backend::kWebGpu;
    default:
      return GpuOptions::Backend::kAutomatic;
  }
}

// Priority for GPU options, the values should match the ones in Kotlin.
enum Priority {
  kPriorityDefault = 0,
  kPriorityLow = 1,
  kPriorityNormal = 2,
  kPriorityHigh = 3,
};

// Converts the priority string to GpuOptions::Priority.
GpuOptions::Priority ToGpuOptionsPriority(const char* priority_str) {
  auto priority = std::stoi(priority_str);
  switch (priority) {
    case kPriorityLow:
      return GpuOptions::Priority::kLow;
    case kPriorityNormal:
      return GpuOptions::Priority::kNormal;
    case kPriorityHigh:
      return GpuOptions::Priority::kHigh;
    default:
      return GpuOptions::Priority::kDefault;
  }
}

// Gets a CompiledModel from the given handles.
CompiledModel& GetCompiledModel(jlong compiled_model_handle) {
  // Extract the actual compiled model from the wrapper
  auto* wrapper =
      reinterpret_cast<CompiledModelWrapper*>(compiled_model_handle);
  ABSL_CHECK(wrapper != nullptr);
  return wrapper->compiled_model;
}

// Populates a CpuOptions from the given cpu options.
// The number of given options must be greater than 0.
Expected<void> PopulateCpuOptions(JNIEnv* env, CpuOptions& cpu_options,
                                  jintArray cpu_options_keys,
                                  jobjectArray cpu_options_values) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, cpu_options_keys);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, cpu_options_values);
  auto cpu_options_keys_size = env->GetArrayLength(cpu_options_keys);
  ABSL_CHECK(cpu_options_keys_size == cpu_options_values_size);

  for (int i = 0; i < cpu_options_keys_size; ++i) {
    if (cpu_options_keys_array[i] == CpuOptionsKey::kNumThreads) {
      LITERT_RETURN_IF_ERROR(
          cpu_options.SetNumThreads(std::stoi(cpu_options_values_vector[i])));
    } else if (cpu_options_keys_array[i] == CpuOptionsKey::kXnnPackFlags) {
      LITERT_RETURN_IF_ERROR(
          cpu_options.SetXNNPackFlags(std::stoi(cpu_options_values_vector[i])));
    } else if (cpu_options_keys_array[i] ==
               CpuOptionsKey::kXnnPackWeightCachePath) {
      LITERT_RETURN_IF_ERROR(
          cpu_options.SetXNNPackWeightCachePath(cpu_options_values_vector[i]));
    } else {
      LITERT_LOG(LITERT_ERROR, "Unknown CPU options key: %d.",
                 cpu_options_keys_array[i]);
    }
  }
  return {};
}

// Populates a GpuOptions from the given gpu options.
// The number of given options must be greater than 0.
Expected<void> PopulateGpuOptions(JNIEnv* env, GpuOptions& gpu_options,
                                  jintArray gpu_options_keys,
                                  jobjectArray gpu_options_values) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, gpu_options_keys);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, gpu_options_values);
  auto gpu_options_keys_size = env->GetArrayLength(gpu_options_keys);
  ABSL_CHECK(gpu_options_keys_size == gpu_options_values_size);

  LiteRtStatus status = kLiteRtStatusOk;
  for (int i = 0; i < gpu_options_keys_size; ++i) {
    switch (gpu_options_keys_array[i]) {
      case GpuOptionsKey::kConstantTensorSharing:
        status = gpu_options.EnableConstantTensorSharing(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options constantTensorSharing.");
        }
        break;
      case GpuOptionsKey::kInfiniteFloatCapping:
        status = gpu_options.EnableInfiniteFloatCapping(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options infiniteFloatCapping.");
        }
        break;
      case GpuOptionsKey::kAllowSrcQuantizedFcConvOps:
        status = gpu_options.EnableAllowSrcQuantizedFcConvOps(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(
              status, "Failed to set GPU options allowSrcQuantizedFcConvOps.");
        }
        break;
      case GpuOptionsKey::kPrecision:
        LITERT_RETURN_IF_ERROR(gpu_options.SetPrecision(
            ToGpuOptionsPrecision(gpu_options_values_vector[i])));
        break;
      case GpuOptionsKey::kBufferStorageType:
        LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(
            ToGpuOptionsBufferStorageType(gpu_options_values_vector[i])));
        break;
      case GpuOptionsKey::kPreferTextureWeights:
        status = gpu_options.SetPreferTextureWeights(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options preferTextureWeights.");
        }
        break;
      case GpuOptionsKey::kSerializationDir:
        status = gpu_options.SetSerializationDir(gpu_options_values_vector[i]);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options serializationDir.");
        }
        break;
      case GpuOptionsKey::kModelCacheKey:
        status = gpu_options.SetModelCacheKey(gpu_options_values_vector[i]);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status, "Failed to set GPU options modelCacheKey.");
        }
        break;
      case GpuOptionsKey::kSerializeProgramCache:
        status = gpu_options.SetSerializeProgramCache(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options serializeProgramCache.");
        }
        break;
      case GpuOptionsKey::kSerializeExternalTensors:
        status = gpu_options.SetSerializeExternalTensors(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(
              status, "Failed to set GPU options serializeExternalTensors.");
        }
        break;
      case GpuOptionsKey::kExternalTensorsMode:
        status = gpu_options.EnableExternalTensorsMode(
            strcmp(gpu_options_values_vector[i], "true") == 0);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options externalTensorsMode.");
        }
        break;
      case GpuOptionsKey::kExternalTensorPattern:
        status =
            gpu_options.AddExternalTensorPattern(gpu_options_values_vector[i]);
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options externalTensorPattern.");
        }
        break;
      case GpuOptionsKey::kBackend:
        LITERT_RETURN_IF_ERROR(gpu_options.SetBackend(
            ToGpuOptionsBackend(gpu_options_values_vector[i])));
        break;
      case GpuOptionsKey::kPriority:
        LITERT_RETURN_IF_ERROR(gpu_options.SetPriority(
            ToGpuOptionsPriority(gpu_options_values_vector[i])));
        break;
      case GpuOptionsKey::kNumStepsOfCommandBufferPreparations:
        status = gpu_options.SetNumStepsOfCommandBufferPreparations(
            std::stoi(gpu_options_values_vector[i]));
        if (status != kLiteRtStatusOk) {
          return Unexpected(status,
                            "Failed to set GPU options "
                            "numStepsOfCommandBufferPreparations.");
        }
        break;
      default:
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          absl::StrCat("Unknown GPU options key: ",
                                       gpu_options_keys_array[i]));
    }
  }
  return {};
}

// Populates a QualcommOptions from the given qualcomm options.
// The number of given options must be greater than 0.
Expected<void> PopulateQualcommOptions(JNIEnv* env,
                                       QualcommOptions& qualcomm_options,
                                       jintArray qualcomm_options_keys,
                                       jobjectArray qualcomm_options_values) {
  AUTO_CLEANUP_JNI_INT_ARRAY(env, qualcomm_options_keys);
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, qualcomm_options_values);
  auto qualcomm_options_keys_size = env->GetArrayLength(qualcomm_options_keys);
  ABSL_CHECK(qualcomm_options_keys_size == qualcomm_options_values_size);

  for (int i = 0; i < qualcomm_options_keys_size; ++i) {
    switch (qualcomm_options_keys_array[i]) {
      case QualcommOptionsKey::kLogLevel:
        qualcomm_options.SetLogLevel(static_cast<QualcommOptions::LogLevel>(
            std::stoi(qualcomm_options_values_vector[i])));
        break;
      case QualcommOptionsKey::kUseHtpPreference:
        qualcomm_options.SetUseHtpPreference(
            strcmp(qualcomm_options_values_vector[i], "true") == 0);
        break;
      case QualcommOptionsKey::kUseQint16AsQuint16:
        qualcomm_options.SetUseQint16AsQuint16(
            strcmp(qualcomm_options_values_vector[i], "true") == 0);
        break;
      case QualcommOptionsKey::kEnableWeightSharing:
        qualcomm_options.SetEnableWeightSharing(
            strcmp(qualcomm_options_values_vector[i], "true") == 0);
        break;
      case QualcommOptionsKey::kDumpTensorIds: {
        std::vector<std::string> ids_str =
            absl::StrSplit(qualcomm_options_values_vector[i], ',');
        std::vector<std::int32_t> ids;
        ids.reserve(ids_str.size());
        for (const auto& id_str : ids_str) {
          ids.push_back(std::stoi(id_str));
        }
        qualcomm_options.SetDumpTensorIds(ids);
        break;
      }
      case QualcommOptionsKey::kUseConvHmx:
        (qualcomm_options.SetUseConvHMX(
            strcmp(qualcomm_options_values_vector[i], "true") == 0));
        break;
      case QualcommOptionsKey::kUseFoldRelu:
        (qualcomm_options.SetUseFoldReLU(
            strcmp(qualcomm_options_values_vector[i], "true") == 0));
        break;
      case QualcommOptionsKey::kHtpPerformanceMode:
        qualcomm_options.SetHtpPerformanceMode(
            static_cast<QualcommOptions::HtpPerformanceMode>(
                std::stoi(qualcomm_options_values_vector[i])));
        break;
      case QualcommOptionsKey::kProfiling:
        qualcomm_options.SetProfiling(static_cast<QualcommOptions::Profiling>(
            std::stoi(qualcomm_options_values_vector[i])));
        break;
      case QualcommOptionsKey::kIrJsonDir:
        qualcomm_options.SetIrJsonDir(qualcomm_options_values_vector[i]);
        break;
      case QualcommOptionsKey::kDlcDir:
        qualcomm_options.SetDlcDir(qualcomm_options_values_vector[i]);
        break;
      case QualcommOptionsKey::kVtcmSize:
        qualcomm_options.SetVtcmSize(
            std::stoi(qualcomm_options_values_vector[i]));
        break;
      case QualcommOptionsKey::kNumHvxThreads:
        qualcomm_options.SetNumHvxThreads(
            std::stoi(qualcomm_options_values_vector[i]));
        break;
      case QualcommOptionsKey::kOptimizationLevel:
        qualcomm_options.SetOptimizationLevel(
            static_cast<QualcommOptions::OptimizationLevel>(
                std::stoi(qualcomm_options_values_vector[i])));
        break;
      default:
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          absl::StrCat("Unknown Qualcomm options key: ",
                                       qualcomm_options_keys_array[i]));
    }
  }
  return {};
}

Expected<Options> CreateOptions(JNIEnv* env, jintArray accelerators,
                                jintArray cpu_options_keys,
                                jobjectArray cpu_options_values,
                                jintArray gpu_options_keys,
                                jobjectArray gpu_options_values,
                                jintArray qualcomm_options_keys,
                                jobjectArray qualcomm_options_values) {
  int accelerators_size = env->GetArrayLength(accelerators);
  AUTO_CLEANUP_JNI_INT_ARRAY(env, accelerators);
  litert::HwAcceleratorSet acceleratorSet =
      litert::HwAcceleratorSet(litert::HwAccelerators::kNone);
  for (int i = 0; i < accelerators_size; ++i) {
    switch (accelerators_array[i]) {
      case litert::jni::kAccelatorNone:
        break;
      case litert::jni::kAccelatorCpu:
        acceleratorSet |= litert::HwAccelerators::kCpu;
        break;
      case litert::jni::kAccelatorGpu:
        acceleratorSet |= litert::HwAccelerators::kGpu;
        break;
      case litert::jni::kAccelatorNpu:
        acceleratorSet |= litert::HwAccelerators::kNpu;
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported accelerator: %d.",
                   accelerators_array[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  LITERT_RETURN_IF_ERROR(
      compilation_options.SetHardwareAccelerators(acceleratorSet));

  if (env->GetArrayLength(cpu_options_keys) > 0) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                            compilation_options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(PopulateCpuOptions(
        env, cpu_options, cpu_options_keys, cpu_options_values));
  }

  if (env->GetArrayLength(gpu_options_keys) > 0) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                            compilation_options.GetGpuOptions());
    LITERT_RETURN_IF_ERROR(PopulateGpuOptions(
        env, gpu_options, gpu_options_keys, gpu_options_values));

    if (env->GetArrayLength(qualcomm_options_keys) > 0) {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              compilation_options.GetQualcommOptions());
      LITERT_RETURN_IF_ERROR(PopulateQualcommOptions(env, qualcomm_options,
                                                     qualcomm_options_keys,
                                                     qualcomm_options_values));
    }
  }
  return std::move(compilation_options);
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

// Converts a C++ ElementType to a Java ElementType object.
jobject ToJavaElementType(JNIEnv* env, ElementType element_type) {
  jclass element_type_class =
      env->FindClass("com/google/ai/edge/litert/TensorType$ElementType");
  ABSL_CHECK(element_type_class != nullptr)
      << "Failed to find ElementType class.";

  std::string element_type_name;
  switch (element_type) {
    case ElementType::Int32:
      element_type_name = "INT";
      break;
    case ElementType::Float32:
      element_type_name = "FLOAT";
      break;
    case ElementType::Int8:
      element_type_name = "INT8";
      break;
    case ElementType::Bool:
      element_type_name = "BOOLEAN";
      break;
    case ElementType::Int64:
      element_type_name = "INT64";
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported element type in Kotlin: %d",
                 element_type);
      ThrowLiteRtException(env, kLiteRtStatusErrorUnsupported,
                           "Unsupported element type in Kotlin");
      return nullptr;
  }
  auto field_id = env->GetStaticFieldID(
      element_type_class, element_type_name.c_str(),
      "Lcom/google/ai/edge/litert/TensorType$ElementType;");
  ABSL_CHECK(field_id != nullptr)
      << "Failed to get field: " << element_type_name;

  auto java_element_type =
      env->GetStaticObjectField(element_type_class, field_id);
  ABSL_CHECK(java_element_type != nullptr)
      << "Failed to get element type: " << element_type_name;
  return java_element_type;
}

// Converts a C++ Layout to a Java Layout object.
jobject ToJavaLayout(JNIEnv* env, const Layout& layout) {
  jclass layout_class =
      env->FindClass("com/google/ai/edge/litert/TensorType$Layout");
  ABSL_CHECK(layout_class != nullptr) << "Failed to find Layout class.";

  auto dimensions = env->NewIntArray(layout.Dimensions().size());
  if (dimensions == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to allocate int array.");
    ThrowLiteRtException(env, kLiteRtStatusErrorMemoryAllocationFailure,
                         "Failed to allocate int array.");
    return nullptr;
  }
  env->SetIntArrayRegion(dimensions, 0, layout.Dimensions().size(),
                         layout.Dimensions().data());

  jobject layout_obj;
  if (layout.HasStrides()) {
    auto strides = env->NewIntArray(layout.Strides().size());
    if (strides == nullptr) {
      LITERT_LOG(LITERT_ERROR, "Failed to allocate int array.");
      ThrowLiteRtException(env, kLiteRtStatusErrorMemoryAllocationFailure,
                           "Failed to allocate int array.");
      return nullptr;
    }
    // Convert unsigned int to int.
    auto strides_vector =
        std::vector<jint>(layout.Strides().begin(), layout.Strides().end());
    env->SetIntArrayRegion(strides, 0, layout.Strides().size(),
                           strides_vector.data());

    auto constructor = env->GetMethodID(layout_class, "<init>", "([I[I)V");
    ABSL_CHECK(constructor != nullptr) << "Failed to get constructor.";
    layout_obj = env->NewObject(layout_class, constructor, dimensions, strides);
  } else {
    auto constructor = env->GetMethodID(layout_class, "<init>", "([I)V");
    ABSL_CHECK(constructor != nullptr) << "Failed to get constructor.";
    layout_obj = env->NewObject(layout_class, constructor, dimensions);
  }
  if (layout_obj == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create layout object.");
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create layout object.");
    return nullptr;
  }
  return layout_obj;
}

// Converts C++ RankedTensorType to a Java TensorType object.
jobject ToJavaTensorType(JNIEnv* env, const RankedTensorType& tensor_type) {
  auto element_type = tensor_type.ElementType();
  auto layout = tensor_type.Layout();
  jclass tensor_type_class =
      env->FindClass("com/google/ai/edge/litert/TensorType");
  ABSL_CHECK(tensor_type_class != nullptr)
      << "Failed to find TensorType class.";

  auto java_element_type = ToJavaElementType(env, element_type);
  if (java_element_type == nullptr) {
    // Exception already thrown.
    return nullptr;
  }
  jobject java_tensor_type;
  auto java_layout = ToJavaLayout(env, layout);
  if (java_layout == nullptr) {
    // Exception already thrown.
    return nullptr;
  }
  auto constructor = env->GetMethodID(
      tensor_type_class, "<init>",
      "(Lcom/google/ai/edge/litert/TensorType$ElementType;Lcom/google/ai/"
      "edge/litert/TensorType$Layout;)V");
  ABSL_CHECK(constructor != nullptr) << "Failed to get constructor.";
  java_tensor_type = env->NewObject(tensor_type_class, constructor,
                                    java_element_type, java_layout);
  env->DeleteLocalRef(java_layout);
  env->DeleteLocalRef(java_element_type);
  if (java_tensor_type == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor type object.");
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create tensor type object.");
    return nullptr;
  }
  return java_tensor_type;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __ANDROID__
JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateFromAsset(
    JNIEnv* env, jclass clazz, jlong env_handle, jobject asset_manager,
    jstring asset_name, jintArray accelerators, jintArray cpu_options_keys,
    jobjectArray cpu_options_values, jintArray gpu_options_keys,
    jobjectArray gpu_options_values, jintArray qualcomm_options_keys,
    jobjectArray qualcomm_options_values) {
  auto am = AAssetManager_fromJava(env, asset_manager);
  AUTO_CLEANUP_JNI_STRING(env, asset_name);
  ABSL_CHECK(asset_name_str != nullptr);
  auto model_asset = AAssetManager_open(am, asset_name_str, AASSET_MODE_BUFFER);
  if (model_asset == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to open asset: %s", asset_name_str);
    ThrowLiteRtException(env, kLiteRtStatusErrorNotFound,
                         "Failed to open asset.");
    return 0;
  }

  auto buffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(model_asset)),
      AAsset_getLength(model_asset));
  AAsset_close(model_asset);

  auto litert_env = reinterpret_cast<Environment*>(env_handle);
  ABSL_CHECK(litert_env != nullptr);

  auto compilation_options = CreateOptions(
      env, accelerators, cpu_options_keys, cpu_options_values, gpu_options_keys,
      gpu_options_values, qualcomm_options_keys, qualcomm_options_values);
  if (!compilation_options) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options: %s",
               compilation_options.Error().Message().c_str());
    ThrowLiteRtException(env, compilation_options.Error().Status(),
                         compilation_options.Error().Message());
    return 0;
  }

  auto compiled_model =
      CompiledModel::Create(*litert_env, buffer, *compilation_options);
  if (!compiled_model) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compiled model: %s",
               compiled_model.Error().Message().c_str());
    ThrowLiteRtException(env, compiled_model.Error().Status(),
                         compiled_model.Error().Message());
    return 0;
  }
  auto* wrapper =
      new CompiledModelWrapper(std::move(*compiled_model), std::move(buffer));
  return reinterpret_cast<jlong>(wrapper);
}
#endif  // __ANDROID__

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateFromFile(
    JNIEnv* env, jclass clazz, jlong env_handle, jstring file_path,
    jintArray accelerators, jintArray cpu_options_keys,
    jobjectArray cpu_options_values, jintArray gpu_options_keys,
    jobjectArray gpu_options_values, jintArray qualcomm_options_keys,
    jobjectArray qualcomm_options_values) {
  auto litert_env = reinterpret_cast<Environment*>(env_handle);
  ABSL_CHECK(litert_env != nullptr);

  auto compilation_options = CreateOptions(
      env, accelerators, cpu_options_keys, cpu_options_values, gpu_options_keys,
      gpu_options_values, qualcomm_options_keys, qualcomm_options_values);
  if (!compilation_options) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options: %s",
               compilation_options.Error().Message().c_str());
    ThrowLiteRtException(env, compilation_options.Error().Status(),
                         compilation_options.Error().Message());
    return 0;
  }

  AUTO_CLEANUP_JNI_STRING(env, file_path);
  ABSL_CHECK(file_path_str != nullptr);
  auto compiled_model =
      CompiledModel::Create(*litert_env, file_path_str, *compilation_options);
  if (!compiled_model) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compiled model: %s",
               compiled_model.Error().Message().c_str());
    ThrowLiteRtException(env, compiled_model.Error().Status(),
                         compiled_model.Error().Message());
    return 0;
  }

  auto* wrapper = new CompiledModelWrapper(std::move(*compiled_model));
  return reinterpret_cast<jlong>(wrapper);
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring input_name) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  ABSL_CHECK(input_name_str != nullptr);
  auto tensor_buffer =
      compiled_model.CreateInputBuffer(signature_str, input_name_str);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input buffer: %s",
               tensor_buffer.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffer.Error().Status(),
                         tensor_buffer.Error().Message());
    return 0;
  }
  auto* tensor_buffer_ptr = new TensorBuffer(std::move(*tensor_buffer));
  return reinterpret_cast<jlong>(tensor_buffer_ptr);
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetInputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring input_name) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  ABSL_CHECK(input_name_str != nullptr);
  auto requirements =
      compiled_model.GetInputBufferRequirements(signature_str, input_name_str);
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
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring output_name) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  ABSL_CHECK(output_name_str != nullptr);
  auto tensor_buffer =
      compiled_model.CreateOutputBuffer(signature_str, output_name_str);
  if (!tensor_buffer) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output buffer: %s",
               tensor_buffer.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_buffer.Error().Status(),
                         tensor_buffer.Error().Message());
    return 0;
  }
  auto* tensor_buffer_ptr = new TensorBuffer(std::move(*tensor_buffer));
  return reinterpret_cast<jlong>(tensor_buffer_ptr);
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetOutputBufferRequirements(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring output_name) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  ABSL_CHECK(output_name_str != nullptr);
  auto requirements = compiled_model.GetOutputBufferRequirements(
      signature_str, output_name_str);
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
    JNIEnv* env, jclass clazz, jlong compiled_model_handle,
    jint signature_index) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

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
    auto* tensor_buffer_ptr =
        new TensorBuffer(std::move(tensor_buffers->at(i)));
    input_tensor_buffers.push_back(reinterpret_cast<jlong>(tensor_buffer_ptr));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          input_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
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
    auto* tensor_buffer_ptr =
        new TensorBuffer(std::move(tensor_buffers->at(i)));
    input_tensor_buffers.push_back(reinterpret_cast<jlong>(tensor_buffer_ptr));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          input_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffers(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle,
    jint signature_index) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

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
    auto* tensor_buffer_ptr =
        new TensorBuffer(std::move(tensor_buffers->at(i)));
    output_tensor_buffers.push_back(reinterpret_cast<jlong>(tensor_buffer_ptr));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          output_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT jlongArray JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOutputBuffersBySignature(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
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
    auto* tensor_buffer_ptr =
        new TensorBuffer(std::move(tensor_buffers->at(i)));
    output_tensor_buffers.push_back(reinterpret_cast<jlong>(tensor_buffer_ptr));
  }
  jlongArray handles_array = env->NewLongArray(tensor_buffers->size());
  env->SetLongArrayRegion(handles_array, 0, tensor_buffers->size(),
                          output_tensor_buffers.data());
  return handles_array;
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_CompiledModel_nativeRun(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle,
    jint signature_index, jlongArray input_buffers, jlongArray output_buffers) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  auto num_inputs = env->GetArrayLength(input_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(input_buffers_array[i]);
    // TODO(niuchl): Use TensorBuffer* when it's possible.
    input_buffer_vector.push_back(litert::TensorBuffer::WrapCObject(
        litert_tensor_buffer->Get(), litert::OwnHandle::kNo));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    output_buffer_vector.push_back(litert::TensorBuffer::WrapCObject(
        litert_tensor_buffer->Get(), litert::OwnHandle::kNo));
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
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jlongArray input_buffers, jlongArray output_buffers) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  auto num_inputs = env->GetArrayLength(input_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  std::vector<litert::TensorBuffer> input_buffer_vector;
  input_buffer_vector.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(input_buffers_array[i]);
    input_buffer_vector.push_back(litert::TensorBuffer::WrapCObject(
        litert_tensor_buffer->Get(), litert::OwnHandle::kNo));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    output_buffer_vector.push_back(litert::TensorBuffer::WrapCObject(
        litert_tensor_buffer->Get(), litert::OwnHandle::kNo));
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
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
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jobjectArray input_keys, jlongArray input_buffers, jobjectArray output_keys,
    jlongArray output_buffers) {
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
    auto* buffer = reinterpret_cast<TensorBuffer*>(input_buffers_array[i]);
    input_buffer_map[key] = litert::TensorBuffer::WrapCObject(
        buffer->Get(), litert::OwnHandle::kNo);
  }

  AUTO_CLEANUP_JNI_STRING_ARRAY(env, output_keys);
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer>
      output_buffer_map;
  output_buffer_map.reserve(output_keys_size);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  for (int i = 0; i < output_keys_size; ++i) {
    auto key = output_keys_vector[i];
    auto* buffer = reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    output_buffer_map[key] = litert::TensorBuffer::WrapCObject(
        buffer->Get(), litert::OwnHandle::kNo);
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature_str != nullptr);
  auto& compiled_model = GetCompiledModel(compiled_model_handle);
  auto result =
      compiled_model.Run(signature_str, input_buffer_map, output_buffer_map);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "Failed to run model: %s",
               result.Error().Message().c_str());
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetInputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name,
    jstring signature) {
  auto& compiled_model = GetCompiledModel(handle);
  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  ABSL_CHECK(input_name != nullptr);
  auto tensor_type =
      compiled_model.GetInputTensorType(signature_str, input_name_str);
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get input tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         tensor_type.Error().Message());
    return nullptr;
  }
  return ToJavaTensorType(env, *tensor_type);
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetOutputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name,
    jstring signature) {
  auto& compiled_model = GetCompiledModel(handle);
  AUTO_CLEANUP_JNI_STRING(env, signature);
  ABSL_CHECK(signature != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  ABSL_CHECK(output_name != nullptr);
  auto tensor_type =
      compiled_model.GetOutputTensorType(signature_str, output_name_str);
  if (!tensor_type) {
    LITERT_LOG(LITERT_ERROR, "Failed to get output tensor type: %s",
               tensor_type.Error().Message().c_str());
    ThrowLiteRtException(env, tensor_type.Error().Status(),
                         tensor_type.Error().Message());
    return nullptr;
  }
  return ToJavaTensorType(env, *tensor_type);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroy(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle) {
  delete reinterpret_cast<CompiledModelWrapper*>(handle);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
