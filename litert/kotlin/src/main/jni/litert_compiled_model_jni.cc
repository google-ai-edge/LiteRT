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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_api_types.h"
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
#include "litert/cc/options/litert_concrete_options_base.h"
#include "litert/core/litert_options_encoder.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "litert/kotlin/src/main/jni/litert_model_wrapper.h"

namespace {

using ::litert::CompiledModel;
using ::litert::ElementType;
using ::litert::Environment;
using ::litert::Expected;
using ::litert::Layout;
using ::litert::Options;
using ::litert::RankedTensorType;
using ::litert::TensorBuffer;
using ::litert::jni::CompiledModelWrapper;
using ::litert::jni::JniOptionsBuilder;
using ::litert::jni::ThrowLiteRtException;

class JniEncodedOpaqueOptions : public litert::ConcreteOptionsBase {
 public:
  JniEncodedOpaqueOptions(std::string identifier, std::string encoded_payload)
      : identifier_(std::move(identifier)),
        encoded_payload_(std::move(encoded_payload)) {}

  LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const override {
    if (identifier == nullptr || payload == nullptr ||
        payload_deleter == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *identifier = identifier_.c_str();
    litert::internal::MakeCStringPayload(encoded_payload_, payload,
                                         payload_deleter);
    return kLiteRtStatusOk;
  }

 private:
  std::string identifier_;
  std::string encoded_payload_;
};

// Gets a JniOptionsBuilder from the given handle.
JniOptionsBuilder& GetOptionsBuilder(jlong options_handle) {
  auto* builder = reinterpret_cast<JniOptionsBuilder*>(options_handle);
  LITERT_INTERNAL_CHECK(builder != nullptr);
  return *builder;
}

// Gets a CompiledModel from the given handles.
CompiledModel& GetCompiledModel(jlong compiled_model_handle) {
  // Extract the actual compiled model from the wrapper
  auto* wrapper =
      reinterpret_cast<CompiledModelWrapper*>(compiled_model_handle);
  LITERT_INTERNAL_CHECK(wrapper != nullptr);
  return wrapper->compiled_model;
}

enum class ResizeInputTensorMode { kStrict, kNonStrict };

void ResizeInputTensor(JNIEnv* env, jlong compiled_model_handle,
                       jstring signature, jstring input_name,
                       jintArray dimensions, ResizeInputTensorMode mode) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  LITERT_INTERNAL_CHECK(input_name_str != nullptr);
  const auto dimensions_size = env->GetArrayLength(dimensions);
  AUTO_CLEANUP_JNI_INT_ARRAY(env, dimensions);
  auto dimensions_span = absl::MakeConstSpan(
      reinterpret_cast<const int*>(dimensions_array), dimensions_size);

  auto result = mode == ResizeInputTensorMode::kStrict
                    ? compiled_model.ResizeInputTensor(
                          signature_str, input_name_str, dimensions_span)
                    : compiled_model.ResizeInputTensorNonStrict(
                          signature_str, input_name_str, dimensions_span);
  if (!result) {
    LITERT_LOG(LITERT_ERROR, "Failed to resize input tensor: %s",
               result.Error().Message().c_str());
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

Expected<Options> CreateOptions(JNIEnv* env, jintArray accelerators) {
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
  if (element_type_class == nullptr) {
    LITERT_FATAL("Failed to find ElementType class.");
  }

  std::string element_type_name;
  switch (element_type) {
    case ElementType::Int32:
      element_type_name = "INT";
      break;
    case ElementType::Int16:
      element_type_name = "INT16";
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
  if (field_id == nullptr) {
    LITERT_FATAL("Failed to get field: %s", element_type_name.c_str());
  }

  auto java_element_type =
      env->GetStaticObjectField(element_type_class, field_id);
  if (java_element_type == nullptr) {
    LITERT_FATAL("Failed to get element type: %s", element_type_name.c_str());
  }
  return java_element_type;
}

// Converts a C++ Layout to a Java Layout object.
jobject ToJavaLayout(JNIEnv* env, const Layout& layout) {
  jclass layout_class =
      env->FindClass("com/google/ai/edge/litert/TensorType$Layout");
  if (layout_class == nullptr) {
    LITERT_FATAL("Failed to find Layout class.");
  }

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
    if (constructor == nullptr) {
      LITERT_FATAL("Failed to get constructor.");
    }
    layout_obj = env->NewObject(layout_class, constructor, dimensions, strides);
  } else {
    auto constructor = env->GetMethodID(layout_class, "<init>", "([I)V");
    if (constructor == nullptr) {
      LITERT_FATAL("Failed to get constructor.");
    }
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
  if (tensor_type_class == nullptr) {
    LITERT_FATAL("Failed to find TensorType class.");
  }

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
  if (constructor == nullptr) {
    LITERT_FATAL("Failed to get constructor.");
  }
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
    jstring asset_name, jlong options_handle) {
  auto am = AAssetManager_fromJava(env, asset_manager);
  AUTO_CLEANUP_JNI_STRING(env, asset_name);
  LITERT_INTERNAL_CHECK(asset_name_str != nullptr);
  auto model_asset = AAssetManager_open(am, asset_name_str, AASSET_MODE_BUFFER);
  if (model_asset == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to open asset: %s", asset_name_str);
    ThrowLiteRtException(env, kLiteRtStatusErrorNotFound,
                         "Failed to open asset.");
    return 0;
  }

  const off_t asset_length = AAsset_getLength(model_asset);
  if (asset_length <= 0) {
    LITERT_LOG(LITERT_ERROR, "Asset '%s' is empty or invalid size: %ld",
               asset_name_str, static_cast<long>(asset_length));
    AAsset_close(model_asset);
    ThrowLiteRtException(env, kLiteRtStatusErrorNotFound,
                         "Empty or invalid asset.");
    return 0;
  }

  litert::OwningBufferRef<uint8_t> buffer(static_cast<size_t>(asset_length));
  int bytes_read = AAsset_read(model_asset, buffer.Data(), asset_length);
  AAsset_close(model_asset);
  if (bytes_read != asset_length) {
    LITERT_LOG(LITERT_ERROR, "Failed to read full asset '%s'", asset_name_str);
    ThrowLiteRtException(env, kLiteRtStatusErrorFileIO,
                         "Failed to read asset data.");
    return 0;
  }

  auto litert_env = reinterpret_cast<Environment*>(env_handle);
  LITERT_INTERNAL_CHECK(litert_env != nullptr);
  auto& builder = GetOptionsBuilder(options_handle);

  auto compiled_model =
      CompiledModel::Create(*litert_env, buffer, builder.options);
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
    jlong options_handle) {
  auto litert_env = reinterpret_cast<Environment*>(env_handle);
  LITERT_INTERNAL_CHECK(litert_env != nullptr);
  auto& builder = GetOptionsBuilder(options_handle);

  AUTO_CLEANUP_JNI_STRING(env, file_path);
  if (file_path_str == nullptr) {
    LITERT_FATAL("Model file path is null.");
  }
  auto compiled_model =
      CompiledModel::Create(*litert_env, file_path_str, builder.options);
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
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateOptions(
    JNIEnv* env, jclass clazz, jintArray accelerators) {
  auto compilation_options = CreateOptions(env, accelerators);
  if (!compilation_options) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options: %s",
               compilation_options.Error().Message().c_str());
    ThrowLiteRtException(env, compilation_options.Error().Status(),
                         compilation_options.Error().Message());
    return 0;
  }
  auto* builder = new JniOptionsBuilder(std::move(*compilation_options));
  return reinterpret_cast<jlong>(builder);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeDestroyOptions(
    JNIEnv* env, jclass clazz, jlong options_handle) {
  delete reinterpret_cast<JniOptionsBuilder*>(options_handle);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeWriteBoolOption(
    JNIEnv* env, jclass clazz, jlong options_handle, jstring key,
    jboolean value) {
  auto& builder = GetOptionsBuilder(options_handle);
  AUTO_CLEANUP_JNI_STRING(env, key);
  if (key_str == nullptr) {
    LITERT_FATAL("Option key is null.");
  }
  LiteRtStatus status = builder.encoder.EncodeBool(key_str, value == JNI_TRUE);
  if (status != kLiteRtStatusOk) {
    ThrowLiteRtException(
        env, status, absl::StrCat("Failed to write boolean option: ", key_str));
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeWriteIntOption(
    JNIEnv* env, jclass clazz, jlong options_handle, jstring key, jlong value) {
  auto& builder = GetOptionsBuilder(options_handle);
  AUTO_CLEANUP_JNI_STRING(env, key);
  if (key_str == nullptr) {
    LITERT_FATAL("Option key is null.");
  }
  LiteRtStatus status =
      builder.encoder.EncodeInt(key_str, static_cast<int64_t>(value));
  if (status != kLiteRtStatusOk) {
    ThrowLiteRtException(
        env, status, absl::StrCat("Failed to write integer option: ", key_str));
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeWriteStringOption(
    JNIEnv* env, jclass clazz, jlong options_handle, jstring key,
    jstring value) {
  auto& builder = GetOptionsBuilder(options_handle);
  AUTO_CLEANUP_JNI_STRING(env, key);
  if (key_str == nullptr) {
    LITERT_FATAL("Option key is null.");
  }
  AUTO_CLEANUP_JNI_STRING(env, value);
  if (value_str == nullptr) {
    LITERT_FATAL("Option value is null.");
  }
  LiteRtStatus status = builder.encoder.EncodeString(key_str, value_str);
  if (status != kLiteRtStatusOk) {
    ThrowLiteRtException(
        env, status, absl::StrCat("Failed to write string option: ", key_str));
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeWriteStringArrayOption(
    JNIEnv* env, jclass clazz, jlong options_handle, jstring key,
    jobjectArray values) {
  auto& builder = GetOptionsBuilder(options_handle);
  AUTO_CLEANUP_JNI_STRING(env, key);
  if (key_str == nullptr) {
    LITERT_FATAL("Option key is null.");
  }
  AUTO_CLEANUP_JNI_STRING_ARRAY(env, values);
  std::vector<absl::string_view> value_views;
  value_views.reserve(values_size);
  for (const char* item : values_vector) {
    if (item == nullptr) {
      LITERT_FATAL("Option array element is null.");
    }
    value_views.emplace_back(item);
  }
  LiteRtStatus status = builder.encoder.EncodeStringArray(key_str, value_views);
  if (status != kLiteRtStatusOk) {
    ThrowLiteRtException(
        env, status,
        absl::StrCat("Failed to write string array option: ", key_str));
  }
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeFlushOpaqueOptions(
    JNIEnv* env, jclass clazz, jlong options_handle, jstring identifier) {
  auto& builder = GetOptionsBuilder(options_handle);
  if (builder.encoder.Empty()) {
    return;
  }
  AUTO_CLEANUP_JNI_STRING(env, identifier);
  if (identifier_str == nullptr) {
    LITERT_FATAL("Opaque options identifier is null.");
  }
  std::string ident(identifier_str);
  std::string encoded_payload = builder.encoder.Release();
  auto result = builder.options.AddOpaqueOptions(
      std::make_unique<JniEncodedOpaqueOptions>(std::move(ident),
                                                std::move(encoded_payload)));
  if (!result) {
    ThrowLiteRtException(env, result.Error().Status(),
                         result.Error().Message());
  }
}

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreateInputBuffer(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring input_name) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);

  AUTO_CLEANUP_JNI_STRING(env, signature);
  if (signature_str == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "Signature is null.");
    return 0;
  }
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  if (input_name_str == nullptr) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "Input name is null.");
    return 0;
  }
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
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  LITERT_INTERNAL_CHECK(input_name_str != nullptr);
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
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  LITERT_INTERNAL_CHECK(output_name_str != nullptr);
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
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  LITERT_INTERNAL_CHECK(output_name_str != nullptr);
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

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeResizeInputTensor(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring input_name, jintArray dimensions) {
  ResizeInputTensor(env, compiled_model_handle, signature, input_name,
                    dimensions, ResizeInputTensorMode::kStrict);
}

JNIEXPORT void JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeResizeInputTensorNonStrict(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle, jstring signature,
    jstring input_name, jintArray dimensions) {
  ResizeInputTensor(env, compiled_model_handle, signature, input_name,
                    dimensions, ResizeInputTensorMode::kNonStrict);
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
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
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
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
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
    auto input_buffer = litert_tensor_buffer->Duplicate();
    if (!input_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate input buffer: %s",
                 input_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, input_buffer.Error().Status(),
                           input_buffer.Error().Message());
      return;
    }
    input_buffer_vector.push_back(std::move(*input_buffer));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    auto output_buffer = litert_tensor_buffer->Duplicate();
    if (!output_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate output buffer: %s",
                 output_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, output_buffer.Error().Status(),
                           output_buffer.Error().Message());
      return;
    }
    output_buffer_vector.push_back(std::move(*output_buffer));
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
    auto input_buffer = litert_tensor_buffer->Duplicate();
    if (!input_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate input buffer: %s",
                 input_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, input_buffer.Error().Status(),
                           input_buffer.Error().Message());
      return;
    };
    input_buffer_vector.push_back(std::move(*input_buffer));
  }

  auto num_outputs = env->GetArrayLength(output_buffers);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  std::vector<litert::TensorBuffer> output_buffer_vector;
  output_buffer_vector.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto* litert_tensor_buffer =
        reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    auto output_buffer = litert_tensor_buffer->Duplicate();
    if (!output_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate output buffer: %s",
                 output_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, output_buffer.Error().Status(),
                           output_buffer.Error().Message());
      return;
    }
    output_buffer_vector.push_back(std::move(*output_buffer));
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
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
  if (env->GetArrayLength(input_keys) != env->GetArrayLength(input_buffers)) {
    LITERT_FATAL("Number of input keys and buffers do not match.");
  }
  if (env->GetArrayLength(output_keys) != env->GetArrayLength(output_buffers)) {
    LITERT_FATAL("Number of output keys and buffers do not match.");
  }

  AUTO_CLEANUP_JNI_STRING_ARRAY(env, input_keys);
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> input_buffer_map;
  input_buffer_map.reserve(input_keys_size);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, input_buffers);
  for (int i = 0; i < input_keys_size; ++i) {
    auto key = input_keys_vector[i];
    auto* buffer = reinterpret_cast<TensorBuffer*>(input_buffers_array[i]);
    auto input_buffer = buffer->Duplicate();
    if (!input_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate input buffer: %s",
                 input_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, input_buffer.Error().Status(),
                           input_buffer.Error().Message());
      return;
    }
    input_buffer_map.try_emplace(key, std::move(*input_buffer));
  }

  AUTO_CLEANUP_JNI_STRING_ARRAY(env, output_keys);
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer>
      output_buffer_map;
  output_buffer_map.reserve(output_keys_size);
  AUTO_CLEANUP_JNI_LONG_ARRAY(env, output_buffers);
  for (int i = 0; i < output_keys_size; ++i) {
    auto key = output_keys_vector[i];
    auto* buffer = reinterpret_cast<TensorBuffer*>(output_buffers_array[i]);
    auto output_buffer = buffer->Duplicate();
    if (!output_buffer) {
      LITERT_LOG(LITERT_ERROR, "Failed to duplicate output buffer: %s",
                 output_buffer.Error().Message().c_str());
      ThrowLiteRtException(env, output_buffer.Error().Status(),
                           output_buffer.Error().Message());
      return;
    }
    output_buffer_map.try_emplace(key, std::move(*output_buffer));
  }

  AUTO_CLEANUP_JNI_STRING(env, signature);
  LITERT_INTERNAL_CHECK(signature_str != nullptr);
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
  LITERT_INTERNAL_CHECK(signature != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, input_name);
  LITERT_INTERNAL_CHECK(input_name != nullptr);
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
  LITERT_INTERNAL_CHECK(signature != nullptr);
  AUTO_CLEANUP_JNI_STRING(env, output_name);
  LITERT_INTERNAL_CHECK(output_name != nullptr);
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

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeGetLastInferenceDurationNanoseconds(
    JNIEnv* env, jclass clazz, jlong compiled_model_handle) {
  auto& compiled_model = GetCompiledModel(compiled_model_handle);
  auto duration = compiled_model.LastInferenceDurationNanoseconds();
  if (!duration) {
    LITERT_LOG(LITERT_ERROR, "Failed to get last inference duration: %s",
               duration.Error().Message().c_str());
    return -1;
  }
  return *duration;
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
