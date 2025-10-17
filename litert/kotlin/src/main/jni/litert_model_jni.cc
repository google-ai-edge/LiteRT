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

#include "litert/kotlin/src/main/jni/litert_model_jni.h"

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <cstdint>

#include "litert/cc/litert_buffer_ref.h"
#endif  // __ANDROID__

#include <jni.h>

#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_model.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"
#include "litert/kotlin/src/main/jni/litert_model_wrapper.h"

namespace {
using litert::ElementType;
using litert::Layout;
using litert::Model;
using litert::jni::ModelWrapper;
using litert::jni::ThrowLiteRtException;

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

// Converts C++ RankedTensorType and UnrankedTensorType to a Java TensorType
// object.
jobject ToJavaTensorType(JNIEnv* env, ElementType element_type,
                         const Layout* layout) {
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
  if (layout != nullptr) {
    auto java_layout = ToJavaLayout(env, *layout);
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
  } else {
    auto constructor = env->GetMethodID(
        tensor_type_class, "<init>",
        "(Lcom/google/ai/edge/litert/TensorType$ElementType;)V");
    ABSL_CHECK(constructor != nullptr) << "Failed to get constructor.";
    java_tensor_type =
        env->NewObject(tensor_type_class, constructor, java_element_type);
  }
  env->DeleteLocalRef(java_element_type);
  if (java_tensor_type == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create tensor type object.");
    ThrowLiteRtException(env, kLiteRtStatusErrorRuntimeFailure,
                         "Failed to create tensor type object.");
    return nullptr;
  }
  return java_tensor_type;
}

jobject GetTensorType(JNIEnv* env, bool is_input, jlong handle, jstring name,
                      jstring signature) {
  AUTO_CLEANUP_JNI_STRING(env, name);
  AUTO_CLEANUP_JNI_STRING(env, signature);
  auto* wrapper = reinterpret_cast<ModelWrapper*>(handle);
  if (!wrapper || !wrapper->model) {
    ThrowLiteRtException(env, kLiteRtStatusErrorInvalidArgument,
                         "Invalid model handle");
    return nullptr;
  }
  auto model = Model::CreateFromNonOwnedHandle(wrapper->model);
  auto subgraph =
      signature_str ? model.Subgraph(signature_str) : model.MainSubgraph();
  if (!subgraph) {
    LITERT_LOG(LITERT_ERROR, "Failed to get subgraph: %s",
               subgraph.Error().Message().c_str());
    ThrowLiteRtException(env, subgraph.Error().Status(),
                         subgraph.Error().Message());
    return nullptr;
  }
  auto tensor =
      is_input ? subgraph->Input(name_str) : subgraph->Output(name_str);
  if (!tensor) {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor: %s",
               tensor.Error().Message().c_str());
    ThrowLiteRtException(env, tensor.Error().Status(),
                         tensor.Error().Message());
    return nullptr;
  }
  if (tensor->TypeId() == kLiteRtRankedTensorType) {
    auto ranked_tensor_type = tensor->RankedTensorType();
    if (!ranked_tensor_type) {
      LITERT_LOG(LITERT_ERROR, "Failed to get ranked tensor type: %s",
                 ranked_tensor_type.Error().Message().c_str());
      ThrowLiteRtException(env, ranked_tensor_type.Error().Status(),
                           ranked_tensor_type.Error().Message());
      return nullptr;
    }
    return ToJavaTensorType(env, tensor->ElementType(),
                            &(ranked_tensor_type->Layout()));
  } else {
    return ToJavaTensorType(env, tensor->ElementType(), nullptr);
  }
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __ANDROID__
JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadAsset(
    JNIEnv* env, jclass clazz, jobject asset_manager, jstring asset_name) {
  auto am = AAssetManager_fromJava(env, asset_manager);
  AUTO_CLEANUP_JNI_STRING(env, asset_name);
  auto g_model_asset =
      AAssetManager_open(am, asset_name_str, AASSET_MODE_BUFFER);
  if (g_model_asset == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to open asset: %s", asset_name_str);
    ThrowLiteRtException(env, kLiteRtStatusErrorNotFound,
                         "Failed to open asset.");
    return 0;
  }

  auto buffer = litert::OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(g_model_asset)),
      AAsset_getLength(g_model_asset));
  AAsset_close(g_model_asset);

  LiteRtModel model = nullptr;
  auto status =
      LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from asset.");
    ThrowLiteRtException(env, status, "Failed to create model from asset.");
    return 0;
  }
  // Create wrapper to keep buffer alive
  auto* wrapper = new ModelWrapper(model, std::move(buffer));
  return reinterpret_cast<jlong>(wrapper);
}
#endif  // __ANDROID__

JNIEXPORT jlong JNICALL Java_com_google_ai_edge_litert_Model_nativeLoadFile(
    JNIEnv* env, jclass clazz, jstring file_path) {
  AUTO_CLEANUP_JNI_STRING(env, file_path);
  LiteRtModel model = nullptr;
  auto status = LiteRtCreateModelFromFile(file_path_str, &model);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create model from file.");
    ThrowLiteRtException(env, status, "Failed to create model from file.");
    return 0;
  }
  // Create wrapper for consistency (no buffer needed for file-based models)
  auto* wrapper = new ModelWrapper(model);
  return reinterpret_cast<jlong>(wrapper);
}

JNIEXPORT void JNICALL Java_com_google_ai_edge_litert_Model_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
  auto* wrapper = reinterpret_cast<ModelWrapper*>(handle);
  delete wrapper;  // Destructor will call LiteRtDestroyModel
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_Model_nativeGetInputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name,
    jstring signature) {
  return GetTensorType(env, /*is_input=*/true, handle, input_name, signature);
}

JNIEXPORT jobject JNICALL
Java_com_google_ai_edge_litert_Model_nativeGetOutputTensorType(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name,
    jstring signature) {
  return GetTensorType(env, /*is_input=*/false, handle, output_name, signature);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
