#include "litert/kotlin/src/main/jni/litert_compiled_model_jni.h"

#include <jni.h>

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compilation_options.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/kotlin/src/main/jni/litert_jni_common.h"

namespace {

using ::litert::jni::ThrowLiteRtException;

// Creates a CompiledModel from the given handles.
// The handles are not owned by the returned CompiledModel.
litert::CompiledModel CreateCompileModel(jlong compiled_model_handle,
                                         jlong model_handle) {
  auto c_model = reinterpret_cast<LiteRtModel>(model_handle);
  ABSL_CHECK(c_model != nullptr);
  auto c_compiled_model =
      reinterpret_cast<LiteRtCompiledModel>(compiled_model_handle);
  ABSL_CHECK(c_compiled_model != nullptr);
  return litert::CompiledModel(c_model, c_compiled_model,
                               litert::OwnHandle::kNo);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_com_google_ai_edge_litert_CompiledModel_nativeCreate(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong env_handle,
                                                          jlong model_handle,
                                                          jintArray options) {
  int options_size = env->GetArrayLength(options);
  AUTO_CLEANUP_JNI_INT_ARRAY(env, options);
  LiteRtHwAcceleratorSet accelerators = kLiteRtHwAcceleratorNone;
  for (int i = 0; i < options_size; ++i) {
    switch (options_array[i]) {
      case litert::jni::kAccelatorNone:
        break;
      case litert::jni::kAccelatorCpu:
        accelerators |= kLiteRtHwAcceleratorCpu;
        break;
      case litert::jni::kAccelatorGpu:
        accelerators |= kLiteRtHwAcceleratorGpu;
        break;
      case litert::jni::kAccelatorNpu:
        accelerators |= kLiteRtHwAcceleratorNpu;
        break;
      default:
        LITERT_LOG(LITERT_ERROR, "Unsupported accelerator: %d.",
                   options_array[i]);
    }
  }

  LiteRtCompilationOptions compilation_options = nullptr;
  auto status = LiteRtCreateCompilationOptions(&compilation_options);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to create compilation options.");
    ThrowLiteRtException(env, status, "Failed to create compilation options.");
    return 0;
  }
  status = LiteRtSetCompilationOptionsHardwareAccelerators(compilation_options,
                                                           accelerators);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to set hardware accelerators.");
    ThrowLiteRtException(env, status, "Failed to set hardware accelerators.");
    return 0;
  }

  auto litert_env = reinterpret_cast<LiteRtEnvironment>(env_handle);
  ABSL_CHECK(litert_env != nullptr);
  auto model = reinterpret_cast<LiteRtModel>(model_handle);
  ABSL_CHECK(model != nullptr);
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
