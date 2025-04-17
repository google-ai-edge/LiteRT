// This sample scenario is mostly taken from:
// https://source.corp.google.com/piper///depot/google3/third_party/tensorflow/lite/c/c_test.c?q=TestSmokeTest

#include "litert/google/sample_app_android/aot/src/native/org_odml_litert_litert_aot_LiteRtJni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model_buffer.h"
#include "litert/core/model/model_load.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/google/sample_app_android/aot/src/native/utils.h"
#include "util/java/jni_helper.h"

namespace {
using litert::CompiledModel;
using litert::Expected;
using litert::Model;
using litert::OwningBufferRef;
using litert::internal::FlatbufferWrapper;
using litert::internal::GetModelBufWithByteCode;
using litert::internal::LoadModelFromBuffer;
using litert::internal::TflModel;
using util::java::ThrowingJniHelper;

constexpr const float kTestInput0Tensor[] = {1, 2};
constexpr const float kTestInput1Tensor[] = {10, 20};
constexpr const float kTestOutputTensor[] = {11, 22};

constexpr const size_t kTestInput0Size =
    sizeof(kTestInput0Tensor) / sizeof(kTestInput0Tensor[0]);
constexpr const size_t kTestInput1Size =
    sizeof(kTestInput1Tensor) / sizeof(kTestInput1Tensor[0]);
constexpr const size_t kTestOutputSize =
    sizeof(kTestOutputTensor) / sizeof(kTestOutputTensor[0]);

Expected<std::unique_ptr<LiteRtModelT>> LoadTfLiteModelFromAsset(
    ThrowingJniHelper* jni_helper, AAssetManager* aAssetManager,
    jstring tflite_model_asset_name) {
  auto g_model_asset = AAssetManager_open(
      aAssetManager,
      jni_helper->GetStringMutf8(tflite_model_asset_name).c_str(),
      AASSET_MODE_BUFFER);
  // RETURN_IF_JNI_FAILURE(jni_helper);

  auto buffer = OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(g_model_asset)),
      AAsset_getLength(g_model_asset));
  return LoadModelFromBuffer(buffer);
}

Expected<OwningBufferRef<uint8_t>> LoadBinaryModelFromAsset(
    ThrowingJniHelper* jni_helper, AAssetManager* aAssetManager,
    jstring npu_model_asset_name) {
  auto g_model_asset = AAssetManager_open(
      aAssetManager, jni_helper->GetStringMutf8(npu_model_asset_name).c_str(),
      AASSET_MODE_BUFFER);
  // RETURN_IF_JNI_FAILURE(jni_helper);

  return OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(AAsset_getBuffer(g_model_asset)),
      AAsset_getLength(g_model_asset));
}

Expected<OwningBufferRef<uint8_t>> GetModelBuf(ThrowingJniHelper* jni_helper,
                                               AAssetManager* aAssetManager,
                                               jstring tflite_model_asset_name,
                                               jstring npu_model_asset_name) {
  auto model = LoadTfLiteModelFromAsset(jni_helper, aAssetManager,
                                        tflite_model_asset_name);
  if (!model) {
    return model.Error();
  }

  auto npu_file_buf =
      LoadBinaryModelFromAsset(jni_helper, aAssetManager, npu_model_asset_name);
  if (!npu_file_buf) {
    return npu_file_buf.Error();
  }

  return GetModelBufWithByteCode(std::move(**model), std::move(*npu_file_buf));
}

}  // namespace

void Java_org_odml_litert_litert_aot_LiteRtJni_runInference(
    JNIEnv* env, jobject litertJni, jobject asset_manager,
    jstring tflite_model_asset_name, jstring npu_model_asset_name,
    jstring native_lib_dir) {
  ThrowingJniHelper jni_helper(env);

  auto model_with_byte_code =
      GetModelBuf(&jni_helper, AAssetManager_fromJava(env, asset_manager),
                  tflite_model_asset_name, npu_model_asset_name);
  if (!model_with_byte_code) {
    LogToCallback(&jni_helper, litertJni, "  Failed to get model buffer");
    return;
  }
  auto model = Model::CreateFromBuffer(*model_with_byte_code);
  if (!model) {
    LogToCallback(&jni_helper, litertJni, "  Failed to create model");
    return;
  }

  auto jit_compilation_options = litert::Options::Create();
  ASSERT_EQ(&jni_helper, jit_compilation_options.HasValue(), true);
  ASSERT_EQ(
      &jni_helper,
      jit_compilation_options->SetHardwareAccelerators(kLiteRtHwAcceleratorNpu)
          .HasValue(),
      true);

  auto litert_env = litert::Environment::Create({});
  // TODO(niuchl): Pass native lib path to Qualcomm NPU runtime.
  auto res_compiled_model =
      CompiledModel::Create(*litert_env, *model, *jit_compilation_options);
  if (!res_compiled_model) {
    LogToCallback(&jni_helper, litertJni, "  Failed to create compiled model");
    return;
  }
  auto& compiled_model = *res_compiled_model;

  auto signatures = model->GetSignatures();
  // ASSERT_NE(&jni_helper, signatures, nullptr);
  ASSERT_EQ(&jni_helper, signatures->size(), 1);
  auto& signature = signatures->at(0);
  auto signature_key = signature.Key();
  ASSERT_EQ(&jni_helper, signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  auto input_buffers_res = compiled_model.CreateInputBuffers(signature_index);
  // ASSERT_NE(&jni_helper, input_buffers_res, nullptr);
  auto& input_buffers = *input_buffers_res;

  auto output_buffers_res = compiled_model.CreateOutputBuffers(signature_index);
  // ASSERT_NE(&jni_helper, output_buffers_res, nullptr);
  auto& output_buffers = *output_buffers_res;

  // Fill model inputs.
  auto input_names = signature.InputNames();
  ASSERT_EQ(&jni_helper, input_names.size(), 2);
  ASSERT_EQ(&jni_helper, input_names.at(0), "arg0");
  ASSERT_EQ(&jni_helper, input_names.at(1), "arg1");
  input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  LogToCallback(&jni_helper, litertJni, "  Input 0 copied");
  input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  LogToCallback(&jni_helper, litertJni, "  Input 1 copied");

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);
  LogToCallback(&jni_helper, litertJni, "  Inference executed");

  // Check model output.
  auto output_names = signature.OutputNames();
  ASSERT_EQ(&jni_helper, output_names.size(), 1);
  ASSERT_EQ(&jni_helper, output_names.at(0), "tfl.custom");
  float output_buffer_data[kTestOutputSize];
  auto output_span = absl::MakeSpan(output_buffer_data, kTestOutputSize);
  output_buffers[0].Read(output_span);
  LogToCallback(&jni_helper, litertJni, "  Output buffer retrieved");
  for (auto i = 0; i < kTestOutputSize; ++i) {
    LogToCallback(&jni_helper, litertJni, "  Output: actual=%f, expected=%f",
                  output_span[i], kTestOutputTensor[i]);
  }
}
