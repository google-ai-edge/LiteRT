// Copyright 2024 Google LLC.
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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer_types.h"

#if defined(__ANDROID__)
#include "platforms/darwinn/tachyon/core/fence/fence.h"
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/darwinn/driver_shared/fence/fence_test_util.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_compiled_model_next.h"
#include "litert/cc/internal/litert_dispatch_delegate.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/runtime/dispatch/dispatch_opaque_options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/signature_runner.h"

using litert::testing::MakeRuntimeFromTestFile;
using testing::FloatNear;
using testing::Pointwise;
using testing::litert::IsOkAndHolds;
using Fence = std::shared_ptr<platforms::darwinn::tachyon::Fence>;
using testing::ElementsAre;

namespace litert {
namespace {

constexpr absl::string_view kPrecompiledTfliteFile =
    "simple_model_npu_google_tensor_precompiled.tflite";
constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

litert::Expected<Environment> CreateDefaultEnvironment() {
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return litert::Environment::Create(absl::MakeConstSpan(environment_options));
}

litert::Expected<Options> CreateDispatchOptions(const uint8_t* base) {
  LITERT_ASSIGN_OR_RETURN(auto dispatch_options,
                          internal::DispatchDelegateOptions::Create());
  LITERT_RETURN_IF_ERROR(dispatch_options.SetAllocBase(base));
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  LITERT_RETURN_IF_ERROR(options.AddOpaqueOptions(std::move(dispatch_options)));
  return options;
}

LiteRtExternalLiteRtBufferContextT CreateBufferContext(
    const LiteRtEnvironment& env, const tflite::Interpreter& interpreter) {
  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) {
      LITERT_LOG(LITERT_ERROR, "Failed to get tensor identifier: %s",
                 tensor_id.Error().Message().c_str());
      constexpr litert::internal::TfLiteTensorIdentifier kInvalidTensorId{-1,
                                                                          -1};
      return kInvalidTensorId;
    }
    return *tensor_id;
  };

  return LiteRtExternalLiteRtBufferContextT(env, get_tensor_id);
}

TEST(DispatchDelegate, CpuBuffer) {
  // The dispatch delegate must be declared before the TFL interpreter so that
  // it gets destroyed only after the interpreter and the dispatch delegate
  // kernels are destroyed. While this order is guaranteed when using
  // litert::CompiledModel, we must handle it manually when using the TFL
  // interpreter directly.
  DispatchDelegatePtr dispatch_delegate = {nullptr, nullptr};

  LITERT_ASSERT_OK_AND_ASSIGN(testing::TflRuntime::Ptr runtime,
                              MakeRuntimeFromTestFile(kPrecompiledTfliteFile));
  tflite::Interpreter& interpreter = runtime->Interpreter();

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LiteRtExternalLiteRtBufferContextT buffer_context =
      CreateBufferContext(env.GetHolder().handle, interpreter);
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateDispatchOptions(runtime->Flatbuffer().Buf().Data()));

  dispatch_delegate = CreateDispatchDelegatePtr(env.Get(), options.Get());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Get the list of signatures and check it.
  auto signature_defs = interpreter.signature_keys();
  ASSERT_EQ(signature_defs.size(), 1);

  tflite::impl::SignatureRunner* runner =
      interpreter.GetSignatureRunner(/*signature_key=*/nullptr);
  ASSERT_NE(runner, nullptr);

  EXPECT_EQ(runner->AllocateTensors(), kTfLiteOk);

  // Fill model inputs.
  ASSERT_STREQ(runner->input_names()[0], "arg0");
  TfLiteTensor* input_0_tensor = runner->input_tensor("arg0");
  ASSERT_NE(input_0_tensor, nullptr);
  float* input_0 = input_0_tensor->data.f;
  std::memcpy(input_0, kTestInput0Tensor, sizeof(kTestInput0Tensor));

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  TfLiteTensor* input_1_tensor = runner->input_tensor("arg1");
  ASSERT_NE(input_1_tensor, nullptr);
  auto* input_1 = input_1_tensor->data.f;
  std::memcpy(input_1, kTestInput1Tensor, sizeof(kTestInput1Tensor));

  EXPECT_EQ(runner->Invoke(), kTfLiteOk);

  // Check model output.
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  auto output_tensor = runner->output_tensor("tfl.custom");
  ASSERT_NE(output_tensor, nullptr);
  auto output = absl::MakeSpan(output_tensor->data.f, kTestOutputSize);
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
  }
  EXPECT_THAT(output, Pointwise(::testing::FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, HwBuffer) {
  // The dispatch delegate must be declared before the TFL interpreter so that
  // it gets destroyed only after the interpreter and the dispatch delegate
  // kernels are destroyed. While this order is guaranteed when using
  // litert::CompiledModel, we must handle it manually when using the TFL
  // interpreter directly.
  DispatchDelegatePtr dispatch_delegate = {nullptr, nullptr};

  LITERT_ASSERT_OK_AND_ASSIGN(testing::TflRuntime::Ptr runtime,
                              MakeRuntimeFromTestFile(kPrecompiledTfliteFile));
  tflite::Interpreter& interpreter = runtime->Interpreter();
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LiteRtExternalLiteRtBufferContextT buffer_context =
      CreateBufferContext(env.GetHolder().handle, interpreter);
  interpreter.SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  EXPECT_EQ(interpreter.nodes_size(), 1);
  EXPECT_EQ(interpreter.inputs().size(), 2);
  EXPECT_EQ(interpreter.outputs().size(), 1);
  ASSERT_EQ(interpreter.execution_plan().size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateDispatchOptions(runtime->Flatbuffer().Buf().Data()));

  dispatch_delegate = CreateDispatchDelegatePtr(env.Get(), options.Get());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(dispatch_delegate.get()),
            kTfLiteOk);

  // Create and register tensor buffers for all inputs and outputs.
  std::vector<LiteRtTensorBufferPtr> input_buffers;
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto* input_buffer_requirements,
        buffer_context.GetBufferRequirements(interpreter.input_tensor(i)));
    const auto& supported_types =
        input_buffer_requirements->SupportedBufferTypes();
    ASSERT_EQ(supported_types.at(0), kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtTensorBufferPtr input_buffer,
        buffer_context.CreateBufferForTensor(interpreter.input_tensor(i)));
    ASSERT_EQ(input_buffer->buffer_type(), kLiteRtTensorBufferTypeAhwb);
    input_buffer->Duplicate();
    LiteRtTensorBufferPtr duplicate_buffer(input_buffer.get());
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.input_tensor(i), std::move(duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    input_buffers.push_back(std::move(input_buffer));
  }

  std::vector<LiteRtTensorBufferPtr> output_buffers;
  for (int i = 0; i < interpreter.outputs().size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto* output_buffer_requirements,
        buffer_context.GetBufferRequirements(interpreter.output_tensor(i)));
    ASSERT_NE(output_buffer_requirements, nullptr);
    const auto& supported_types =
        output_buffer_requirements->SupportedBufferTypes();
    ASSERT_EQ(supported_types.at(0), kLiteRtTensorBufferTypeAhwb);
    LITERT_ASSERT_OK_AND_ASSIGN(
        LiteRtTensorBufferPtr output_buffer,
        buffer_context.CreateBufferForTensor(interpreter.output_tensor(i)));
    ASSERT_EQ(output_buffer->buffer_type(), kLiteRtTensorBufferTypeAhwb);
    output_buffer->Duplicate();
    LiteRtTensorBufferPtr duplicate_buffer(output_buffer.get());
    auto status = buffer_context.RegisterTensorBuffer(
        interpreter.output_tensor(i), std::move(duplicate_buffer));
    ASSERT_EQ(status, kLiteRtStatusOk);
    output_buffers.push_back(std::move(output_buffer));
  }

  // Get the list of signatures and check it.
  auto signature_defs = interpreter.signature_keys();
  ASSERT_EQ(signature_defs.size(), 1);

  tflite::impl::SignatureRunner* runner =
      interpreter.GetSignatureRunner(/*signature_key=*/nullptr);
  ASSERT_NE(runner, nullptr);

  EXPECT_EQ(runner->AllocateTensors(), kTfLiteOk);

  // Fill model inputs.
  ASSERT_STREQ(runner->input_names()[0], "arg0");
  auto& input_0_buffer = input_buffers[0];
  LITERT_ASSERT_OK_AND_ASSIGN(
      void* host_mem_addr,
      input_0_buffer->Lock(kLiteRtTensorBufferLockModeWrite));
  std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
  LITERT_ASSERT_OK(input_0_buffer->Unlock());

  ASSERT_STREQ(runner->input_names()[1], "arg1");
  auto& input_1_buffer = input_buffers[1];
  LITERT_ASSERT_OK_AND_ASSIGN(
      host_mem_addr, input_1_buffer->Lock(kLiteRtTensorBufferLockModeWrite));
  std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
  LITERT_ASSERT_OK(input_1_buffer->Unlock());

  EXPECT_EQ(runner->Invoke(), kTfLiteOk);

  // Check model output.
  ASSERT_STREQ(runner->output_names()[0], "tfl.custom");
  LITERT_ASSERT_OK_AND_ASSIGN(
      void* output_mem_addr,
      output_buffers[0]->Lock(kLiteRtTensorBufferLockModeRead));
  absl::Span<const float> output = absl::MakeConstSpan(
      reinterpret_cast<const float*>(output_mem_addr), kTestOutputSize);
  for (auto i = 0; i < kTestOutputSize; ++i) {
    ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
  }
  EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
}

TEST(DispatchDelegate, CompiledModel) {
  // Create Model and check signatures.
  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

  // Check CompiledModel buffer requirements. Input and output are supposed to
  // be Ahwb.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg0,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg0,
      input_buffer_requirements_arg0.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg0, ElementsAre(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg1,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg1,
      input_buffer_requirements_arg1.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg1, ElementsAre(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(
          /*output_name=*/"tfl.custom"));
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBufferType> output_buffer_types,
                              output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types, ElementsAre(TensorBufferType::kAhwb));

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute compiled model.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(DispatchDelegate, CompiledModelMultiRun) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));

  // Check signatures.
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

  // ///////////////////////////////////////////////////////////////////////////
  // First inference.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "First inference";

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute the model once and check the outputs.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Second inference, reusing the same input tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "Second inference";

  // Fill in new model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor_2, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor_2, kTestInput1Size)));

  // Execute model a second time by reusing the same I/O buffers, to verify if
  // the buffer registration is working.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kTestOutputTensor_2[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor_2));
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Third inference, using new input and output tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "Third inference";

  LITERT_ASSERT_OK_AND_ASSIGN(input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor_3, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor_3, kTestInput1Size)));

  // Execute model a third time by using new I/O buffers, to verify if buffer
  // registration and attachment is working.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kTestOutputTensor_3[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor_3));
  }
}

TEST(DispatchDelegate, CompiledModelSharedInput) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  std::string model_file_path = testing::GetTestFilePath(
      "shared_input_cpu_npu_google_tensor_precompiled.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kCpu));

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.add", "tfl.custom"));

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  // Check model outputs.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[1], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(DispatchDelegate, CompiledModelWithMetrics) {
  // Create Model and check signatures.
  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);
  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromFile(model_file_path));

  EXPECT_EQ(model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "GoogleTensor eTPU";
#endif

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModelNext::Create(env, model, HwAccelerators::kNpu));

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  bool metrics_supported =
      compiled_model.StartMetricsCollection(/*detail_level=*/100).HasValue();

  // Execute compiled model.
  LITERT_ASSERT_OK(compiled_model.Run(input_buffers, output_buffers));

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }

  // Check collected metrics.
  if (metrics_supported) {
    auto metrics = compiled_model.StopMetricsCollection();
    ASSERT_TRUE(metrics);
    for (int i = 0; i < metrics->metrics.size(); ++i) {
      auto& metric = metrics->metrics[i];
      ABSL_LOG(INFO) << "Metric[" << i << "]: " << metric.name << " = "
                     << metric.value.int_value;
    }
  }
}

TEST(DispatchDelegate, CompiledModelAsync) {
#if !defined(__ANDROID__)
  GTEST_SKIP()
      << "The rest of this test is specific to Android devices with  a "
         "GoogleTensor eTPU";
#endif
  // Create Model and check signatures.
  std::string model_file_path =
      testing::GetTestFilePath(kPrecompiledTfliteFile);

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, CreateDefaultEnvironment());

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model_file_path, HwAccelerators::kNpu));

  // Check signatures.
  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom"));

  // Create and fill input and output tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_0_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[0],
                                     TensorBuffer::LockMode::kWrite));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_1_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[1],
                                     TensorBuffer::LockMode::kWrite));

  // Attach events to input buffers.
  Fence input_fence_0 = platforms::darwinn::fence_util::CreateFence();
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_0,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_0->GetFd(),
                                           /*owns_fd=*/false));
  input_buffers[0].SetEvent(std::move(input_event_0));

  Fence input_fence_1 = platforms::darwinn::fence_util::CreateFence();
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_1,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_1->GetFd(),
                                           /*owns_fd=*/false));
  input_buffers[1].SetEvent(std::move(input_event_1));

  // Start the model asynchronously.
  bool async;
  ASSERT_TRUE(compiled_model.RunAsync(input_buffers, output_buffers, async));
  ASSERT_TRUE(async);
  ASSERT_TRUE(output_buffers[0].HasEvent());

  // Set input values.
  std::memcpy(input_0_cpu_addr_and_lock.second, kTestInput0Tensor,
              sizeof(kTestInput0Tensor));
  std::memcpy(input_1_cpu_addr_and_lock.second, kTestInput1Tensor,
              sizeof(kTestInput1Tensor));

  // Signal input fences so that the inference can start.
  ASSERT_OK(input_fence_0->Signal(/*success=*/true));
  ASSERT_OK(input_fence_1->Signal(/*success=*/true));

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert
