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

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

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
#include "litert/c/internal/litert_dispatch_delegate.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/internal/litert_dispatch_delegate.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/model/model_buffer.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_cascade_model_test_vectors.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/signature_runner.h"

using litert::testing::MakeRuntimeFromTestFileWithNpuModel;
using testing::ElementsAre;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

using Fence = std::shared_ptr<platforms::darwinn::tachyon::Fence>;

constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";
constexpr absl::string_view kNpuBytecodeFileName =
    kGoogleTensorNpuBytecodeFileName;

TEST(DispatchDelegate, CompiledModel) {
  absl::flat_hash_map<std::string, std::string> custom_code_to_npu_file = {
      {"DISPATCH_OP_1", testing::GetTestFilePath(kNpuBytecodeFileName)},
      {"DISPATCH_OP_2", testing::GetTestFilePath(kNpuBytecodeFileName)},
      {"DISPATCH_OP_3", testing::GetTestFilePath(kNpuBytecodeFileName)},
  };

  // Create Model and check signatures.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model_with_byte_code,
      internal::GetModelBufWithByteCode(
          testing::GetTestFilePath(kModelFileName), custom_code_to_npu_file));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Google Tensor TPU";
#endif

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env,
      litert::Environment::Create(absl::MakeConstSpan(environment_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(env, model_with_byte_code, HwAccelerators::kNpu));

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1", "arg2", "arg3"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom2"));

  // Check CompiledModel buffer requirements. Input and output are supposed to
  // be Ahwb DmaBuf.
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
      TensorBufferRequirements input_buffer_requirements_arg2,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg2"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg2,
      input_buffer_requirements_arg2.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg2, ElementsAre(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements input_buffer_requirements_arg3,
      compiled_model.GetInputBufferRequirements(
          /*input_name=*/"arg3"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBufferType> input_buffer_types_arg3,
      input_buffer_requirements_arg3.SupportedTypes());
  EXPECT_THAT(input_buffer_types_arg3, ElementsAre(TensorBufferType::kAhwb));

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBufferRequirements output_buffer_requirements,
      compiled_model.GetOutputBufferRequirements(
          /*output_name=*/"tfl.custom2"));
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBufferType> output_buffer_types,
                              output_buffer_requirements.SupportedTypes());
  EXPECT_THAT(output_buffer_types, ElementsAre(TensorBufferType::kAhwb));

  // ///////////////////////////////////////////////////////////////////////////
  // First inference.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "First inference";

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  ASSERT_EQ(input_buffers.size(), 4);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());
  ASSERT_EQ(output_buffers.size(), 1);

  // Fill model inputs.
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor_1, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor_1, kTestInput1Size)));
  ASSERT_TRUE(input_buffers[2].Write<float>(
      absl::MakeConstSpan(kTestInput2Tensor_1, kTestInput2Size)));
  ASSERT_TRUE(input_buffers[3].Write<float>(
      absl::MakeConstSpan(kTestInput3Tensor_1, kTestInput3Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kTestOutputTensor_1[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor_1));
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Second inference, reusing the same input tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "Second inference";

  // Fill model inputs.
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor_2, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor_2, kTestInput1Size)));
  ASSERT_TRUE(input_buffers[2].Write<float>(
      absl::MakeConstSpan(kTestInput2Tensor_2, kTestInput2Size)));
  ASSERT_TRUE(input_buffers[3].Write<float>(
      absl::MakeConstSpan(kTestInput3Tensor_2, kTestInput3Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
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
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor_2));
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
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor_3, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor_3, kTestInput1Size)));
  ASSERT_TRUE(input_buffers[2].Write<float>(
      absl::MakeConstSpan(kTestInput2Tensor_3, kTestInput2Size)));
  ASSERT_TRUE(input_buffers[3].Write<float>(
      absl::MakeConstSpan(kTestInput3Tensor_3, kTestInput3Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
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
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor_3));
  }
}

TEST(DispatchDelegate, CompiledModelAsync) {
  absl::flat_hash_map<std::string, std::string> custom_code_to_npu_file = {
      {"DISPATCH_OP_1",
       testing::GetTestFilePath(kGoogleTensorNpuBytecodeFileName)},
      {"DISPATCH_OP_2",
       testing::GetTestFilePath(kGoogleTensorNpuBytecodeFileName)},
      {"DISPATCH_OP_3",
       testing::GetTestFilePath(kGoogleTensorNpuBytecodeFileName)},
  };

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model_with_byte_code,
      internal::GetModelBufWithByteCode(
          testing::GetTestFilePath(kModelFileName), custom_code_to_npu_file));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Google Tensor TPU";
#endif

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env,
      litert::Environment::Create(absl::MakeConstSpan(environment_options)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(env, model_with_byte_code, HwAccelerators::kNpu));

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1", "arg2", "arg3"));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_THAT(output_names, ElementsAre("tfl.custom2"));

  // Create I/O tensor buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());
  ASSERT_EQ(input_buffers.size(), 4);

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());
  ASSERT_EQ(output_buffers.size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_0_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[0],
                                     TensorBuffer::LockMode::kWrite));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_1_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[1],
                                     TensorBuffer::LockMode::kWrite));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_2_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[2],
                                     TensorBuffer::LockMode::kWrite));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_3_cpu_addr_and_lock,
      TensorBufferScopedLock::Create(input_buffers[3],
                                     TensorBuffer::LockMode::kWrite));

  // ///////////////////////////////////////////////////////////////////////////
  // First inference.
  // ///////////////////////////////////////////////////////////////////////////
  ABSL_LOG(INFO) << "First inference";

  // Attach events to input buffers.
  Fence input_fence_0 = platforms::darwinn::fence_util::CreateFence();
  Fence input_fence_1 = platforms::darwinn::fence_util::CreateFence();
  Fence input_fence_2 = platforms::darwinn::fence_util::CreateFence();
  Fence input_fence_3 = platforms::darwinn::fence_util::CreateFence();

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_0,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_0->GetFd(),
                                           /*owns_fd=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_1,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_1->GetFd(),
                                           /*owns_fd=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_2,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_2->GetFd(),
                                           /*owns_fd=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event input_event_3,
      litert::Event::CreateFromSyncFenceFd(env, input_fence_3->GetFd(),
                                           /*owns_fd=*/false));

  input_buffers[0].SetEvent(std::move(input_event_0));
  input_buffers[1].SetEvent(std::move(input_event_1));
  input_buffers[2].SetEvent(std::move(input_event_2));
  input_buffers[3].SetEvent(std::move(input_event_3));

  ABSL_LOG(INFO) << "Start async";

  // Start the model asynchronously.
  bool async;
  compiled_model.RunAsync(input_buffers, output_buffers, async);
  ASSERT_TRUE(async);
  ASSERT_TRUE(output_buffers[0].HasEvent());

  ABSL_LOG(INFO) << "Fill inputs";

  // Fill model inputs.
  std::memcpy(input_0_cpu_addr_and_lock.second, kTestInput0Tensor_1,
              sizeof(kTestInput0Tensor_1));
  std::memcpy(input_1_cpu_addr_and_lock.second, kTestInput1Tensor_1,
              sizeof(kTestInput1Tensor_1));
  std::memcpy(input_2_cpu_addr_and_lock.second, kTestInput2Tensor_1,
              sizeof(kTestInput2Tensor_1));
  std::memcpy(input_3_cpu_addr_and_lock.second, kTestInput3Tensor_1,
              sizeof(kTestInput3Tensor_1));

  ABSL_LOG(INFO) << "Trigger input sync fences";

  // Signal input fences so that the inference can start.
  ASSERT_OK(input_fence_0->Signal(/*success=*/true));
  ASSERT_OK(input_fence_1->Signal(/*success=*/true));
  ASSERT_OK(input_fence_2->Signal(/*success=*/true));
  ASSERT_OK(input_fence_3->Signal(/*success=*/true));

  ABSL_LOG(INFO) << "Check outputs";

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[0], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kTestOutputTensor_1[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-3), kTestOutputTensor_1));
  }
}

}  // namespace
}  // namespace litert
