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

#include "litert/runtime/compiled_model.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/core/model/model.h"
#include "litert/runtime/open_cl_memory.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace litert {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::litert::IsError;

Expected<LiteRtEnvironment> CreateGpuEnabledEnvironment() {
  LiteRtEnvironment env;
  LITERT_RETURN_IF_ERROR(
      LiteRtCreateEnvironment(/*num_options=*/0, /*options=*/nullptr, &env));

  LITERT_ASSIGN_OR_RETURN(auto gpu_env,
                          litert::internal::GpuEnvironment::Create(env));
  LITERT_RETURN_IF_ERROR(env->SetGpuEnvironment(std::move(gpu_env)));
  return env;
}

// Creates a tensor buffer of the given tensor, buffer type, and size.
Expected<LiteRtTensorBufferT*> CreateBufferOfType(
    LiteRtEnvironment env, const LiteRtTensorT& tensor,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  const LiteRtRankedTensorType ranked_tensor_type =
      tensor.Type().second.ranked_tensor_type;

  LiteRtTensorBufferT* tensor_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      env, buffer_type, &ranked_tensor_type, bytes, &tensor_buffer));

  return tensor_buffer;
}

// Creates input or output tensor buffers of the given model, buffer type and
// size.
Expected<std::vector<LiteRtTensorBufferT*>> CreateInputOutputBuffersOfType(
    LiteRtEnvironment env, LiteRtModelT& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes, bool is_input) {
  LITERT_ASSIGN_OR_RETURN(const LiteRtSignatureT& signature,
                          model.FindSignature(signature_key));
  const LiteRtSubgraphT& subgraph = signature.GetSubgraph();

  const std::vector<LiteRtTensorT*>& tensors =
      is_input ? subgraph.Inputs() : subgraph.Outputs();

  std::vector<LiteRtTensorBufferT*> tensor_buffers;
  tensor_buffers.reserve(tensors.size());

  for (int i = 0; i < tensors.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        LiteRtTensorBufferT * tensor_buffer,
        CreateBufferOfType(env, *tensors[i], buffer_type, bytes));
    tensor_buffers.push_back(tensor_buffer);
  }
  return tensor_buffers;
}

// Creates input buffers of the given model, buffer type, and size.
Expected<std::vector<LiteRtTensorBufferT*>> CreateInputBuffersOfType(
    LiteRtEnvironment env, LiteRtModelT& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  return CreateInputOutputBuffersOfType(env, model, signature_key, buffer_type,
                                        bytes, /*is_input=*/true);
}

// Creates output buffers of the given model, buffer type, and size.
Expected<std::vector<LiteRtTensorBufferT*>> CreateOutputBuffersOfType(
    LiteRtEnvironment env, LiteRtModelT& model, absl::string_view signature_key,
    LiteRtTensorBufferType buffer_type, size_t bytes) {
  return CreateInputOutputBuffersOfType(env, model, signature_key, buffer_type,
                                        bytes, /*is_input=*/false);
}

// Creates a tensor buffer of the given tensor and buffer requirements.
Expected<LiteRtTensorBufferT*> CreateBufferFromRequirements(
    LiteRtEnvironment env, const LiteRtTensorT& tensor,
    const LiteRtTensorBufferRequirementsT& requirements) {
  return CreateBufferOfType(env, tensor,
                            requirements.SupportedBufferTypes().at(0),
                            requirements.BufferSize());
}

// Creates input or output tensor buffers of the given model and requirements.
Expected<std::vector<LiteRtTensorBufferT*>>
CreateInputOutputBuffersFromRequirements(LiteRtEnvironment env,
                                         LiteRtModelT& model,
                                         absl::string_view signature_key,
                                         LiteRtCompiledModelT& compiled_model,
                                         bool is_input) {
  LITERT_ASSIGN_OR_RETURN(const LiteRtSignatureT& signature,
                          model.FindSignature(signature_key));
  const LiteRtSubgraphT& subgraph = signature.GetSubgraph();

  const std::vector<LiteRtTensorT*>& tensors =
      is_input ? subgraph.Inputs() : subgraph.Outputs();

  std::vector<LiteRtTensorBufferT*> tensor_buffers;
  tensor_buffers.reserve(tensors.size());

  for (int i = 0; i < tensors.size(); ++i) {
    Expected<LiteRtTensorBufferRequirementsT*> requirements_expected =
        is_input ? compiled_model.GetInputBufferRequirements(signature_key, i)
                 : compiled_model.GetOutputBufferRequirements(signature_key, i);
    LITERT_ASSIGN_OR_RETURN(LiteRtTensorBufferRequirementsT * requirements,
                            requirements_expected);

    LITERT_ASSIGN_OR_RETURN(
        LiteRtTensorBufferT * tensor_buffer,
        CreateBufferFromRequirements(env, *tensors[i], *requirements));
    tensor_buffers.push_back(tensor_buffer);
  }
  return tensor_buffers;
}

// Creates input buffers of the given model and requirements.
Expected<std::vector<LiteRtTensorBufferT*>> CreateInputBuffersFromRequirements(
    LiteRtEnvironment env, LiteRtModelT& model, absl::string_view signature_key,
    LiteRtCompiledModelT& compiled_model) {
  return CreateInputOutputBuffersFromRequirements(env, model, signature_key,
                                                  compiled_model,
                                                  /*is_input=*/true);
}

// Creates output buffers of the given model and requirements.
Expected<std::vector<LiteRtTensorBufferT*>> CreateOutputBuffersFromRequirements(
    LiteRtEnvironment env, LiteRtModelT& model, absl::string_view signature_key,
    LiteRtCompiledModelT& compiled_model) {
  return CreateInputOutputBuffersFromRequirements(env, model, signature_key,
                                                  compiled_model,
                                                  /*is_input=*/false);
}

TEST(CompiledModelTest, Basic) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output LiteRtTensorBuffers. Buffers are
  // created to match CompiledModel's TensorBufferRequirements.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersFromRequirements(env_ptr, *model, signature_key,
                                         *compiled_model));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersFromRequirements(env_ptr, *model, signature_key,
                                          *compiled_model));

  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  {
    TensorBuffer cpu_buffer(input_0_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer(input_1_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest,
     CompilationFailsWhenUnacceleratedOpsRemainWithoutCpuFallback) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);
  EXPECT_THAT(
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options),
      IsError(kLiteRtStatusErrorCompilation));

  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, UseAhwbBuffer) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices";
#endif
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Check input and output buffer requirements expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers. CompiledModel's
  // TensorBufferRequirements expect host memory,but we create AHWB buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersOfType(env_ptr, *model, signature_key,
                               kLiteRtTensorBufferTypeAhwb,
                               sizeof(float) * kTestInput0Size));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersOfType(env_ptr, *model, signature_key,
                                kLiteRtTensorBufferTypeAhwb,
                                sizeof(float) * kTestOutputSize));

  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  EXPECT_EQ(input_0_buffer->buffer_type(), kLiteRtTensorBufferTypeAhwb);
  {
    TensorBuffer ahwb_buffer(input_0_buffer, OwnHandle::kNo);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer ahwb_buffer(input_1_buffer, OwnHandle::kNo);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, UseOpenCLBuffer) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported In msan";
#endif

  if (!litert::internal::OpenClMemory::IsSupported()) {
    GTEST_SKIP() << "OpenCL memory is not supported on this platform; "
                    "skipping the test";
  }
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_ptr, CreateGpuEnabledEnvironment());

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Check ComiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output buffers. CompiledModel's
  // TensorBufferRequirements expect host memory,but we create OpenCL buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersOfType(env_ptr, *model, signature_key,
                               kLiteRtTensorBufferTypeOpenClBuffer,
                               sizeof(float) * kTestInput0Size));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersOfType(env_ptr, *model, signature_key,
                                kLiteRtTensorBufferTypeOpenClBuffer,
                                sizeof(float) * kTestOutputSize));

  // Fill model inputs.
  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  EXPECT_EQ(input_0_buffer->buffer_type(), kLiteRtTensorBufferTypeOpenClBuffer);
  {
    TensorBuffer opencl_buffer(input_0_buffer, OwnHandle::kNo);
    opencl_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer opencl_buffer(input_1_buffer, OwnHandle::kNo);
    opencl_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));

    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, WithProfiler) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(path.c_str(), &model), kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, LiteRtSignatureT::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  const std::vector<std::string>& output_names = signatures[0]->OutputNames();
  EXPECT_THAT(output_names, ElementsAre("tfl.add"));

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Create profiler.
  LiteRtProfiler profiler = nullptr;
  ASSERT_EQ(LiteRtCreateProfiler(100, &profiler), kLiteRtStatusOk);
  compiled_model->SetProfiler(profiler);
  ASSERT_EQ(LiteRtStartProfiler(profiler), kLiteRtStatusOk);

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferRequirementsT * output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/LiteRtSignatureT::kDefaultSignatureKey,
          /*output_index=*/0));
  const std::vector<LiteRtTensorBufferType>& output_buffer_types =
      output_buffer_requirements->SupportedBufferTypes();
  EXPECT_THAT(output_buffer_types,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  // Create and fill input and output LiteRtTensorBuffers. Buffers are
  // created to match CompiledModel's TensorBufferRequirements.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersFromRequirements(env_ptr, *model, signature_key,
                                         *compiled_model));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersFromRequirements(env_ptr, *model, signature_key,
                                          *compiled_model));

  LiteRtTensorBuffer& input_0_buffer = input_buffers[0];
  {
    TensorBuffer cpu_buffer(input_0_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer(input_1_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model.
  bool async = false;
  compiled_model->Run(signature_key, input_buffers, output_buffers, async);

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  ASSERT_EQ(LiteRtStopProfiler(profiler), kLiteRtStatusOk);
  ASSERT_GT(profiler->GetNumEvents(), 2);
  ABSL_LOG(INFO) << "Profiler events: " << profiler->GetProfiledEventsString();
  // Since Buffers in LiteRtTensorBuffer, we need to destroy them explicitly.
  for (auto& input_buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(input_buffer);
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyProfiler(profiler);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}
}  // namespace
}  // namespace litert
