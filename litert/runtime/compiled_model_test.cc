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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_runtime_options.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/core/model/model.h"
#include "litert/core/options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/open_cl_memory.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "ml_drift_delegate/delegate/buffer_handler_opencl.h"
#include "tflite/interpreter.h"

namespace litert {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::litert::IsError;

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
    Expected<const LiteRtTensorBufferRequirementsT*> requirements_expected =
        is_input ? compiled_model.GetInputBufferRequirements(signature_key, i)
                 : compiled_model.GetOutputBufferRequirements(signature_key, i);
    LITERT_ASSIGN_OR_RETURN(const LiteRtTensorBufferRequirementsT* requirements,
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
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, litert::kDefaultSignatureKey);

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
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
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
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(input_0_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(input_1_buffer, OwnHandle::kNo);
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
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

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
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, litert::kDefaultSignatureKey);

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
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
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
    TensorBuffer ahwb_buffer =
        TensorBuffer::WrapCObject(input_0_buffer, OwnHandle::kNo);
    ahwb_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer ahwb_buffer =
        TensorBuffer::WrapCObject(input_1_buffer, OwnHandle::kNo);
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
#if defined(_WIN32)
  GTEST_SKIP() << "OpenCL buffer coverage is not linked on Windows.";
#else
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
  if (!ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL could not be loaded; skipping the test";
  }

  ml_drift::cl::Environment cl_env;
  LITERT_ASSERT_OK(ml_drift::cl::CreateEnvironment(&cl_env));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny context_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(cl_env.context().context()))));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny queue_id,
      litert::ToLiteRtAny(litert::LiteRtVariant(
          reinterpret_cast<int64_t>(cl_env.queue()->queue()))));

  const std::array<LiteRtEnvOption, 2> environment_options = {
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClContext,
          /*.value=*/context_id,
      },
      LiteRtEnvOption{
          /*.tag=*/kLiteRtEnvOptionTagOpenClCommandQueue,
          /*.value=*/queue_id,
      },
  };

  LiteRtEnvironment env_ptr;
  LITERT_ASSERT_OK(LiteRtCreateEnvironment(
      environment_options.size(), environment_options.data(), &env_ptr));

  LiteRtRegisterTensorBufferHandlers(
      env_ptr, kLiteRtTensorBufferTypeOpenClBuffer, LiteRtCreateOpenClMemory,
      LiteRtDestroyOpenClMemory, LiteRtLockOpenClMemory,
      LiteRtUnlockOpenClMemory, LiteRtClearOpenClMemory,
      LiteRtImportOpenClMemory, kLiteRtEnvOptionTagOpenClContext,
      kLiteRtEnvOptionTagOpenClCommandQueue);

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, litert::kDefaultSignatureKey);

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
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
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
    TensorBuffer opencl_buffer =
        TensorBuffer::WrapCObject(input_0_buffer, OwnHandle::kNo);
    opencl_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer opencl_buffer =
        TensorBuffer::WrapCObject(input_1_buffer, OwnHandle::kNo);
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
#endif
}

TEST(CompiledModelTest, WithProfiler) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, litert::kDefaultSignatureKey);

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
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetEnableProfiling(/*enabled=*/true);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options.Get(), &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(jit_compilation_options,
                                   opaque_runtime_options.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Create profiler.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtProfiler profiler,
                              compiled_model->GetProfiler());
  ASSERT_EQ(LiteRtStartProfiler(profiler), kLiteRtStatusOk);

  // Check CompiledModel buffer requirements.
  // input and output expect host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg0,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/0));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg0 =
      input_buffer_requirements_arg0->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg0,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* input_buffer_requirements_arg1,
      compiled_model->GetInputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
          /*input_index=*/1));
  const std::vector<LiteRtTensorBufferType>& input_buffer_types_arg1 =
      input_buffer_requirements_arg1->SupportedBufferTypes();
  EXPECT_THAT(input_buffer_types_arg1,
              ElementsAre(kLiteRtTensorBufferTypeHostMemory));

  LITERT_ASSERT_OK_AND_ASSIGN(
      const LiteRtTensorBufferRequirementsT* output_buffer_requirements,
      compiled_model->GetOutputBufferRequirements(
          /*signature_key=*/litert::kDefaultSignatureKey,
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
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(input_0_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  LiteRtTensorBuffer& input_1_buffer = input_buffers[1];
  {
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(input_1_buffer, OwnHandle::kNo);
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

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, WithCpuOptions) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);

  LITERT_ASSIGN_OR_ABORT(auto cpu_options, CpuOptions::Create());
  ASSERT_TRUE(cpu_options.SetNumThreads(2));
  const char* identifier;
  void* payload;
  void (*payload_deleter)(void*);
  ASSERT_EQ(LrtGetOpaqueCpuOptionsData(cpu_options.Get(), &identifier, &payload,
                                       &payload_deleter),
            kLiteRtStatusOk);

  LiteRtOpaqueOptions opaque_cpu_options_handle;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_cpu_options_handle),
            kLiteRtStatusOk);

  ASSERT_EQ(LiteRtAddOpaqueOptions(jit_compilation_options,
                                   opaque_cpu_options_handle),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // We can't easily verify the internal state of the interpreter here without
  // exposing more internals, but successful creation implies the options were
  // parsed without error. In a real integration test, we might check
  // performance or use a mock interpreter if available. For now, we rely on the
  // fact that `compiled_model.cc` parses the options and would log a warning or
  // fail if parsing failed (though currently it warns).

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterBufferMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeBuffer);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options.Get(), &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(jit_compilation_options,
                                   opaque_runtime_options.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Report some errors
  compiled_model->ReportError("Test error 1: %d", 100);
  compiled_model->ReportError("Test error 2: %s", "failed operation");
  compiled_model->ReportError("Test error 3");

  // Get error messages
  auto messages_result = compiled_model->GetErrorMessages();
  ASSERT_TRUE(messages_result);
  std::string messages = *messages_result;

  // Verify all errors are captured
  EXPECT_THAT(messages, ::testing::HasSubstr("Test error 1: 100"));
  EXPECT_THAT(messages, ::testing::HasSubstr("Test error 2: failed operation"));
  EXPECT_THAT(messages, ::testing::HasSubstr("Test error 3"));

  // Clear errors
  auto clear_result = compiled_model->ClearErrors();
  ASSERT_TRUE(clear_result);

  // Verify errors are cleared
  messages_result = compiled_model->GetErrorMessages();
  ASSERT_TRUE(messages_result);
  EXPECT_EQ(*messages_result, "");

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterStderrMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeStderr);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options.Get(), &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(jit_compilation_options,
                                   opaque_runtime_options.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Report errors (they will go to stderr)
  compiled_model->ReportError("Test stderr error: %d", 42);

  // ClearErrors and GetErrorMessages should fail for stderr mode
  auto clear_result = compiled_model->ClearErrors();
  EXPECT_FALSE(clear_result);
  EXPECT_EQ(clear_result.Error().Status(), kLiteRtStatusErrorUnsupported);

  auto messages_result = compiled_model->GetErrorMessages();
  EXPECT_FALSE(messages_result);
  EXPECT_EQ(messages_result.Error().Status(), kLiteRtStatusErrorUnsupported);

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterNoneMode) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeNone);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options.Get(), &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(jit_compilation_options,
                                   opaque_runtime_options.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Error reporter should be null (no getter available)

  // Report errors (should be no-op)
  compiled_model->ReportError("This error should be ignored");

  // ClearErrors and GetErrorMessages should fail for none mode
  auto clear_result = compiled_model->ClearErrors();
  EXPECT_FALSE(clear_result);
  EXPECT_EQ(clear_result.Error().Status(), kLiteRtStatusErrorInvalidArgument);

  auto messages_result = compiled_model->GetErrorMessages();
  EXPECT_FALSE(messages_result);
  EXPECT_EQ(messages_result.Error().Status(),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterWithMultipleModels) {
  // Test that each model has its own error reporter
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  std::string path = testing::GetTestFilePath(kModelFileName);

  // Create first model with buffer reporter
  LiteRtModel model1;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model1),
            kLiteRtStatusOk);

  LiteRtOptions options1;
  ASSERT_EQ(LiteRtCreateOptions(&options1), kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options1, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options1, RuntimeOptions::Create());
  runtime_options1.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeBuffer);
  const char* identifier1;
  void* payload1 = nullptr;
  void (*payload_deleter1)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options1.Get(), &identifier1,
                                           &payload1, &payload_deleter1),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts1 = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier1, payload1, payload_deleter1,
                                      &opaque_opts1),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options1 =
      litert::OpaqueOptions::WrapCObject(opaque_opts1, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(options1, opaque_runtime_options1.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model1,
      LiteRtCompiledModelT::Create(env_ptr, model1, options1));
  LiteRtDestroyOptions(options1);

  // Create second model with stderr reporter
  LiteRtModel model2;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model2),
            kLiteRtStatusOk);

  LiteRtOptions options2;
  ASSERT_EQ(LiteRtCreateOptions(&options2), kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options2, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);
  LITERT_ASSIGN_OR_ABORT(auto runtime_options2, RuntimeOptions::Create());
  runtime_options2.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeStderr);
  const char* identifier2;
  void* payload2 = nullptr;
  void (*payload_deleter2)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options2.Get(), &identifier2,
                                           &payload2, &payload_deleter2),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts2 = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier2, payload2, payload_deleter2,
                                      &opaque_opts2),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options2 =
      litert::OpaqueOptions::WrapCObject(opaque_opts2, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(options2, opaque_runtime_options2.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model2,
      LiteRtCompiledModelT::Create(env_ptr, model2, options2));
  LiteRtDestroyOptions(options2);

  // Report errors to both models
  compiled_model1->ReportError("Model 1 error");
  compiled_model2->ReportError("Model 2 error");

  // Model 1 should have buffer functionality
  auto messages1 = compiled_model1->GetErrorMessages();
  ASSERT_TRUE(messages1);
  EXPECT_THAT(*messages1, ::testing::HasSubstr("Model 1 error"));

  // Model 2 should not support buffer operations
  auto messages2 = compiled_model2->GetErrorMessages();
  EXPECT_FALSE(messages2);

  LiteRtDestroyModel(model1);
  LiteRtDestroyModel(model2);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterDefaultMode) {
  // Test default error reporter mode when not explicitly set
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel without setting error reporter mode
  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, options));
  LiteRtDestroyOptions(options);

  // Default should be stderr mode
  compiled_model->ReportError("Default mode error");

  // Should not support buffer operations
  auto messages = compiled_model->GetErrorMessages();
  EXPECT_FALSE(messages);

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, ErrorReporterWithProfilingEnabled) {
  // Test that error reporter and profiler can coexist, ensure the runtime
  // options are correctly applied.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);

  // Enable both profiling and buffer error reporter
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetEnableProfiling(true);
  runtime_options.SetErrorReporterMode(
      LiteRtErrorReporterMode::kLiteRtErrorReporterModeBuffer);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueRuntimeOptionsData(runtime_options.Get(), &identifier,
                                           &payload, &payload_deleter),
            kLiteRtStatusOk);
  LiteRtOpaqueOptions opaque_opts = nullptr;
  ASSERT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_opts),
            kLiteRtStatusOk);
  litert::OpaqueOptions opaque_runtime_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  ASSERT_EQ(LiteRtAddOpaqueOptions(options, opaque_runtime_options.Release()),
            kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, options));
  LiteRtDestroyOptions(options);

  // Both should work independently
  compiled_model->ReportError("Error with profiling enabled");

  auto messages = compiled_model->GetErrorMessages();
  ASSERT_TRUE(messages);
  EXPECT_THAT(*messages, ::testing::HasSubstr("Error with profiling enabled"));

  // Profiler should also be available
  auto profiler = compiled_model->GetProfiler();
  EXPECT_TRUE(profiler);
  EXPECT_NE(*profiler, nullptr);

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, BindExternalWeightBuffer) {
  // This test verifies that an external buffer can be bound to a weight tensor
  // using the runtime options.

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);
  std::string signature_key = std::string(model->Signatures()[0]->Key());

  // Define the external weight buffer. The values should be added to the input.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) float kWeightTensor[] = {1.0f,
                                                                        2.0f};

  // Create the tensor binding structure.
  LiteRtExternalTensorBinding weight_binding = {
      .signature_name = "",
      .tensor_name = "arg1",
      .data = kWeightTensor,
      .size_bytes = sizeof(kWeightTensor),
  };

  // Create CompiledModel with options that include the external binding.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  // Add the binding to the options.
  reinterpret_cast<LiteRtOptionsT*>(jit_compilation_options)
      ->external_tensor_bindings.push_back(weight_binding);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  auto cc_env = litert::Environment::WrapCObject(env_ptr, OwnHandle::kNo);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      TensorBuffer::CreateManaged(
          cc_env, TensorBufferType::kHostMemory,
          RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
          2 * sizeof(float)));
  // The model has two inputs: "input" and "weight". We only provide the buffer
  // for "input". The "weight" buffer is bound externally.
  std::vector<LiteRtTensorBuffer> input_buffers;
  input_buffers.push_back(std::move(input_buffer.Get()));
  input_buffers.push_back(nullptr);  // The weight buffer is bound externally.

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersFromRequirements(
          env_ptr, *model, litert::kDefaultSignatureKey, *compiled_model));
  ASSERT_EQ(output_buffers.size(), 1);

  // Provide data for the non-weight input tensor.
  std::vector<float> input_data = {5.0f, 6.0f};
  // The first input is the data input.
  input_buffer.Write<float>(absl::MakeConstSpan(input_data));

  // Execute model.
  bool async = false;
  // We only need to pass the buffer for the "input" tensor. The "weight" tensor
  // is already bound.
  LITERT_ASSERT_OK(
      compiled_model->Run(signature_key, input_buffers, output_buffers, async));

  // Check model output.
  {
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output =
        absl::MakeSpan(static_cast<const float*>(host_mem_addr), 2);
    std::vector<float> expected_output = {6.0f, 8.0f};  // input_data + weights
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), expected_output));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Cleanup.
  for (auto& input_buffer : input_buffers) {
    if (input_buffer != nullptr) {
      LiteRtDestroyTensorBuffer(input_buffer);
    }
  }
  for (auto& output_buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(output_buffer);
  }

  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, NonExternalModelKeepsWeightLoaderNull) {
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  LiteRtOptions options;
  ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
  ASSERT_EQ(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu),
      kLiteRtStatusOk);
  auto* options_impl = reinterpret_cast<LiteRtOptionsT*>(options);
  ASSERT_NE(options_impl, nullptr);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, options));
  EXPECT_EQ(options_impl->weight_loader, nullptr);

  LiteRtDestroyOptions(options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, GetInterpreter) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

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
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);

  LITERT_ASSERT_OK_AND_ASSIGN(tflite::Interpreter * interpreter,
                              GetInterpreter(compiled_model.get()));
  EXPECT_NE(interpreter, nullptr);
}

TEST(CompiledModelTest, GetOutputTensorShapes) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));

  std::vector<LiteRtLayout> output_layouts(1);
  auto output_tensor_shapes = absl::MakeSpan(output_layouts);
  LITERT_ASSERT_OK(compiled_model->GetOutputTensorShapes(
      litert::kDefaultSignatureKey, output_tensor_shapes));
  ASSERT_EQ(output_tensor_shapes.size(), 1);
  // The output tensor shape is [[2]]
  EXPECT_EQ(output_tensor_shapes[0].rank, 1);
  EXPECT_EQ(output_tensor_shapes[0].dimensions[0], 2);

  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, CheckResize) {
  constexpr absl::string_view kSimpleAddDynamicShapeModel =
      "simple_add_dynamic_shape.tflite";

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kSimpleAddDynamicShapeModel);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto interp,
                              GetInterpreter(compiled_model.get()));
  auto input_tensor0 = interp->input_tensor(0);
  auto input_tensor1 = interp->input_tensor(1);
  {
    // Check that the input tensors need resize with {2, 128, 4}.
    LITERT_ASSERT_OK_AND_ASSIGN(
        bool input0_need_resize,
        InputTensorNeedsResize(compiled_model.get(), input_tensor0,
                               {2, 128, 4}));
    LITERT_ASSERT_OK_AND_ASSIGN(
        bool input1_need_resize,
        InputTensorNeedsResize(compiled_model.get(), input_tensor1,
                               {2, 128, 4}));
    EXPECT_TRUE(input0_need_resize);
    EXPECT_TRUE(input1_need_resize);
  }
  {
    // Check that the input tensors don't need resize with {1, 128, 4}.
    LITERT_ASSERT_OK_AND_ASSIGN(
        bool input0_need_resize,
        InputTensorNeedsResize(compiled_model.get(), input_tensor0,
                               {1, 128, 4}));
    LITERT_ASSERT_OK_AND_ASSIGN(
        bool input1_need_resize,
        InputTensorNeedsResize(compiled_model.get(), input_tensor1,
                               {1, 128, 4}));
    EXPECT_FALSE(input0_need_resize);
    EXPECT_FALSE(input1_need_resize);
  }

  // Check output tensor shape.
  std::vector<LiteRtLayout> output_layouts(1);
  auto output_tensor_shapes = absl::MakeSpan(output_layouts);
  LITERT_ASSERT_OK(compiled_model->GetOutputTensorShapes(
      litert::kDefaultSignatureKey, output_tensor_shapes));
  ASSERT_EQ(output_tensor_shapes.size(), 1);
  // The output tensor shape is [1, 128, 4]
  EXPECT_EQ(output_tensor_shapes[0].rank, 3);
  EXPECT_EQ(output_tensor_shapes[0].dimensions[0], 1);
  EXPECT_EQ(output_tensor_shapes[0].dimensions[1], 128);
  EXPECT_EQ(output_tensor_shapes[0].dimensions[2], 4);
  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, CheckResizeFail) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto interp,
                              GetInterpreter(compiled_model.get()));
  auto input_tensor0 = interp->input_tensor(0);
  auto input_tensor1 = interp->input_tensor(1);
  // Check that the input tensors need resize with {2, 128, 4}.
  // The input tensor has static dimensions, so the resize should fail.
  EXPECT_THAT(
      InputTensorNeedsResize(compiled_model.get(), input_tensor0, {2, 128, 4}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      InputTensorNeedsResize(compiled_model.get(), input_tensor1, {2, 128, 4}),
      IsError(kLiteRtStatusErrorInvalidArgument));

  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

TEST(CompiledModelTest, DynamicResizeWithCustomAllocationsSimple) {
  constexpr absl::string_view kSimpleAddDynamicShapeModel =
      "simple_add_dynamic_shape.tflite";

  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kSimpleAddDynamicShapeModel);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();
  EXPECT_EQ(signature_key, litert::kDefaultSignatureKey);

  const std::vector<std::string>& input_names = signatures[0]->InputNames();
  EXPECT_THAT(input_names, ElementsAre("arg0", "arg1"));

  LITERT_ASSERT_OK_AND_ASSIGN(const LiteRtSignatureT& signature,
                              model->FindSignature(signature_key));
  const LiteRtSubgraphT& subgraph = signature.GetSubgraph();
  const LiteRtTensorT& tensor0 = *subgraph.Inputs()[0];
  const LiteRtTensorT& tensor1 = *subgraph.Inputs()[1];
  const LiteRtTensorT& tensor_out = *subgraph.Outputs()[0];

  // Create CompiledModel with options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));

  // 1. Resize to Shape A: {1, 128, 4}
  std::vector<int> shape_a = {1, 128, 4};
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto resize = compiled_model->ResizeInputTensor(
        /*signature_index=*/0, /*input_index=*/i, shape_a);
    ASSERT_TRUE(resize.HasValue()) << resize.Error().Message();
  }

  // 2. Create and Run with Shape A
  LiteRtRankedTensorType type_a_0 = tensor0.Type().second.ranked_tensor_type;
  type_a_0.layout.dimensions[0] = 1;
  type_a_0.layout.dimensions[1] = 128;
  type_a_0.layout.dimensions[2] = 4;
  type_a_0.layout.rank = 3;

  LiteRtRankedTensorType type_a_1 = tensor1.Type().second.ranked_tensor_type;
  type_a_1.layout.dimensions[0] = 1;
  type_a_1.layout.dimensions[1] = 128;
  type_a_1.layout.dimensions[2] = 4;
  type_a_1.layout.rank = 3;

  LiteRtRankedTensorType type_out_a =
      tensor_out.Type().second.ranked_tensor_type;
  type_out_a.layout.dimensions[0] = 1;
  type_out_a.layout.dimensions[1] = 128;
  type_out_a.layout.dimensions[2] = 4;
  type_out_a.layout.rank = 3;

  size_t bytes_a = 1 * 128 * 4 * sizeof(float);

  LiteRtTensorBuffer input_a_0, input_a_1, output_a;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_a_0, bytes_a, &input_a_0),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_a_1, bytes_a, &input_a_1),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_out_a, bytes_a, &output_a),
            kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> inputs_a = {input_a_0, input_a_1};
  std::vector<LiteRtTensorBuffer> outputs_a = {output_a};

  bool async = false;
  auto run_status_a =
      compiled_model->Run(signature_key, inputs_a, outputs_a, async);
  ASSERT_TRUE(run_status_a.HasValue()) << run_status_a.Error().Message();

  // 3. Resize to Shape B: {2, 128, 4} (larger)
  std::vector<int> shape_b = {2, 128, 4};
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto resize = compiled_model->ResizeInputTensor(
        /*signature_index=*/0, /*input_index=*/i, shape_b);
    ASSERT_TRUE(resize.HasValue()) << resize.Error().Message();
  }

  // 4. Create and Run with Shape B
  LiteRtRankedTensorType type_b_0 = tensor0.Type().second.ranked_tensor_type;
  type_b_0.layout.dimensions[0] = 2;
  type_b_0.layout.dimensions[1] = 128;
  type_b_0.layout.dimensions[2] = 4;
  type_b_0.layout.rank = 3;

  LiteRtRankedTensorType type_b_1 = tensor1.Type().second.ranked_tensor_type;
  type_b_1.layout.dimensions[0] = 2;
  type_b_1.layout.dimensions[1] = 128;
  type_b_1.layout.dimensions[2] = 4;
  type_b_1.layout.rank = 3;

  LiteRtRankedTensorType type_out_b =
      tensor_out.Type().second.ranked_tensor_type;
  type_out_b.layout.dimensions[0] = 2;
  type_out_b.layout.dimensions[1] = 128;
  type_out_b.layout.dimensions[2] = 4;
  type_out_b.layout.rank = 3;

  size_t bytes_b = 2 * 128 * 4 * sizeof(float);

  LiteRtTensorBuffer input_b_0, input_b_1, output_b;
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_b_0, bytes_b, &input_b_0),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_b_1, bytes_b, &input_b_1),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtCreateManagedTensorBuffer(env_ptr,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            &type_out_b, bytes_b, &output_b),
            kLiteRtStatusOk);

  std::vector<LiteRtTensorBuffer> inputs_b = {input_b_0, input_b_1};
  std::vector<LiteRtTensorBuffer> outputs_b = {output_b};

  auto run_status_b =
      compiled_model->Run(signature_key, inputs_b, outputs_b, async);
  ASSERT_TRUE(run_status_b.HasValue()) << run_status_b.Error().Message();

  // Cleanup
  for (auto& buf : inputs_a) LiteRtDestroyTensorBuffer(buf);
  for (auto& buf : outputs_a) LiteRtDestroyTensorBuffer(buf);
  for (auto& buf : inputs_b) LiteRtDestroyTensorBuffer(buf);
  for (auto& buf : outputs_b) LiteRtDestroyTensorBuffer(buf);

  LiteRtDestroyOptions(jit_compilation_options);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

namespace {

struct CustomMemory : public HwMemoryInfo {
  void* aligned_data = nullptr;
  size_t size = 0;
};

LiteRtStatus CreateCustomMemory(LiteRtGpuDeviceId, LiteRtGpuQueueId,
                                const LiteRtRankedTensorType*,
                                LiteRtTensorBufferType, size_t bytes,
                                size_t packed_bytes,
                                HwMemoryInfoPtr* hw_memory_info) {
  auto* mem = new CustomMemory();
  mem->size = bytes;
  if (posix_memalign(&mem->aligned_data, 64, bytes) != 0) {
    delete mem;
    return kLiteRtStatusErrorRuntimeFailure;
  }
  mem->raw_handle = mem->aligned_data;
  mem->memory_handle = mem;
  *hw_memory_info = mem;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyCustomMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* mem = static_cast<CustomMemory*>(hw_memory_info);
  if (mem != nullptr) {
    if (mem->aligned_data != nullptr) {
      free(mem->aligned_data);
    }
    delete mem;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LockCustomMemory(HwMemoryInfoPtr hw_memory_info,
                              LiteRtTensorBufferLockMode,
                              void** host_memory_ptr) {
  auto* mem = static_cast<CustomMemory*>(hw_memory_info);
  *host_memory_ptr = mem->aligned_data;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockCustomMemory(HwMemoryInfoPtr) { return kLiteRtStatusOk; }

// Custom memory handler that invalidates (clears) host memory upon unlock,
// simulating hardware buffer unmapping (e.g. munmap / AHardwareBuffer_unlock).
struct InvalidatingCustomMemory : public HwMemoryInfo {
  std::vector<uint8_t> device_storage;
  void* host_mapped_ptr = nullptr;
  size_t size = 0;
};

LiteRtStatus CreateInvalidatingCustomMemory(LiteRtGpuDeviceId, LiteRtGpuQueueId,
                                            const LiteRtRankedTensorType*,
                                            LiteRtTensorBufferType,
                                            size_t bytes, size_t packed_bytes,
                                            HwMemoryInfoPtr* hw_memory_info) {
  auto* mem = new InvalidatingCustomMemory();
  mem->size = bytes;
  mem->device_storage.resize(bytes, 0);
  if (posix_memalign(&mem->host_mapped_ptr, 64, bytes) != 0) {
    delete mem;
    return kLiteRtStatusErrorRuntimeFailure;
  }
  mem->raw_handle = mem->host_mapped_ptr;
  mem->memory_handle = mem;
  *hw_memory_info = mem;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyInvalidatingCustomMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* mem = static_cast<InvalidatingCustomMemory*>(hw_memory_info);
  if (mem != nullptr) {
    if (mem->host_mapped_ptr != nullptr) {
      free(mem->host_mapped_ptr);
    }
    delete mem;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LockInvalidatingCustomMemory(HwMemoryInfoPtr hw_memory_info,
                                         LiteRtTensorBufferLockMode,
                                         void** host_memory_ptr) {
  auto* mem = static_cast<InvalidatingCustomMemory*>(hw_memory_info);
  // Restore host memory contents from device storage upon lock.
  std::memcpy(mem->host_mapped_ptr, mem->device_storage.data(), mem->size);
  *host_memory_ptr = mem->host_mapped_ptr;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockInvalidatingCustomMemory(HwMemoryInfoPtr hw_memory_info) {
  auto* mem = static_cast<InvalidatingCustomMemory*>(hw_memory_info);
  // Persist current host memory to device storage.
  std::memcpy(mem->device_storage.data(), mem->host_mapped_ptr, mem->size);
  // Invalidate host memory to simulate driver unmapping / cache invalidation.
  std::memset(mem->host_mapped_ptr, 0, mem->size);
  return kLiteRtStatusOk;
}

}  // namespace

// This test verifies that CPU-shared input tensor buffers remain mapped and
// valid throughout model execution in CompiledModel::Run().
//
// Lifecycle Diagram:
//
//   Expected (Correct) Lifecycle:
//
//     +--------------------------------+
//     |     CompiledModel::Run()       |
//     |     - RegisterBuffer()         |  Locks buffer -> host_mapped_ptr valid
//     |       (SetCustomAllocation)    |  (is_locked_ = true)
//     |                                |
//     |     - Interpreter::Invoke()    |  CPU kernels execute and read from
//     |                                |  host_mapped_ptr [MUST be valid!]
//     |                                |
//     |     - Cleanup / Return         |  Unlocks buffer (unlock_buffers RAII)
//     |                                |  (is_locked_ = false)
//     +--------------------------------+
//                   |
//                   v
//     +--------------------------------+
//     |        Run() Completed         |  Correct inference results produced
//     +--------------------------------+
//
//   Premature Unlock (Buggy) Lifecycle:
//
//     +--------------------------------+
//     |     CompiledModel::Run()       |
//     |     - RegisterBuffer()         |  Locks buffer -> host_mapped_ptr valid
//     |       (SetCustomAllocation)    |
//     |       (UnlockTensorBuffer)     |  Premature unlock! Invalidate memory
//     |                                |  (e.g. munmap/AHardwareBuffer_unlock)
//     |                                |
//     |     - Interpreter::Invoke()    |  CPU kernels read INVALID host memory!
//     |                                |  ---> Corrupted output or SIGSEGV
//     |                                |
//     |     - Cleanup / Return         |
//     +--------------------------------+
//
// When a non-host-memory buffer (e.g. AHardwareBuffer, OpenCL, GPU, DMA-buf,
// or custom hardware buffer) is passed to a CPU subgraph, LiteRT locks the
// buffer to obtain a CPU virtual address for TfLiteCustomAllocation.
// Unlocking hardware buffers can invalidate or unmap the host virtual address.
// This test uses `InvalidatingCustomMemory` to simulate unmapping/clearing host
// memory upon unlock, ensuring input buffers remain valid during `Invoke()`.
TEST(CompiledModelTest, CpuSharedInputBufferRemainsValidDuringRun) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  ASSERT_EQ(
      LiteRtRegisterTensorBufferHandlers(
          env_ptr, kLiteRtTensorBufferTypeUserCustomBuffer,
          CreateInvalidatingCustomMemory, DestroyInvalidatingCustomMemory,
          LockInvalidatingCustomMemory, UnlockInvalidatingCustomMemory,
          /*clear_func=*/nullptr, /*import_func=*/nullptr,
          kLiteRtEnvOptionTagNull, kLiteRtEnvOptionTagNull),
      kLiteRtStatusOk);

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();

  // Create CompiledModel with CPU accelerator options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Register buffer requirements on buffer_context for input tensor to route
  // through cpu_tensors_ registration path.
  LITERT_ASSERT_OK_AND_ASSIGN(auto interp,
                              GetInterpreter(compiled_model.get()));
  auto* input_tensor0 = interp->input_tensor(0);
  ASSERT_NE(input_tensor0, nullptr);

  LiteRtTensorBufferRequirements requirements;
  LiteRtTensorBufferType supported_types[] = {
      kLiteRtTensorBufferTypeHostMemory};
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                /*num_supported_tensor_buffer_types=*/1, supported_types,
                /*buffer_size=*/sizeof(float) * kTestInput0Size,
                /*num_strides=*/0, /*strides=*/nullptr, &requirements),
            kLiteRtStatusOk);
  ASSERT_EQ(compiled_model->GetBufferContext()->RegisterBufferRequirements(
                input_tensor0, LiteRtTensorBufferRequirementsPtr(requirements)),
            kLiteRtStatusOk);

  // Create input buffers with kLiteRtTensorBufferTypeUserCustomBuffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersOfType(env_ptr, *model, signature_key,
                               kLiteRtTensorBufferTypeUserCustomBuffer,
                               sizeof(float) * kTestInput0Size));

  {
    void* host_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(input_buffers[0], &host_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_addr, kTestInput0Tensor, sizeof(float) * kTestInput0Size);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_buffers[0]), kLiteRtStatusOk);
  }
  {
    void* host_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(input_buffers[1], &host_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_addr, kTestInput1Tensor, sizeof(float) * kTestInput1Size);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_buffers[1]), kLiteRtStatusOk);
  }

  // Create output buffers matching requirements.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersFromRequirements(env_ptr, *model, signature_key,
                                          *compiled_model));

  // Run model.
  bool async = false;
  auto run_res =
      compiled_model->Run(signature_key, input_buffers, output_buffers, async);
  ASSERT_TRUE(run_res.HasValue()) << run_res.Error().Message();

  // Verify that model output was computed correctly using the input buffer
  // data.
  {
    void* host_mem_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Cleanup.
  for (auto& buf : input_buffers) {
    LiteRtDestroyTensorBuffer(buf);
  }
  for (auto& buf : output_buffers) {
    LiteRtDestroyTensorBuffer(buf);
  }
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

// This test guards against buffer lock leaks for CPU-shared tensors in
// CompiledModel::Run() (e.g. b/540540866).
//
// Lifecycle Diagram:
//
//   +--------------------------+
//   |  TensorBuffer Allocated  |  (is_locked_ = false)
//   +--------------------------+
//                 |
//                 v
//   +--------------------------+
//   |   CompiledModel::Run()   |
//   |   - RegisterBuffer()     |  Locks buffer for custom allocation
//   |                          |  (is_locked_ = true)
//   |   - Interpreter::Invoke()|  Executes model kernels
//   |   - Cleanup / Return     |  Unlocks buffer
//   +--------------------------+
//                 |
//                 v
//   +--------------------------+
//   |     Run() Completed      |  (is_locked_ = false)
//   +--------------------------+
//                 |
//                 +---> Caller can immediately lock, read, or reuse buffer
//                       without "Tensor buffer is already locked" error.
//
// When a tensor is routed through the `cpu_tensors_` registration path
// (e.g. non-host-memory buffers shared with CPU subgraphs), CompiledModel
// locks the buffer to obtain host memory for TfLiteCustomAllocation.
// This test verifies that CPU-shared buffers are cleanly unlocked when Run()
// returns, preventing lock leaks across inference steps or pipeline stages.
TEST(CompiledModelTest, CpuSharedTensorBufferUnlockedAfterRun) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  ASSERT_EQ(
      LiteRtRegisterTensorBufferHandlers(
          env_ptr, kLiteRtTensorBufferTypeUserCustomBuffer, CreateCustomMemory,
          DestroyCustomMemory, LockCustomMemory, UnlockCustomMemory,
          /*clear_func=*/nullptr, /*import_func=*/nullptr,
          kLiteRtEnvOptionTagNull, kLiteRtEnvOptionTagNull),
      kLiteRtStatusOk);

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();

  // Create CompiledModel with CPU accelerator options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Register buffer requirements on buffer_context for input tensor to route
  // through cpu_tensors_ registration path.
  LITERT_ASSERT_OK_AND_ASSIGN(auto interp,
                              GetInterpreter(compiled_model.get()));
  auto* input_tensor0 = interp->input_tensor(0);
  ASSERT_NE(input_tensor0, nullptr);

  LiteRtTensorBufferRequirements requirements;
  LiteRtTensorBufferType supported_types[] = {
      kLiteRtTensorBufferTypeHostMemory};
  ASSERT_EQ(LiteRtCreateTensorBufferRequirements(
                /*num_supported_tensor_buffer_types=*/1, supported_types,
                /*buffer_size=*/sizeof(float) * kTestInput0Size,
                /*num_strides=*/0, /*strides=*/nullptr, &requirements),
            kLiteRtStatusOk);
  ASSERT_EQ(compiled_model->GetBufferContext()->RegisterBufferRequirements(
                input_tensor0, LiteRtTensorBufferRequirementsPtr(requirements)),
            kLiteRtStatusOk);

  // Create input buffers with kLiteRtTensorBufferTypeUserCustomBuffer.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersOfType(env_ptr, *model, signature_key,
                               kLiteRtTensorBufferTypeUserCustomBuffer,
                               sizeof(float) * kTestInput0Size));

  {
    void* host_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(input_buffers[0], &host_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_addr, kTestInput0Tensor, sizeof(float) * kTestInput0Size);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_buffers[0]), kLiteRtStatusOk);
  }
  {
    void* host_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(input_buffers[1], &host_addr,
                                     kLiteRtTensorBufferLockModeWrite),
              kLiteRtStatusOk);
    std::memcpy(host_addr, kTestInput1Tensor, sizeof(float) * kTestInput1Size);
    ASSERT_EQ(LiteRtUnlockTensorBuffer(input_buffers[1]), kLiteRtStatusOk);
  }

  // Create output buffers matching requirements.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> output_buffers,
      CreateOutputBuffersFromRequirements(env_ptr, *model, signature_key,
                                          *compiled_model));

  // Run model.
  bool async = false;
  auto run_res =
      compiled_model->Run(signature_key, input_buffers, output_buffers, async);
  ASSERT_TRUE(run_res.HasValue()) << run_res.Error().Message();

  // Verify that the input buffer is NOT left locked after Run().
  void* host_mem_addr = nullptr;
  EXPECT_EQ(LiteRtLockTensorBuffer(input_buffers[0], &host_mem_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk)
      << "Input buffer was left locked after CompiledModel::Run()";
  if (host_mem_addr != nullptr) {
    EXPECT_EQ(LiteRtUnlockTensorBuffer(input_buffers[0]), kLiteRtStatusOk);
  }

  // Verify that the output buffer is NOT left locked after Run().
  void* host_out_addr = nullptr;
  EXPECT_EQ(LiteRtLockTensorBuffer(output_buffers[0], &host_out_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk)
      << "Output buffer was left locked after CompiledModel::Run()";
  if (host_out_addr != nullptr) {
    EXPECT_EQ(LiteRtUnlockTensorBuffer(output_buffers[0]), kLiteRtStatusOk);
  }

  // Cleanup.
  for (auto& buf : input_buffers) {
    LiteRtDestroyTensorBuffer(buf);
  }
  for (auto& buf : output_buffers) {
    LiteRtDestroyTensorBuffer(buf);
  }
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

// This test verifies that a tensor buffer can be passed as both an input and an
// output in the same `CompiledModel::Run()` call (e.g. in-place KV cache update
// in LLM execution pipelines).
//
// Lifecycle Diagram:
//
//   +-------------------------------------------------------------+
//   | Shared Buffer (as input_buffers[0] and output_buffers[0])   |
//   +-------------------------------------------------------------+
//                                  |
//                                  v
//   +-------------------------------------------------------------+
//   | CompiledModel::Run()                                        |
//   | - Register Input 0:  Locks shared buffer (first time)       |
//   | - Register Output 0: Reuses locked host address without     |
//   |                      double-locking error                   |
//   | - Interpreter::Invoke(): Executes kernel (in-place)         |
//   | - Cleanup / Return:  Unlocks shared buffer exactly once     |
//   +-------------------------------------------------------------+
//                                  |
//                                  v
//   +-------------------------------------------------------------+
//   | Run() Completed: Buffer is unlocked and usable              |
//   +-------------------------------------------------------------+
TEST(CompiledModelTest, SharedInputAndOutputBuffer) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  LiteRtEnvironmentT* env_ptr = env.release();

  // Create LiteRtModel and check signatures.
  std::string path = testing::GetTestFilePath(kModelFileName);
  LiteRtModel model;
  ASSERT_EQ(LiteRtCreateModelFromFile(env_ptr, path.c_str(), &model),
            kLiteRtStatusOk);

  absl::Span<LiteRtSignature> signatures = model->Signatures();
  ASSERT_EQ(signatures.size(), 1);
  absl::string_view signature_key = signatures[0]->Key();

  // Create CompiledModel with CPU accelerator options.
  LiteRtOptions jit_compilation_options;
  ASSERT_EQ(LiteRtCreateOptions(&jit_compilation_options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetOptionsHardwareAccelerators(jit_compilation_options,
                                                 kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtCompiledModelT::Ptr compiled_model,
      LiteRtCompiledModelT::Create(env_ptr, model, jit_compilation_options));
  LiteRtDestroyOptions(jit_compilation_options);

  // Create input/output buffers matching model dimensions.
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<LiteRtTensorBuffer> input_buffers,
      CreateInputBuffersFromRequirements(env_ptr, *model, signature_key,
                                         *compiled_model));
  ASSERT_EQ(input_buffers.size(), 2);

  // Use input_buffers[0] as the shared buffer for both input 0 and output 0.
  LiteRtTensorBuffer shared_buffer = input_buffers[0];
  std::vector<LiteRtTensorBuffer> output_buffers = {shared_buffer};

  {
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(shared_buffer, OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size));
  }
  {
    TensorBuffer cpu_buffer =
        TensorBuffer::WrapCObject(input_buffers[1], OwnHandle::kNo);
    cpu_buffer.Write<float>(
        absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size));
  }

  // Execute model with the shared buffer passed as both input and output.
  bool async = false;
  auto run_res =
      compiled_model->Run(signature_key, input_buffers, output_buffers, async);
  EXPECT_TRUE(run_res.HasValue()) << run_res.Error().Message();

  if (run_res.HasValue()) {
    // Verify in-place computation output (input0 + input1).
    void* host_mem_addr = nullptr;
    ASSERT_EQ(LiteRtLockTensorBuffer(shared_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    absl::Span<const float> output = absl::MakeSpan(
        static_cast<const float*>(host_mem_addr), kTestOutputSize);
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(shared_buffer), kLiteRtStatusOk);
  }

  // Verify shared_buffer is NOT left locked after Run().
  void* check_addr = nullptr;
  EXPECT_EQ(LiteRtLockTensorBuffer(shared_buffer, &check_addr,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk)
      << "Shared buffer was left locked after CompiledModel::Run()";
  if (check_addr != nullptr) {
    EXPECT_EQ(LiteRtUnlockTensorBuffer(shared_buffer), kLiteRtStatusOk);
  }

  // Cleanup.
  LiteRtDestroyTensorBuffer(input_buffers[0]);
  LiteRtDestroyTensorBuffer(input_buffers[1]);
  LiteRtDestroyModel(model);
  LiteRtDestroyEnvironment(env_ptr);
}

}  // namespace
}  // namespace litert
