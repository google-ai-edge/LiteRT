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

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_profiler_event.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_platform_support.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_runtime_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include <CL/cl.h>

#if LITERT_HAS_OPENGL_SUPPORT
#include "tflite/delegates/gpu/cl/cl_device.h"
#include "tflite/delegates/gpu/cl/gl_interop.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#include "tflite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

using testing::ElementsAre;
using testing::Eq;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

Expected<Options> CreateGpuOptions(bool external_tensors_mode) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_options, GpuOptions::Create());

  LITERT_RETURN_IF_ERROR(
      gpu_options.EnableExternalTensorsMode(external_tensors_mode));
  LITERT_ASSIGN_OR_RETURN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));
  return std::move(options);
}

void BasicTest(bool external_tensors_mode) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options,
                              CreateGpuOptions(external_tensors_mode));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

class CompiledModelGpuTest : public ::testing::TestWithParam<bool> {};

TEST_P(CompiledModelGpuTest, Basic) {
  BasicTest(CompiledModelGpuTest::GetParam());
}

TEST_P(CompiledModelGpuTest, Basic2nd) {
  // Run the test twice to verify that the CL environment is shared between
  // instances.
  BasicTest(CompiledModelGpuTest::GetParam());
}

TEST_P(CompiledModelGpuTest, WithProfiler) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateGpuOptions(/*no_immutable_external_tensors_mode=*/true));
  LITERT_ASSIGN_OR_ABORT(auto runtime_options, RuntimeOptions::Create());
  runtime_options.SetEnableProfiling(/*enabled=*/true);
  options.AddOpaqueOptions(std::move(runtime_options));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto profiler, compiled_model.GetProfiler());
  ASSERT_TRUE(profiler.StartProfiling());

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check the profiler.
  ASSERT_TRUE(profiler.StopProfiling());
  LITERT_ASSERT_OK_AND_ASSIGN(auto num_events, profiler.GetNumEvents());
  ASSERT_GT(num_events, 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto events, profiler.GetEvents());
  ASSERT_EQ(events[0].event_source, ProfiledEventSource::LITERT);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST_P(CompiledModelGpuTest, GpuEnvironment) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateGpuOptions(CompiledModelGpuTest::GetParam()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options, env->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opencl_device_id,
      env_options.GetOption(kLiteRtEnvOptionTagOpenClDeviceId));
  ABSL_LOG(INFO) << "OpenCL device id: "
                 << reinterpret_cast<cl_device_id>(
                        std::get<int64_t>(opencl_device_id));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opencl_platform_id,
      env_options.GetOption(kLiteRtEnvOptionTagOpenClPlatformId));
  ABSL_LOG(INFO) << "OpenCL platform id: "
                 << reinterpret_cast<cl_platform_id>(
                        std::get<int64_t>(opencl_platform_id));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opencl_context,
      env_options.GetOption(kLiteRtEnvOptionTagOpenClContext));
  ABSL_LOG(INFO) << "OpenCL context: "
                 << reinterpret_cast<cl_context>(
                        std::get<int64_t>(opencl_context));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto opencl_command_queue,
      env_options.GetOption(kLiteRtEnvOptionTagOpenClCommandQueue));
  ABSL_LOG(INFO) << "OpenCL command queue: "
                 << reinterpret_cast<cl_command_queue>(
                        std::get<int64_t>(opencl_command_queue));
}

TEST_P(CompiledModelGpuTest, Async) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateGpuOptions(CompiledModelGpuTest::GetParam()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_event,
      Event::CreateManaged(env->Get(), LiteRtEventTypeOpenCl));
  // Copy of the event to trigger the signal since the ownership of the
  // input_event is transferred to the input_buffers[0].
  LiteRtEvent litert_input_event = input_event.Get();

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Bind the input event to the input buffers.
  // Note: The task should be done after the input buffers are filled.
  // Otherwise the input_buffers[0].Write<> will be blocked by the associated
  // event.
  input_buffers[0].SetEvent(std::move(input_event));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Execute model asynchronously.
  bool async_execution_mode = true;
  compiled_model.RunAsync(signature_index, input_buffers, output_buffers,
                          async_execution_mode);

  // Signal the input event to resume the async execution.
  LiteRtSignalEvent(litert_input_event);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST_P(CompiledModelGpuTest, PartialDelegation) {
  constexpr const char* kModelPartilaFileName = "simple_cast_and_add_op.tflite";
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelPartilaFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LiteRtHwAcceleratorSet accelerator_flags =
      kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu;
  auto compilation_options = Options::Create();
  compilation_options->SetHardwareAccelerators(accelerator_flags);
  LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options, litert::GpuOptions::Create());
  LITERT_ASSERT_OK(
      gpu_options.EnableExternalTensorsMode(CompiledModelGpuTest::GetParam()));
  compilation_options->AddOpaqueOptions(std::move(gpu_options));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, model, *compilation_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 3);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_EQ(input_names.at(2), "arg2");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));
  int64_t arg2_data[1] = {1};
  ASSERT_TRUE(
      input_buffers[2].Write<int64_t>(absl::MakeConstSpan(arg2_data, 1)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add1");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    float expected_output[2] = {12.0f, 23.0f};
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << expected_output[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), expected_output));
  }
}

TEST_P(CompiledModelGpuTest, BasicAdd3dCstInt32) {
  constexpr const char* kInt32ModelFileName = "simple_add3d_cst_int32.tflite";
  constexpr const int32_t kInt32TestInput0Tensor[] = {1, 2, 3, 4, 5, 6};
  constexpr const int32_t kInt32TestOutputTensor[] = {11, 22, 33, 44, 55, 66};
  constexpr const size_t kInt32TestInput0Size = 6;
  constexpr const size_t kInt32TestOutputSize = 6;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kInt32ModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateGpuOptions(CompiledModelGpuTest::GetParam()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers, compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 1);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_TRUE(input_buffers[0].IsOpenClMemory());
  ASSERT_TRUE(input_buffers[0].Write<int32_t>(
      absl::MakeConstSpan(kInt32TestInput0Tensor, kInt32TestInput0Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsOpenClMemory());
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const int32_t>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kInt32TestOutputSize);
    for (auto i = 0; i < kInt32TestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kInt32TestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(Eq(), kInt32TestOutputTensor));
  }
}

// TODO(b/383176413): Add API to CompiledModel to create buffers of custom
// buffer type.
Expected<std::vector<TensorBuffer>> CreateGlInputBuffers(
    CompiledModel& compiled_model, Signature& signature) {
  LITERT_ASSIGN_OR_RETURN(Environment env, compiled_model.GetEnvironment());
  LiteRtSubgraph subgraph_handle = signature.Subgraph();
  Subgraph subgraph = Subgraph(subgraph_handle);

  std::vector<TensorBuffer> input_buffers;
  input_buffers.reserve(subgraph.Inputs().size());
  for (Tensor& input_tensor : subgraph.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(TensorBufferRequirements input_buffer_requirements,
                            compiled_model.GetInputBufferRequirements(
                                signature.Key(), input_tensor.Name()));
    LITERT_ASSIGN_OR_RETURN(RankedTensorType ranked_tensor_type,
                            input_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            input_buffer_requirements.BufferSize());
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        TensorBuffer::CreateManaged(env.Get(), kLiteRtTensorBufferTypeGlBuffer,
                                    ranked_tensor_type, buffer_size));
    input_buffers.push_back(std::move(input_buffer));
  }
  return input_buffers;
}

// TODO(b/383176413): Add API to CompiledModel to create buffers of custom
// buffer type.
Expected<std::vector<TensorBuffer>> CreateGlOutputBuffers(
    CompiledModel& compiled_model, Signature& signature) {
  LITERT_ASSIGN_OR_RETURN(Environment env, compiled_model.GetEnvironment());
  LiteRtSubgraph subgraph_handle = signature.Subgraph();
  Subgraph subgraph = Subgraph(subgraph_handle);

  std::vector<TensorBuffer> output_buffers;
  output_buffers.reserve(subgraph.Outputs().size());
  for (Tensor& output_tensor : subgraph.Outputs()) {
    LITERT_ASSIGN_OR_RETURN(TensorBufferRequirements input_buffer_requirements,
                            compiled_model.GetOutputBufferRequirements(
                                signature.Key(), output_tensor.Name()));
    LITERT_ASSIGN_OR_RETURN(RankedTensorType ranked_tensor_type,
                            output_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            input_buffer_requirements.BufferSize());
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffer,
        TensorBuffer::CreateManaged(env.Get(), kLiteRtTensorBufferTypeGlBuffer,
                                    ranked_tensor_type, buffer_size));
    output_buffers.push_back(std::move(output_buffer));
  }
  return output_buffers;
}

bool IsGlClInteropSupported() {
  if (!HasOpenGlSupport() || !HasOpenClSupport()) {
    return false;
  }
#if LITERT_HAS_OPENCL_SUPPORT && LITERT_HAS_OPENGL_SUPPORT
  if (!tflite::gpu::cl::LoadOpenCL().ok()) {
    return false;
  }
  tflite::gpu::cl::CLDevice device;
  if (!tflite::gpu::cl::CreateDefaultGPUDevice(&device).ok()) {
    return false;
  }
  return tflite::gpu::cl::IsGlSharingSupported(device);
#else
  return false;
#endif  // LITERT_HAS_OPENCL_SUPPORT && LITERT_HAS_OPENGL_SUPPORT
}

// Runs model synchronously on OpenCL with GL input/output buffers.
TEST_P(CompiledModelGpuTest, SyncWithGlClInterop) {
  if (!IsGlClInteropSupported()) {
    GTEST_SKIP() << "GPU tests are not supported in this configuration";
  }

  // Check input and output path
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options,
                              litert::GpuOptions::Create());
  LITERT_ASSERT_OK(
      gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32));
  LITERT_ASSERT_OK(
      gpu_options.SetBufferStorageType(kLiteRtDelegateBufferStorageTypeBuffer));
  LITERT_ASSERT_OK(
      gpu_options.EnableExternalTensorsMode(CompiledModelGpuTest::GetParam()));

  LITERT_ASSERT_OK_AND_ASSIGN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, CreateGlInputBuffers(compiled_model, signatures[0]));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // After GL buffers are filled, create and set egl sync fence event to each
  // buffer. This ensures proper synchronization in the GPU command queue.
  for (int i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto buffer_type,
                                input_buffers[i].BufferType());
    ASSERT_EQ(buffer_type, kLiteRtTensorBufferTypeGlBuffer);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_event,
        Event::CreateManaged(env->Get(), LiteRtEventTypeEglSyncFence));
    input_buffers[i].SetEvent(std::move(input_event));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      CreateGlOutputBuffers(compiled_model, signatures[0]));

  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

// Runs model asynchronously on OpenCL with GL input/output buffers.
TEST(CompiledModelGpuTest, AsyncWithGlClInterop) {
  if (!IsGlClInteropSupported()) {
    GTEST_SKIP() << "GPU tests are not supported in this configuration";
  }

  // Check input and output path
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(auto gpu_options,
                              litert::GpuOptions::Create());
  LITERT_ASSERT_OK(
      gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32));
  LITERT_ASSERT_OK(
      gpu_options.SetBufferStorageType(kLiteRtDelegateBufferStorageTypeBuffer));

  LITERT_ASSERT_OK_AND_ASSIGN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffers, CreateGlInputBuffers(compiled_model, signatures[0]));

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // After GL buffers are filled, create and set egl sync fence event to each
  // buffer. This ensures proper synchronization in the GPU command queue.
  for (int i = 0; i < input_buffers.size(); ++i) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto buffer_type,
                                input_buffers[i].BufferType());
    ASSERT_EQ(buffer_type, kLiteRtTensorBufferTypeGlBuffer);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto input_event,
        Event::CreateManaged(env->Get(), LiteRtEventTypeEglSyncFence));
    input_buffers[i].SetEvent(std::move(input_event));
  }

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffers,
      CreateGlOutputBuffers(compiled_model, signatures[0]));

  // Execute model asynchronously.
  bool async_execution_mode = true;
  compiled_model.RunAsync(signature_index, input_buffers, output_buffers,
                          async_execution_mode);

  ASSERT_TRUE(output_buffers[0].HasEvent());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_event, output_buffers[0].GetEvent());
  ASSERT_TRUE(output_event.Wait());

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

// Test for constant output tensor support
TEST(CompiledModelTest, ConstantOutputTensor) {
  // Create Model with constant output tensor.
  Model model = testing::LoadTestFileModel(kConstantOutputTensorModelFileName);
  ASSERT_TRUE(model);

  // Environment setup
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create CompiledModel
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, kLiteRtHwAcceleratorCpu));

  // Get signatures
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  ASSERT_EQ(signatures.size(), 1);
  size_t signature_index = 0;

  // Create input and output buffers
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));
  ASSERT_EQ(input_buffers.size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));
  ASSERT_EQ(output_buffers.size(), 2);  // normal_output and constant_output

  // Set input values
  const float input_data[] = {5.0f, 10.0f};
  ASSERT_TRUE(
      input_buffers[0].Write<float>(absl::MakeConstSpan(input_data, 2)));

  // Run the model
  LITERT_ASSERT_OK(
      compiled_model.Run(signature_index, input_buffers, output_buffers));

  // Note: TFLite might reorder outputs - check which is which by size
  // The constant output has 4 elements, the normal output has 2 elements
  int constant_output_idx = -1;
  int normal_output_idx = -1;

  // Determine which output is which based on size
  for (int i = 0; i < 2; i++) {
    LITERT_ASSERT_OK_AND_ASSIGN(auto size, output_buffers[i].Size());
    if (size == 4 * sizeof(float)) {
      constant_output_idx = i;
    } else if (size == 2 * sizeof(float)) {
      normal_output_idx = i;
    }
  }

  ASSERT_NE(constant_output_idx, -1) << "Could not find constant output";
  ASSERT_NE(normal_output_idx, -1) << "Could not find normal output";

  // Check normal output (should be [10.0, 20.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[normal_output_idx], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 2);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(10.0f, 1e-5), FloatNear(20.0f, 1e-5)));
  }

  // Check constant output (should always be [1.0, 2.0, 3.0, 4.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_buffers[constant_output_idx],
                                TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(1.0f, 1e-5), FloatNear(2.0f, 1e-5),
                            FloatNear(3.0f, 1e-5), FloatNear(4.0f, 1e-5)));
    ABSL_LOG(INFO) << "Constant output tensor test passed. Values: ["
                   << output[0] << ", " << output[1] << ", " << output[2]
                   << ", " << output[3] << "]";
  }

  // Run again with different input to verify constant output doesn't change
  const float input_data2[] = {100.0f, 200.0f};
  ASSERT_TRUE(
      input_buffers[0].Write<float>(absl::MakeConstSpan(input_data2, 2)));
  LITERT_ASSERT_OK(
      compiled_model.Run(signature_index, input_buffers, output_buffers));

  // Check normal output changed (should be [200.0, 400.0])
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(
            output_buffers[normal_output_idx], TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 2);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(200.0f, 1e-5), FloatNear(400.0f, 1e-5)))
        << "Normal output should reflect new input values";
  }

  // Check that constant output is still [1.0, 2.0, 3.0, 4.0]
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_buffers[constant_output_idx],
                                TensorBuffer::LockMode::kRead));
    auto output = absl::MakeSpan(lock_and_addr.second, 4);
    EXPECT_THAT(output,
                ElementsAre(FloatNear(1.0f, 1e-5), FloatNear(2.0f, 1e-5),
                            FloatNear(3.0f, 1e-5), FloatNear(4.0f, 1e-5)))
        << "Constant output should not change with different inputs";
  }
}

TEST(CompiledModelTest, ExternalTensorBinding) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  // Create Model.
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  // Create weight tensor buffer.
  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) float kWeightTensor[] = {1.0f,
                                                                        2.0f};
  constexpr int kWeightSize = sizeof(kWeightTensor);

  // Create Compilation options and bind weight tensor.
  LITERT_ASSERT_OK_AND_ASSIGN(Options compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  LITERT_ASSERT_OK(compilation_options.AddExternalTensorBinding(
      /*signature_name=*/"", /*tensor_name=*/"arg1", kWeightTensor,
      kWeightSize));

  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled_model,
      CompiledModel::Create(env, model, compilation_options));

  // Create and fill input and output buffers.
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<TensorBuffer> output_buffers,
                              compiled_model.CreateOutputBuffers());
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffers[0]);

  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  float kInputTensor[] = {1.0f, 1.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBufferRequirements requirements,
                              compiled_model.GetInputBufferRequirements(0));
  LITERT_ASSERT_OK_AND_ASSIGN(auto buffer_type, requirements.SupportedTypes());
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer arg0_buffer,
      TensorBuffer::CreateManaged(
          env.Get(), buffer_type[0],
          RankedTensorType(ElementType::Float32, Layout(Dimensions({2}))),
          sizeof(kInputTensor)));
  LITERT_ASSERT_OK(
      arg0_buffer.Write<float>(absl::MakeConstSpan(kInputTensor, 2)));
  input_map["arg0"] = std::move(arg0_buffer);

  // Execute model with input and output buffers.
  LITERT_ASSERT_OK(compiled_model.Run(input_map, output_map));

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      litert::TensorBufferScopedLock::Create<const float>(
          output_map["tfl.add"], TensorBuffer::LockMode::kRead));
  auto output = absl::MakeSpan(lock_and_addr.second, 2);
  constexpr float kExpectedOutput[] = {2.0f, 3.0f};
  EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kExpectedOutput));
}

INSTANTIATE_TEST_SUITE_P(CompiledModelGpuTest, CompiledModelGpuTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace litert
