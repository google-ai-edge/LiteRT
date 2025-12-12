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
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "webgpu/webgpu.h"  // from @dawn
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

struct TestParams {
  using TupleType = std::tuple<bool, bool, GpuOptions::Precision,
                               GpuOptions::BufferStorageType>;

  bool async;
  bool external_tensors_mode;
  GpuOptions::Precision precision;
  GpuOptions::BufferStorageType buffer_storage_type;

  explicit TestParams(TupleType params)
      : async(std::get<0>(params)),
        external_tensors_mode(std::get<1>(params)),
        precision(std::get<2>(params)),
        buffer_storage_type(std::get<3>(params)) {}
};

Expected<Options> CreateGpuOptions(
    bool external_tensors_mode,
    GpuOptions::Precision precision,
    GpuOptions::BufferStorageType buffer_storage_type) {
  LITERT_ASSIGN_OR_RETURN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(HwAccelerators::kGpu);
  LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
  LITERT_RETURN_IF_ERROR(
      gpu_options.EnableExternalTensorsMode(external_tensors_mode));
  LITERT_RETURN_IF_ERROR(gpu_options.SetPrecision(precision));
  LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(buffer_storage_type));
  return std::move(options);
}

class ParameterizedTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(ParameterizedTest, Basic) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  auto param = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(
      param.external_tensors_mode, param.precision, param.buffer_storage_type));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, testing::GetTestFilePath(kModelFileName),
                            options));

  EXPECT_EQ(compiled_model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model.GetSignatureInputNames());
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  if (param.async) {
    bool async = false;
    auto result = compiled_model.RunAsync(input_buffers, output_buffers, async);
    EXPECT_TRUE(result);
    EXPECT_TRUE(async);
  } else {
    auto result = compiled_model.Run(input_buffers, output_buffers);
    EXPECT_TRUE(result);
  }

  // Check model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model.GetSignatureOutputNames());
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsWebGpuMemory());
  if (param.async) {
    EXPECT_TRUE(output_buffers[0].HasEvent());
    auto event = output_buffers[0].GetEvent();
    EXPECT_TRUE(event);
    auto result = event->IsSignaled();
    EXPECT_TRUE(result);
    EXPECT_TRUE(*result);  // Webgpu event is signaled immediately.
  }
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

INSTANTIATE_TEST_SUITE_P(
    CompiledModelWebGpuTest, ParameterizedTest,
    ::testing::ConvertGenerator<TestParams::TupleType>(::testing::Combine(
        ::testing::ValuesIn<bool>({false, true}),
        ::testing::ValuesIn<bool>({false, true}),
        ::testing::ValuesIn<GpuOptions::Precision>({
            GpuOptions::Precision::kDefault,
            GpuOptions::Precision::kFp16,
            GpuOptions::Precision::kFp32,
        }),
        ::testing::ValuesIn<GpuOptions::BufferStorageType>({
            GpuOptions::BufferStorageType::kDefault,
            GpuOptions::BufferStorageType::kBuffer,
            GpuOptions::BufferStorageType::kTexture2D,
        }))),
    [](const ::testing::TestParamInfo<TestParams>& info) {
      std::string precision;
      switch (info.param.precision) {
        case GpuOptions::Precision::kDefault:
          precision = "Default";
          break;
        case GpuOptions::Precision::kFp16:
          precision = "Fp16";
          break;
        case GpuOptions::Precision::kFp32:
          precision = "Fp32";
          break;
      }
      std::string buffer_storage_type;
      switch (info.param.buffer_storage_type) {
        case GpuOptions::BufferStorageType::kDefault:
          buffer_storage_type = "Default";
          break;
        case GpuOptions::BufferStorageType::kBuffer:
          buffer_storage_type = "Buffer";
          break;
        case GpuOptions::BufferStorageType::kTexture2D:
          buffer_storage_type = "Texture2D";
          break;
      }
      return absl::StrCat(info.param.async ? "Async" : "Sync", "_",
                          info.param.external_tensors_mode ? "External_" : "",
                          precision, "_", buffer_storage_type);
    });

#ifdef ADDRESS_SANITIZER
// Currently, it's working only when --config=asan is given. Without it, it
// complains about leaks of 56 bytes in 1 object at the end of whole test.

struct PipelineTestParams {
  using TupleType = std::tuple<bool, bool, bool>;

  bool async_1st_model;
  bool async_2nd_model;
  bool external_tensors_mode;

  explicit PipelineTestParams(TupleType params)
      : async_1st_model(std::get<0>(params)),
        async_2nd_model(std::get<1>(params)),
        external_tensors_mode(std::get<2>(params)) {}
};

class ParameterizedPipelineTest
    : public ::testing::TestWithParam<PipelineTestParams> {};

TEST_P(ParameterizedPipelineTest, Pipeline) {
  constexpr const float kTestOutputTensorForPipelineTest[] = {21, 42};

  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  auto param = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(
      param.external_tensors_mode, GpuOptions::Precision::kDefault,
      GpuOptions::BufferStorageType::kDefault));

  // Create 1st model.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_1,
      CompiledModel::Create(*env, testing::GetTestFilePath(kModelFileName),
                            options));
  EXPECT_EQ(compiled_model_1.GetNumSignatures(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers_1,
                              compiled_model_1.CreateInputBuffers());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers_1,
                              compiled_model_1.CreateOutputBuffers());

  // Create 2nd model.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_2,
      CompiledModel::Create(*env, testing::GetTestFilePath(kModelFileName),
                            options));
  EXPECT_EQ(compiled_model_2.GetNumSignatures(), 1);

  // One of input buffers of 2nd model is same as output of 1st model.
  // Set rest of the input buffers of 2nd model same as 1st model's input
  // buffers.
  std::vector<TensorBuffer> input_buffers_2(2);
  LITERT_ASSERT_OK_AND_ASSIGN(input_buffers_2[0],
                              output_buffers_1[0].Duplicate());
  LITERT_ASSERT_OK_AND_ASSIGN(input_buffers_2[1],
                              input_buffers_1[1].Duplicate());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers_2,
                              compiled_model_2.CreateOutputBuffers());

  // Fill model inputs for 1st model.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names,
                              compiled_model_1.GetSignatureInputNames());
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers_1[0].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers_1[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers_1[1].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers_1[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute 1st model.
  if (param.async_1st_model) {
    bool async = false;
    auto result =
        compiled_model_1.RunAsync(input_buffers_1, output_buffers_1, async);
    EXPECT_TRUE(result);
    EXPECT_TRUE(async);
  } else {
    auto result = compiled_model_1.Run(input_buffers_1, output_buffers_1);
    EXPECT_TRUE(result);
  }

  // Execute 2nd model.
  if (param.async_2nd_model) {
    bool async = false;
    auto result =
        compiled_model_2.RunAsync(input_buffers_2, output_buffers_2, async);
    EXPECT_TRUE(result);
    EXPECT_TRUE(async);
  } else {
    auto result = compiled_model_2.Run(input_buffers_2, output_buffers_2);
    EXPECT_TRUE(result);
  }

  // Check 2nd model output.
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              compiled_model_2.GetSignatureOutputNames());
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers_2[0].IsWebGpuMemory());
  if (param.async_2nd_model) {
    EXPECT_TRUE(output_buffers_2[0].HasEvent());
    auto event = output_buffers_2[0].GetEvent();
    EXPECT_TRUE(event);
    auto result = event->IsSignaled();
    EXPECT_TRUE(result);
    EXPECT_TRUE(*result);  // Webgpu event is signaled immediately.
  }
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_buffers_2[0], TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t"
                     << kTestOutputTensorForPipelineTest[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5),
                kTestOutputTensorForPipelineTest));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModelWebGpuTest, ParameterizedPipelineTest,
    ::testing::ConvertGenerator<PipelineTestParams::TupleType>(
        ::testing::Combine(::testing::ValuesIn<bool>({false, true}),
                           ::testing::ValuesIn<bool>({false, true}),
                           ::testing::ValuesIn<bool>({false, true})))
    [](const ::testing::TestParamInfo<PipelineTestParams>& info) {
      return absl::StrCat(
          info.param.async_1st_model ? "Async1st" : "Sync1st", "_",
          info.param.async_2nd_model ? "Async2nd" : "Sync2nd",
          info.param.external_tensors_mode ? "_External" : "");
    });
#endif  // MEMORY_SANITIZER

TEST(CompiledModelWebGpuTest, GpuEnvironment) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto env_1 = litert::Environment::Create({});
  ASSERT_TRUE(env_1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_1,
      CreateGpuOptions(false, GpuOptions::Precision::kDefault,
                       GpuOptions::BufferStorageType::kDefault));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_1,
      CompiledModel::Create(*env_1, testing::GetTestFilePath(kModelFileName),
                            options_1));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options_1, env_1->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_device_id_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagWebGpuDevice));
  ABSL_LOG(INFO) << "WebGPU device id: "
                 << reinterpret_cast<WGPUDevice>(
                        std::get<int64_t>(wegpu_device_id_1));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_command_queue_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagWebGpuQueue));
  ABSL_LOG(INFO) << "WebGPU command queue: "
                 << reinterpret_cast<WGPUQueue>(
                        std::get<int64_t>(wegpu_command_queue_1));

  // Check if the 2nd LiteRT environment can get the same WebGPU device and
  // command queue.
  auto env_2 = litert::Environment::Create({});
  ASSERT_TRUE(env_2);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_2,
      CreateGpuOptions(true, GpuOptions::Precision::kFp32,
                       GpuOptions::BufferStorageType::kTexture2D));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_2,
      CompiledModel::Create(*env_2, testing::GetTestFilePath(kModelFileName),
                            options_2));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options_2, env_2->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_device_id_2,
      env_options_2.GetOption(kLiteRtEnvOptionTagWebGpuDevice));
  EXPECT_EQ(std::get<int64_t>(wegpu_device_id_1),
            std::get<int64_t>(wegpu_device_id_2));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_command_queue_2,
      env_options_2.GetOption(kLiteRtEnvOptionTagWebGpuQueue));
  EXPECT_EQ(std::get<int64_t>(wegpu_command_queue_1),
            std::get<int64_t>(wegpu_command_queue_2));
}

TEST(CompiledModelWebGpuTest, ConstructMlDriftWebGpuEnvironment) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto env_1 = litert::Environment::Create({});
  ASSERT_TRUE(env_1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_1,
      CreateGpuOptions(false, GpuOptions::Precision::kDefault,
                       GpuOptions::BufferStorageType::kDefault));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model_1,
      CompiledModel::Create(*env_1, testing::GetTestFilePath(kModelFileName),
                            options_1));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options_1, env_1->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_device_id_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagWebGpuDevice));
  WGPUDevice wgpu_device =
      reinterpret_cast<WGPUDevice>(std::get<int64_t>(wegpu_device_id_1));
  ABSL_LOG(INFO) << "WebGPU device id: " << wgpu_device;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_command_queue_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagWebGpuQueue));
  WGPUQueue wgpu_queue =
      reinterpret_cast<WGPUQueue>(std::get<int64_t>(wegpu_command_queue_1));
  ABSL_LOG(INFO) << "WebGPU command queue: " << wgpu_queue;

  auto webgpu_env = std::make_unique<ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );

  // Initialize MLD WebGPU environment from the resources in LiteRtEnvironment.
  wgpu::Device device = wgpu_device;
  wgpu::AdapterInfo adapter_info;
  device.GetAdapterInfo(&adapter_info);
  ASSERT_OK(webgpu_env->Initialize(device, adapter_info));

  auto wgpu_device_id_2 = reinterpret_cast<int64_t>(webgpu_env->device().Get());
  auto wegpu_command_queue_2 =
      reinterpret_cast<int64_t>(webgpu_env->queue().Get());

  EXPECT_EQ(std::get<int64_t>(wegpu_device_id_1), wgpu_device_id_2);
  EXPECT_EQ(std::get<int64_t>(wegpu_command_queue_1), wegpu_command_queue_2);
}

TEST(CompiledModelWebGpuTest, ShareWebGpuResources) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto webgpu_env = std::make_unique<ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );

  auto status = webgpu_env->Initialize({.enable_host_mapped_pointer = true});
  ASSERT_OK(status);

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::WebGpuDevice,
          reinterpret_cast<int64_t>(webgpu_env->device().Get()),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::WebGpuQueue,
          reinterpret_cast<int64_t>(webgpu_env->queue().Get()),
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateGpuOptions(false, GpuOptions::Precision::kDefault,
                       GpuOptions::BufferStorageType::kDefault));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, testing::GetTestFilePath(kModelFileName),
                            options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, compiled_model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  EXPECT_TRUE(input_buffers[0].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsWebGpuMemory());
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

TEST(CompiledModelWebGpuTest, ShareWebGpuResourcesExternalBuffer) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  auto webgpu_env = std::make_unique<ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );

  auto status = webgpu_env->Initialize({.enable_host_mapped_pointer = true});
  ASSERT_OK(status);

  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::WebGpuDevice,
          reinterpret_cast<int64_t>(webgpu_env->device().Get()),
      },
      litert::Environment::Option{
          litert::Environment::OptionTag::WebGpuQueue,
          reinterpret_cast<int64_t>(webgpu_env->queue().Get()),
      },
  };
  auto env =
      litert::Environment::Create(absl::MakeConstSpan(environment_options));
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options,
      CreateGpuOptions(false, GpuOptions::Precision::kDefault,
                       GpuOptions::BufferStorageType::kDefault));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto compiled_model,
      CompiledModel::Create(*env, testing::GetTestFilePath(kModelFileName),
                            options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signatures, compiled_model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  // Create a tensor buffer from the existing WebGPU buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tensor_type0,
                              compiled_model.GetInputTensorType(
                                  /*signature_index=*/0, /*tensor_index=*/0));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input0_bytes, input_tensor_type0.Bytes());
  EXPECT_EQ(input0_bytes, 8);
  wgpu::BufferDescriptor mappable_buffer_desc0{
      .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst |
               wgpu::BufferUsage::Storage,
      .size = input0_bytes,
  };
  wgpu::Buffer wgpu_buffer0 =
      webgpu_env->device().CreateBuffer(&mappable_buffer_desc0);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto webpu_tensor_buffer0,
      litert::TensorBuffer::CreateFromWebGpuBuffer(
          *env, input_tensor_type0, TensorBufferType::kWebGpuBufferPacked,
          wgpu_buffer0.Get(), input0_bytes));
  EXPECT_TRUE(webpu_tensor_buffer0.IsWebGpuMemory());
  ASSERT_TRUE(webpu_tensor_buffer0.Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tensor_type1,
                              compiled_model.GetInputTensorType(
                                  /*signature_index=*/0, /*tensor_index=*/1));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input1_bytes, input_tensor_type1.Bytes());
  EXPECT_EQ(input1_bytes, 8);
  wgpu::BufferDescriptor mappable_buffer_desc1{
      .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst |
               wgpu::BufferUsage::Storage,
      .size = input1_bytes,
  };
  wgpu::Buffer wgpu_buffer1 =
      webgpu_env->device().CreateBuffer(&mappable_buffer_desc1);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto webpu_tensor_buffer1,
      litert::TensorBuffer::CreateFromWebGpuBuffer(
          *env, input_tensor_type1, TensorBufferType::kWebGpuBufferPacked,
          wgpu_buffer1.Get(), input1_bytes));
  EXPECT_TRUE(webpu_tensor_buffer1.IsWebGpuMemory());
  ASSERT_TRUE(webpu_tensor_buffer1.Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  std::vector<TensorBuffer> input_buffers;
  input_buffers.push_back(std::move(webpu_tensor_buffer0));
  input_buffers.push_back(std::move(webpu_tensor_buffer1));

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Execute model.
  compiled_model.Run(input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsWebGpuMemory());
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

}  // namespace
}  // namespace litert
