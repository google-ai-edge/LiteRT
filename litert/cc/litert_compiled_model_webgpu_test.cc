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
#include <memory>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "webgpu/webgpu.h"  // from @dawn
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

using testing::Eq;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

using TestParams =
    std::tuple<bool, GpuOptions::Precision, GpuOptions::BufferStorageType>;

Expected<Options> CreateGpuOptions(const TestParams& params) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_options, GpuOptions::Create());
  LITERT_RETURN_IF_ERROR(
      gpu_options.EnableExternalTensorsMode(std::get<0>(params)));
  LITERT_RETURN_IF_ERROR(gpu_options.SetPrecision(std::get<1>(params)));
  LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(std::get<2>(params)));

  LITERT_ASSIGN_OR_RETURN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));
  return std::move(options);
}

class ParameterizedTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(ParameterizedTest, Basic) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CreateGpuOptions(GetParam()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));

  EXPECT_EQ(model.GetNumSignatures(), 1);

  LITERT_ASSERT_OK_AND_ASSIGN(auto input_buffers,
                              compiled_model.CreateInputBuffers());

  LITERT_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              compiled_model.CreateOutputBuffers());

  // Fill model inputs.
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_names, model.GetSignatureInputNames());
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_names,
                              model.GetSignatureOutputNames());
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

INSTANTIATE_TEST_SUITE_P(
    CompiledModelWebGpuTest, ParameterizedTest,
    ::testing::Combine(::testing::ValuesIn<bool>({false, true}),
                       ::testing::ValuesIn<GpuOptions::Precision>({
                           GpuOptions::Precision::kDefault,
                           GpuOptions::Precision::kFp16,
                           GpuOptions::Precision::kFp32,
                       }),
                       ::testing::ValuesIn<GpuOptions::BufferStorageType>({
                           GpuOptions::BufferStorageType::kDefault,
                           GpuOptions::BufferStorageType::kBuffer,
                           GpuOptions::BufferStorageType::kTexture2D,
                       })));

TEST(CompiledModelWebGpuTest, GpuEnvironment) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env_1 = litert::Environment::Create({});
  ASSERT_TRUE(env_1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_1,
      CreateGpuOptions({false, GpuOptions::Precision::kDefault,
                        GpuOptions::BufferStorageType::kDefault}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model_1,
                              CompiledModel::Create(*env_1, model, options_1));
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
      CreateGpuOptions({true, GpuOptions::Precision::kFp32,
                        GpuOptions::BufferStorageType::kTexture2D}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model_2,
                              CompiledModel::Create(*env_2, model, options_2));
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

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env_1 = litert::Environment::Create({});
  ASSERT_TRUE(env_1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_1,
      CreateGpuOptions({false, GpuOptions::Precision::kDefault,
                        GpuOptions::BufferStorageType::kDefault}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model_1,
                              CompiledModel::Create(*env_1, model, options_1));
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

}  // namespace
}  // namespace litert
