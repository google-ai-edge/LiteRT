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
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

using TestParams =
    std::tuple<LiteRtDelegatePrecision, LiteRtDelegateBufferStorageType>;

Expected<Options> CreateGpuOptions(const TestParams& params) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_options, GpuOptions::Create());
  LITERT_RETURN_IF_ERROR(gpu_options.EnableExternalTensorsMode(true));
  LITERT_RETURN_IF_ERROR(gpu_options.SetDelegatePrecision(std::get<0>(params)));
  LITERT_RETURN_IF_ERROR(gpu_options.SetBufferStorageType(std::get<1>(params)));

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
  EXPECT_TRUE(input_buffers[0].IsVulkanMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsVulkanMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  EXPECT_TRUE(output_buffers[0].IsVulkanMemory());
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
    ::testing::Combine(::testing::ValuesIn<LiteRtDelegatePrecision>({
                           kLiteRtDelegatePrecisionDefault,
                           kLiteRtDelegatePrecisionFp16,
                           kLiteRtDelegatePrecisionFp32,
                       }),
                       ::testing::ValuesIn<LiteRtDelegateBufferStorageType>({
                           kLiteRtDelegateBufferStorageTypeDefault,
                           kLiteRtDelegateBufferStorageTypeBuffer,
                           kLiteRtDelegateBufferStorageTypeTexture2D,
                       })));

TEST(CompiledModelVulkanTest, GpuEnvironment) {
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env_1 = litert::Environment::Create({});
  ASSERT_TRUE(env_1);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_1,
      CreateGpuOptions({kLiteRtDelegatePrecisionDefault,
                        kLiteRtDelegateBufferStorageTypeDefault}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model_1,
                              CompiledModel::Create(*env_1, model, options_1));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options_1, env_1->GetOptions());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vulkan_env_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagVulkanEnvironment));
  ABSL_LOG(INFO) << "Vulkan env: "
                 << reinterpret_cast<void*>(std::get<int64_t>(vulkan_env_1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto command_pool_1,
      env_options_1.GetOption(kLiteRtEnvOptionTagVulkanCommandPool));
  ABSL_LOG(INFO) << "Vulkan command pool: "
                 << reinterpret_cast<void*>(std::get<int64_t>(command_pool_1));

  // Check if the 2nd LiteRT environment can get the same WebGPU device and
  // command queue.
  auto env_2 = litert::Environment::Create({});
  ASSERT_TRUE(env_2);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options_2,
      CreateGpuOptions({kLiteRtDelegatePrecisionFp32,
                        kLiteRtDelegateBufferStorageTypeTexture2D}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model_2,
                              CompiledModel::Create(*env_2, model, options_2));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options_2, env_2->GetOptions());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vulkan_env_2,
      env_options_2.GetOption(kLiteRtEnvOptionTagVulkanEnvironment));
  EXPECT_EQ(std::get<int64_t>(vulkan_env_1), std::get<int64_t>(vulkan_env_2));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto command_pool_2,
      env_options_2.GetOption(kLiteRtEnvOptionTagVulkanCommandPool));
  EXPECT_EQ(std::get<int64_t>(command_pool_1),
            std::get<int64_t>(command_pool_2));
}

}  // namespace
}  // namespace litert
