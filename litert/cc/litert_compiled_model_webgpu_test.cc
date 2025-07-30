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

#include <any>
#include <cstdint>
#include <cstring>
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

using testing::Eq;
using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

Expected<Options> CreateGpuOptions(bool no_immutable_external_tensors_mode) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_options, GpuOptions::Create());

  LITERT_RETURN_IF_ERROR(gpu_options.EnableNoExternalTensorsMode(
      no_immutable_external_tensors_mode));
  LITERT_ASSIGN_OR_RETURN(litert::Options options, Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));
  return std::move(options);
}

void BasicTest(bool no_immutable_external_tensors_mode) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateGpuOptions(no_immutable_external_tensors_mode));
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
  EXPECT_TRUE(input_buffers[0].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  EXPECT_TRUE(input_buffers[1].IsWebGpuMemory());
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

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

class CompiledModelWebGpuTest : public ::testing::TestWithParam<bool> {};

TEST_P(CompiledModelWebGpuTest, Basic) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  BasicTest(CompiledModelWebGpuTest::GetParam());
}

TEST_P(CompiledModelWebGpuTest, Basic2nd) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  // Run the test twice to verify that the CL environment is shared between
  // instances.
  BasicTest(CompiledModelWebGpuTest::GetParam());
}

typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;

TEST_P(CompiledModelWebGpuTest, GpuEnvironment) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  // To workaround the memory leak in Nvidia's driver
  absl::LeakCheckDisabler disable_leak_check;

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      Model::CreateFromFile(testing::GetTestFilePath(kModelFileName)));

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto options, CreateGpuOptions(CompiledModelWebGpuTest::GetParam()));
  LITERT_ASSERT_OK_AND_ASSIGN(auto compiled_model,
                              CompiledModel::Create(*env, model, options));
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options, env->GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_device_id,
      env_options.GetOption(kLiteRtEnvOptionTagWebGpuDevice));
  ABSL_LOG(INFO) << "WebGPU device id: "
                 << reinterpret_cast<WGPUDevice>(
                        std::any_cast<int64_t>(wegpu_device_id));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto wegpu_command_queue,
      env_options.GetOption(kLiteRtEnvOptionTagWebGpuQueue));
  ABSL_LOG(INFO) << "WebGPU command queue: "
                 << reinterpret_cast<WGPUQueue>(
                        std::any_cast<int64_t>(wegpu_command_queue));
}

INSTANTIATE_TEST_SUITE_P(CompiledModelWebGpuTest, CompiledModelWebGpuTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace litert
