/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/runners/litert/runner_options.h"

#include <cstdio>
#include <filesystem>  // NOLINT
#include <random>
#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/cc/litert_options.h"
#include "litert/test/matchers.h"

namespace litert::tensor {
namespace {

TEST(RunnerOptionsTest, CreateWorks) {
  auto options_or = RunnerOptions::Create();
  ASSERT_TRUE(options_or.ok());
}

TEST(RunnerOptionsTest, SetAcceleratorWorks) {
  auto options_or = RunnerOptions::Create();
  ASSERT_TRUE(options_or.ok());
  auto& options = *options_or;

  ASSERT_TRUE(options.SetAccelerator(RunnerOptions::Accelerator::kGpu).ok());

  auto hw_or = options.litert_options().GetHardwareAccelerators();
  ASSERT_TRUE(hw_or.HasValue());
  EXPECT_EQ(*hw_or, kLiteRtHwAcceleratorGpu);

  ASSERT_TRUE(options.SetAccelerator(RunnerOptions::Accelerator::kCpu).ok());
  hw_or = options.litert_options().GetHardwareAccelerators();
  ASSERT_TRUE(hw_or.HasValue());
  EXPECT_EQ(*hw_or, kLiteRtHwAcceleratorCpu);
}

TEST(RunnerOptionsTest, SetCacheDirectoryWorks) {
  auto options_or = RunnerOptions::Create();
  ASSERT_TRUE(options_or.ok());
  auto& options = *options_or;

  std::string temp_dir = ::testing::TempDir();
  std::mt19937 gen(12345);
  std::string model_key = "test_model_" + std::to_string(gen());

  ASSERT_TRUE(options.SetCacheDirectory(temp_dir, model_key).ok());

  // Verify options are set in the underlying GPU options.
  auto gpu_options_or = options.litert_options().GetGpuOptions();
  ASSERT_TRUE(gpu_options_or.HasValue());
  LrtGpuOptions* payload = gpu_options_or->Get();

  int program_fd = -1;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsProgramCacheFd(
      &program_fd, payload));
  EXPECT_GE(program_fd, 0);

  int weight_fd = -1;
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsWeightCacheFd(&weight_fd, payload));
  EXPECT_GE(weight_fd, 0);

  bool serialize_program = false;
  LITERT_ASSERT_OK(LrtGetGpuAcceleratorCompilationOptionsSerializeProgramCache(
      &serialize_program, payload));
  EXPECT_TRUE(serialize_program);

  bool serialize_external = false;
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsSerializeExternalTensors(
          &serialize_external, payload));
  EXPECT_TRUE(serialize_external);

  const char* cache_key = nullptr;
  LITERT_ASSERT_OK(
      LrtGetGpuAcceleratorCompilationOptionsModelCacheKey(&cache_key, payload));
  EXPECT_STREQ(cache_key, model_key.c_str());

  // Verify files are created.
  std::filesystem::path program_path =
      std::filesystem::path(temp_dir) / (model_key + ".program.bin");
  std::filesystem::path weight_path =
      std::filesystem::path(temp_dir) / (model_key + ".weights.bin");

  EXPECT_TRUE(std::filesystem::exists(program_path));
  EXPECT_TRUE(std::filesystem::exists(weight_path));

  // Clean up files.
  std::remove(program_path.string().c_str());
  std::remove(weight_path.string().c_str());
}

}  // namespace
}  // namespace litert::tensor
