// Copyright 2026 Google LLC.
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

// Include the executable's internal option helpers without its entry point.
#define LITERT_IMAGENET_EVAL_TEST
#include "litert/tools/imagenet_eval_runner.cc"
#include <gtest/gtest.h>

namespace {
TEST(EvaluationOptionsTest, CpuBaselineDoesNotInheritVendorOptions) {
  Config cfg;
  cfg.baseline_model = "same.tflite";
  cfg.baseline_accelerator = "cpu";
  cfg.accelerator = "npu,gpu";
  cfg.arm_vulkan_ml_enable_jit = 1;
  cfg.compiler_plugin_library_dir = "/plugins";
  cfg.dispatch_library_dir = "/dispatch";
  cfg.openvino_device = "npu";
  cfg.gpu_use_ir_model = "true";
  cfg.gpu_backend = "opencl";
  cfg.require_full_acceleration = 1;
  cfg.num_threads = 4;
  cfg.xnnpack_only = 1;
  EXPECT_TRUE(ValidateBackendOptions(cfg));
  auto baseline = BaselineConfig(cfg);
  EXPECT_EQ(baseline.accelerator, "cpu");
  EXPECT_EQ(baseline.num_threads, 1);
  EXPECT_EQ(baseline.arm_vulkan_ml_enable_jit, 0);
  EXPECT_TRUE(baseline.compiler_plugin_library_dir.empty());
  EXPECT_TRUE(baseline.dispatch_library_dir.empty());
  EXPECT_TRUE(baseline.openvino_device.empty());
  EXPECT_EQ(baseline.gpu_use_ir_model, "default");
  EXPECT_EQ(baseline.require_full_acceleration, 0);
  auto options = CreateOptions(baseline, true);
  ASSERT_TRUE(options);
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  auto mode = cpu->GetKernelMode();
  ASSERT_TRUE(mode);
  EXPECT_EQ(*mode, kLiteRtCpuKernelModeReference);
}

TEST(EvaluationOptionsTest, InheritedGpuBaselinePreservesSettings) {
  Config cfg;
  cfg.baseline_model = "baseline.tflite";
  cfg.accelerator = "gpu";
  cfg.gpu_backend = "opencl";
  cfg.gpu_use_ir_model = "true";
  cfg.gpu_precision = "fp32";
  cfg.require_full_acceleration = 1;
  const auto baseline = BaselineConfig(cfg);
  EXPECT_EQ(baseline.model, cfg.baseline_model);
  EXPECT_EQ(DescribeExecution(cfg, false), DescribeExecution(baseline, false));
  EXPECT_TRUE(CreateOptions(cfg));
  EXPECT_TRUE(CreateOptions(baseline));
}

TEST(EvaluationOptionsTest, ArmJitIsOptInAndOwnedIndependently) {
  Config cfg;
  cfg.accelerator = "npu";
  cfg.arm_vulkan_ml_enable_jit = 1;
  auto candidate = CreateOptions(cfg);
  ASSERT_TRUE(candidate);
  cfg.arm_vulkan_ml_enable_jit = 0;
  auto baseline = CreateOptions(cfg);
  ASSERT_TRUE(baseline);
  auto arm = candidate->GetOptions<litert::arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(arm);
  ASSERT_TRUE(arm->GetEnableJustInTime());
  auto enabled = arm->GetEnableJustInTime();
  EXPECT_TRUE(*enabled);
  auto other = baseline->GetOptions<litert::arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(other);
  ASSERT_TRUE(other->GetEnableJustInTime());
  auto disabled = other->GetEnableJustInTime();
  EXPECT_FALSE(*disabled);
}

TEST(EvaluationOptionsTest, RejectsInvalidBackendCombinations) {
  Config cfg;
  EXPECT_TRUE(ValidateBackendOptions(cfg));
  cfg.arm_vulkan_ml_enable_jit = 1;
  EXPECT_FALSE(ValidateBackendOptions(cfg));
  cfg.accelerator = "npu";
  EXPECT_TRUE(ValidateBackendOptions(cfg));
  cfg.arm_vulkan_ml_enable_jit = 2;
  EXPECT_FALSE(ValidateBackendOptions(cfg));
  cfg.arm_vulkan_ml_enable_jit = 0;
  cfg.baseline_accelerator = "cpu";
  EXPECT_FALSE(ValidateBackendOptions(cfg));
  cfg.baseline_model = "baseline.tflite";
  cfg.baseline_num_threads = 0;
  EXPECT_FALSE(ValidateBackendOptions(cfg));
  cfg.baseline_num_threads = 1;
  cfg.accelerator = "typo";
  EXPECT_FALSE(ValidateBackendOptions(cfg));
}
TEST(EvaluationOptionsTest, RejectsMalformedNumbersWithoutExceptions) {
  for (const std::string& bad : {"--max_samples=3oops", "--num_threads=99999999999999999999",
                                 "--atol=invalid", "--std=0.1,no,0.3"}) {
    std::vector<std::string> args = {"eval", "--model=model.tflite", "--manifest=manifest.tsv",
                                     "--image_root=images", bad};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    Config cfg;
    if (bad.rfind("--std=", 0) == 0) {
      ASSERT_TRUE(ParseArgs(argv.size(), argv.data(), cfg));
      EXPECT_FALSE(ParseTriple(cfg.stddev));
    } else {
      EXPECT_FALSE(ParseArgs(argv.size(), argv.data(), cfg));
    }
  }
}
}  // namespace
