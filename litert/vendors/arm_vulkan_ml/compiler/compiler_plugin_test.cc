// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0

#include <utility>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/core/model/model.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"

namespace litert {
namespace {

struct JitPlugin {
  Environment environment;
  StaticallyLinkedPlugin plugin;
  internal::LiteRtOptionsPtr options;
};

JitPlugin CreateJitPlugin() {
  auto options = Options::Create();
  EXPECT_TRUE(options);
  auto arm_vulkan_ml_options =
      options->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  EXPECT_TRUE(arm_vulkan_ml_options);
  EXPECT_TRUE(arm_vulkan_ml_options->SetEnableJustInTime(true));

  auto env = Environment::Create({});
  EXPECT_TRUE(env);
  auto c_options =
      internal::LiteRtOptionsPtrBuilder::Build(*options, env->GetHolder());
  EXPECT_TRUE(c_options);
  auto plugin = StaticallyLinkedPlugin::Create(LrtGetCompilerContext(), nullptr,
                                               c_options->get());
  EXPECT_TRUE(plugin);
  return {std::move(*env), std::move(*plugin), std::move(*c_options)};
}

TEST(ArmVulkanMLCompilerPluginPartitionTest, SelectsSupportedOperations) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel(
      "arm_vulkan_ml_conv2d_int8.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  LiteRtOpListT selected_ops;

  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->compiler_plugin_partition(
      jit_plugin.plugin.Get(), nullptr, subgraph.Get(), &selected_ops));

  EXPECT_FALSE(selected_ops.Values().empty());
  for (const auto& [op, partition_index] : selected_ops.Values()) {
    EXPECT_EQ(partition_index, 0);
    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflConv2d);
  }
}

TEST(ArmVulkanMLCompilerPluginCompileTest, OwnsExecutableAfterModelDestruction) {
  auto jit_plugin = CreateJitPlugin();
  LiteRtCompiledResult compiled = nullptr;
  {
    auto model = testing::LoadTestFileModel("arm_vulkan_ml_conv2d_int8.tflite");
    LITERT_ASSERT_OK(jit_plugin.plugin.Api()->compiler_plugin_compile(
        jit_plugin.plugin.Get(), nullptr, model.Get(), &compiled));
  }
  ASSERT_NE(compiled, nullptr);
  LiteRtParamIndex count = 0;
  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->get_num_compiled_result_calls(compiled, &count));
  EXPECT_EQ(count, 1);
  LiteRtJitExecutable handle = nullptr;
  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->get_compiled_result_handle(compiled, 0, &handle));
  ASSERT_NE(handle, nullptr);
  auto artifact = reinterpret_cast<arm_vulkan_ml::ArmVulkanMLJitExecutable*>(handle)->partition;
  jit_plugin.plugin.Api()->destroy_compiled_result(compiled);
  ASSERT_TRUE(artifact);
  ASSERT_FALSE(artifact->spirv.empty());
  EXPECT_EQ(artifact->spirv[0], 0x07230203);
  ASSERT_GE(artifact->constants.size(), 4);
  EXPECT_EQ(artifact->constants[0].bytes.size(), 54);
  EXPECT_EQ(artifact->inputs.size(), 1);
  EXPECT_EQ(artifact->outputs.size(), 1);
}

TEST(ArmVulkanMLCompilerPluginCompileTest, RejectsUnimplementedRatherThanEmptySuccess) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");
  LiteRtCompiledResult compiled = nullptr;
  EXPECT_EQ(jit_plugin.plugin.Api()->compiler_plugin_compile(
      jit_plugin.plugin.Get(), nullptr, model.Get(), &compiled), kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(compiled, nullptr);
  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph);
  LiteRtOpListT selected;
  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->compiler_plugin_partition(
      jit_plugin.plugin.Get(), nullptr, subgraph->Get(), &selected));
  EXPECT_TRUE(selected.Values().empty());
}

TEST(ArmVulkanMLCompilerPluginPartitionTest, RejectsUnsupportedOperation) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  LiteRtOpListT selected_ops;

  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->compiler_plugin_partition(
      jit_plugin.plugin.Get(), nullptr, subgraph.Get(), &selected_ops));

  EXPECT_TRUE(selected_ops.Values().empty());
}

TEST(ArmVulkanMLCompilerPluginPartitionTest, ValidatesArgumentsAndJitMode) {
  auto jit_plugin = CreateJitPlugin();
  LiteRtOpListT selected_ops;
  EXPECT_EQ(jit_plugin.plugin.Api()->compiler_plugin_partition(
                nullptr, nullptr, nullptr, &selected_ops),
            kLiteRtStatusErrorInvalidArgument);

  auto model = testing::LoadTestFileModel(
      "single_add_default_a8w8_recipe_quantized.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  EXPECT_EQ(jit_plugin.plugin.Api()->compiler_plugin_partition(
                nullptr, nullptr, subgraph.Get(), &selected_ops),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert
