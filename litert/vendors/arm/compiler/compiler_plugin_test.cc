// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <utility>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_options.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

struct JitPlugin {
  Environment environment;
  PluginPtr plugin;
  internal::LiteRtOptionsPtr options;
};

JitPlugin CreateJitPlugin() {
  auto options = Options::Create();
  EXPECT_TRUE(options);
  auto arm_options = options->GetOptions<arm::ArmOptions>();
  EXPECT_TRUE(arm_options);
  EXPECT_TRUE(arm_options->SetEnableJustInTime(true));

  auto env = Environment::Create({});
  EXPECT_TRUE(env);
  auto c_options =
      internal::LiteRtOptionsPtrBuilder::Build(*options, env->GetHolder());
  EXPECT_TRUE(c_options);
  return {std::move(*env),
          CreatePlugin(LrtGetCompilerContext(), nullptr, c_options->get()),
          std::move(*c_options)};
}

TEST(ArmCompilerPluginPartitionTest, SelectsSupportedOperations) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel(
      "single_add_default_a8w8_recipe_quantized.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  LiteRtOpListT selected_ops;

  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      jit_plugin.plugin.get(), nullptr, subgraph.Get(), &selected_ops));

  EXPECT_FALSE(selected_ops.Values().empty());
  for (const auto& [op, partition_index] : selected_ops.Values()) {
    EXPECT_EQ(partition_index, 0);
    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflAdd);
  }
}

TEST(ArmCompilerPluginPartitionTest, RejectsUnsupportedOperation) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  LiteRtOpListT selected_ops;

  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      jit_plugin.plugin.get(), nullptr, subgraph.Get(), &selected_ops));

  EXPECT_TRUE(selected_ops.Values().empty());
}

TEST(ArmCompilerPluginPartitionTest, ValidatesArgumentsAndJitMode) {
  LiteRtOpListT selected_ops;
  EXPECT_EQ(
      LiteRtCompilerPluginPartition(nullptr, nullptr, nullptr, &selected_ops),
      kLiteRtStatusErrorInvalidArgument);

  auto model = testing::LoadTestFileModel(
      "single_add_default_a8w8_recipe_quantized.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  EXPECT_EQ(LiteRtCompilerPluginPartition(nullptr, nullptr, subgraph.Get(),
                                          &selected_ops),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert
