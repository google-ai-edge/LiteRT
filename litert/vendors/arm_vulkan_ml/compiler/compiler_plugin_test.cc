// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
// <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <string>
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
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

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
  auto model = testing::LoadTestFileModel("arm_vulkan_ml_conv2d_int8.tflite");
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

TEST(ArmVulkanMLCompilerPluginCompileTest,
     OwnsExecutableAfterModelDestruction) {
  auto jit_plugin = CreateJitPlugin();
  LiteRtCompiledResult compiled = nullptr;
  {
    auto model = testing::LoadTestFileModel("arm_vulkan_ml_conv2d_int8.tflite");
    LITERT_ASSERT_OK(jit_plugin.plugin.Api()->compiler_plugin_compile(
        jit_plugin.plugin.Get(), nullptr, model.Get(), &compiled));
  }
  ASSERT_NE(compiled, nullptr);
  LiteRtParamIndex count = 0;
  LITERT_ASSERT_OK(
      jit_plugin.plugin.Api()->get_num_compiled_result_calls(compiled, &count));
  EXPECT_EQ(count, 1);
  LiteRtJitExecutable handle = nullptr;
  LITERT_ASSERT_OK(jit_plugin.plugin.Api()->get_compiled_result_handle(
      compiled, 0, &handle));
  ASSERT_NE(handle, nullptr);
  auto artifact =
      reinterpret_cast<arm_vulkan_ml::ArmVulkanMLJitExecutable*>(handle)
          ->partition;
  jit_plugin.plugin.Api()->destroy_compiled_result(compiled);
  ASSERT_TRUE(artifact);
  ASSERT_FALSE(artifact->spirv.empty());
  EXPECT_EQ(artifact->spirv[0], 0x07230203);
  ASSERT_GE(artifact->constants.size(), 4);
  EXPECT_EQ(artifact->constants[0].bytes.size(), 54);
  EXPECT_EQ(artifact->inputs.size(), 1);
  EXPECT_EQ(artifact->outputs.size(), 1);
}

TEST(ArmVulkanMLCompilerPluginCompileTest,
     RejectsUnimplementedRatherThanEmptySuccess) {
  auto jit_plugin = CreateJitPlugin();
  auto model = testing::LoadTestFileModel("simple_topk_op.tflite");
  LiteRtCompiledResult compiled = nullptr;
  EXPECT_EQ(jit_plugin.plugin.Api()->compiler_plugin_compile(
                jit_plugin.plugin.Get(), nullptr, model.Get(), &compiled),
            kLiteRtStatusErrorUnsupported);
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

TEST(ArmVulkanMLCompilerPluginApiTest,
     MetadataCompatibilityByteCodeCallInfoAndDumpTimings) {
  EXPECT_EQ(LiteRtCompilerPluginQueryInterface(
                kLiteRtCompilerPluginInterfaceBasic, {1, 0, 0}, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtInterface raw_iface = nullptr;
  EXPECT_EQ(LiteRtCompilerPluginQueryInterface(
                kLiteRtCompilerPluginInterfaceBasic, {0, 1, 0}, &raw_iface),
            kLiteRtStatusErrorUnsupported);
  LITERT_ASSERT_OK(LiteRtCompilerPluginQueryInterface(
      kLiteRtCompilerPluginInterfaceBasic, {1, 0, 0}, &raw_iface));
  ASSERT_NE(raw_iface, nullptr);

  auto jit_plugin = CreateJitPlugin();
  const auto* api = jit_plugin.plugin.Api();
  LiteRtCompilerPlugin handle = jit_plugin.plugin.Get();

  EXPECT_EQ(api->get_compiler_plugin_version(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtApiVersion version{};
  LITERT_ASSERT_OK(api->get_compiler_plugin_version(&version));
  EXPECT_EQ(version.major, LITERT_API_VERSION_MAJOR);
  EXPECT_EQ(version.minor, LITERT_API_VERSION_MINOR);
  EXPECT_EQ(version.patch, LITERT_API_VERSION_PATCH);

  EXPECT_STREQ(api->get_compiler_plugin_soc_manufacturer(), "ArmVulkanML");

  const char* sdk_version = nullptr;
  EXPECT_EQ(api->get_compiler_plugin_sdk_version(nullptr, &sdk_version),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiler_plugin_sdk_version(handle, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(api->get_compiler_plugin_sdk_version(handle, &sdk_version));
  EXPECT_STREQ(sdk_version, "TFSC-2.0.0/TOSA.001000.1-direct-v1");

  EXPECT_EQ(api->create_compiler_plugin(LrtGetCompilerContext(), nullptr,
                                        nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtHwAccelerators hw = kLiteRtHwAcceleratorNone;
  EXPECT_EQ(api->get_compiler_plugin_supported_hardware(nullptr, &hw),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiler_plugin_supported_hardware(handle, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(api->get_compiler_plugin_supported_hardware(handle, &hw));
  EXPECT_EQ(hw, kLiteRtHwAcceleratorNpu);

  LiteRtParamIndex num_models = 0;
  EXPECT_EQ(api->get_num_compiler_plugin_supported_models(nullptr, &num_models),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_num_compiler_plugin_supported_models(handle, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(
      api->get_num_compiler_plugin_supported_models(handle, &num_models));
  EXPECT_EQ(num_models, 1);

  const char* soc_model = nullptr;
  EXPECT_EQ(
      api->get_compiler_plugin_supported_soc_model(nullptr, 0, &soc_model),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiler_plugin_supported_soc_model(handle, 0, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiler_plugin_supported_soc_model(handle, 1, &soc_model),
            kLiteRtStatusErrorIndexOOB);
  LITERT_ASSERT_OK(
      api->get_compiler_plugin_supported_soc_model(handle, 0, &soc_model));
  EXPECT_STREQ(soc_model, "Generic");

  LiteRtTransformation* transformations = nullptr;
  LiteRtParamIndex num_patterns = 99;
  EXPECT_EQ(api->register_all_transformations(nullptr, &transformations,
                                              &num_patterns),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->register_all_transformations(handle, nullptr, &num_patterns),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      api->register_all_transformations(handle, &transformations, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(api->register_all_transformations(handle, &transformations,
                                                     &num_patterns));
  EXPECT_EQ(transformations, nullptr);
  EXPECT_EQ(num_patterns, 0);

  LITERT_EXPECT_OK(api->check_compiler_compatibility(
      {LITERT_API_VERSION_MAJOR, 0, 0}, handle, nullptr, nullptr, "Generic"));
  EXPECT_EQ(
      api->check_compiler_compatibility({LITERT_API_VERSION_MAJOR + 99, 0, 0},
                                        handle, nullptr, nullptr, "Generic"),
      kLiteRtStatusErrorUnsupportedCompilerVersion);

  // Plugin created with null options does not have JIT enabled and rejects
  // partition and compile calls cleanly.
  LiteRtCompilerPlugin non_jit = nullptr;
  LITERT_ASSERT_OK(api->create_compiler_plugin(LrtGetCompilerContext(),
                                               &non_jit, nullptr, nullptr));
  ASSERT_NE(non_jit, nullptr);
  auto model = testing::LoadTestFileModel("arm_vulkan_ml_conv2d_int8.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  LiteRtOpListT selected_ops;
  EXPECT_EQ(api->compiler_plugin_partition(non_jit, nullptr, subgraph.Get(),
                                           &selected_ops),
            kLiteRtStatusErrorUnsupported);
  LiteRtCompiledResult non_jit_result = nullptr;
  EXPECT_EQ(
      api->compiler_plugin_compile(non_jit, nullptr, nullptr, &non_jit_result),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->compiler_plugin_compile(non_jit, nullptr, model.Get(),
                                         &non_jit_result),
            kLiteRtStatusErrorUnsupported);
  api->destroy_compiler_plugin(non_jit);

  // Verify compiled result bytecode, call info, and bounds checking.
  auto timed_opts = Options::Create();
  ASSERT_TRUE(timed_opts);
  auto arm_opts = timed_opts->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(arm_opts);
  LITERT_ASSERT_OK(arm_opts->SetEnableJustInTime(true));
  auto env = Environment::Create({});
  ASSERT_TRUE(env);
  auto c_timed_opts =
      internal::LiteRtOptionsPtrBuilder::Build(*timed_opts, env->GetHolder());
  ASSERT_TRUE(c_timed_opts);
  auto timed_plugin = StaticallyLinkedPlugin::Create(
      LrtGetCompilerContext(), nullptr, c_timed_opts->get());
  ASSERT_TRUE(timed_plugin);

  LiteRtOpListT timed_ops;
  LITERT_ASSERT_OK(timed_plugin->Api()->compiler_plugin_partition(
      timed_plugin->Get(), nullptr, subgraph.Get(), &timed_ops));
  EXPECT_FALSE(timed_ops.Values().empty());

  LiteRtCompiledResult compiled = nullptr;
  LITERT_ASSERT_OK(timed_plugin->Api()->compiler_plugin_compile(
      timed_plugin->Get(), nullptr, model.Get(), &compiled));
  ASSERT_NE(compiled, nullptr);

  LiteRtParamIndex num_byte_code = 0;
  EXPECT_EQ(api->get_compiled_result_num_byte_code(nullptr, &num_byte_code),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_num_byte_code(compiled, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSERT_OK(
      api->get_compiled_result_num_byte_code(compiled, &num_byte_code));
  EXPECT_EQ(num_byte_code, 1);

  const void* byte_code = nullptr;
  size_t byte_code_size = 99;
  EXPECT_EQ(api->get_compiled_result_byte_code(nullptr, 0, &byte_code,
                                               &byte_code_size),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      api->get_compiled_result_byte_code(compiled, 0, nullptr, &byte_code_size),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      api->get_compiled_result_byte_code(compiled, 0, &byte_code, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_byte_code(compiled, 1, &byte_code,
                                               &byte_code_size),
            kLiteRtStatusErrorIndexOOB);
  LITERT_ASSERT_OK(api->get_compiled_result_byte_code(compiled, 0, &byte_code,
                                                      &byte_code_size));
  EXPECT_EQ(byte_code_size, 0u);

  const void* call_info = nullptr;
  size_t call_info_size = 0;
  LiteRtParamIndex bc_idx = 99;
  EXPECT_EQ(api->get_compiled_result_call_info(nullptr, 0, &call_info,
                                               &call_info_size, &bc_idx),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_call_info(compiled, 0, nullptr,
                                               &call_info_size, &bc_idx),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_call_info(compiled, 0, &call_info, nullptr,
                                               &bc_idx),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_call_info(compiled, 0, &call_info,
                                               &call_info_size, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_call_info(compiled, 1, &call_info,
                                               &call_info_size, &bc_idx),
            kLiteRtStatusErrorIndexOOB);
  LITERT_ASSERT_OK(api->get_compiled_result_call_info(
      compiled, 0, &call_info, &call_info_size, &bc_idx));
  EXPECT_EQ(std::string(static_cast<const char*>(call_info), call_info_size),
            "arm_vulkan_ml_0");
  EXPECT_EQ(bc_idx, 0);

  LiteRtJitExecutable exec_handle = nullptr;
  EXPECT_EQ(api->get_compiled_result_handle(nullptr, 0, &exec_handle),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_handle(compiled, 0, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_compiled_result_handle(compiled, 1, &exec_handle),
            kLiteRtStatusErrorIndexOOB);
  EXPECT_EQ(api->get_num_compiled_result_calls(nullptr, &num_byte_code),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(api->get_num_compiled_result_calls(compiled, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  api->destroy_compiled_result(compiled);
}

}  // namespace
}  // namespace litert
