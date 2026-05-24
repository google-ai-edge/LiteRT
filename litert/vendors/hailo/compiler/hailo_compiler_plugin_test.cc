#include <cstddef>
#include <cstdint>
#include <string>
#include <fstream>
#include <cstdlib>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

TEST(TestHailoPlugin, PluginMetadata) {
  auto plugin = CreatePlugin(LrtGetCompilerContext());
  ASSERT_TRUE(plugin);

  EXPECT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Hailo");

  LiteRtHwAccelerators supported_hw;
  LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedHardware(plugin.get(), &supported_hw));
  EXPECT_EQ(supported_hw, kLiteRtHwAcceleratorNpu);

  LiteRtParamIndex num_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(plugin.get(), &num_models));
  EXPECT_EQ(num_models, 4);

  const char* model_name;
  LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0, &model_name));
  EXPECT_STREQ(model_name, "Hailo-8");
}

TEST(TestHailoPlugin, PartitionAllGraph) {
  auto plugin = CreatePlugin(LrtGetCompilerContext());
  auto model = testing::LoadTestFileModel("add_simple.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  // All ops should be selected into a single partition for wrapping.
  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(selected_ops[0].second, 0); // Partition index 0
}

TEST(TestHailoPlugin, CompileWithPrecompiledHef) {
  // Create a dummy HEF file.
  const std::string dummy_hef_path = "dummy_model.hef";
  std::ofstream dummy_file(dummy_hef_path, std::ios::binary);
  std::string expected_bytecode = "HAILO_EXEC_BYTECODE_DUMMY_DATA";
  dummy_file.write(expected_bytecode.data(), expected_bytecode.size());
  dummy_file.close();

  // Set the environment variable.
  ::setenv("LITERT_HAILO_HEF_PATH", dummy_hef_path.c_str(), 1);

  auto plugin = CreatePlugin(LrtGetCompilerContext());
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "Hailo-8", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view compiled_bytecode(reinterpret_cast<const char*>(byte_code), byte_code_size);
  EXPECT_EQ(compiled_bytecode, expected_bytecode);

  LiteRtDestroyCompiledResult(compiled);
  std::remove(dummy_hef_path.c_str());
  ::unsetenv("LITERT_HAILO_HEF_PATH");
}

}  // namespace
}  // namespace litert
