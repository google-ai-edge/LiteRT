// Copyright 2024 Google LLC.
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

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"

namespace google_tensor {
enum class FilterOutcome { kRunOnTpu, kDoNotRunOnTpu };
FilterOutcome GetFilterOutcome(
    const ::litert::Op& op, const ::third_party::odml::litert::litert::vendors::
                                google_tensor::compiler::OpFilters& op_filters);
}  // namespace google_tensor

namespace litert {
namespace {

using ::google_tensor::FilterOutcome;
using ::litert::google_tensor::GoogleTensorOptions;
using ::testing::UnorderedElementsAre;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    OpFilters;

TEST(TestGoogleTensorPlugin, GetConfigInfo) {
  ASSERT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Google");

  PluginPtr plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_THAT(num_supported_soc_models, 4);

  std::vector<std::string> soc_model_names;
  for (int i = 0; i < num_supported_soc_models; ++i) {
    const char* soc_model_name;
    LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), i,
                                                              &soc_model_name));
    soc_model_names.push_back(soc_model_name);
  }
  EXPECT_THAT(soc_model_names, UnorderedElementsAre("Tensor_G3", "Tensor_G4",
                                                    "Tensor_G5", "Tensor_G6"));
}

TEST(TestCallGoogleTensorPlugin, PartitionSimpleMultiAdd) {
  PluginPtr plugin = CreatePlugin();
  ExtendedModel model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  ASSERT_THAT(selected_ops.size(), 4);
  ASSERT_THAT(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
  ASSERT_THAT(selected_ops[1].first->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestCallGoogleTensorPlugin, PartitionWithOpFiltersRunOnCpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(auto& google_tensor_options,
                              options.GetGoogleTensorOptions());

  std::string temp_file =
      ::testing::TempDir() + "/test_op_filters_cpu.textproto";
  std::ofstream out(temp_file);
  out << "filter_behavior: MATCHES_NOT_RUN_ON_TPU\n";
  out << "filters {\n";
  out << "  op_name_pattern: \".*\"\n";  // Block all operations
  out << "}\n";
  out.close();

  google_tensor_options.SetOpFiltersProto(temp_file);
  const char* identifier;
  void* payload;
  void (*deleter)(void*);
  LITERT_ASSERT_OK(google_tensor_options.GetOpaqueOptionsData(
      &identifier, &payload, &deleter));
  LiteRtOpaqueOptions opaque;
  LITERT_ASSERT_OK(
      LiteRtCreateOpaqueOptions(identifier, payload, deleter, &opaque));
  LITERT_ASSERT_OK(LiteRtAddOpaqueOptions(options.Get(), opaque));

  auto plugin = CreatePlugin(/*env=*/nullptr, options.Get());
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  // All ops should be blocked by the filter and fall back to CPU.
  ASSERT_EQ(selected_ops.size(), 0);
}

TEST(TestGoogleTensorPlugin, GetFilterOutcome) {
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));
  auto ops = subgraph.Ops();
  ASSERT_FALSE(ops.empty());
  auto op = ops[0];
  auto outputs = op.Outputs();
  ASSERT_FALSE(outputs.empty());
  auto output_name = outputs[0].Name();

  OpFilters filters;

  // 1. Empty filter list (Default Blocklist) -> kRunOnTpu
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kRunOnTpu);

  // 2. Empty filter list (Allowlist) -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_RUN_ON_TPU);
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 3. MATCHES_NOT_RUN_ON_TPU (Blocklist) - Match -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_NOT_RUN_ON_TPU);
  auto* filter1 = filters.add_filters();
  filter1->set_op_name_pattern(std::string(output_name));
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 4. MATCHES_NOT_RUN_ON_TPU (Blocklist) - No Match -> kRunOnTpu
  filter1->set_op_name_pattern("some_unmatched_name");
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kRunOnTpu);

  // 5. MATCHES_RUN_ON_TPU (Allowlist) - No Match -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_RUN_ON_TPU);
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 6. MATCHES_RUN_ON_TPU (Allowlist) - Match -> kRunOnTpu
  filter1->set_op_name_pattern(std::string(output_name));
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kRunOnTpu);

  // 7. Empty op_name_pattern (should be ignored) -> kRunOnTpu
  filters.clear_filters();
  filters.set_filter_behavior(OpFilters::MATCHES_NOT_RUN_ON_TPU);
  auto* filter2 = filters.add_filters();
  filter2->set_op_name_pattern("");
  EXPECT_EQ(::google_tensor::GetFilterOutcome(op, filters),
            FilterOutcome::kRunOnTpu);
}

TEST(TestCallGoogleTensorPlugin, PartitionWithOpFiltersRunOnTpu) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(auto& google_tensor_options,
                              options.GetGoogleTensorOptions());

  std::string temp_file =
      ::testing::TempDir() + "/test_op_filters_tpu.textproto";
  std::ofstream out(temp_file);
  out << "filter_behavior: MATCHES_RUN_ON_TPU\n";
  out << "filters {\n";
  out << "  op_name_pattern: \"will_not_match_anything\"\n";  // Allow nothing
  out << "}\n";
  out.close();

  google_tensor_options.SetOpFiltersProto(temp_file);
  const char* identifier;
  void* payload;
  void (*deleter)(void*);
  LITERT_ASSERT_OK(google_tensor_options.GetOpaqueOptionsData(
      &identifier, &payload, &deleter));
  LiteRtOpaqueOptions opaque;
  LITERT_ASSERT_OK(
      LiteRtCreateOpaqueOptions(identifier, payload, deleter, &opaque));
  LITERT_ASSERT_OK(LiteRtAddOpaqueOptions(options.Get(), opaque));

  auto plugin = CreatePlugin(/*env=*/nullptr, options.Get());
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  // Since nothing matches the allowlist, everything should fall back to CPU.
  ASSERT_EQ(selected_ops.size(), 0);
}

TEST(TestCallGoogleTensorPlugin, CompileMulSubgraph) {
  PluginPtr plugin = CreatePlugin();
  ExtendedModel model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "Tensor_G5",
                                               model.Get(), &compiled));
  absl::Cleanup compiled_cleanup = [&compiled] {
    LiteRtDestroyCompiledResult(compiled);
  };

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(compiled, 0, &byte_code,
                                                   &byte_code_size));
  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, 0, &op_data, &op_data_size, &byte_code_idx));
  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_THAT(op_data_string, "subgraph_0_fn");
}

TEST(TestCallGoogleTensorPlugin, CompileMulSubgraphWithOptions) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(GoogleTensorOptions& google_tensor_options,
                              options.GetGoogleTensorOptions());
  google_tensor_options.SetFloatTruncationType(
      kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  PluginPtr plugin =
      CreatePlugin(/*runtime_context=*/nullptr, /*env=*/nullptr, options.Get());
  auto model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "Tensor_G5",
                                               model.Get(), &compiled));
  absl::Cleanup compiled_cleanup = [&compiled] {
    LiteRtDestroyCompiledResult(compiled);
  };

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(compiled, 0, &byte_code,
                                                   &byte_code_size));
  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, 0, &op_data, &op_data_size, &byte_code_idx));
  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_THAT(op_data_string, "subgraph_0_fn");
}

TEST(TestCallGoogleTensorPlugin, CompileWithTestingFlags) {
  LITERT_ASSERT_OK_AND_ASSIGN(Options options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(GoogleTensorOptions & google_tensor_options,
                              options.GetGoogleTensorOptions());

  // Set testing flags. Use a safe flag that doesn't involve the filesystem.
  google_tensor_options.SetTestingFlags("enable_reference=true");

  PluginPtr plugin =
      CreatePlugin(/*runtime_context=*/nullptr, /*env=*/nullptr, options.Get());
  auto model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  // Verify the plugin handles testing_flags without crashing.
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "Tensor_G5",
                                               model.Get(), &compiled));
  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestCallGoogleTensorPlugin, PartitionRmsNormCompositeOp) {
  PluginPtr plugin = CreatePlugin();
  ExtendedModel model = testing::LoadTestFileModel(
      "stablehlo/stablehlo_composite_rms_norm.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  ASSERT_THAT(selected_ops.size(), 1);
  ASSERT_THAT(selected_ops[0].first->OpCode(), kLiteRtOpCodeShloComposite);
}

TEST(TestCallGoogleTensorPlugin, PartitionUnsupportedCompositeOp) {
  PluginPtr plugin = CreatePlugin();
  ExtendedModel model = testing::LoadTestFileModel(
      "stablehlo/stablehlo_composite_softmax.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  ASSERT_THAT(selected_ops.size(), 0);
}

}  // namespace
}  // namespace litert
