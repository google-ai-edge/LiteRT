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
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"

namespace google_tensor {
enum class FilterOutcome { kRunOnTpu, kDoNotRunOnTpu };
FilterOutcome GetFilterOutcome(
    const ::litert::compiler::Op& op,
    const ::third_party::odml::litert::litert::vendors::google_tensor::
        compiler::OpFilters& op_filters);
}  // namespace google_tensor

namespace litert {
namespace {

using ::google_tensor::FilterOutcome;
using ::litert::google_tensor::GoogleTensorOptions;
using ::testing::UnorderedElementsAre;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    OpFilters;

TEST(TestGoogleTensorPlugin, GetConfigInfo) {
  auto plugin_or = StaticallyLinkedPlugin::Create(LrtGetCompilerContext());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;

  ASSERT_STREQ(plugin.Api()->get_compiler_plugin_soc_manufacturer(), "Google");

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(plugin.Api()->get_num_compiler_plugin_supported_models(
      plugin.Get(), &num_supported_soc_models));
#ifdef EDGETPU_EXTERNAL_RELEASE_COMPILER
  ASSERT_THAT(num_supported_soc_models, 4);
#else
  ASSERT_THAT(num_supported_soc_models, 5);
#endif

  std::vector<std::string> soc_model_names;
  for (int i = 0; i < num_supported_soc_models; ++i) {
    const char* soc_model_name;
    LITERT_ASSERT_OK(plugin.Api()->get_compiler_plugin_supported_soc_model(
        plugin.Get(), i, &soc_model_name));
    soc_model_names.push_back(soc_model_name);
  }
#ifdef EDGETPU_EXTERNAL_RELEASE_COMPILER
  EXPECT_THAT(soc_model_names, UnorderedElementsAre("Tensor_G3", "Tensor_G4",
                                                    "Tensor_G5", "Tensor_G6"));
#else
  EXPECT_THAT(soc_model_names,
              UnorderedElementsAre("Tensor_G3", "Tensor_G4", "Tensor_G5",
                                   "Tensor_G6", "Tensor_G7"));
#endif
}

TEST(TestCallGoogleTensorPlugin, PartitionSimpleMultiAdd) {
  auto plugin_or = StaticallyLinkedPlugin::Create(LrtGetCompilerContext());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  ExtendedModel model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
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

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_opts,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));

  LITERT_ASSERT_OK(LiteRtAddOpaqueOptions(litert_opts.get(), opaque));

  auto plugin_or = StaticallyLinkedPlugin::Create(
      LrtGetCompilerContext(), /*env=*/nullptr, litert_opts.get());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
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

  litert::compiler::Op compiler_op(LrtGetCompilerContext(), op.Get());

  OpFilters filters;

  // 1. Empty filter list (Default Blocklist) -> kRunOnTpu
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kRunOnTpu);

  // 2. Empty filter list (Allowlist) -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_RUN_ON_TPU);
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 3. MATCHES_NOT_RUN_ON_TPU (Blocklist) - Match -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_NOT_RUN_ON_TPU);
  auto* filter1 = filters.add_filters();
  filter1->set_op_name_pattern(std::string(output_name));
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 4. MATCHES_NOT_RUN_ON_TPU (Blocklist) - No Match -> kRunOnTpu
  filter1->set_op_name_pattern("some_unmatched_name");
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kRunOnTpu);

  // 5. MATCHES_RUN_ON_TPU (Allowlist) - No Match -> kDoNotRunOnTpu
  filters.set_filter_behavior(OpFilters::MATCHES_RUN_ON_TPU);
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kDoNotRunOnTpu);

  // 6. MATCHES_RUN_ON_TPU (Allowlist) - Match -> kRunOnTpu
  filter1->set_op_name_pattern(std::string(output_name));
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
            FilterOutcome::kRunOnTpu);

  // 7. Empty op_name_pattern (should be ignored) -> kRunOnTpu
  filters.clear_filters();
  filters.set_filter_behavior(OpFilters::MATCHES_NOT_RUN_ON_TPU);
  auto* filter2 = filters.add_filters();
  filter2->set_op_name_pattern("");
  EXPECT_EQ(::google_tensor::GetFilterOutcome(compiler_op, filters),
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

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, Environment::Create({}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_opts,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));

  LITERT_ASSERT_OK(LiteRtAddOpaqueOptions(litert_opts.get(), opaque));

  auto plugin_or = StaticallyLinkedPlugin::Create(
      LrtGetCompilerContext(), /*env=*/nullptr, litert_opts.get());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  // Since nothing matches the allowlist, everything should fall back to CPU.
  ASSERT_EQ(selected_ops.size(), 0);
}

TEST(TestCallGoogleTensorPlugin, CompileMulSubgraph) {
  auto plugin_or = StaticallyLinkedPlugin::Create(LrtGetCompilerContext());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  ExtendedModel model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_compile(
      plugin.Get(), "Tensor_G5", model.Get(), &compiled));
  absl::Cleanup compiled_cleanup = [&plugin, &compiled] {
    plugin.Api()->destroy_compiled_result(compiled);
  };

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_byte_code(
      compiled, 0, &byte_code, &byte_code_size));
  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;
  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_call_info(
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

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_opts,
      internal::LiteRtOptionsPtrBuilder::Build(options, env.GetHolder()));

  auto plugin_or = StaticallyLinkedPlugin::Create(
      LrtGetCompilerContext(), /*env=*/nullptr, litert_opts.get());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  auto model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_compile(
      plugin.Get(), "Tensor_G5", model.Get(), &compiled));
  absl::Cleanup compiled_cleanup = [&plugin, &compiled] {
    plugin.Api()->destroy_compiled_result(compiled);
  };

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_byte_code(
      compiled, 0, &byte_code, &byte_code_size));
  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;
  LITERT_ASSERT_OK(plugin.Api()->get_compiled_result_call_info(
      compiled, 0, &op_data, &op_data_size, &byte_code_idx));
  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_THAT(op_data_string, "subgraph_0_fn");
}

TEST(TestCallGoogleTensorPlugin, PartitionRmsNormCompositeOp) {
  auto plugin_or = StaticallyLinkedPlugin::Create(LrtGetCompilerContext());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  ExtendedModel model = testing::LoadTestFileModel(
      "stablehlo/stablehlo_composite_rms_norm.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  ASSERT_THAT(selected_ops.size(), 1);
  ASSERT_THAT(selected_ops[0].first->OpCode(), kLiteRtOpCodeShloComposite);
}

TEST(TestCallGoogleTensorPlugin, PartitionUnsupportedCompositeOp) {
  auto plugin_or = StaticallyLinkedPlugin::Create(LrtGetCompilerContext());
  ASSERT_TRUE(plugin_or.HasValue());
  auto& plugin = *plugin_or;
  ExtendedModel model = testing::LoadTestFileModel(
      "stablehlo/stablehlo_composite_softmax.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(plugin.Api()->compiler_plugin_partition(
      plugin.Get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  ASSERT_THAT(selected_ops.size(), 0);
}

}  // namespace
}  // namespace litert
