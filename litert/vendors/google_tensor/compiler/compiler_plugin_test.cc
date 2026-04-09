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

namespace litert {
namespace {

using ::litert::google_tensor::GoogleTensorOptions;
using ::testing::UnorderedElementsAre;

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

  PluginPtr plugin = CreatePlugin(/*env=*/nullptr, options.Get());
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

  PluginPtr plugin = CreatePlugin(/*env=*/nullptr, options.Get());
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
