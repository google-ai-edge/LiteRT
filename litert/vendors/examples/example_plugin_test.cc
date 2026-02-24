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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

TEST(TestDummyPlugin, GetConfigInfo) {
  ASSERT_STREQ(LiteRtGetCompilerPluginSocManufacturer(),
               "ExampleSocManufacturer");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 1);

  const char* soc_model_name;
  LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0,
                                                            &soc_model_name));
  ASSERT_STREQ(soc_model_name, "ExampleSocModel");
}

TEST(TestCallDummyPlugin, PartitionSimpleMultiAdd) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 2);
  ASSERT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(selected_ops[1].first->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestCallDummyPlugin, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(
      plugin.get(), /*soc_model=*/nullptr, model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_EQ(
      byte_code_string,
      "SUBGRAPH "
      "partition_0\nversion:1\ninputs:0,1\noutputs:3\nconst_map:\ntensors:["
      "2x2],[2x2],[2x2],[2x2]"
      "\nops:mul(0,"
      "0)(2)~mul(2,1)(3)\n");

  LiteRtParamIndex byte_code_idx;
  const void* op_data;
  size_t op_data_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ(op_data_string, "partition_0");

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestCallDummyPlugin, RegisterAllTransformations) {
  auto plugin = CreatePlugin();
  LiteRtTransformation* transformations;
  LiteRtParamIndex num_transformations;
  LITERT_ASSERT_OK(LiteRtCompilerPluginRegisterAllTransformations(
      plugin.get(), &transformations, &num_transformations));
  ASSERT_EQ(num_transformations, 2);
  ASSERT_STREQ(transformations[0].name, "MyTransformation0");
  ASSERT_EQ(transformations[0].benefit, 100);
}

TEST(TestCallDummyPlugin, CheckCompilerCompatibility) {
  auto plugin = CreatePlugin();
  LiteRtApiVersion api_version = {.major = 1, .minor = 0, .patch = 0};
  LiteRtEnvironmentOptions env = nullptr;
  LiteRtOptions options = nullptr;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCheckCompilerCompatibility(
      api_version, plugin.get(), env, options, "ExampleSocModel"));

  EXPECT_EQ(
      kLiteRtStatusErrorUnsupportedCompilerVersion,
      LiteRtCompilerPluginCheckCompilerCompatibility(
          api_version, plugin.get(), env, options, "UnsupportedSocModel"));
}

TEST(TestCallDummyPlugin, CompileMultiSubgraphWithSharedWeights) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("cst_multi_subgraph.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(
      plugin.get(), /*soc_model=*/nullptr, model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);

  EXPECT_THAT(byte_code_string, ::testing::HasSubstr("BUFFER "));
  EXPECT_THAT(byte_code_string, ::testing::HasSubstr("SUBGRAPH partition_0"));
  EXPECT_THAT(byte_code_string, ::testing::HasSubstr("SUBGRAPH partition_1"));
  EXPECT_THAT(byte_code_string, ::testing::HasSubstr("const_map:"));

  LiteRtDestroyCompiledResult(compiled);
}

}  // namespace
}  // namespace litert
