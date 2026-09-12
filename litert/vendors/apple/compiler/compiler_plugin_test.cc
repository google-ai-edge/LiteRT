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

#include <cstddef>
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
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/vendors/apple/bytecode.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

using ::testing::Eq;

TEST(TestAppleMlxPlugin, GetConfigInfo) {
  ASSERT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Apple");

  PluginPtr plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_THAT(num_supported_soc_models, 1);

  const char* soc_model_name;
  LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0,
                                                            &soc_model_name));
  EXPECT_STREQ(soc_model_name, "apple_mlx");
}

TEST(TestCallAppleMlxPlugin, PartitionSimpleFullyConnected) {
  PluginPtr plugin = CreatePlugin(LrtGetCompilerContext());
  ExtendedModel model =
      testing::LoadTestFileModel("fully_connected_cst.tflite");
  LITERT_ASSERT_OK_AND_ASSIGN(Subgraph subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const std::vector<LiteRtOpWithPartitionIndex> selected_ops =
      selected_op_list.Values();

  // The model has one FullyConnected op, which should be selected.
  ASSERT_THAT(selected_ops.size(), 1);
  ASSERT_THAT(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflFullyConnected);
}

TEST(TestCallAppleMlxPlugin, CompileFullyConnected) {
  PluginPtr plugin = CreatePlugin(LrtGetCompilerContext());
  ExtendedModel model =
      testing::LoadTestFileModel("fully_connected_cst.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), "apple_mlx",
                                               model.Get(), &compiled));
  absl::Cleanup compiled_cleanup = [&compiled] {
    LiteRtDestroyCompiledResult(compiled);
  };

  LiteRtParamIndex num_calls;
  LITERT_ASSERT_OK(LiteRtGetNumCompiledResultCalls(compiled, &num_calls));
  ASSERT_EQ(num_calls, 1);

  const void* call_info;
  size_t call_info_size;
  LiteRtParamIndex byte_code_idx;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, 0, &call_info, &call_info_size, &byte_code_idx));
  absl::string_view call_info_string(reinterpret_cast<const char*>(call_info),
                                     call_info_size);
  ASSERT_THAT(call_info_string, "mlx_partition_0");
  ASSERT_EQ(byte_code_idx, 0);

  const void* byte_code;
  size_t byte_code_size;
  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(compiled, 0, &byte_code,
                                                   &byte_code_size));

  // Parse bytecode and verify
  auto bytecode_or = litert::apple::ParseMlxBytecode(byte_code, byte_code_size);
  LITERT_ASSERT_OK(bytecode_or);
  const auto& bytecode = *bytecode_or;

  // Verify weights (shape is 2x4)
  EXPECT_EQ(bytecode.weights_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(bytecode.weights_dims.size(), 2);
  EXPECT_EQ(bytecode.weights_dims[0], 2);
  EXPECT_EQ(bytecode.weights_dims[1], 4);
  EXPECT_EQ(bytecode.weights_data.size(), 2 * 4 * sizeof(float));

  // Verify bias (shape is 2)
  EXPECT_TRUE(bytecode.has_bias);
  EXPECT_EQ(bytecode.bias_type, kLiteRtElementTypeFloat32);
  ASSERT_EQ(bytecode.bias_dims.size(), 1);
  EXPECT_EQ(bytecode.bias_dims[0], 2);
  EXPECT_EQ(bytecode.bias_data.size(), 2 * sizeof(float));

  // Verify activation (None)
  EXPECT_EQ(bytecode.activation, 0);  // kActivationFunctionTypeNone
}

}  // namespace
}  // namespace litert
