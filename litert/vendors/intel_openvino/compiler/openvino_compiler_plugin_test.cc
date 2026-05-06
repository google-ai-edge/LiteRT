// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
#include <cstdint>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

namespace litert {
namespace {

using ::testing::Values;

const auto kSupportedOps = Values("add_simple.tflite");
const auto kSupportedSocModels = Values("LNL", "PTL");

TEST(TestOVPlugin, PartitionAddGraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  LITERT_ASSERT_OK_AND_ASSIGN(auto subgraph, model.Subgraph(0));

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, subgraph.Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflAdd);
}

TEST(TestOVPlugin, CompileAddSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "PTL", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);

  LiteRtDestroyCompiledResult(compiled);
}

// Tests for GetSocModelConfig
TEST(TestOVPlugin, GetSocModelConfigLNL) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("LNL", config_map),
            kLiteRtStatusOk);
  ASSERT_GT(config_map.count("NPU_PLATFORM"), 0u);
  EXPECT_EQ(config_map.at("NPU_PLATFORM").as<std::string>(), "NPU4000");
}

TEST(TestOVPlugin, GetSocModelConfigPTL) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("PTL", config_map),
            kLiteRtStatusOk);
  ASSERT_GT(config_map.count("NPU_PLATFORM"), 0u);
  EXPECT_EQ(config_map.at("NPU_PLATFORM").as<std::string>(), "NPU5010");
}

TEST(TestOVPlugin, GetSocModelConfigUnknownReturnsError) {
  ov::AnyMap config_map;
  EXPECT_EQ(litert::openvino::GetSocModelConfig("UNKNOWN_SOC", config_map),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(config_map.count("NPU_PLATFORM"), 0u);
}

// ===== Negative tests for compiled result getters =====

// Null compiled_result returns InvalidArgument.
TEST(TestOVPlugin, GetByteCodeNullCompiledResult) {
  const void* byte_code = nullptr;
  size_t byte_code_size = 0;
  EXPECT_EQ(LiteRtGetCompiledResultByteCode(
                /*compiled_result=*/nullptr, /*byte_code_idx=*/0, &byte_code,
                &byte_code_size),
            kLiteRtStatusErrorInvalidArgument);
}

// Null output pointer returns InvalidArgument.
TEST(TestOVPlugin, GetByteCodeNullOutputPointer) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");
  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "PTL", model.Get(), &compiled));

  size_t byte_code_size = 0;
  EXPECT_EQ(LiteRtGetCompiledResultByteCode(compiled, /*byte_code_idx=*/0,
                                            /*byte_code=*/nullptr,
                                            &byte_code_size),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDestroyCompiledResult(compiled);
}

// OOB byte_code_idx returns IndexOOB.
TEST(TestOVPlugin, GetByteCodeOobIndex) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");
  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "PTL", model.Get(), &compiled));

  const void* byte_code = nullptr;
  size_t byte_code_size = 0;
  EXPECT_EQ(LiteRtGetCompiledResultByteCode(compiled, /*byte_code_idx=*/999,
                                            &byte_code, &byte_code_size),
            kLiteRtStatusErrorIndexOOB);
  LiteRtDestroyCompiledResult(compiled);
}

// Null compiled_result for CallInfo returns InvalidArgument.
TEST(TestOVPlugin, GetCallInfoNullCompiledResult) {
  const void* call_info = nullptr;
  size_t call_info_size = 0;
  LiteRtParamIndex byte_code_idx = 0;
  EXPECT_EQ(LiteRtGetCompiledResultCallInfo(
                /*compiled_result=*/nullptr, /*call_idx=*/0, &call_info,
                &call_info_size, &byte_code_idx),
            kLiteRtStatusErrorInvalidArgument);
}

// OOB call_idx returns IndexOOB.
TEST(TestOVPlugin, GetCallInfoOobIndex) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("add_simple.tflite");
  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "PTL", model.Get(), &compiled));

  const void* call_info = nullptr;
  size_t call_info_size = 0;
  LiteRtParamIndex byte_code_idx = 0;
  EXPECT_EQ(LiteRtGetCompiledResultCallInfo(compiled, /*call_idx=*/999,
                                            &call_info, &call_info_size,
                                            &byte_code_idx),
            kLiteRtStatusErrorIndexOOB);
  LiteRtDestroyCompiledResult(compiled);
}

// Null compiled_result for NumCalls returns InvalidArgument.
TEST(TestOVPlugin, GetNumCallsNullCompiledResult) {
  LiteRtParamIndex num_calls = 0;
  EXPECT_EQ(LiteRtGetNumCompiledResultCalls(
                /*compiled_result=*/nullptr, &num_calls),
            kLiteRtStatusErrorInvalidArgument);
}

// Null compiled_result for NumByteCodeModules returns InvalidArgument.
TEST(TestOVPlugin, NumByteCodeModulesNullCompiledResult) {
  LiteRtParamIndex num_byte_code = 0;
  EXPECT_EQ(LiteRtCompiledResultNumByteCodeModules(
                /*compiled_result=*/nullptr, &num_byte_code),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert
