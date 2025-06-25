// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <openvino/frontend/tensorflow_lite/frontend.hpp>
#include <string>

#include "absl/strings/string_view.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/test_models.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

using ::testing::Values;

const auto kSupportedOps =
                  Values(
                    "add_simple.tflite"
                    );
const auto kSupportedSocModels = Values(
		"NPU2700"
);

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
      LiteRtCompilerPluginCompile(plugin.get(), "NPU2700", model.Get(), &compiled));

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

}  // namespace
}  // namespace litert

