// Copyright 2025 Google LLC.
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
#include "litert/tools/flags/apply_plugin_flags.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_compiler_options.h"

namespace litert {
namespace {
TEST(ApplyPluginFlagsTest, MalformedPartitionStrategyFailsToParse) {
  std::string error;
  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  EXPECT_FALSE(AbslParseFlag("not", &partition_strategy, &error));
  EXPECT_FALSE(AbslParseFlag("a real", &partition_strategy, &error));
  EXPECT_FALSE(AbslParseFlag("flag", &partition_strategy, &error));
}

TEST(ApplyPluginFlagsTest, ParsePartitionStrategySuccess) {
  std::string error;
  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  EXPECT_TRUE(AbslParseFlag("default", &partition_strategy, &error));
  EXPECT_EQ(partition_strategy, kLiteRtCompilerOptionsPartitionStrategyDefault);
  EXPECT_EQ(AbslUnparseFlag(partition_strategy), "default");
  EXPECT_TRUE(AbslParseFlag("weakly_connected", &partition_strategy, &error));
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  EXPECT_EQ(AbslUnparseFlag(partition_strategy), "weakly_connected");
}

TEST(ApplyPluginFlagsTest, UnparsePartitionStrategySuccess) {
  EXPECT_EQ(AbslUnparseFlag(kLiteRtCompilerOptionsPartitionStrategyDefault),
            "default");
  EXPECT_EQ(
      AbslUnparseFlag(kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected),
      "weakly_connected");
}

TEST(ApplyPluginFlagsTest, ParseFlagsAndGetPartitionStrategySuccess) {
  absl::SetFlag(&FLAGS_partition_strategy,
                kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);

  Expected<CompilerOptions> options = CompilerOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateCompilerOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetPartitionStrategy().HasValue());
  LITERT_ASSIGN_OR_ABORT(auto partition_strategy,
                         options.Value().GetPartitionStrategy());
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

TEST(ApplyPluginFlagsTest, ParseFlagsAndAddCustomOpInfoSuccess) {
  absl::SetFlag(&FLAGS_npu_custom_op_info,
                std::vector<std::string>{"op1,path1", "op2,path2"});

  Expected<CompilerOptions> options = CompilerOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateCompilerOptionsFromFlags(options.Value()).HasValue());

  LiteRtCompilerOptions c_options;
  ASSERT_EQ(LiteRtFindCompilerOptions(options.Value().Get(), &c_options),
            kLiteRtStatusOk);

  LiteRtParamIndex num_custom_ops;
  ASSERT_EQ(LiteRtGetCompilerOptionsNumCustomOpInfo(c_options, &num_custom_ops),
            kLiteRtStatusOk);
  EXPECT_EQ(num_custom_ops, 2);

  const char* name;
  const char* path;

  ASSERT_EQ(LiteRtGetCompilerOptionsCustomOpInfo(c_options, 0, &name, &path),
            kLiteRtStatusOk);
  EXPECT_EQ(std::string(name), "op1");
  EXPECT_EQ(std::string(path), "path1");

  ASSERT_EQ(LiteRtGetCompilerOptionsCustomOpInfo(c_options, 1, &name, &path),
            kLiteRtStatusOk);
  EXPECT_EQ(std::string(name), "op2");
  EXPECT_EQ(std::string(path), "path2");
}

TEST(ApplyPluginFlagsTest, ParseFlagsAndAddCustomOpInfoFailure) {
  absl::SetFlag(&FLAGS_npu_custom_op_info,
                std::vector<std::string>{"invalid_format"});

  Expected<CompilerOptions> options = CompilerOptions::Create();
  ASSERT_TRUE(options.HasValue());
  auto result = UpdateCompilerOptionsFromFlags(options.Value());
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert
