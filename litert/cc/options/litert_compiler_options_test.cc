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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/cc/options/litert_compiler_options.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {

TEST(CompilerOptionsTest, CreateSetAndGetDummyOptionWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(::litert::CompilerOptions options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetDummyOption(true));
  EXPECT_TRUE(options.GetDummyOption());
}

TEST(CompilerOptionsTest, SetAndGetPartitionStrategyReturnsSetValue) {
  LITERT_ASSERT_OK_AND_ASSIGN(::litert::CompilerOptions options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK_AND_ASSIGN(auto partition_strategy,
                              options.GetPartitionStrategy());
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

TEST(CompilerOptionsTest, AddCustomOpInfo) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options,
                              ::litert::CompilerOptions::Create());
  LITERT_EXPECT_OK(options.AddCustomOpInfo("op1", "path1"));
  LITERT_EXPECT_OK(options.AddCustomOpInfo("op2", "path2"));

  LiteRtCompilerOptions c_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options.Get(), &c_options));

  LiteRtParamIndex num_custom_ops;
  LITERT_ASSERT_OK(
      LiteRtGetCompilerOptionsNumCustomOpInfo(c_options, &num_custom_ops));
  EXPECT_EQ(num_custom_ops, 2);

  const char* name;
  const char* path;

  LITERT_ASSERT_OK(
      LiteRtGetCompilerOptionsCustomOpInfo(c_options, 0, &name, &path));
  EXPECT_STREQ(name, "op1");
  EXPECT_STREQ(path, "path1");

  LITERT_ASSERT_OK(
      LiteRtGetCompilerOptionsCustomOpInfo(c_options, 1, &name, &path));
  EXPECT_STREQ(name, "op2");
  EXPECT_STREQ(path, "path2");
}

}  // namespace
}  // namespace litert
