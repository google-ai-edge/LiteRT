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

#include "litert/c/options/litert_compiler_options.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert::compiler {
namespace {

TEST(LiteRtCompilerOptionsTest, CreateAndGet) {
  EXPECT_NE(LiteRtCreateCompilerOptions(nullptr), kLiteRtStatusOk);

  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options));

  const char* id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(options, &id));
  EXPECT_STREQ(id, "litert_compiler");

  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options, &compiler_options));

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtCompilerOptionsTest, DummyOptions) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options));

  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options, &compiler_options));

  LITERT_ASSERT_OK(LiteRtSetDummyCompilerOptions(compiler_options, true));

  bool dummy_option;
  LITERT_ASSERT_OK(
      LiteRtGetDummyCompilerOptions(compiler_options, &dummy_option));
  EXPECT_TRUE(dummy_option);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtCompilerOptionsTest, GetDefaultPartitionStrategy) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options));
  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options, &compiler_options));

  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  LITERT_ASSERT_OK(LiteRtGetCompilerOptionsPartitionStrategy(
      compiler_options, &partition_strategy));
  EXPECT_EQ(partition_strategy, kLiteRtCompilerOptionsPartitionStrategyDefault);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtCompilerOptionsTest, SetAndGetWCPartitionStrategy) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options));
  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options, &compiler_options));

  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  LITERT_ASSERT_OK(LiteRtSetCompilerOptionsPartitionStrategy(
      compiler_options,
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK(LiteRtGetCompilerOptionsPartitionStrategy(
      compiler_options, &partition_strategy));
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtCompilerOptionsTest, Hash) {
  LiteRtOpaqueOptions options1;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options1));
  LiteRtOpaqueOptions options2;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options2));

  uint64_t hash1, hash2;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options1, &hash1));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options2, &hash2));
  EXPECT_EQ(hash1, hash2);

  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options1, &compiler_options));
  LITERT_ASSERT_OK(LiteRtSetDummyCompilerOptions(compiler_options, true));
  LITERT_ASSERT_OK(LiteRtSetCompilerOptionsPartitionStrategy(
      compiler_options,
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options1, &hash1));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options2, &hash2));
  EXPECT_NE(hash1, hash2);

  LiteRtDestroyOpaqueOptions(options1);
  LiteRtDestroyOpaqueOptions(options2);
}

TEST(LiteRtCompilerOptionsTest, CustomOpInfo) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCompilerOptions(&options));
  LiteRtCompilerOptions compiler_options;
  LITERT_ASSERT_OK(LiteRtFindCompilerOptions(options, &compiler_options));

  LiteRtParamIndex num_custom_ops;
  LITERT_ASSERT_OK(LiteRtGetCompilerOptionsNumCustomOpInfo(compiler_options,
                                                           &num_custom_ops));
  EXPECT_EQ(num_custom_ops, 0);

  LITERT_ASSERT_OK(
      LiteRtAddCompilerOptionCustomOpInfo(compiler_options, "op1", "path1"));
  LITERT_ASSERT_OK(
      LiteRtAddCompilerOptionCustomOpInfo(compiler_options, "op2", "path2"));

  LITERT_ASSERT_OK(LiteRtGetCompilerOptionsNumCustomOpInfo(compiler_options,
                                                           &num_custom_ops));
  EXPECT_EQ(num_custom_ops, 2);

  const char* name;
  const char* path;
  LITERT_ASSERT_OK(
      LiteRtGetCompilerOptionsCustomOpInfo(compiler_options, 0, &name, &path));
  EXPECT_STREQ(name, "op1");
  EXPECT_STREQ(path, "path1");

  LITERT_ASSERT_OK(
      LiteRtGetCompilerOptionsCustomOpInfo(compiler_options, 1, &name, &path));
  EXPECT_STREQ(name, "op2");
  EXPECT_STREQ(path, "path2");

  EXPECT_EQ(
      LiteRtGetCompilerOptionsCustomOpInfo(compiler_options, 2, &name, &path),
      kLiteRtStatusErrorIndexOOB);

  LiteRtDestroyOpaqueOptions(options);
}
}  // namespace
}  // namespace litert::compiler
