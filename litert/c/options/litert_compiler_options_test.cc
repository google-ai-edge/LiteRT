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

#include <cstddef>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert::compiler {
namespace {

TEST(LiteRtCompilerOptionsTest, CreateAndGet) {
  EXPECT_NE(LrtCreateCompilerOptions(nullptr), kLiteRtStatusOk);

  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  const char* id;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(options, &id, &payload,
                                                   &payload_deleter));
  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_ASSERT_OK(
      LiteRtCreateOpaqueOptions(id, payload, payload_deleter, &opaque_options));
  auto opaque_options_cleanup = absl::MakeCleanup(
      [opaque_options] { LiteRtDestroyOpaqueOptions(opaque_options); });

  const char* get_id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(opaque_options, &get_id));
  EXPECT_STREQ(get_id, LrtGetCompilerOptionsIdentifier());
}

TEST(LiteRtCompilerOptionsTest, GetOpaqueDataFailsWithNullArgs) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  const char* id = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  EXPECT_EQ(
      LrtGetOpaqueCompilerOptionsData(nullptr, &id, &payload, &payload_deleter),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtGetOpaqueCompilerOptionsData(options, nullptr, &payload,
                                            &payload_deleter),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LrtGetOpaqueCompilerOptionsData(options, &id, nullptr, &payload_deleter),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtGetOpaqueCompilerOptionsData(options, &id, &payload, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtCompilerOptionsTest, GetOpaqueDataSerializesSetFields) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  LITERT_ASSERT_OK(LrtSetCompilerOptionsPartitionStrategy(
      options, kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK(LrtSetCompilerOptionsDummyOption(options, true));
  LITERT_ASSERT_OK(LrtSetCompilerOptionsMaxPartitions(options, 3));

  const char* id = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(options, &id, &payload,
                                                   &payload_deleter));

  EXPECT_STREQ(id, LrtGetCompilerOptionsIdentifier());
  EXPECT_STREQ(static_cast<const char*>(payload),
               "partition_strategy = 1\n"
               "dummy_option = true\n"
               "max_partitions = 3\n");

  payload_deleter(payload);
}

TEST(LiteRtCompilerOptionsTest, DummyOptions) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  LITERT_ASSERT_OK(LrtSetCompilerOptionsDummyOption(options, true));

  bool dummy_option;
  LITERT_ASSERT_OK(LrtGetCompilerOptionsDummyOption(options, &dummy_option));
  EXPECT_TRUE(dummy_option);
}

TEST(LiteRtCompilerOptionsTest, MaxPartitions) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  size_t max_partitions;
  EXPECT_EQ(LrtGetCompilerOptionsMaxPartitions(options, &max_partitions),
            kLiteRtStatusErrorNotFound);

  LITERT_ASSERT_OK(LrtSetCompilerOptionsMaxPartitions(options, 5));

  LITERT_ASSERT_OK(
      LrtGetCompilerOptionsMaxPartitions(options, &max_partitions));
  EXPECT_EQ(max_partitions, 5);
}

TEST(LiteRtCompilerOptionsTest, GetDefaultPartitionStrategy) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  EXPECT_EQ(
      LrtGetCompilerOptionsPartitionStrategy(options, &partition_strategy),
      kLiteRtStatusErrorNotFound);
}

TEST(LiteRtCompilerOptionsTest, SetAndGetWCPartitionStrategy) {
  LrtCompilerOptions* options;
  LITERT_ASSERT_OK(LrtCreateCompilerOptions(&options));
  auto options_cleanup =
      absl::MakeCleanup([options] { LrtDestroyCompilerOptions(options); });

  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  LITERT_ASSERT_OK(LrtSetCompilerOptionsPartitionStrategy(
      options, kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected));
  LITERT_ASSERT_OK(
      LrtGetCompilerOptionsPartitionStrategy(options, &partition_strategy));
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

}  // namespace
}  // namespace litert::compiler
