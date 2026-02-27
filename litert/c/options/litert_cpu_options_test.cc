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

#include "litert/c/options/litert_cpu_options.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/test/matchers.h"

using testing::IsNull;
using testing::litert::IsError;

namespace {

auto CleanupOptionsOnScopeExit(LrtCpuOptions* options) {
  return absl::Cleanup([options] { LrtDestroyCpuOptions(options); });
}

TEST(LiteRtCpuOptionsTest, CreateErrorsOutWithNullptrParam) {
  EXPECT_THAT(LrtCreateCpuOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtCpuOptionsTest, CreateWorks) {
  LrtCpuOptions* options = nullptr;
  LITERT_ASSERT_OK(LrtCreateCpuOptions(&options));
  const auto _ = CleanupOptionsOnScopeExit(options);
  EXPECT_THAT(options, Not(IsNull()));
}

TEST(LiteRtCpuOptionsTest, GetOpaqueCpuOptionsDataWorks) {
  LrtCpuOptions* options = nullptr;
  LITERT_ASSERT_OK(LrtCreateCpuOptions(&options));
  const auto _ = CleanupOptionsOnScopeExit(options);

  const char* identifier;
  void* payload;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueCpuOptionsData(options, &identifier, &payload,
                                              &payload_deleter));
  EXPECT_STREQ(identifier, LrtGetCpuOptionsIdentifier());

  LiteRtOpaqueOptions opaque_options;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_options));
  LiteRtDestroyOpaqueOptions(opaque_options);
}

class LiteRtCpuOptionsFieldsTest : public testing::Test {
  void SetUp() override {
    LITERT_ASSERT_OK(LrtCreateCpuOptions(&cpu_options_));
  }

  void TearDown() override { LrtDestroyCpuOptions(cpu_options_); }

 protected:
  LrtCpuOptions* cpu_options_;
};

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetNumThreads) {
  const int expected_num_threads = 4;
  int num_threads = -1;

  // Initial state should be NotFound (or default if we decided to set defaults,
  // but struct has optional) Actually the struct has optional, so Get should
  // return NotFound if not set.
  EXPECT_THAT(LrtGetCpuOptionsNumThread(cpu_options_, &num_threads),
              IsError(kLiteRtStatusErrorNotFound));

  // Actual test.
  LITERT_EXPECT_OK(
      LrtSetCpuOptionsNumThread(cpu_options_, expected_num_threads));
  LITERT_EXPECT_OK(LrtGetCpuOptionsNumThread(cpu_options_, &num_threads));
  ASSERT_EQ(num_threads, expected_num_threads);
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackFlags) {
  const uint32_t expected_flags = 4;
  uint32_t flags = 0;

  EXPECT_THAT(LrtGetCpuOptionsXNNPackFlags(cpu_options_, &flags),
              IsError(kLiteRtStatusErrorNotFound));

  LITERT_EXPECT_OK(LrtSetCpuOptionsXNNPackFlags(cpu_options_, expected_flags));
  LITERT_EXPECT_OK(LrtGetCpuOptionsXNNPackFlags(cpu_options_, &flags));
  ASSERT_EQ(flags, expected_flags);
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackWeightCachePath) {
  const absl::string_view expected_path = "a/path/to/the/cache";
  const char* path = nullptr;

  EXPECT_THAT(LrtGetCpuOptionsXnnPackWeightCachePath(cpu_options_, &path),
              IsError(kLiteRtStatusErrorNotFound));

  LITERT_EXPECT_OK(LrtSetCpuOptionsXnnPackWeightCachePath(
      cpu_options_, expected_path.data()));
  LITERT_EXPECT_OK(LrtGetCpuOptionsXnnPackWeightCachePath(cpu_options_, &path));
  ASSERT_EQ(path, expected_path);
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackWeightCacheDescriptor) {
  const int expected_fd = 1234;
  int fd;

  EXPECT_THAT(
      LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, &fd),
      IsError(kLiteRtStatusErrorNotFound));

  LITERT_EXPECT_OK(LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(
      cpu_options_, expected_fd));
  LITERT_EXPECT_OK(
      LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, &fd));
  ASSERT_EQ(fd, expected_fd);
}

TEST_F(LiteRtCpuOptionsFieldsTest,
       SetXNNPackWeightCacheFailsIfBothPathAndDescriptorAreSet) {
  const absl::string_view path = "a/path/to/the/cache";
  const int fd = 1234;

  LITERT_EXPECT_OK(
      LrtSetCpuOptionsXnnPackWeightCachePath(cpu_options_, path.data()));
  EXPECT_THAT(
      LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, fd),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

}  // namespace
