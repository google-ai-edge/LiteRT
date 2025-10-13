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
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/test/matchers.h"

using testing::IsNull;
using testing::litert::IsError;

namespace {

auto CleanupOptionsOnScopeExit(LiteRtOpaqueOptions options) {
  return absl::Cleanup([options] { LiteRtDestroyOpaqueOptions(options); });
}

TEST(LiteRtCpuOptionsTest, CreateErrorsOutWithNullptrParam) {
  EXPECT_THAT(LiteRtCreateCpuOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtCpuOptionsTest, CreateWorks) {
  LiteRtOpaqueOptions options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateCpuOptions(&options));
  const auto _ = CleanupOptionsOnScopeExit(options);

  const char* id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(options, &id));
  EXPECT_THAT(id, testing::StrEq(LiteRtGetCpuOptionsIdentifier()));
}

TEST(LiteRtCpuOptionsTest, FindErrorsOutWithInvalidParameters) {
  LiteRtOpaqueOptions options = nullptr;
  LiteRtCpuOptions cpu_options = nullptr;
  EXPECT_THAT(LiteRtFindCpuOptions(nullptr, &cpu_options),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_ASSERT_OK(LiteRtCreateCpuOptions(&options));
  const auto _ = CleanupOptionsOnScopeExit(options);
  EXPECT_THAT(LiteRtFindCpuOptions(options, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtCpuOptionsTest, FindInSingleOpaqueOptionWorks) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCpuOptions(&options));
  const auto _ = CleanupOptionsOnScopeExit(options);

  LiteRtCpuOptions cpu_options = nullptr;
  LITERT_EXPECT_OK(LiteRtFindCpuOptions(options, &cpu_options));
  EXPECT_THAT(cpu_options, Not(IsNull()));
}

LiteRtStatus CreateIntOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  auto options_data = std::make_unique<int>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      "int-option", options_data.get(),
      [](void* payload) { delete reinterpret_cast<int*>(payload); }, options));
  options_data.release();
  return kLiteRtStatusOk;
}

TEST(LiteRtCpuOptionsTest, FindInOpaqueOptionListWorks) {
  // Create the list of options.
  LiteRtOpaqueOptions option_list;
  LITERT_ASSERT_OK(CreateIntOptions(&option_list));
  const auto _1 = CleanupOptionsOnScopeExit(option_list);

  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtCreateCpuOptions(&options));
  auto _2 = CleanupOptionsOnScopeExit(option_list);

  LITERT_ASSERT_OK(LiteRtAppendOpaqueOptions(&option_list, options));
  std::move(_2).Cancel();  // Cleanup is now handled by the list

  LiteRtCpuOptions cpu_options = nullptr;
  LITERT_EXPECT_OK(LiteRtFindCpuOptions(options, &cpu_options));
  EXPECT_THAT(cpu_options, Not(IsNull()));
}

class LiteRtCpuOptionsFieldsTest : public testing::Test {
  void SetUp() override {
    LITERT_ASSERT_OK(LiteRtCreateCpuOptions(&opaque_options_));
    LITERT_EXPECT_OK(LiteRtFindCpuOptions(opaque_options_, &cpu_options_));
  }

  void TearDown() override { LiteRtDestroyOpaqueOptions(opaque_options_); }

 protected:
  LiteRtOpaqueOptions opaque_options_;
  LiteRtCpuOptions cpu_options_;
};

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetNumThreads) {
  const int expected_num_threads = 4;
  int num_threads = -1;

  // Avoid a no-op test.
  LITERT_EXPECT_OK(LiteRtGetCpuOptionsNumThread(cpu_options_, &num_threads));
  ASSERT_NE(num_threads, expected_num_threads);

  // Actual test.
  LITERT_EXPECT_OK(
      LiteRtSetCpuOptionsNumThread(cpu_options_, expected_num_threads));
  LITERT_EXPECT_OK(LiteRtGetCpuOptionsNumThread(cpu_options_, &num_threads));
  ASSERT_EQ(num_threads, expected_num_threads);
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetNumThreadsFailsWithInvalidArgument) {
  EXPECT_THAT(
      LiteRtSetCpuOptionsNumThread(/*options=*/nullptr, /*num_threads=*/1),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest, GetNumThreadsFailsWithInvalidArgument) {
  int num_threads = 1;
  EXPECT_THAT(LiteRtGetCpuOptionsNumThread(/*options=*/nullptr, &num_threads),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LiteRtGetCpuOptionsNumThread(cpu_options_, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackFlags) {
  const int expected_flags = 4;
  uint32_t flags = 0;

  // Avoid a no-op test.
  LITERT_EXPECT_OK(LiteRtGetCpuOptionsXNNPackFlags(cpu_options_, &flags));
  ASSERT_NE(flags, expected_flags);

  // Actual test.
  LITERT_EXPECT_OK(
      LiteRtSetCpuOptionsXNNPackFlags(cpu_options_, expected_flags));
  LITERT_EXPECT_OK(LiteRtGetCpuOptionsXNNPackFlags(cpu_options_, &flags));
  ASSERT_EQ(flags, expected_flags);
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetXNNPackFlagsFailsWithInvalidArgument) {
  EXPECT_THAT(
      LiteRtSetCpuOptionsXNNPackFlags(/*options=*/nullptr, /*num_threads=*/1),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest, GetXNNPackFlagsFailsWithInvalidArgument) {
  uint32_t flags = 1;
  EXPECT_THAT(LiteRtGetCpuOptionsXNNPackFlags(/*options=*/nullptr, &flags),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LiteRtGetCpuOptionsXNNPackFlags(cpu_options_, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackWeightCachePath) {
  const absl::string_view expected_path = "a/path/to/the/cache";
  const char* path = nullptr;

  // Avoid a no-op test.
  LITERT_EXPECT_OK(
      LiteRtGetCpuOptionsXnnPackWeightCachePath(cpu_options_, &path));
  ASSERT_NE(absl::NullSafeStringView(path), expected_path);

  // Actual test.
  LITERT_EXPECT_OK(LiteRtSetCpuOptionsXnnPackWeightCachePath(
      cpu_options_, expected_path.data()));
  LITERT_EXPECT_OK(
      LiteRtGetCpuOptionsXnnPackWeightCachePath(cpu_options_, &path));
  ASSERT_EQ(path, expected_path);
}

TEST_F(LiteRtCpuOptionsFieldsTest,
       SetXNNPackWeightCachePathFailsWithInvalidArgument) {
  EXPECT_THAT(LiteRtSetCpuOptionsXnnPackWeightCachePath(/*options=*/nullptr,
                                                        /*path=*/"a/path"),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest,
       GetXNNPackWeightCachePathFailsWithInvalidArgument) {
  const char* path = nullptr;
  EXPECT_THAT(
      LiteRtGetCpuOptionsXnnPackWeightCachePath(/*options=*/nullptr, &path),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LiteRtGetCpuOptionsXnnPackWeightCachePath(cpu_options_, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtCpuOptionsFieldsTest, SetAndGetXNNPackWeightCacheDescriptor) {
  const int expected_fd = 1234;
  int fd;

  // Avoid a no-op test.
  LITERT_EXPECT_OK(
      LiteRtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, &fd));
  ASSERT_NE(fd, expected_fd);

  // Actual test.
  LITERT_EXPECT_OK(LiteRtSetCpuOptionsXnnPackWeightCacheFileDescriptor(
      cpu_options_, expected_fd));
  LITERT_EXPECT_OK(
      LiteRtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, &fd));
  ASSERT_EQ(fd, expected_fd);
}

TEST_F(LiteRtCpuOptionsFieldsTest,
       SetXNNPackWeightCacheFailsIfBothPathAndDescriptorAreSet) {
  const absl::string_view path = "a/path/to/the/cache";
  const int fd = 1234;

  LITERT_EXPECT_OK(
      LiteRtSetCpuOptionsXnnPackWeightCachePath(cpu_options_, path.data()));
  EXPECT_THAT(
      LiteRtSetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options_, fd),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

}  // namespace
