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

#include "litert/c/options/litert_arm_vulkan_ml_options.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::IsNull;
using ::testing::Not;
using ::testing::litert::IsError;

auto CleanupOptionsOnScopeExit(LrtArmVulkanMLOptions options) {
  return absl::Cleanup([options] { LrtDestroyArmVulkanMLOptions(options); });
}

TEST(LiteRtArmVulkanMLOptionsTest, CreateAndNullValidation) {
  EXPECT_THAT(LrtCreateArmVulkanMLOptions(nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtArmVulkanMLOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateArmVulkanMLOptions(&options));
  const auto cleanup = CleanupOptionsOnScopeExit(options);
  ASSERT_THAT(options, Not(IsNull()));

  bool enable_jit = true;
  bool dump_timings = true;
  LITERT_EXPECT_OK(
      LrtArmVulkanMLOptionsGetEnableJustInTime(options, &enable_jit));
  LITERT_EXPECT_OK(
      LrtArmVulkanMLOptionsGetDumpOpTimings(options, &dump_timings));
  EXPECT_FALSE(enable_jit);
  EXPECT_FALSE(dump_timings);

  EXPECT_THAT(LrtArmVulkanMLOptionsSetEnableJustInTime(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtArmVulkanMLOptionsGetEnableJustInTime(nullptr, &enable_jit),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtArmVulkanMLOptionsGetEnableJustInTime(options, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtArmVulkanMLOptionsSetDumpOpTimings(nullptr, true),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtArmVulkanMLOptionsGetDumpOpTimings(nullptr, &dump_timings),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtArmVulkanMLOptionsGetDumpOpTimings(options, nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(LiteRtArmVulkanMLOptionsTest, SetGetAndRoundTripToml) {
  LrtArmVulkanMLOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateArmVulkanMLOptions(&options));
  const auto cleanup = CleanupOptionsOnScopeExit(options);

  LITERT_EXPECT_OK(LrtArmVulkanMLOptionsSetEnableJustInTime(options, true));
  LITERT_EXPECT_OK(LrtArmVulkanMLOptionsSetDumpOpTimings(options, true));

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  EXPECT_THAT(LrtGetOpaqueArmVulkanMLOptionsData(nullptr, &identifier, &payload,
                                                 &payload_deleter),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtGetOpaqueArmVulkanMLOptionsData(options, nullptr, &payload,
                                                 &payload_deleter),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(LrtGetOpaqueArmVulkanMLOptionsData(options, &identifier, nullptr,
                                                 &payload_deleter),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      LrtGetOpaqueArmVulkanMLOptionsData(options, &identifier, &payload,
                                         nullptr),
      IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_ASSERT_OK(LrtGetOpaqueArmVulkanMLOptionsData(
      options, &identifier, &payload, &payload_deleter));
  const auto payload_cleanup =
      absl::Cleanup([payload, payload_deleter] { payload_deleter(payload); });
  EXPECT_STREQ(identifier, LrtArmVulkanMLOptionsGetIdentifier());
  EXPECT_STREQ(static_cast<const char*>(payload),
               "enable_just_in_time = true\n"
               "dump_op_timings = true\n");

  LrtArmVulkanMLOptions parsed = nullptr;
  LITERT_ASSERT_OK(LrtCreateArmVulkanMLOptionsFromToml(
      static_cast<const char*>(payload), &parsed));
  const auto parsed_cleanup = CleanupOptionsOnScopeExit(parsed);

  bool enable_jit = false;
  bool dump_timings = false;
  LITERT_EXPECT_OK(
      LrtArmVulkanMLOptionsGetEnableJustInTime(parsed, &enable_jit));
  LITERT_EXPECT_OK(
      LrtArmVulkanMLOptionsGetDumpOpTimings(parsed, &dump_timings));
  EXPECT_TRUE(enable_jit);
  EXPECT_TRUE(dump_timings);
}

TEST(LiteRtArmVulkanMLOptionsTest, CreateFromTomlHandlesEmptyAndInvalidInput) {
  EXPECT_THAT(LrtCreateArmVulkanMLOptionsFromToml("", nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LrtArmVulkanMLOptions empty_opts = nullptr;
  LITERT_ASSERT_OK(LrtCreateArmVulkanMLOptionsFromToml("", &empty_opts));
  const auto empty_cleanup = CleanupOptionsOnScopeExit(empty_opts);
  bool dump_timings = true;
  LITERT_EXPECT_OK(
      LrtArmVulkanMLOptionsGetDumpOpTimings(empty_opts, &dump_timings));
  EXPECT_FALSE(dump_timings);

  LrtArmVulkanMLOptions invalid_opts = nullptr;
  EXPECT_NE(LrtCreateArmVulkanMLOptionsFromToml("dump_op_timings = not_a_bool",
                                                &invalid_opts),
            kLiteRtStatusOk);
  EXPECT_THAT(invalid_opts, IsNull());
}

TEST(LiteRtArmVulkanMLOptionsTest, CppWrapperRoundTripsOpaqueOptions) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::arm_vulkan_ml::ArmVulkanMLOptions options,
      litert::arm_vulkan_ml::ArmVulkanMLOptions::Create());
  LITERT_EXPECT_OK(options.SetEnableJustInTime(true));
  LITERT_EXPECT_OK(options.SetDumpOpTimings(true));

  LITERT_ASSERT_OK_AND_ASSIGN(bool enable_jit, options.GetEnableJustInTime());
  LITERT_ASSERT_OK_AND_ASSIGN(bool dump_timings, options.GetDumpOpTimings());
  EXPECT_TRUE(enable_jit);
  EXPECT_TRUE(dump_timings);

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(
      options.GetOpaqueOptionsData(&identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_options));
  LiteRtDestroyOpaqueOptions(opaque_options);

  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::arm_vulkan_ml::ArmVulkanMLOptions other_options,
      litert::arm_vulkan_ml::ArmVulkanMLOptions::Create());
  other_options = std::move(options);
  LITERT_ASSERT_OK_AND_ASSIGN(bool moved_jit,
                              other_options.GetEnableJustInTime());
  LITERT_ASSERT_OK_AND_ASSIGN(bool moved_timings,
                              other_options.GetDumpOpTimings());
  EXPECT_TRUE(moved_jit);
  EXPECT_TRUE(moved_timings);
  EXPECT_STREQ(litert::arm_vulkan_ml::ArmVulkanMLOptions::Discriminator(),
               LrtArmVulkanMLOptionsGetIdentifier());
}

}  // namespace
