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

#include "litert/tools/flags/vendors/arm_vulkan_ml_flags.h"

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/test/matchers.h"
#include "litert/tools/flags/options_parser_registry.h"

namespace litert::arm_vulkan_ml {
namespace {

TEST(ArmVulkanMLFlagsTest, DefaultValuesLeaveOptionsFalse) {
  LITERT_ASSERT_OK_AND_ASSIGN(ArmVulkanMLOptions options,
                              ArmVulkanMLOptions::Create());
  LITERT_ASSERT_OK(UpdateArmVulkanMLOptionsFromFlags(options));

  LITERT_ASSERT_OK_AND_ASSIGN(bool enable_jit, options.GetEnableJustInTime());
  LITERT_ASSERT_OK_AND_ASSIGN(bool dump_timings, options.GetDumpOpTimings());
  EXPECT_FALSE(enable_jit);
  EXPECT_FALSE(dump_timings);
}

TEST(ArmVulkanMLFlagsTest, UpdateOptionsFromFlagsAppliesBothFlags) {
  absl::SetFlag(&FLAGS_arm_vulkan_ml_enable_just_in_time, true);
  absl::SetFlag(&FLAGS_arm_vulkan_ml_dump_op_timings, true);

  LITERT_ASSERT_OK_AND_ASSIGN(ArmVulkanMLOptions options,
                              ArmVulkanMLOptions::Create());
  LITERT_ASSERT_OK(UpdateArmVulkanMLOptionsFromFlags(options));

  LITERT_ASSERT_OK_AND_ASSIGN(bool enable_jit, options.GetEnableJustInTime());
  LITERT_ASSERT_OK_AND_ASSIGN(bool dump_timings, options.GetDumpOpTimings());
  EXPECT_TRUE(enable_jit);
  EXPECT_TRUE(dump_timings);
}

TEST(ArmVulkanMLFlagsTest, RegisteredParserSkipsWhenUnsetAndAppliesWhenSet) {
  LITERT_ASSERT_OK_AND_ASSIGN(Options unset_options, Options::Create());
  LITERT_ASSERT_OK_AND_ASSIGN(ArmVulkanMLOptions& preconfigured,
                              unset_options.GetOptions<ArmVulkanMLOptions>());
  LITERT_ASSERT_OK(preconfigured.SetEnableJustInTime(true));
  LITERT_ASSERT_OK(
      tools::OptionsParserRegistry::GetInstance().RunAllParsers(unset_options));
  LITERT_ASSERT_OK_AND_ASSIGN(bool preserved_jit,
                              preconfigured.GetEnableJustInTime());
  EXPECT_TRUE(preserved_jit);

  absl::SetFlag(&FLAGS_arm_vulkan_ml_dump_op_timings, true);
  LITERT_ASSERT_OK_AND_ASSIGN(Options set_options, Options::Create());
  LITERT_ASSERT_OK(
      tools::OptionsParserRegistry::GetInstance().RunAllParsers(set_options));
  LITERT_ASSERT_OK_AND_ASSIGN(ArmVulkanMLOptions& arm_opts,
                              set_options.GetOptions<ArmVulkanMLOptions>());
  LITERT_ASSERT_OK_AND_ASSIGN(bool enable_jit, arm_opts.GetEnableJustInTime());
  LITERT_ASSERT_OK_AND_ASSIGN(bool dump_timings, arm_opts.GetDumpOpTimings());
  EXPECT_FALSE(enable_jit);
  EXPECT_TRUE(dump_timings);
}

}  // namespace
}  // namespace litert::arm_vulkan_ml
