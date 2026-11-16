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

#include "absl/flags/flag.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"
#include "litert/tools/flags/options_parser_registry.h"

ABSL_FLAG(bool, arm_vulkan_ml_enable_just_in_time, false,
          "Whether to enable Just-In-Time compilation in Arm Vulkan ML.");

ABSL_FLAG(bool, arm_vulkan_ml_dump_op_timings, false,
          "Whether to dump detailed compilation and initialization timings in "
          "Arm Vulkan ML.");

namespace litert::arm_vulkan_ml {

Expected<void> UpdateArmVulkanMLOptionsFromFlags(ArmVulkanMLOptions& options) {
  LITERT_RETURN_IF_ERROR(options.SetEnableJustInTime(
      absl::GetFlag(FLAGS_arm_vulkan_ml_enable_just_in_time)));
  LITERT_RETURN_IF_ERROR(options.SetDumpOpTimings(
      absl::GetFlag(FLAGS_arm_vulkan_ml_dump_op_timings)));
  return {};
}

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  if (!absl::GetFlag(FLAGS_arm_vulkan_ml_enable_just_in_time) &&
      !absl::GetFlag(FLAGS_arm_vulkan_ml_dump_op_timings)) {
    return {};
  }
  LITERT_ASSIGN_OR_RETURN(auto& arm_opts,
                          options.GetOptions<ArmVulkanMLOptions>());
  return UpdateArmVulkanMLOptionsFromFlags(arm_opts);
});

}  // namespace litert::arm_vulkan_ml
