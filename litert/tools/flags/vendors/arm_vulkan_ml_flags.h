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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_ARM_VULKAN_ML_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_ARM_VULKAN_ML_FLAGS_H_

#include "absl/flags/declare.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"

ABSL_DECLARE_FLAG(bool, arm_vulkan_ml_enable_just_in_time);
ABSL_DECLARE_FLAG(bool, arm_vulkan_ml_dump_op_timings);

namespace litert::arm_vulkan_ml {

Expected<void> UpdateArmVulkanMLOptionsFromFlags(ArmVulkanMLOptions& options);

}  // namespace litert::arm_vulkan_ml

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_ARM_VULKAN_ML_FLAGS_H_
