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
//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_ARM_VULKAN_ML_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_ARM_VULKAN_ML_OPTIONS_H_

#include <stdbool.h>

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtArmVulkanMLOptions);

LiteRtStatus LrtCreateArmVulkanMLOptions(LrtArmVulkanMLOptions* options);

LiteRtStatus LrtCreateArmVulkanMLOptionsFromToml(
    const char* toml_payload, LrtArmVulkanMLOptions* options);

void LrtDestroyArmVulkanMLOptions(LrtArmVulkanMLOptions options);

LiteRtStatus LrtGetOpaqueArmVulkanMLOptionsData(
    LrtArmVulkanMLOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*));

const char* LrtArmVulkanMLOptionsGetIdentifier();

LiteRtStatus LrtArmVulkanMLOptionsSetEnableJustInTime(
    LrtArmVulkanMLOptions options, bool enable_just_in_time);

LiteRtStatus LrtArmVulkanMLOptionsGetEnableJustInTime(
    LrtArmVulkanMLOptions options, bool* enable_just_in_time);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_ARM_VULKAN_ML_OPTIONS_H_
