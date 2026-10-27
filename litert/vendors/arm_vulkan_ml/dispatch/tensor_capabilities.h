// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_DISPATCH_TENSOR_CAPABILITIES_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_DISPATCH_TENSOR_CAPABILITIES_H_

#include <string>

#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "vulkan/vulkan.h"

namespace litert::arm_vulkan_ml {

VkFormat TensorFormat(TensorElement element);

// Checks the actual linear-tiled data-graph resources before device selection.
// TOSA operator/type legality is separately established by the compiler and the
// engine's profile/extension contract; tensor format support alone is not enough.
bool QueryTensorCapabilities(
    VkPhysicalDevice device, const CompiledPartition& artifact,
    PFN_vkGetPhysicalDeviceProperties2 get_properties,
    PFN_vkGetPhysicalDeviceFormatProperties2 get_format_properties,
    std::string& reason);

}  // namespace litert::arm_vulkan_ml
#endif
