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

#ifndef LITERT_ARM_VULKAN_ML_DEVICE_CAPABILITIES_H_
#define LITERT_ARM_VULKAN_ML_DEVICE_CAPABILITIES_H_
#include <string>
#include <vector>
#include "litert/vendors/arm_vulkan_ml/dispatch/tensor_capabilities.h"
namespace litert::arm_vulkan_ml {
struct TosaName { std::string name; VkDataGraphTOSAQualityFlagsARM quality; };
struct GraphEngine {
  uint32_t queue_family;
  std::string instruction_set;
  VkDataGraphTOSALevelARM level;
  std::vector<TosaName> profiles, extensions;
};
struct DeviceCapabilities {
  VkPhysicalDeviceProperties properties{};
  bool tensors_feature = false, graph_feature = false, synchronization2 = false;
  bool maintenance5 = false, replicated_composites = false;
  std::vector<VkExtensionProperties> extensions;
  VkPhysicalDeviceTensorPropertiesARM tensors{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM};
  std::vector<std::pair<VkFormat, VkFormatFeatureFlags2>> formats;
  std::vector<GraphEngine> engines;
};
const char* VulkanLibraryName();
bool QueryDeviceCapabilities(PFN_vkGetInstanceProcAddr get, VkInstance instance,
    VkPhysicalDevice physical, DeviceCapabilities& caps, std::string& error);
bool CheckEngineRequirements(const CompiledPartition& artifact, const DeviceCapabilities& device,
    const GraphEngine& engine, bool allow_emulation, std::string& reason);
bool CheckDeviceRequirements(const DeviceCapabilities& device, std::string& reason);
}  // namespace litert::arm_vulkan_ml
#endif
