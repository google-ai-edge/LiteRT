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

#include "litert/vendors/arm_vulkan_ml/dispatch/device_capabilities.h"
#include <algorithm>
#include <cstring>
namespace litert::arm_vulkan_ml {
const char* VulkanLibraryName() {
#if defined(__ANDROID__)
  return "libvulkan.so";
#else
  return "libvulkan.so.1";
#endif
}
bool CheckDeviceRequirements(const DeviceCapabilities& device, std::string& reason) {
  const char* names[] = {VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_DATA_GRAPH_EXTENSION_NAME,
      VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME, VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME};
  for (auto name : names) {
    const uint32_t revision = std::strcmp(name, VK_ARM_TENSORS_EXTENSION_NAME) == 0 ? 2 : 1;
    if (!std::any_of(device.extensions.begin(), device.extensions.end(), [&](const auto& e) {
          return name == std::string(e.extensionName) && e.specVersion >= revision;
        })) { reason = std::string("Missing extension/revision: ") + name; return false; }
  }
  if (!device.tensors_feature || !device.graph_feature || !device.synchronization2 ||
      !device.maintenance5 || !device.replicated_composites) {
    reason = "Required Vulkan graph/tensor/synchronization features unavailable";
    return false;
  }
  return true;
}
bool CheckEngineRequirements(const CompiledPartition& artifact, const DeviceCapabilities& device,
    const GraphEngine& engine, bool allow_emulation, std::string& reason) {
  auto has = [](const auto& names, const std::string& name) {
    return std::any_of(names.begin(), names.end(), [&](const auto& e) {
      return e.name == name && (e.quality & VK_DATA_GRAPH_TOSA_QUALITY_CONFORMANT_ARM);
    });
  };
  if (engine.instruction_set != artifact.instruction_set) {
    reason = "Unsupported instruction set: " + artifact.instruction_set; return false;
  }
  const bool emulator = allow_emulation && device.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU &&
      has(engine.profiles, "Emulation Layer");
  for (const auto& name : artifact.profiles)
    if (!has(engine.profiles, name) && !(emulator && (name == "PRO-INT" || name == "PRO-FP"))) {
      reason = "TOSA profile is not advertised as conformant: " + name; return false;
    }
  for (const auto& name : artifact.extensions)
    if (!has(engine.extensions, name)) {
      reason = "TOSA extension is not advertised as conformant: " + name; return false;
    }
  return true;
}
bool QueryDeviceCapabilities(PFN_vkGetInstanceProcAddr get, VkInstance instance,
    VkPhysicalDevice physical, DeviceCapabilities& caps, std::string& error) {
#define FN(name) auto name = reinterpret_cast<PFN_vk##name>(get(instance, "vk" #name));   if (!name) { error = "Missing vk" #name; return false; }
#define CALL(expr) do { VkResult result = (expr); if (result != VK_SUCCESS) {   error = #expr " returned " + std::to_string(result); return false; } } while (false)
  FN(GetPhysicalDeviceProperties);
  FN(EnumerateDeviceExtensionProperties);
  GetPhysicalDeviceProperties(physical, &caps.properties);
  uint32_t count = 0;
  CALL(EnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
  caps.extensions.resize(count);
  CALL(EnumerateDeviceExtensionProperties(physical, nullptr, &count, caps.extensions.data()));
  auto has = [&](const char* name) { return std::any_of(caps.extensions.begin(), caps.extensions.end(),
      [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
  FN(GetPhysicalDeviceFeatures2);
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceTensorFeaturesARM tensors{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
  VkPhysicalDeviceDataGraphFeaturesARM graph{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
  VkPhysicalDeviceMaintenance5FeaturesKHR maintenance{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
  VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicated{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT};
  void** tail = &features.pNext;
  auto append = [&](auto& value) { *tail = &value; tail = &value.pNext; };
  if (caps.properties.apiVersion >= VK_API_VERSION_1_3) append(f13);
  if (has(VK_ARM_TENSORS_EXTENSION_NAME)) append(tensors);
  if (has(VK_ARM_DATA_GRAPH_EXTENSION_NAME)) append(graph);
  if (has(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) append(maintenance);
  if (has(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME)) append(replicated);
  GetPhysicalDeviceFeatures2(physical, &features);
  caps.tensors_feature = tensors.tensors;
  caps.graph_feature = graph.dataGraph;
  caps.synchronization2 = f13.synchronization2;
  caps.maintenance5 = maintenance.maintenance5;
  caps.replicated_composites = replicated.shaderReplicatedComposites;
  if (has(VK_ARM_TENSORS_EXTENSION_NAME)) {
    FN(GetPhysicalDeviceProperties2);
    FN(GetPhysicalDeviceFormatProperties2);
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &caps.tensors; GetPhysicalDeviceProperties2(physical, &properties);
    for (auto format : {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_SFLOAT}) {
      VkTensorFormatPropertiesARM tensor{VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM};
      VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2}; properties.pNext = &tensor;
      GetPhysicalDeviceFormatProperties2(physical, format, &properties);
      caps.formats.push_back({format, tensor.linearTilingTensorFeatures});
    }
  }
  if (!has(VK_ARM_DATA_GRAPH_EXTENSION_NAME) || !has(VK_ARM_DATA_GRAPH_INSTRUCTION_SET_TOSA_EXTENSION_NAME)) return true;
  FN(GetPhysicalDeviceQueueFamilyProperties);
  FN(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM);
  FN(GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM);
  GetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> queues(count);
  GetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
  for (uint32_t q = 0; q < count; ++q) {
    if (!queues[q].queueCount || !(queues[q].queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM)) continue;
    uint32_t n = 0;
    CALL(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(physical, q, &n, nullptr));
    std::vector<VkQueueFamilyDataGraphPropertiesARM> engines(n, {VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM});
    CALL(GetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(physical, q, &n, engines.data()));
    for (const auto& e : engines) {
      if (e.engine.isForeign || e.engine.type != VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM ||
          e.operation.operationType != VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM) continue;
      if (std::strcmp(e.operation.name, "TOSA.001000.1") != 0) continue;
      VkQueueFamilyDataGraphTOSAPropertiesARM tosa{VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM};
      CALL(GetPhysicalDeviceQueueFamilyDataGraphEngineOperationPropertiesARM(physical, q, &e,
          reinterpret_cast<VkBaseOutStructure*>(&tosa)));
      GraphEngine engine{q, e.operation.name, tosa.level, {}, {}};
      for (uint32_t i = 0; i < tosa.profileCount; ++i) engine.profiles.push_back({tosa.pProfiles[i].name, tosa.pProfiles[i].qualityFlags});
      for (uint32_t i = 0; i < tosa.extensionCount; ++i) engine.extensions.push_back({tosa.pExtensions[i].name, tosa.pExtensions[i].qualityFlags});
      caps.engines.push_back(std::move(engine));
    }
  }
#undef FN
#undef CALL
  return true;
}
}  // namespace litert::arm_vulkan_ml
