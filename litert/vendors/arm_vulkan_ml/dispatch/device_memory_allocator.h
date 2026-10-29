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

#ifndef LITERT_ARM_VULKAN_ML_DEVICE_MEMORY_ALLOCATOR_H_
#define LITERT_ARM_VULKAN_ML_DEVICE_MEMORY_ALLOCATOR_H_
#include <algorithm>
#include <string>
#include "vulkan/vulkan.h"
namespace litert::arm_vulkan_ml {
// One allocator per owned VkDevice. Imports and owned allocations share the device allocation budget.
class DeviceMemoryAllocator {
 public:
  DeviceMemoryAllocator(VkDevice device, VkPhysicalDeviceMemoryProperties properties,
      uint32_t limit, PFN_vkAllocateMemory allocate)
      : device_(device), properties_(properties), limit_(limit), allocate_(allocate) {}
  bool Allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags,
                VkDeviceMemory& memory, std::string& error, const void* next = nullptr) {
    if (count_ >= limit_) { error = "Device memory allocation count limit exceeded"; return false; }
    for (uint32_t i = 0; i < properties_.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1u << i)) &&
          !(properties_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT) &&
          (properties_.memoryTypes[i].propertyFlags & flags) == flags) {
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = std::max(VkDeviceSize{1}, requirements.size);
        info.memoryTypeIndex = i;
        info.pNext = next;
        auto status = allocate_(device_, &info, nullptr, &memory);
        if (status != VK_SUCCESS) { error = "AllocateMemory returned VkResult " + std::to_string(status); return false; }
        ++count_; return true;
      }
    }
    error = "No compatible Vulkan memory type"; return false;
  }
  void Released() { if(count_)--count_; }
 private:
  VkDevice device_;
  VkPhysicalDeviceMemoryProperties properties_;
  uint32_t limit_, count_ = 0;
  PFN_vkAllocateMemory allocate_;
};
}
#endif
