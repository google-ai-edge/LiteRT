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

#ifndef LITERT_ARM_VULKAN_ML_VULKAN_VALIDATION_H_
#define LITERT_ARM_VULKAN_ML_VULKAN_VALIDATION_H_
#include <cstdio>
#include <string>
#include "absl/synchronization/mutex.h"
#include "vulkan/vulkan.h"
namespace litert::arm_vulkan_ml {
// Optional development diagnostics. Each instance owns a separate event stream;
// the CLI aggregates streams after dispatch/plugin teardown, including failures.
class VulkanValidation {
 public:
  ~VulkanValidation();
  bool Prepare(PFN_vkGetInstanceProcAddr get, VkInstanceCreateInfo& info, std::string& error);
  bool Attach(PFN_vkGetInstanceProcAddr get, VkInstance instance, std::string& error);
  void Close();
 private:
  static VKAPI_ATTR VkBool32 VKAPI_CALL Callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
      VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);
  absl::Mutex mutex_;
  FILE* file_ = nullptr;
  size_t messages_ = 0;
  bool enabled_ = false, attached_ = false, lost_ = false;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_ = nullptr;
  const char* layer_ = "VK_LAYER_KHRONOS_validation";
  const char* extensions_[2] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};
  VkValidationFeatureEnableEXT sync_ = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
  VkValidationFeaturesEXT features_{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  VkDebugUtilsMessengerCreateInfoEXT debug_{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
};
}
#endif
