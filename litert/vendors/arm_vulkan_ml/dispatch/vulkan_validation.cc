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

#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_validation.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers
#include "absl/synchronization/mutex.h"  // from @com_google_absl
namespace litert::arm_vulkan_ml {
namespace {
std::string EscapeJsonString(const char* raw) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out = "\"";
  const auto* s = reinterpret_cast<const unsigned char*>(raw ? raw : "");
  for (size_t i = 0; s[i] != 0;) {
    const unsigned char c = s[i];
    if (c == '"') {
      out += "\\\"";
      ++i;
    } else if (c == '\\') {
      out += "\\\\";
      ++i;
    } else if (c == '\b') {
      out += "\\b";
      ++i;
    } else if (c == '\f') {
      out += "\\f";
      ++i;
    } else if (c == '\n') {
      out += "\\n";
      ++i;
    } else if (c == '\r') {
      out += "\\r";
      ++i;
    } else if (c == '\t') {
      out += "\\t";
      ++i;
    } else if (c < 0x20) {
      out += {'\\', 'u', '0', '0', kHex[c >> 4], kHex[c & 0xf]};
      ++i;
    } else if (c < 0x80) {
      out.push_back(static_cast<char>(c));
      ++i;
    } else {
      auto cont = [&](size_t off) {
        return s[i + off] >= 0x80 && s[i + off] <= 0xbf;
      };
      size_t len = 0;
      if (c >= 0xc2 && c <= 0xdf && cont(1)) {
        len = 2;
      } else if (c >= 0xe0 && c <= 0xef && cont(1) && cont(2) &&
                 !(c == 0xe0 && s[i + 1] < 0xa0) &&
                 !(c == 0xed && s[i + 1] >= 0xa0)) {
        len = 3;
      } else if (c >= 0xf0 && c <= 0xf4 && cont(1) && cont(2) && cont(3) &&
                 !(c == 0xf0 && s[i + 1] < 0x90) &&
                 !(c == 0xf4 && s[i + 1] >= 0x90)) {
        len = 4;
      }
      if (len > 0) {
        out.append(reinterpret_cast<const char*>(s + i), len);
        i += len;
      } else {
        out += "\\ufffd";
        ++i;
      }
    }
  }
  out.push_back('"');
  return out;
}
}  // namespace
VulkanValidation::~VulkanValidation() { Close(); }
bool VulkanValidation::Prepare(PFN_vkGetInstanceProcAddr get,
                               VkInstanceCreateInfo& info, std::string& error) {
  const char* mode = std::getenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  if (!mode || !std::strcmp(mode, "off")) return true;
  if (std::strcmp(mode, "core") && std::strcmp(mode, "sync")) {
    error = "Invalid Vulkan validation mode";
    return false;
  }
  const char* directory =
      std::getenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
  if (!directory) {
    error = "Validation log directory missing";
    return false;
  }
  std::string pattern = std::string(directory) + "/instance-XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  int fd = mkstemp(name.data());
  if (fd < 0 || !(file_ = fdopen(fd, "w"))) {
    if (fd >= 0) close(fd);
    error = "Cannot open Vulkan validation log";
    return false;
  }
  auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
      get(nullptr, "vkEnumerateInstanceLayerProperties"));
  uint32_t count = 0;
  if (!enumerate || enumerate(&count, nullptr) != VK_SUCCESS) {
    error = "Cannot enumerate validation layers";
    return false;
  }
  std::vector<VkLayerProperties> layers(count);
  if (enumerate(&count, layers.data()) != VK_SUCCESS) {
    error = "Cannot enumerate validation layers";
    return false;
  }
  const VkLayerProperties* layer = nullptr;
  for (const auto& candidate : layers)
    if (!std::strcmp(candidate.layerName, layer_)) layer = &candidate;
  if (!layer) {
    error = "Requested VK_LAYER_KHRONOS_validation unavailable";
    return false;
  }
  auto extension_properties =
      reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
          get(nullptr, "vkEnumerateInstanceExtensionProperties"));
  if (!extension_properties) {
    error = "Cannot enumerate instance extensions";
    return false;
  }
  std::vector<VkExtensionProperties> available;
  for (const char* owner : {static_cast<const char*>(nullptr), layer_}) {
    count = 0;
    if (extension_properties(owner, &count, nullptr) != VK_SUCCESS) {
      error = "Extension enumeration failed";
      return false;
    }
    std::vector<VkExtensionProperties> list(count);
    if (extension_properties(owner, &count, list.data()) != VK_SUCCESS) {
      error = "Extension enumeration failed";
      return false;
    }
    available.insert(available.end(), list.begin(), list.end());
  }
  for (const char* required : extensions_) {
    bool found = false;
    for (const auto& e : available)
      found |= !std::strcmp(e.extensionName, required);
    if (!found) {
      error =
          std::string("Required validation extension unavailable: ") + required;
      return false;
    }
  }
  std::string line = "{\"event\":\"instance\",\"mode\":" +
                     EscapeJsonString(mode) +
                     ",\"layer\":" + EscapeJsonString(layer_) +
                     ",\"api_version\":" + std::to_string(layer->specVersion) +
                     ",\"implementation_version\":" +
                     std::to_string(layer->implementationVersion) + "}";
  if (std::fprintf(file_, "%s\n", line.c_str()) < 0) lost_ = true;
  debug_.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debug_.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debug_.pfnUserCallback = Callback;
  debug_.pUserData = this;
  features_.enabledValidationFeatureCount = !std::strcmp(mode, "sync") ? 1 : 0;
  features_.pEnabledValidationFeatures = &sync_;
  features_.pNext = &debug_;
  info.enabledLayerCount = 1;
  info.ppEnabledLayerNames = &layer_;
  info.enabledExtensionCount = 2;
  info.ppEnabledExtensionNames = extensions_;
  features_.pNext = info.pNext;
  debug_.pNext = &features_;
  info.pNext = &debug_;
  enabled_ = true;
  return true;
}
bool VulkanValidation::Attach(PFN_vkGetInstanceProcAddr get,
                              VkInstance instance, std::string& error) {
  if (!enabled_) return true;
  auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      get(instance, "vkCreateDebugUtilsMessengerEXT"));
  destroy_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      get(instance, "vkDestroyDebugUtilsMessengerEXT"));
  debug_.pNext = nullptr;
  if (!create || !destroy_ ||
      create(instance, &debug_, nullptr, &messenger_) != VK_SUCCESS) {
    error = "Cannot attach validation messenger";
    return false;
  }
  instance_ = instance;
  attached_ = true;
  return true;
}
VkBool32 VulkanValidation::Callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
  auto& self = *static_cast<VulkanValidation*>(user);
  absl::MutexLock lock(&self.mutex_);
  if (!self.file_ || self.messages_++ >= 4096) {
    self.lost_ = true;
    return VK_FALSE;
  }
  std::string line =
      "{\"event\":\"message\",\"severity\":" +
      std::to_string(static_cast<uint32_t>(severity)) +
      ",\"type\":" + std::to_string(type) + ",\"id\":" +
      EscapeJsonString(data->pMessageIdName ? data->pMessageIdName : "") +
      ",\"message\":" +
      EscapeJsonString(data->pMessage ? data->pMessage : "") + "}";
  if (std::fprintf(self.file_, "%s\n", line.c_str()) < 0 ||
      std::fflush(self.file_) != 0)
    self.lost_ = true;
  return VK_FALSE;
}
void VulkanValidation::Close() {
  if (messenger_ && destroy_) destroy_(instance_, messenger_, nullptr);
  messenger_ = VK_NULL_HANDLE;
  absl::MutexLock lock(&mutex_);
  if (file_) {
    std::string line = std::string("{\"event\":\"closed\",\"enabled\":") +
                       (enabled_ ? "true" : "false") + ",\"attached\":" +
                       (attached_ ? "true" : "false") + ",\"lost\":" +
                       (lost_ ? "true" : "false") + "}";
    std::fprintf(file_, "%s\n", line.c_str());
    std::fclose(file_);
    file_ = nullptr;
  }
}
}  // namespace litert::arm_vulkan_ml
