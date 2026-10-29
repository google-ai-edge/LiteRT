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
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>
#define JSON_NOEXCEPTION 1
#include "nlohmann/json.hpp"
namespace litert::arm_vulkan_ml {
using Json = nlohmann::json;
VulkanValidation::~VulkanValidation() { Close(); }
bool VulkanValidation::Prepare(PFN_vkGetInstanceProcAddr get, VkInstanceCreateInfo& info, std::string& error) {
  const char* mode = std::getenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  if (!mode || !std::strcmp(mode,"off")) return true;
  if (std::strcmp(mode,"core") && std::strcmp(mode,"sync")) { error="Invalid Vulkan validation mode"; return false; }
  const char* directory=std::getenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
  if(!directory) { error="Validation log directory missing";return false; }
  std::string pattern=std::string(directory)+"/instance-XXXXXX";
  std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
  int fd=mkstemp(name.data());
  if(fd<0 || !(file_=fdopen(fd,"w"))) { if(fd>=0)close(fd);error="Cannot open Vulkan validation log";return false; }
  auto enumerate=reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(get(nullptr,"vkEnumerateInstanceLayerProperties"));
  uint32_t count=0;
  if(!enumerate || enumerate(&count,nullptr)!=VK_SUCCESS) { error="Cannot enumerate validation layers";return false; }
  std::vector<VkLayerProperties> layers(count);
  if(enumerate(&count,layers.data())!=VK_SUCCESS) { error="Cannot enumerate validation layers";return false; }
  const VkLayerProperties* layer=nullptr;
  for(const auto& candidate:layers)if(!std::strcmp(candidate.layerName,layer_))layer=&candidate;
  if(!layer) { error="Requested VK_LAYER_KHRONOS_validation unavailable";return false; }
  auto extension_properties=reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(get(nullptr,"vkEnumerateInstanceExtensionProperties"));
  if(!extension_properties) { error="Cannot enumerate instance extensions";return false; }
  std::vector<VkExtensionProperties> available;
  for(const char* owner:{static_cast<const char*>(nullptr),layer_}) {
    count=0;if(extension_properties(owner,&count,nullptr)!=VK_SUCCESS){error="Extension enumeration failed";return false;}
    std::vector<VkExtensionProperties> list(count);
    if(extension_properties(owner,&count,list.data())!=VK_SUCCESS){error="Extension enumeration failed";return false;}
    available.insert(available.end(),list.begin(),list.end());
  }
  for(const char* required:extensions_) {
    bool found=false;for(const auto& e:available)found|=!std::strcmp(e.extensionName,required);
    if(!found){error=std::string("Required validation extension unavailable: ")+required;return false;}
  }
  Json event={{"event","instance"},{"mode",mode},{"layer",layer_},{"api_version",layer->specVersion},
      {"implementation_version",layer->implementationVersion}};
  auto line=event.dump();if(std::fprintf(file_,"%s\n",line.c_str())<0)lost_=true;
  debug_.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debug_.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debug_.pfnUserCallback=Callback;debug_.pUserData=this;
  features_.enabledValidationFeatureCount=!std::strcmp(mode,"sync")?1:0;
  features_.pEnabledValidationFeatures=&sync_;features_.pNext=&debug_;
  info.enabledLayerCount=1;info.ppEnabledLayerNames=&layer_;
  info.enabledExtensionCount=2;info.ppEnabledExtensionNames=extensions_;
  features_.pNext=info.pNext;debug_.pNext=&features_;info.pNext=&debug_;
  enabled_=true;return true;
}
bool VulkanValidation::Attach(PFN_vkGetInstanceProcAddr get,VkInstance instance,std::string& error) {
  if(!enabled_)return true;
  auto create=reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(get(instance,"vkCreateDebugUtilsMessengerEXT"));
  destroy_=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(get(instance,"vkDestroyDebugUtilsMessengerEXT"));
  debug_.pNext=nullptr;
  if(!create || !destroy_ || create(instance,&debug_,nullptr,&messenger_)!=VK_SUCCESS){error="Cannot attach validation messenger";return false;}
  instance_=instance;attached_=true;return true;
}
VkBool32 VulkanValidation::Callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,const VkDebugUtilsMessengerCallbackDataEXT* data,void* user) {
  auto& self=*static_cast<VulkanValidation*>(user);absl::MutexLock lock(&self.mutex_);
  if(!self.file_ || self.messages_++>=4096){self.lost_=true;return VK_FALSE;}
  Json event={{"event","message"},{"severity",severity},{"type",type},
      {"id",data->pMessageIdName?data->pMessageIdName:""},{"message",data->pMessage?data->pMessage:""}};
  auto line=event.dump(-1,' ',false,Json::error_handler_t::replace);
  if(std::fprintf(self.file_,"%s\n",line.c_str())<0 || std::fflush(self.file_)!=0)self.lost_=true;
  return VK_FALSE;
}
void VulkanValidation::Close() {
  if(messenger_ && destroy_)destroy_(instance_,messenger_,nullptr);
  messenger_=VK_NULL_HANDLE;
  absl::MutexLock lock(&mutex_);
  if(file_) {
    Json event={{"event","closed"},{"enabled",enabled_},{"attached",attached_},{"lost",lost_}};
    auto line=event.dump();std::fprintf(file_,"%s\n",line.c_str());std::fclose(file_);file_=nullptr;
  }
}
}
