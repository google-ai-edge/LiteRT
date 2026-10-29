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
#include <filesystem>
#include <fstream>
#include "gtest/gtest.h"
namespace litert::arm_vulkan_ml {
namespace {
VkDebugUtilsMessengerCreateInfoEXT callback;
bool available=true;
VKAPI_ATTR VkResult VKAPI_CALL Layers(uint32_t* count,VkLayerProperties* out) {
  if(out && available){std::strcpy(out[0].layerName,"VK_LAYER_KHRONOS_validation");out[0].specVersion=VK_MAKE_API_VERSION(0,1,4,349);}
  *count=available?1:0;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL Extensions(const char*,uint32_t* count,VkExtensionProperties* out) {
  if(out){std::strcpy(out[0].extensionName,VK_EXT_DEBUG_UTILS_EXTENSION_NAME);std::strcpy(out[1].extensionName,VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);}
  *count=2;return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL Create(VkInstance,const VkDebugUtilsMessengerCreateInfoEXT* info,const VkAllocationCallbacks*,VkDebugUtilsMessengerEXT*) {
  callback=*info;return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL Destroy(VkInstance,VkDebugUtilsMessengerEXT,const VkAllocationCallbacks*) {}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Get(VkInstance,const char* name) {
#define FN(text, fn) if(!std::strcmp(name,text))return reinterpret_cast<PFN_vkVoidFunction>(fn)
  FN("vkEnumerateInstanceLayerProperties",Layers);
  FN("vkEnumerateInstanceExtensionProperties",Extensions);
  FN("vkCreateDebugUtilsMessengerEXT",Create);
  FN("vkDestroyDebugUtilsMessengerEXT",Destroy);
#undef FN
  return nullptr;
}
TEST(VulkanValidation, CapturesMessagesAndSyncConfiguration) {
  const auto dir=testing::TempDir()+"/validation-capture";
  std::filesystem::create_directories(dir);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION","sync",1);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR",dir.c_str(),1);
  available=true;
  VulkanValidation context; VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};std::string error;
  ASSERT_TRUE(context.Prepare(Get,info,error))<<error;
  auto* debug=static_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(info.pNext);
  auto* features=static_cast<const VkValidationFeaturesEXT*>(debug->pNext);
  ASSERT_EQ(features->enabledValidationFeatureCount,1);
  EXPECT_EQ(features->pEnabledValidationFeatures[0],VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
  ASSERT_TRUE(context.Attach(Get,VK_NULL_HANDLE,error));
  VkDebugUtilsMessengerCallbackDataEXT data{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT};
  data.pMessageIdName="VUID-test-09921";data.pMessage="quoted \"message\"\nline";
  EXPECT_EQ(callback.pfnUserCallback(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,&data,callback.pUserData),VK_FALSE);
  context.Close();
  bool found=false;
  for(const auto& file:std::filesystem::directory_iterator(dir)) {
    std::ifstream stream(file.path());std::string text((std::istreambuf_iterator<char>(stream)),{});
    found|=text.find("VUID-test-09921")!=std::string::npos && text.find("\\\"message\\\"")!=std::string::npos && text.find("closed")!=std::string::npos;
  }
  EXPECT_TRUE(found);
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}
TEST(VulkanValidation, ExplicitRequestFailsWhenLayerUnavailable) {
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION","core",1);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR",testing::TempDir().c_str(),1);
  available=false;VulkanValidation context;VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};std::string error;
  EXPECT_FALSE(context.Prepare(Get,info,error));EXPECT_NE(error.find("unavailable"),std::string::npos);
  available=true;unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}
}
}
