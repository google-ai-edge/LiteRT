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

#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "vulkan/vk_platform.h"  // from @arm_vulkan_ml_dep_vulkan_headers
#include "vulkan/vulkan_core.h"  // from @arm_vulkan_ml_dep_vulkan_headers

namespace litert::arm_vulkan_ml {
namespace {
std::string MakeLogSubdir(const char* name) {
  std::string dir = testing::TempDir() + "/" + name;
  mkdir(dir.c_str(), 0700);
  return dir;
}

std::string ReadDirectoryFiles(const std::string& dir) {
  std::string combined;
  DIR* handle = opendir(dir.c_str());
  if (!handle) return combined;
  while (struct dirent* entry = readdir(handle)) {
    if (std::strcmp(entry->d_name, ".") == 0 ||
        std::strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    std::ifstream stream(dir + "/" + entry->d_name);
    combined += std::string((std::istreambuf_iterator<char>(stream)), {});
  }
  closedir(handle);
  return combined;
}

VkDebugUtilsMessengerCreateInfoEXT callback;
bool available = true;
int fail_layers_step = 0;
int fail_extensions_step = 0;
bool omit_extensions_fn = false;
bool omit_required_extension = false;
bool fail_messenger_create = false;
VKAPI_ATTR VkResult VKAPI_CALL Layers(uint32_t* count, VkLayerProperties* out) {
  if (!out && fail_layers_step == 1) return VK_ERROR_INITIALIZATION_FAILED;
  if (out && fail_layers_step == 2) return VK_ERROR_INITIALIZATION_FAILED;
  if (out && available) {
    std::snprintf(out[0].layerName, sizeof(out[0].layerName), "%s",
                  "VK_LAYER_KHRONOS_validation");
    out[0].specVersion = VK_MAKE_API_VERSION(0, 1, 4, 349);
  }
  *count = available ? 1 : 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL Extensions(const char*, uint32_t* count,
                                          VkExtensionProperties* out) {
  if (!out && fail_extensions_step == 1) return VK_ERROR_INITIALIZATION_FAILED;
  if (out && fail_extensions_step == 2) return VK_ERROR_INITIALIZATION_FAILED;
  const uint32_t ext_count = omit_required_extension ? 1 : 2;
  if (out) {
    std::snprintf(out[0].extensionName, sizeof(out[0].extensionName), "%s",
                  VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (!omit_required_extension) {
      std::snprintf(out[1].extensionName, sizeof(out[1].extensionName), "%s",
                    VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    }
  }
  *count = ext_count;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL
Create(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT* info,
       const VkAllocationCallbacks*, VkDebugUtilsMessengerEXT* out) {
  if (fail_messenger_create) return VK_ERROR_INITIALIZATION_FAILED;
  callback = *info;
  if (out) *out = reinterpret_cast<VkDebugUtilsMessengerEXT>(uintptr_t{1});
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL Destroy(VkInstance, VkDebugUtilsMessengerEXT,
                                   const VkAllocationCallbacks*) {}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Get(VkInstance, const char* name) {
#define FN(text, fn) \
  if (!std::strcmp(name, text)) return reinterpret_cast<PFN_vkVoidFunction>(fn)
  FN("vkEnumerateInstanceLayerProperties", Layers);
  if (!omit_extensions_fn) {
    FN("vkEnumerateInstanceExtensionProperties", Extensions);
  }
  FN("vkCreateDebugUtilsMessengerEXT", Create);
  FN("vkDestroyDebugUtilsMessengerEXT", Destroy);
#undef FN
  return nullptr;
}
TEST(VulkanValidation, CapturesMessagesAndSyncConfiguration) {
  const auto dir = MakeLogSubdir("validation-capture");
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "sync", 1);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", dir.c_str(), 1);
  available = true;
  VulkanValidation context;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  std::string error;
  ASSERT_TRUE(context.Prepare(Get, info, error)) << error;
  auto* debug =
      static_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(info.pNext);
  auto* features = static_cast<const VkValidationFeaturesEXT*>(debug->pNext);
  ASSERT_EQ(features->enabledValidationFeatureCount, 1);
  EXPECT_EQ(features->pEnabledValidationFeatures[0],
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
  ASSERT_TRUE(context.Attach(Get, VK_NULL_HANDLE, error));
  VkDebugUtilsMessengerCallbackDataEXT data{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT};
  data.pMessageIdName = "VUID-test-09921";
  data.pMessage = "quoted \"message\"\nline";
  EXPECT_EQ(
      callback.pfnUserCallback(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
                               &data, callback.pUserData),
      VK_FALSE);
  context.Close();
  const std::string text = ReadDirectoryFiles(dir);
  EXPECT_TRUE(absl::StrContains(text, "VUID-test-09921"));
  EXPECT_TRUE(absl::StrContains(text, "\\\"message\\\""));
  EXPECT_TRUE(absl::StrContains(text, "closed"));
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}
TEST(VulkanValidation, ExplicitRequestFailsWhenLayerUnavailable) {
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "core", 1);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", testing::TempDir().c_str(),
         1);
  available = false;
  VulkanValidation context;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  std::string error;
  EXPECT_FALSE(context.Prepare(Get, info, error));
  EXPECT_NE(error.find("unavailable"), std::string::npos);
  available = true;
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}

TEST(VulkanValidation, EscapesControlCharactersValidUtf8AndInvalidUtf8) {
  const auto dir = MakeLogSubdir("validation-utf8");
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "core", 1);
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", dir.c_str(), 1);
  VulkanValidation context;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  std::string error;
  ASSERT_TRUE(context.Prepare(Get, info, error)) << error;
  ASSERT_TRUE(context.Attach(Get, VK_NULL_HANDLE, error)) << error;

  const std::string rich_msg =
      std::string("slash\\bs\bff\fcr\rtab\tctrl\x07") +
      "utf8:\xc3\xa9|\xe2\x82\xac|\xf0\x9f\x9a\x80|" +
      "bad:\x80|\xe0\x80\x80|\xed\xa0\x80|\xf4\x90\x80\x80";
  VkDebugUtilsMessengerCallbackDataEXT data{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT};
  data.pMessageIdName = nullptr;
  data.pMessage = rich_msg.c_str();
  EXPECT_EQ(
      callback.pfnUserCallback(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
                               VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
                               &data, callback.pUserData),
      VK_FALSE);

  // Exceeding 4096 messages marks the log stream as having lost messages.
  data.pMessage = "flood";
  for (int i = 0; i < 4097; ++i) {
    callback.pfnUserCallback(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
                             VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, &data,
                             callback.pUserData);
  }
  context.Close();

  const std::string combined = ReadDirectoryFiles(dir);
  EXPECT_TRUE(absl::StrContains(combined, "slash\\\\bs\\bff\\fcr\\rtab\\t"));
  EXPECT_TRUE(absl::StrContains(combined, "ctrl\\u0007"));
  EXPECT_TRUE(absl::StrContains(
      combined, "utf8:\xc3\xa9|\xe2\x82\xac|\xf0\x9f\x9a\x80"));
  EXPECT_TRUE(absl::StrContains(combined, "bad:\\ufffd"));
  EXPECT_TRUE(absl::StrContains(combined, "\"lost\":true"));

  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}

TEST(VulkanValidation, ReportsPrepareAndAttachConfigurationErrors) {
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  std::string error;

  // Off mode is a no-op and Attach succeeds without calling Vulkan.
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "off", 1);
  {
    VulkanValidation off_ctx;
    EXPECT_TRUE(off_ctx.Prepare(Get, info, error));
    EXPECT_TRUE(off_ctx.Attach(Get, VK_NULL_HANDLE, error));
  }

  // Invalid mode string.
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "bogus", 1);
  {
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Invalid Vulkan validation mode");
  }

  // Missing log directory env var.
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION", "core", 1);
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
  {
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Validation log directory missing");
  }

  // Non-existent log directory.
  const std::string missing_dir = testing::TempDir() + "/missing-log-subdir";
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", missing_dir.c_str(), 1);
  {
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Cannot open Vulkan validation log");
  }

  const std::string valid_dir = MakeLogSubdir("validation-errors");
  setenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR", valid_dir.c_str(), 1);

  for (int step : {1, 2}) {
    fail_layers_step = step;
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Cannot enumerate validation layers");
    fail_layers_step = 0;
  }

  omit_extensions_fn = true;
  {
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Cannot enumerate instance extensions");
  }
  omit_extensions_fn = false;

  for (int step : {1, 2}) {
    fail_extensions_step = step;
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_EQ(error, "Extension enumeration failed");
    fail_extensions_step = 0;
  }

  omit_required_extension = true;
  {
    VulkanValidation ctx;
    EXPECT_FALSE(ctx.Prepare(Get, info, error));
    EXPECT_TRUE(
        absl::StrContains(error, "Required validation extension unavailable"));
  }
  omit_required_extension = false;

  fail_messenger_create = true;
  {
    VulkanValidation ctx;
    VkInstanceCreateInfo local_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ASSERT_TRUE(ctx.Prepare(Get, local_info, error)) << error;
    EXPECT_FALSE(ctx.Attach(Get, VK_NULL_HANDLE, error));
    EXPECT_EQ(error, "Cannot attach validation messenger");
  }
  fail_messenger_create = false;

  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION");
  unsetenv("LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR");
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
