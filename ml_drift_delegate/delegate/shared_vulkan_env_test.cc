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

#include "ml_drift_delegate/delegate/shared_vulkan_env.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "vulkan/vk_platform.h"  // from @arm_vulkan_ml_dep_vulkan_headers

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Each;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::NotNull;

const VkQueueFamilyProperties* g_fake_queue_families = nullptr;
uint32_t g_fake_queue_family_count = 0;

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice /*physical_device*/, uint32_t* count,
    VkQueueFamilyProperties* properties) {
  if (properties == nullptr) {
    *count = g_fake_queue_family_count;
    return;
  }
  const uint32_t to_copy = std::min(*count, g_fake_queue_family_count);
  std::copy_n(g_fake_queue_families, to_copy, properties);
  *count = to_copy;
}

uint32_t g_valid_device_queue_family_index = 0;
uint32_t g_last_command_pool_queue_family_index = 0;
PFN_vkCreateCommandPool g_real_create_command_pool = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL InterceptCreateCommandPool(
    VkDevice device, const VkCommandPoolCreateInfo* create_info,
    const VkAllocationCallbacks* allocator, VkCommandPool* command_pool) {
  g_last_command_pool_queue_family_index = create_info->queueFamilyIndex;
  VkCommandPoolCreateInfo valid_create_info = *create_info;
  valid_create_info.queueFamilyIndex = g_valid_device_queue_family_index;
  return g_real_create_command_pool(device, &valid_create_info, allocator,
                                    command_pool);
}

class SharedVulkanEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
    GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
    ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
    ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env_), IsOk());
    ASSERT_THAT(vk_env_.GetQueue().WithQueue([&](VkQueue queue) {
      vk_queue_ = queue;
      return absl::OkStatus();
    }),
                IsOk());

    saved_get_instance_proc_addr_ = ::ml_drift::syrtis::vkGetInstanceProcAddr;
    saved_get_queue_families_ =
        ::ml_drift::syrtis::vkGetPhysicalDeviceQueueFamilyProperties;
    saved_create_command_pool_ = ::ml_drift::syrtis::vkCreateCommandPool;
    g_real_create_command_pool = saved_create_command_pool_;
    g_valid_device_queue_family_index = vk_env_.GetQueue().QueueFamilyIndex();
    g_last_command_pool_queue_family_index = 0;
    ::ml_drift::syrtis::vkCreateCommandPool = &InterceptCreateCommandPool;
  }

  void TearDown() override {
    if (::ml_drift::syrtis::vkGetInstanceProcAddr == nullptr &&
        saved_get_instance_proc_addr_ != nullptr) {
      ::ml_drift::syrtis::vkGetInstanceProcAddr = saved_get_instance_proc_addr_;
    }
    if (saved_get_queue_families_ != nullptr) {
      ::ml_drift::syrtis::vkGetPhysicalDeviceQueueFamilyProperties =
          saved_get_queue_families_;
    }
    if (saved_create_command_pool_ != nullptr) {
      ::ml_drift::syrtis::vkCreateCommandPool = saved_create_command_pool_;
    }
    g_fake_queue_families = nullptr;
    g_fake_queue_family_count = 0;
    g_real_create_command_pool = nullptr;
  }

  ::ml_drift::syrtis::Environment vk_env_;
  VkQueue vk_queue_ = VK_NULL_HANDLE;
  PFN_vkGetInstanceProcAddr saved_get_instance_proc_addr_ = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties saved_get_queue_families_ =
      nullptr;
  PFN_vkCreateCommandPool saved_create_command_pool_ = nullptr;
};

TEST_F(SharedVulkanEnvTest, CreateInitializesVulkanAndSubmitsWork) {
  ::ml_drift::syrtis::vkGetInstanceProcAddr = nullptr;
  std::unique_ptr<SharedVulkanEnv> shared_env = SharedVulkanEnv::Create();
  ASSERT_NE(shared_env, nullptr);
  EXPECT_NE(::ml_drift::syrtis::vkGetInstanceProcAddr, nullptr);
  EXPECT_NE(shared_env->vulkan_env().GetPhysicalDevice(), VK_NULL_HANDLE);
  EXPECT_NE(shared_env->vulkan_env().GetDevice(), VK_NULL_HANDLE);
  EXPECT_NE(shared_env->command_pool(), VK_NULL_HANDLE);

  ASSERT_THAT(shared_env->GetCommandBuffer(), IsOkAndHolds(NotNull()));
  EXPECT_THAT(shared_env->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());
}

TEST_F(SharedVulkanEnvTest,
       CreateFromExternalHandlesWithNullRequiredHandlesReturnsNull) {
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, VK_NULL_HANDLE, vk_env_.GetDevice(), vk_queue_),
            nullptr);
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(), VK_NULL_HANDLE,
                vk_queue_),
            nullptr);
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(),
                vk_env_.GetDevice(), VK_NULL_HANDLE),
            nullptr);
}

TEST_F(SharedVulkanEnvTest,
       CreateFromExternalHandlesResolvesComputeQueueFamilyIndexAndSubmitsWork) {
  ::ml_drift::syrtis::vkGetInstanceProcAddr = nullptr;
  std::unique_ptr<SharedVulkanEnv> shared_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(), vk_env_.GetDevice(),
          vk_queue_);
  ASSERT_NE(shared_env, nullptr);
  EXPECT_NE(::ml_drift::syrtis::vkGetInstanceProcAddr, nullptr);

  EXPECT_EQ(shared_env->vulkan_env().GetPhysicalDevice(),
            vk_env_.GetPhysicalDevice());
  EXPECT_EQ(shared_env->vulkan_env().GetDevice(), vk_env_.GetDevice());
  EXPECT_EQ(shared_env->vulkan_env().GetQueue().QueueFamilyIndex(),
            vk_env_.GetQueue().QueueFamilyIndex());
  EXPECT_NE(shared_env->command_pool(), VK_NULL_HANDLE);

  uint32_t expected_extension_count = 0;
  ::ml_drift::syrtis::vkEnumerateDeviceExtensionProperties(
      vk_env_.GetPhysicalDevice(), /*pLayerName=*/nullptr,
      &expected_extension_count, /*pProperties=*/nullptr);
  EXPECT_GT(expected_extension_count, 0u);
  EXPECT_GE(shared_env->vulkan_env().GetInfo().vulkan_info.extensions.size(),
            expected_extension_count);
  EXPECT_THAT(shared_env->vulkan_env().GetInfo().vulkan_info.extensions,
              Each(Not(IsEmpty())));

  VkQueue observed_queue = VK_NULL_HANDLE;
  EXPECT_THAT(shared_env->vulkan_env().GetQueue().WithQueue([&](VkQueue queue) {
    observed_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());
  EXPECT_EQ(observed_queue, vk_queue_);

  ASSERT_THAT(shared_env->GetCommandBuffer(), IsOkAndHolds(NotNull()));
  EXPECT_THAT(shared_env->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());

  // Destroying SharedVulkanEnv must not destroy the caller-owned VkDevice.
  shared_env.reset();
  EXPECT_EQ(::ml_drift::syrtis::vkQueueWaitIdle(vk_queue_), VK_SUCCESS);
}

TEST_F(SharedVulkanEnvTest,
       CreateFromExternalHandlesWithExplicitInstanceAndQueueFamilyIndex) {
  const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "SharedVulkanEnvTest",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_MAKE_VERSION(1, 1, 0),
  };
  const VkInstanceCreateInfo instance_create_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
  };
  VkInstance vk_instance = VK_NULL_HANDLE;
  ASSERT_EQ(::ml_drift::syrtis::vkCreateInstance(&instance_create_info, nullptr,
                                                 &vk_instance),
            VK_SUCCESS);
  absl::Cleanup instance_cleanup = [&]() {
    if (vk_instance != VK_NULL_HANDLE) {
      ::ml_drift::syrtis::vkDestroyInstance(vk_instance, nullptr);
    }
  };

  std::unique_ptr<SharedVulkanEnv> shared_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          vk_instance, vk_env_.GetPhysicalDevice(), vk_env_.GetDevice(),
          vk_queue_, /*queue_family_index=*/1);
  ASSERT_NE(shared_env, nullptr);
  EXPECT_EQ(shared_env->vulkan_env().GetQueue().QueueFamilyIndex(), 1u);
  EXPECT_EQ(g_last_command_pool_queue_family_index, 1u);
}

TEST_F(SharedVulkanEnvTest,
       CreateFromExternalHandlesDiscoversComputeQueueFamilyFromProperties) {
  ::ml_drift::syrtis::vkGetPhysicalDeviceQueueFamilyProperties =
      &FakeGetPhysicalDeviceQueueFamilyProperties;

  const VkQueueFamilyProperties transfer_only[] = {
      VkQueueFamilyProperties{.queueFlags = VK_QUEUE_TRANSFER_BIT,
                              .queueCount = 1},
  };
  g_fake_queue_families = transfer_only;
  g_fake_queue_family_count = 1;
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(),
                vk_env_.GetDevice(), vk_queue_),
            nullptr);

  const VkQueueFamilyProperties transfer_then_compute[] = {
      VkQueueFamilyProperties{.queueFlags = VK_QUEUE_TRANSFER_BIT,
                              .queueCount = 1},
      VkQueueFamilyProperties{.queueFlags = VK_QUEUE_COMPUTE_BIT,
                              .queueCount = 1},
  };
  g_fake_queue_families = transfer_then_compute;
  g_fake_queue_family_count = 2;
  std::unique_ptr<SharedVulkanEnv> auto_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(), vk_env_.GetDevice(),
          vk_queue_);
  ASSERT_NE(auto_env, nullptr);
  EXPECT_EQ(auto_env->vulkan_env().GetQueue().QueueFamilyIndex(), 1u);
  EXPECT_EQ(g_last_command_pool_queue_family_index, 1u);

  std::unique_ptr<SharedVulkanEnv> explicit_zero_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          VK_NULL_HANDLE, vk_env_.GetPhysicalDevice(), vk_env_.GetDevice(),
          vk_queue_, /*queue_family_index=*/0);
  ASSERT_NE(explicit_zero_env, nullptr);
  EXPECT_EQ(explicit_zero_env->vulkan_env().GetQueue().QueueFamilyIndex(), 0u);
  EXPECT_EQ(g_last_command_pool_queue_family_index, 0u);
}

}  // namespace
}  // namespace litert::ml_drift
