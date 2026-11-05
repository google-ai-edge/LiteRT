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

#include <cstdint>
#include <memory>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "ml_drift/syrtis/command_buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;

TEST(SharedVulkanEnvTest, CreateFromExternalHandlesWithNullHandlesReturnsNull) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue queue) {
    vk_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());

  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, VK_NULL_HANDLE, vk_env.GetDevice(), vk_queue,
                /*queue_family_index=*/-1),
            nullptr);
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, vk_env.GetPhysicalDevice(), VK_NULL_HANDLE,
                vk_queue, /*queue_family_index=*/-1),
            nullptr);
  EXPECT_EQ(SharedVulkanEnv::CreateFromExternalHandles(
                VK_NULL_HANDLE, vk_env.GetPhysicalDevice(), vk_env.GetDevice(),
                VK_NULL_HANDLE, /*queue_family_index=*/-1),
            nullptr);
}

TEST(SharedVulkanEnvTest,
     CreateFromExternalHandlesResolvesComputeQueueFamilyIndexAndSubmitsWork) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue queue) {
    vk_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());

  std::unique_ptr<SharedVulkanEnv> shared_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          VK_NULL_HANDLE, vk_env.GetPhysicalDevice(), vk_env.GetDevice(),
          vk_queue, /*queue_family_index=*/-1);
  ASSERT_NE(shared_env, nullptr);
  EXPECT_TRUE(shared_env->EnsureVulkanInitialized());
  EXPECT_EQ(shared_env->vulkan_env().GetPhysicalDevice(),
            vk_env.GetPhysicalDevice());
  EXPECT_EQ(shared_env->vulkan_env().GetDevice(), vk_env.GetDevice());
  EXPECT_EQ(shared_env->vulkan_env().GetQueue().QueueFamilyIndex(),
            vk_env.GetQueue().QueueFamilyIndex());
  EXPECT_NE(shared_env->command_pool(), VK_NULL_HANDLE);

  VkQueue observed_queue = VK_NULL_HANDLE;
  EXPECT_THAT(shared_env->vulkan_env().GetQueue().WithQueue([&](VkQueue queue) {
    observed_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());
  EXPECT_EQ(observed_queue, vk_queue);

  ASSERT_OK_AND_ASSIGN(::ml_drift::syrtis::CommandBuffer * cb,
                       shared_env->GetCommandBuffer());
  ASSERT_NE(cb, nullptr);
  EXPECT_THAT(shared_env->SubmitCommandBuffer(/*wait_for_completion=*/true),
              IsOk());

  // Destroying SharedVulkanEnv must not destroy the caller-owned VkDevice.
  shared_env.reset();
  EXPECT_EQ(::ml_drift::syrtis::vkQueueWaitIdle(vk_queue), VK_SUCCESS);
}

TEST(SharedVulkanEnvTest,
     CreateFromExternalHandlesWithExplicitInstanceAndQueueFamilyIndex) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN/TSAN";
#endif
  ASSERT_EQ(::ml_drift::syrtis::InitVulkan(), 1);
  ::ml_drift::syrtis::Environment vk_env;
  ASSERT_THAT(::ml_drift::syrtis::CreateEnvironment(&vk_env), IsOk());
  VkQueue vk_queue = VK_NULL_HANDLE;
  ASSERT_THAT(vk_env.GetQueue().WithQueue([&](VkQueue queue) {
    vk_queue = queue;
    return absl::OkStatus();
  }),
              IsOk());

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

  const uint32_t expected_queue_family_index =
      vk_env.GetQueue().QueueFamilyIndex();
  std::unique_ptr<SharedVulkanEnv> shared_env =
      SharedVulkanEnv::CreateFromExternalHandles(
          vk_instance, vk_env.GetPhysicalDevice(), vk_env.GetDevice(), vk_queue,
          static_cast<int64_t>(expected_queue_family_index));
  ASSERT_NE(shared_env, nullptr);
  EXPECT_TRUE(shared_env->EnsureVulkanInitialized());
  EXPECT_EQ(shared_env->vulkan_env().GetQueue().QueueFamilyIndex(),
            expected_queue_family_index);
}

}  // namespace
}  // namespace litert::ml_drift
