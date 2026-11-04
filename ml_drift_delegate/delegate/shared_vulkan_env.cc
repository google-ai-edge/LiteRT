// Copyright 2025 Google LLC.
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
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/syrtis/command_buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/queue.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_device.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift

namespace litert::ml_drift {
namespace {

class ExternalVulkanQueue final : public ::ml_drift::syrtis::Queue {
 public:
  ExternalVulkanQueue(VkQueue queue, uint32_t queue_family_index)
      : queue_(queue), queue_family_index_(queue_family_index) {}
  ~ExternalVulkanQueue() final = default;

  uint32_t QueueFamilyIndex() final { return queue_family_index_; }
  absl::Status WithQueue(
      absl::FunctionRef<absl::Status(VkQueue)> action) final {
    return action(queue_);
  }

 private:
  VkQueue queue_;
  uint32_t queue_family_index_;
};

absl::StatusOr<uint32_t> GetComputeQueueFamilyIndex(
    VkPhysicalDevice physical_device) {
  uint32_t count = 0;
  ::ml_drift::syrtis::vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &count, /*pQueueFamilyProperties=*/nullptr);
  std::vector<VkQueueFamilyProperties> queue_family_properties(count);
  ::ml_drift::syrtis::vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &count, queue_family_properties.data());
  for (uint32_t i = 0; i < count; ++i) {
    if (queue_family_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
  }
  return absl::NotFoundError("No Vulkan queue family with compute support.");
}

bool InitVulkanLibrary(VkInstance instance = VK_NULL_HANDLE) {
  if (::ml_drift::syrtis::vkGetInstanceProcAddr == nullptr &&
      ::ml_drift::syrtis::InitVulkan() == 0) {
    ABSL_LOG(ERROR) << "Failed to locate Vulkan library.";
    return false;
  }
  if (instance != VK_NULL_HANDLE) {
    ::ml_drift::syrtis::LoadInstanceFunctions(instance);
  }
  return true;
}

}  // namespace

using ::ml_drift::syrtis::vkDestroyCommandPool;
using ::ml_drift::syrtis::vkQueueWaitIdle;
using ::ml_drift::syrtis::VkResultToStatus;

SharedVulkanEnv::SharedVulkanEnv() = default;

SharedVulkanEnv::~SharedVulkanEnv() {
  pending_command_buffers_.clear();
  current_command_buffer_ = ::ml_drift::syrtis::CommandBuffer();
  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(vulkan_env_.GetDevice(), command_pool_, nullptr);
  }
}

std::unique_ptr<SharedVulkanEnv> SharedVulkanEnv::Create() {
  if (!InitVulkanLibrary()) {
    return nullptr;
  }

  auto env = std::make_unique<SharedVulkanEnv>();
  if (absl::Status status =
          ::ml_drift::syrtis::CreateEnvironment(&env->vulkan_env_);
      !status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create Vulkan environment: " << status;
    return nullptr;
  }

  const VkCommandPoolCreateInfo command_pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = env->vulkan_env_.GetQueue().QueueFamilyIndex()};
  if (VkResult result = ::ml_drift::syrtis::vkCreateCommandPool(
          env->vulkan_env_.GetDevice(), &command_pool_create_info,
          /*pAllocator=*/nullptr, &env->command_pool_);
      result != VK_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to create Vulkan command pool: " << result;
    return nullptr;
  }

  ABSL_LOG(INFO) << "Created a Vulkan environment.";
  return env;
}

std::unique_ptr<SharedVulkanEnv> SharedVulkanEnv::CreateFromExternalHandles(
    VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
    VkQueue queue, int64_t queue_family_index) {
  if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE ||
      queue == VK_NULL_HANDLE) {
    ABSL_LOG(ERROR) << "External Vulkan physical_device, device, and queue "
                       "must be non-null.";
    return nullptr;
  }

  if (!InitVulkanLibrary(instance)) {
    return nullptr;
  }

  auto env = std::make_unique<SharedVulkanEnv>();

  uint32_t resolved_queue_family_index = 0;
  if (queue_family_index >= 0) {
    resolved_queue_family_index = static_cast<uint32_t>(queue_family_index);
  } else {
    absl::StatusOr<uint32_t> compute_queue_family_index =
        GetComputeQueueFamilyIndex(physical_device);
    if (!compute_queue_family_index.ok()) {
      ABSL_LOG(ERROR) << compute_queue_family_index.status();
      return nullptr;
    }
    resolved_queue_family_index = *compute_queue_family_index;
  }

  uint32_t extension_count = 0;
  ::ml_drift::syrtis::vkEnumerateDeviceExtensionProperties(
      physical_device, /*pLayerName=*/nullptr, &extension_count,
      /*pProperties=*/nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  ::ml_drift::syrtis::vkEnumerateDeviceExtensionProperties(
      physical_device, /*pLayerName=*/nullptr, &extension_count,
      extension_properties.data());
  std::vector<std::string> supported_extensions;
  supported_extensions.reserve(extension_count);
  for (const VkExtensionProperties& extension_property : extension_properties) {
    supported_extensions.emplace_back(extension_property.extensionName);
  }

  env->external_queue_ =
      std::make_unique<ExternalVulkanQueue>(queue, resolved_queue_family_index);
  ::ml_drift::syrtis::VulkanDevice vk_device(device, physical_device,
                                             supported_extensions,
                                             /*owns_device=*/false);
  env->vulkan_env_ = ::ml_drift::syrtis::Environment(
      std::move(vk_device), env->external_queue_.get(), VK_NULL_HANDLE);

  const VkCommandPoolCreateInfo command_pool_create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = resolved_queue_family_index};
  if (VkResult result = ::ml_drift::syrtis::vkCreateCommandPool(
          device, &command_pool_create_info, /*pAllocator=*/nullptr,
          &env->command_pool_);
      result != VK_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to create Vulkan command pool: " << result;
    return nullptr;
  }

  ABSL_LOG(INFO) << "Created a Vulkan environment from external handles.";
  return env;
}

absl::StatusOr<::ml_drift::syrtis::CommandBuffer*>
SharedVulkanEnv::GetCommandBuffer(bool new_command_pool) {
  if (!current_command_buffer_) {
    ABSL_ASSIGN_OR_RETURN(
        current_command_buffer_,
        ::ml_drift::syrtis::CreateOneUseCommandBuffer(
            vulkan_env_.GetDevice(), vulkan_env_.GetQueue().QueueFamilyIndex(),
            new_command_pool ? VK_NULL_HANDLE : command_pool_));
  }
  return &current_command_buffer_;
}

absl::Status SharedVulkanEnv::SubmitCommandBuffer(bool wait_for_completion) {
  if (current_command_buffer_) {
    pending_command_buffers_.push_back(std::move(current_command_buffer_));
    ABSL_RETURN_IF_ERROR(vulkan_env_.GetQueue().WithQueue([&](VkQueue queue) {
      return pending_command_buffers_.back().AddToQueue(queue,
                                                        wait_for_completion);
    }));
  } else if (wait_for_completion && !pending_command_buffers_.empty()) {
    ABSL_RETURN_IF_ERROR(vulkan_env_.GetQueue().WithQueue([](VkQueue queue) {
      return VkResultToStatus(vkQueueWaitIdle(queue));
    }));
  }
  return absl::OkStatus();
}

absl::Status SharedVulkanEnv::AddToPendingCommandBuffers(
    std::vector<::ml_drift::syrtis::CommandBuffer> command_buffers) {
  for (auto& command_buffer : command_buffers) {
    pending_command_buffers_.push_back(std::move(command_buffer));
  }
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
