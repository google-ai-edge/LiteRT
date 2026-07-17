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

#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/syrtis/command_buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift

namespace litert::ml_drift {

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

absl::StatusOr<::ml_drift::syrtis::CommandBuffer*>
SharedVulkanEnv::GetCommandBuffer(bool new_command_pool) {
  if (!current_command_buffer_) {
    ASSIGN_OR_RETURN(
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
    RETURN_IF_ERROR(vulkan_env_.GetQueue().WithQueue([&](VkQueue queue) {
      return pending_command_buffers_.back().AddToQueue(queue,
                                                        wait_for_completion);
    }));
  } else if (wait_for_completion && !pending_command_buffers_.empty()) {
    RETURN_IF_ERROR(vulkan_env_.GetQueue().WithQueue([](VkQueue queue) {
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
