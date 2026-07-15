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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_VULKAN_ENV_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_VULKAN_ENV_H_

#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/syrtis/command_buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift

namespace litert::ml_drift {

class SharedVulkanEnv {
 public:
  SharedVulkanEnv();
  ~SharedVulkanEnv();

  // Gets the command buffer. If no command buffers are ready for use, creates
  // one.
  // When a new command buffer is created, a new command pool will be created
  // if |new_command_pool| is true.
  // Otherwise, the default command pool, |command_pool_|, will be used.
  absl::StatusOr<::ml_drift::syrtis::CommandBuffer*> GetCommandBuffer(
      bool new_command_pool = false);

  // Submits the current command buffer.
  //
  // If wait_for_completion is false, the function returns immediately after
  // submitting the current command buffer to the queue. The command buffer is
  // kept alive until SubmitCommandBuffer() is called with wait_for_completion
  // true.
  //
  // If wait_for_completion is true, the function will wait for the command
  // buffer to be completed. It returns all pending command buffers as their
  // commands have been executed.
  absl::Status SubmitCommandBuffer(bool wait_for_completion = false);

  // Adds command buffers to pending command buffers. New command buffers must
  // have been submitted to the queue asynchronously.
  absl::Status AddToPendingCommandBuffers(
      std::vector<::ml_drift::syrtis::CommandBuffer> command_buffers);

  ::ml_drift::syrtis::Environment& vulkan_env() { return vulkan_env_; }
  const ::ml_drift::syrtis::Environment& vulkan_env() const {
    return vulkan_env_;
  }

  VkCommandPool& command_pool() { return command_pool_; }
  std::vector<::ml_drift::syrtis::CommandBuffer>& pending_command_buffers() {
    return pending_command_buffers_;
  }

 private:
  ::ml_drift::syrtis::Environment vulkan_env_;
  // A command pool used for every data transfer and compute works.
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  ::ml_drift::syrtis::CommandBuffer current_command_buffer_;
  std::vector<::ml_drift::syrtis::CommandBuffer> pending_command_buffers_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_VULKAN_ENV_H_
