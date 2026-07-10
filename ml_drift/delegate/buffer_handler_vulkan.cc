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

#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_vulkan.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/syrtis/buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/memory.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_spatial_tensor.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_wrapper.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_utils.h"
#include "third_party/odml/litert/ml_drift/delegate/shared_vulkan_env.h"

using ::litert::internal::LockState;

namespace {

// A `HwMemoryInfo` implementation for Vulkan Custom Buffer integration.
struct VulkanMemoryInfo : public HwMemoryInfo {
  // MLD Tensor owns a Vulkan memory.
  ::ml_drift::syrtis::VulkanSpatialTensor vk_tensor;
  LiteRtRankedTensorType tensor_type;
  LiteRtTensorBufferType buffer_type;
  size_t packed_bytes;
  void* host_memory;
  // Buffer visible from the host, used for data IO from/to the Vulkan device.
  ::ml_drift::syrtis::StagingBuffer staging_buffer;
  LockState lock_state;
  ::litert::ml_drift::SharedVulkanEnv* vk_env;
};

}  // namespace

LiteRtStatus LiteRtCreateVulkanMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* vulkan_memory_info) {
  int fp16_scale = IsGpuFloat16Memory(buffer_type) ? 2 : 1;
  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  } else if (tensor_desc->GetMemorySizeInBytes() * fp16_scale < packed_bytes) {
    ABSL_LOG(ERROR) << "Too big memory requested: max_size="
                    << tensor_desc->GetMemorySizeInBytes()
                    << " vs requested=" << packed_bytes;
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (device_id == nullptr) {
    ABSL_LOG(ERROR) << "Vulkan environment is missing in parameters";
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto* vk_env = static_cast<::litert::ml_drift::SharedVulkanEnv*>(device_id);

  auto memory_info = std::make_unique<VulkanMemoryInfo>(
      VulkanMemoryInfo{.tensor_type = *tensor_type,
                       .buffer_type = buffer_type,
                       .packed_bytes = packed_bytes,
                       .lock_state = LockState::kUnlocked,
                       .vk_env = vk_env});
  LITERT_RETURN_IF_ERROR(::ml_drift::syrtis::CreateTensor(
      *tensor_desc, &vk_env->vulkan_env(), &memory_info->vk_tensor));

  // Returns `VulkanSpatialTensor*` as the `memory_handle`.
  memory_info->memory_handle = &memory_info->vk_tensor;

  *vulkan_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyVulkanMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<VulkanMemoryInfo*>(hw_memory_info);
  if (memory_info->host_memory != nullptr) {
    litert_aligned_free(memory_info->host_memory);
  }
  delete memory_info;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockVulkanMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<VulkanMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state != LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The Vulkan memory is already unlocked.");
  absl::Cleanup unlock = [&memory_info] {
    memory_info->lock_state = LockState::kUnlocked;
  };

  // Don't upload data if read only.
  if (memory_info->lock_state == LockState::kReadLocked) {
    return kLiteRtStatusOk;
  }

  auto* vk_env = memory_info->vk_env;
  auto& gpu_tensor = memory_info->vk_tensor;

  LITERT_ASSIGN_OR_RETURN(auto* command_buffer, vk_env->GetCommandBuffer());
  if (memory_info->staging_buffer.IsValid() &&
      (memory_info->staging_buffer.GetUsageFlags() &
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
    memory_info->staging_buffer.SetCommandBuffer(command_buffer);
  } else {
    if (memory_info->staging_buffer.IsValid()) {
      memory_info->staging_buffer = ::ml_drift::syrtis::StagingBuffer();
    }
    LITERT_RETURN_IF_ERROR(memory_info->staging_buffer.Initialize(
        vk_env->vulkan_env().GetPhysicalDevice(), command_buffer,
        gpu_tensor.GetMemorySizeInBytes(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  }

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeVulkanBufferPacked) {
    LITERT_RETURN_IF_ERROR(::ml_drift::syrtis::CopyDataToBuffer(
        vk_env->vulkan_env().GetPhysicalDevice(),
        vk_env->vulkan_env().GetDevice(), vk_env->vulkan_env().GetQueue(),
        gpu_tensor.GetBufferHandle(), memory_info->packed_bytes,
        memory_info->host_memory, &memory_info->staging_buffer));
    return kLiteRtStatusOk;
  }

  // TODO: b/426869066 - Use Tensor Converter for better performance.
  auto descriptor_with_data = gpu_tensor.GetDescriptor();
  ::litert::ml_drift::ConvertDataToDescriptor(
      memory_info->host_memory, descriptor_with_data,
      memory_info->tensor_type.element_type);
  auto status = gpu_tensor.UploadDescriptorData(
      descriptor_with_data, vk_env->vulkan_env().GetVulkanDevice(),
      vk_env->vulkan_env().GetQueue(), &memory_info->staging_buffer);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to upload data to tensor: " << status;
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockVulkanMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr) {
  if (hw_memory_info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = reinterpret_cast<VulkanMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state == LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The Vulkan memory is already locked.");

  LockState new_lock_state = litert::internal::ToLockState(mode);

  if (memory_info->host_memory == nullptr) {
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&memory_info->host_memory,
                                 LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 memory_info->packed_bytes);
        rc) {
      ABSL_LOG(ERROR) << "Failed to allocate aligned memory";
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }
  *host_memory_ptr = memory_info->host_memory;

  // Don't download data if write only.
  if (new_lock_state == LockState::kWriteLocked) {
    memory_info->lock_state = new_lock_state;
    return kLiteRtStatusOk;
  }

  auto* vk_env = memory_info->vk_env;
  auto& gpu_tensor = memory_info->vk_tensor;

  LITERT_ASSIGN_OR_RETURN(auto* command_buffer, vk_env->GetCommandBuffer());
  if (memory_info->staging_buffer.IsValid() &&
      (memory_info->staging_buffer.GetUsageFlags() &
           VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
    memory_info->staging_buffer.SetCommandBuffer(command_buffer);
  } else {
    if (memory_info->staging_buffer.IsValid()) {
      memory_info->staging_buffer = ::ml_drift::syrtis::StagingBuffer();
    }
    LITERT_RETURN_IF_ERROR(memory_info->staging_buffer.Initialize(
        vk_env->vulkan_env().GetPhysicalDevice(), command_buffer,
        gpu_tensor.GetMemorySizeInBytes(), VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  }

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeVulkanBufferPacked) {
    LITERT_RETURN_IF_ERROR(::ml_drift::syrtis::CopyDataFromBuffer(
        vk_env->vulkan_env().GetPhysicalDevice(),
        vk_env->vulkan_env().GetDevice(), vk_env->vulkan_env().GetQueue(),
        gpu_tensor.GetBufferHandle(), memory_info->packed_bytes,
        memory_info->host_memory, &memory_info->staging_buffer));
    LITERT_RETURN_IF_ERROR(
        vk_env->SubmitCommandBuffer(/*wait_for_completion=*/true));
    LITERT_RETURN_IF_ERROR(::ml_drift::syrtis::CopyDataFromMemory(
        vk_env->vulkan_env().GetDevice(),
        memory_info->staging_buffer.GetMemory(), memory_info->packed_bytes,
        memory_info->host_memory));
  } else {
    // TODO: b/426869066 - Use Tensor Converter for better performance.
    auto descriptor_with_data = gpu_tensor.GetDescriptor();
    LITERT_RETURN_IF_ERROR(gpu_tensor.ToDescriptor(
        vk_env->vulkan_env().GetVulkanDevice(), vk_env->vulkan_env().GetQueue(),
        &descriptor_with_data, &memory_info->staging_buffer));
    LITERT_RETURN_IF_ERROR(
        vk_env->SubmitCommandBuffer(/*wait_for_completion=*/true));
    LITERT_RETURN_IF_ERROR(::ml_drift::syrtis::CopyDataFromMemory(
        vk_env->vulkan_env().GetDevice(),
        memory_info->staging_buffer.GetMemory(),
        descriptor_with_data.GetMemorySizeInBytes(),
        const_cast<unsigned char*>(descriptor_with_data.GetData().data())));
    ::litert::ml_drift::ConvertDataFromDescriptor(
        descriptor_with_data, memory_info->host_memory,
        memory_info->tensor_type.element_type);
  }

  memory_info->lock_state = new_lock_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtClearVulkanMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = reinterpret_cast<VulkanMemoryInfo*>(hw_memory_info);
  auto* vk_env = memory_info->vk_env;
  LITERT_ASSIGN_OR_RETURN(auto* command_buffer, vk_env->GetCommandBuffer());
  auto& gpu_tensor = memory_info->vk_tensor;
  if (gpu_tensor.GetBufferHandle() != VK_NULL_HANDLE) {
    ::ml_drift::syrtis::vkCmdFillBuffer(
        command_buffer->VkCB(), gpu_tensor.GetBufferHandle(), /*offset=*/0,
        VK_WHOLE_SIZE, /*data=*/0);
  } else if (gpu_tensor.GetImageHandle() != VK_NULL_HANDLE) {
    // Use a temporary buffer to copy zeros as the texture is not configured as
    // a rendering target.
    ::ml_drift::syrtis::StagingBuffer temp_buffer;
    LITERT_RETURN_IF_ERROR(temp_buffer.Initialize(
        vk_env->vulkan_env().GetPhysicalDevice(), command_buffer,
        memory_info->packed_bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
    ::ml_drift::syrtis::vkCmdFillBuffer(command_buffer->VkCB(),
                                        temp_buffer.GetBuffer(), /*offset=*/0,
                                        VK_WHOLE_SIZE, /*data=*/0);
    std::vector<uint64_t> storage_dims =
        gpu_tensor.GetDescriptor().GetStorageDims();
    uint32_t image_width = storage_dims[0];
    uint32_t image_height = storage_dims[1];
    uint32_t image_depth = storage_dims.size() > 2 ? storage_dims[2] : 1;
    VkBufferImageCopy copy_region{
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .layerCount = 1},
        .imageExtent = VkExtent3D{image_width, image_height, image_depth}};
    ::ml_drift::syrtis::vkCmdCopyBufferToImage(
        command_buffer->VkCB(), temp_buffer.GetBuffer(),
        gpu_tensor.GetImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        /*regionCount=*/1, &copy_region);
    // Need to be synchronous to destroy the temporary buffer safely.
    LITERT_RETURN_IF_ERROR(
        vk_env->SubmitCommandBuffer(/*wait_for_completion=*/true));
  } else {
    return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}
