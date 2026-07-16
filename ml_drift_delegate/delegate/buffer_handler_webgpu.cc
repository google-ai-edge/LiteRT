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

#include "ml_drift_delegate/delegate/buffer_handler_webgpu.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_tensor_buffer_utils.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "ml_drift_delegate/delegate/buffer_handler_utils.h"

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>  // IWYU pragma: export
#else
#include "webgpu/webgpu.h"  // from @dawn
#endif  // __EMSCRIPTEN__

using ::litert::internal::LockState;

namespace {

// A `HwMemoryInfo` implementation for WebGPU Custom Buffer integration.
struct WebGpuMemoryInfo : public HwMemoryInfo {
  // MLD Tensor owns a GPU memory.
  ::ml_drift::webgpu::SpatialTensor wgpu_tensor;
  bool owns_tensor;
  LiteRtRankedTensorType tensor_type;
  LiteRtTensorBufferType buffer_type;
  size_t packed_bytes;
  void* host_memory;
  LockState lock_state;
  WGPUDevice device;
  WGPUQueue queue;
};

}  // namespace

LiteRtStatus LiteRtCreateWebGpuMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* webgpu_memory_info) {
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
    return kLiteRtStatusErrorInvalidArgument;
  }
  WGPUDevice wgpu_device = static_cast<WGPUDevice>(device_id);
  wgpu::Device device(wgpu_device);
  auto memory_info = std::make_unique<WebGpuMemoryInfo>(
      WebGpuMemoryInfo{.wgpu_tensor = {},
                       .owns_tensor = true,
                       .tensor_type = *tensor_type,
                       .buffer_type = buffer_type,
                       .packed_bytes = packed_bytes,
                       .host_memory = nullptr,
                       .lock_state = LockState::kUnlocked,
                       .device = wgpu_device,
                       .queue = static_cast<WGPUQueue>(queue_id)});
  LITERT_RETURN_IF_ERROR(::ml_drift::webgpu::CreateTensor(
      device, *tensor_desc, &memory_info->wgpu_tensor));

  // Returns `SpatialTensor*` as the `memory_handle`.
  memory_info->memory_handle = &memory_info->wgpu_tensor;

  *webgpu_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyWebGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<WebGpuMemoryInfo*>(hw_memory_info);
  if (memory_info->host_memory != nullptr) {
    litert_aligned_free(memory_info->host_memory);
  }
  delete memory_info;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockWebGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<WebGpuMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state != LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The WebGPU memory is already unlocked.");
  absl::Cleanup unlock = [&memory_info] {
    memory_info->lock_state = LockState::kUnlocked;
  };

  // Don't upload data if read only.
  if (memory_info->lock_state == LockState::kReadLocked) {
    return kLiteRtStatusOk;
  }

  WGPUQueue wgpu_queue = memory_info->queue;
  auto& gpu_tensor = memory_info->wgpu_tensor;

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeWebGpuBufferPacked) {
    ::ml_drift::webgpu::WriteDataToBuffer(
        wgpu_queue, gpu_tensor.GetBufferHandle(),
        /* webgpu::WriteDataToBuffer() requires a size of multiple of 4 */
        ml_drift::AlignByN(memory_info->packed_bytes, 4),
        memory_info->host_memory);
    return kLiteRtStatusOk;
  }

  // TODO: b/413431454 - Use Tensor Converter for better performance.
  auto desc_with_data = gpu_tensor.GetDescriptor();
  ::litert::ml_drift::ConvertDataToDescriptor(
      memory_info->host_memory, desc_with_data,
      memory_info->tensor_type.element_type);
  LITERT_RETURN_IF_ERROR(
      gpu_tensor.UploadDescriptorData(wgpu_queue, desc_with_data));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockWebGpuMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr) {
  if (hw_memory_info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<WebGpuMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state == LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The WebGPU memory is already locked.");

  LockState new_lock_state = litert::internal::ToLockState(mode);

  if (memory_info->host_memory == nullptr) {
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(
            &memory_info->host_memory, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
            ml_drift::AlignByN(memory_info->packed_bytes, 4));
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

  WGPUDevice wgpu_device = memory_info->device;
  auto& gpu_tensor = memory_info->wgpu_tensor;

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeWebGpuBufferPacked) {
    WGPUQueue wgpu_queue = memory_info->queue;
    LITERT_RETURN_IF_ERROR(::ml_drift::webgpu::ReadDataFromBuffer(
        wgpu_device, wgpu_queue, gpu_tensor.GetBufferHandle(),
        memory_info->packed_bytes, memory_info->host_memory));
  } else {
    // TODO: b/413431454 - Use Tensor Converter for better performance.
    auto descriptor_with_data = gpu_tensor.GetDescriptor();
    LITERT_RETURN_IF_ERROR(
        gpu_tensor.ToDescriptor(wgpu_device, &descriptor_with_data));
    ::litert::ml_drift::ConvertDataFromDescriptor(
        descriptor_with_data, memory_info->host_memory,
        memory_info->tensor_type.element_type);
  }

  memory_info->lock_state = new_lock_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtClearWebGpuMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<WebGpuMemoryInfo*>(hw_memory_info);
  wgpu::Device device(memory_info->device);
  wgpu::Queue queue(memory_info->queue);

  auto& gpu_tensor = memory_info->wgpu_tensor;
  auto command_encoder = device.CreateCommandEncoder();
  // Temporary buffer to clear textures. Need to keep it until the command
  // buffer is submitted.
  ::ml_drift::webgpu::Buffer temp_buffer;
  if (auto buffer = gpu_tensor.GetBufferHandle(); buffer) {
    command_encoder.ClearBuffer(buffer);
  } else if (auto texture = gpu_tensor.GetTextureHandle(); texture) {
    // Use a temporary buffer to copy zeros as the texture is not configured as
    // a rendering target.
    temp_buffer = ml_drift::webgpu::CreateBuffer(
        device, wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
        memory_info->packed_bytes);
    command_encoder.ClearBuffer(temp_buffer.GetMemoryHandle());
    wgpu::TexelCopyBufferInfo src{
        .layout =
            wgpu::TexelCopyBufferLayout{
                .bytesPerRow = static_cast<uint32_t>(memory_info->packed_bytes /
                                                     texture.GetHeight()),
                .rowsPerImage = texture.GetHeight()},
        .buffer = temp_buffer.GetMemoryHandle()};
    wgpu::TexelCopyTextureInfo dst{.texture = texture};
    wgpu::Extent3D copy_size{
        .width = texture.GetWidth(),
        .height = texture.GetHeight(),
        .depthOrArrayLayers = texture.GetDepthOrArrayLayers()};
    command_encoder.CopyBufferToTexture(&src, &dst, &copy_size);
  } else {
    return kLiteRtStatusErrorUnsupported;
  }
  auto command_buffer = command_encoder.Finish();
  queue.Submit(1, &command_buffer);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtImportWebGpuMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      HwMemoryHandle hw_buffer_handle,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info) {
  if (hw_buffer_handle == nullptr || hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor for import: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }

  if (device_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto memory_info = std::make_unique<WebGpuMemoryInfo>(
      WebGpuMemoryInfo{.wgpu_tensor = {},
                       .owns_tensor = false,
                       .tensor_type = *tensor_type,
                       .buffer_type = buffer_type,
                       .packed_bytes = packed_bytes,
                       .host_memory = nullptr,
                       .lock_state = LockState::kUnlocked,
                       .device = static_cast<WGPUDevice>(device_id),
                       .queue = static_cast<WGPUQueue>(queue_id)});
  if (buffer_type == kLiteRtTensorBufferTypeWebGpuBuffer ||
      buffer_type == kLiteRtTensorBufferTypeWebGpuBufferFp16 ||
      buffer_type == kLiteRtTensorBufferTypeWebGpuBufferPacked) {
    WGPUBuffer wgpu_buffer = static_cast<WGPUBuffer>(hw_buffer_handle);
    if (!wgpu_buffer) {
      ABSL_LOG(ERROR) << "Passed-in HwMemoryHandle is not a valid WGPUBuffer.";
      return kLiteRtStatusErrorInvalidArgument;
    }

    LITERT_RETURN_IF_ERROR(::ml_drift::webgpu::CreateSharedTensor(
        wgpu_buffer, *tensor_desc, &memory_info->wgpu_tensor));
  } else if (buffer_type == kLiteRtTensorBufferTypeWebGpuTexture) {
    WGPUTexture wgpu_texture = static_cast<WGPUTexture>(hw_buffer_handle);
    if (!wgpu_texture) {
      ABSL_LOG(ERROR) << "Passed-in HwMemoryHandle is not a valid WGPUTexture.";
      return kLiteRtStatusErrorInvalidArgument;
    }
    LITERT_RETURN_IF_ERROR(::ml_drift::webgpu::CreateSharedTensor(
        wgpu_texture, nullptr, *tensor_desc, &memory_info->wgpu_tensor));
  } else {
    ABSL_LOG(ERROR) << "Unsupported buffer type for import: "
                    << litert::BufferTypeToString(buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  // Returns `SpatialTensor*` as the `memory_handle`.
  memory_info->memory_handle = &memory_info->wgpu_tensor;
  *hw_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}
