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

#import <Metal/Metal.h>
#include "ml_drift/metal/buffer.h"  // from @ml_drift
#include "ml_drift/metal/common.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_tensor_buffer_utils.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "ml_drift_delegate/delegate/buffer_handler_metal.h"
#include "ml_drift_delegate/delegate/buffer_handler_utils.h"
#include "ml_drift_delegate/delegate/kv_cache_metal.h"

using ::litert::internal::LockState;

namespace {

// A `HwMemoryInfo` implementation for Metal Custom Buffer integration.
struct MetalMemoryInfo : public HwMemoryInfo {
  // MLD Tensor owns a GPU memory.
  ::ml_drift::metal::MetalSpatialTensor metal_tensor;
  bool owns_tensor;
  LiteRtRankedTensorType tensor_type;
  LiteRtTensorBufferType buffer_type;
  size_t packed_bytes;
  void* host_memory;
  LockState lock_state;
  // Managed through ARC. ARC retains the underlying Objective-C objects during assignment
  // to these strong members, ensuring they remain alive as long as this struct is alive.
  id<MTLDevice> metal_device;
  id<MTLCommandQueue> command_queue;
};



void UpdateRawHandle(MetalMemoryInfo* memory_info) {
  if (memory_info->metal_tensor.GetBufferHandle() != nil) {
    memory_info->raw_handle = (__bridge void*)memory_info->metal_tensor.GetBufferHandle();
  } else if (memory_info->metal_tensor.GetTextureHandle() != nil) {
    memory_info->raw_handle = (__bridge void*)memory_info->metal_tensor.GetTextureHandle();
  } else {
    memory_info->raw_handle = nullptr;
  }
}

// Function to read data from a Metal buffer.
void ReadDataFromBuffer(id<MTLBuffer> buffer, int buffer_offset, id<MTLCommandQueue> command_queue,
                        void* data, int data_size) {
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    // If the buffer is already in shared storage mode, copy the data directly to
    // the host memory.
    if (buffer.storageMode == MTLStorageModeShared) {
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      std::memcpy(data, reinterpret_cast<uint8_t*>([buffer contents]) + buffer_offset, data_size);
      return;
    }
    // Create a temporary buffer with shared storage mode to be accessible by the
    // CPU.
    id<MTLBuffer> temp_buffer =
        [command_queue.device newBufferWithLength:data_size options:MTLResourceStorageModeShared];

    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];

    // Enqueue a command to copy data from the source buffer to the temporary
    // buffer.
    [blitCommandEncoder copyFromBuffer:buffer
                          sourceOffset:buffer_offset
                              toBuffer:temp_buffer
                     destinationOffset:0
                                  size:data_size];
    [blitCommandEncoder endEncoding];

    // Commit the command buffer and wait for it to complete.
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy the data from the temporary buffer to the host memory (data pointer).
    // The .contents property is accessible here because temp_buffer uses
    // MTLResourceStorageModeShared.
    std::memcpy(data, [temp_buffer contents], data_size);
  }
}

void WriteDataToBuffer(id<MTLBuffer> buffer, int buffer_offset, id<MTLCommandQueue> command_queue,
                       const void* data, int data_size, bool wait_for_completion) {
  @autoreleasepool {
    // Check if CPU can access the memory directly.
    // Shared is always accessible. Managed (macOS) is also accessible.
    bool isAccessible = (buffer.storageMode == MTLStorageModeShared);

#if TARGET_OS_OSX
  if (buffer.storageMode == MTLStorageModeManaged) {
    isAccessible = true;
  }
#endif

  // Direct CPU Write
  if (isAccessible) {
    uint8_t* contents = (uint8_t*)[buffer contents];
    std::memcpy(contents + buffer_offset, data, data_size);

    // specific to macOS Managed buffers to inform GPU of changes
#if TARGET_OS_OSX
    if (buffer.storageMode == MTLStorageModeManaged) {
      [buffer didModifyRange:NSMakeRange(buffer_offset, data_size)];
    }
#endif
    return;
  }

  // Fallback to using a temporary buffer if the buffer is not accessible.
  id<MTLBuffer> temp_buffer =
      [[command_queue device] newBufferWithBytes:data
                                          length:data_size
                                         options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];

  [blitCommandEncoder copyFromBuffer:temp_buffer
                        sourceOffset:0
                            toBuffer:buffer
                   destinationOffset:buffer_offset
                                size:data_size];

  [blitCommandEncoder endEncoding];
  [command_buffer commit];

  if (wait_for_completion) {
    [command_buffer waitUntilCompleted];
  }
  }
}

}  // namespace

LiteRtStatus LiteRtCreateMetalMemory(LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type, size_t bytes,
                                     size_t packed_bytes, HwMemoryInfoPtr* metal_memory_info) {
  absl::StatusOr<::ml_drift::TensorDescriptor> tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor: " << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }

  if (device_id == nullptr) {
    ABSL_LOG(ERROR) << "Metal device is missing";
    return kLiteRtStatusErrorInvalidArgument;
  }
  absl::Status absl_status = absl::OkStatus();
  id<MTLDevice> metal_device = (__bridge id<MTLDevice>)(device_id);
  id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)(queue_id);

  auto memory_info =
      std::make_unique<MetalMemoryInfo>(MetalMemoryInfo{.owns_tensor = true,
                                                        .tensor_type = *tensor_type,
                                                        .buffer_type = buffer_type,
                                                        .packed_bytes = packed_bytes,
                                                        .lock_state = LockState::kUnlocked,
                                                        .metal_device = metal_device,
                                                        .command_queue = command_queue});
  absl_status =
      ::ml_drift::metal::CreateTensor(metal_device, *tensor_desc, &memory_info->metal_tensor);
  if (absl_status.ok()) {
    // Returns `SpatialTensor*` as the `memory_handle`.
    memory_info->memory_handle = &memory_info->metal_tensor;
    UpdateRawHandle(memory_info.get());
    *metal_memory_info = memory_info.release();
  }
  LITERT_RETURN_IF_ERROR(absl_status);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtImportMetalMemory(LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     HwMemoryHandle hw_buffer_handle, size_t bytes,
                                     size_t packed_bytes, HwMemoryInfoPtr* metal_memory_info) {
  if (hw_buffer_handle == nullptr || metal_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  absl::StatusOr<::ml_drift::TensorDescriptor> tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor for import: " << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }

  absl::Status absl_status = absl::OkStatus();
  id<MTLDevice> metal_device = device_id ? (__bridge id<MTLDevice>)(device_id) : nil;
  id<MTLCommandQueue> command_queue = queue_id ? (__bridge id<MTLCommandQueue>)(queue_id) : nil;

  auto memory_info =
      std::make_unique<MetalMemoryInfo>(MetalMemoryInfo{.owns_tensor = false,  // Set no ownership
                                                        .tensor_type = *tensor_type,
                                                        .buffer_type = buffer_type,
                                                        .packed_bytes = packed_bytes,
                                                        .lock_state = LockState::kUnlocked,
                                                        .metal_device = metal_device,
                                                        .command_queue = command_queue});

  if (buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16 ||
      buffer_type == kLiteRtTensorBufferTypeMetalTexture) {
    id<MTLTexture> metal_texture = (__bridge id<MTLTexture>)hw_buffer_handle;
    if (!metal_texture) {
      ABSL_LOG(ERROR) << "Passed-in HwMemoryHandle is not a valid MTLTexture.";
      absl_status =
          absl::InvalidArgumentError("Passed-in HwMemoryHandle is not a valid MTLTexture");
    } else {
      absl_status = ::ml_drift::metal::CreateTensorSharedTexture(metal_texture, *tensor_desc,
                                                                 &memory_info->metal_tensor);
    }
  } else if (buffer_type == kLiteRtTensorBufferTypeMetalBufferFp16 ||
             buffer_type == kLiteRtTensorBufferTypeMetalBuffer ||
             buffer_type == kLiteRtTensorBufferTypeMetalBufferPacked) {
    id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)hw_buffer_handle;
    if (metal_buffer == nil) {
      ABSL_LOG(ERROR) << "Passed-in HwMemoryHandle is not a valid MTLBuffer.";
      absl_status = absl::InvalidArgumentError("Passed-in HwMemoryHandle is not a valid MTLBuffer");
    } else {
      absl_status = ::ml_drift::metal::CreateTensorSharedBuffer(metal_buffer, *tensor_desc,
                                                                &memory_info->metal_tensor);
    }
  } else {
    ABSL_LOG(ERROR) << "Unsupported buffer type for import: "
                    << litert::BufferTypeToString(buffer_type);
    absl_status = absl::UnimplementedError("Unsupported buffer type for import");
  }

  if (absl_status.ok()) {
    // Returns `SpatialTensor*` as the `memory_handle`.
    memory_info->memory_handle = &memory_info->metal_tensor;
    UpdateRawHandle(memory_info.get());
    *metal_memory_info = memory_info.release();
  }
  LITERT_RETURN_IF_ERROR(absl_status);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyMetalMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  MetalMemoryInfo* memory_info = static_cast<MetalMemoryInfo*>(hw_memory_info);
  if (memory_info->host_memory != nullptr) {
    litert_aligned_free(memory_info->host_memory);
  }
  delete memory_info;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockMetalMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  MetalMemoryInfo* memory_info = static_cast<MetalMemoryInfo*>(hw_memory_info);
  LITERT_RETURN_IF_ERROR(memory_info->lock_state != LockState::kUnlocked,
                         litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
                             << "The Metal memory is already unlocked.");
  absl::Cleanup unlock = [&memory_info] { memory_info->lock_state = LockState::kUnlocked; };

  // Don't upload data if read only.
  if (memory_info->lock_state == LockState::kReadLocked) {
    return kLiteRtStatusOk;
  }

  id<MTLDevice> metal_device = memory_info->metal_device;
  id<MTLCommandQueue> command_queue = memory_info->command_queue;
  ::ml_drift::metal::MetalSpatialTensor& gpu_tensor = memory_info->metal_tensor;

  absl::Status absl_status = absl::OkStatus();
  if (memory_info->buffer_type == kLiteRtTensorBufferTypeMetalBufferPacked) {
    WriteDataToBuffer(gpu_tensor.GetBufferHandle(), 0, command_queue, memory_info->host_memory,
                      memory_info->packed_bytes, false);
  } else {
    // TODO: b/413431454 - Use Tensor Converter for better performance.
    ::ml_drift::TensorDescriptor desc_with_data = gpu_tensor.GetDescriptor();
    ::litert::ml_drift::ConvertDataToDescriptor(memory_info->host_memory, desc_with_data,
                                                memory_info->tensor_type.element_type);
    absl_status = gpu_tensor.UploadDescriptorData(desc_with_data, metal_device);
  }
  LITERT_RETURN_IF_ERROR(absl_status);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockMetalMemory(HwMemoryInfoPtr hw_memory_info, LiteRtTensorBufferLockMode mode,
                                   void** host_memory_ptr) {
  if (hw_memory_info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  MetalMemoryInfo* memory_info = static_cast<MetalMemoryInfo*>(hw_memory_info);
  LITERT_RETURN_IF_ERROR(memory_info->lock_state == LockState::kUnlocked,
                         litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
                             << "The Metal memory is already locked.");

  LockState new_lock_state = litert::internal::ToLockState(mode);

  if (memory_info->host_memory == nullptr) {
    // Ensure the data is aligned.
    if (int rc = posix_memalign(&memory_info->host_memory, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
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

  id<MTLDevice> metal_device = memory_info->metal_device;
  ::ml_drift::metal::MetalSpatialTensor& gpu_tensor = memory_info->metal_tensor;

  absl::Status absl_status = absl::OkStatus();
  if (memory_info->buffer_type == kLiteRtTensorBufferTypeMetalBufferPacked) {
    id<MTLCommandQueue> command_queue = memory_info->command_queue;
    ReadDataFromBuffer(gpu_tensor.GetBufferHandle(), 0, command_queue, memory_info->host_memory,
                       memory_info->packed_bytes);
  } else {
    // TODO: b/413431454 - Use Tensor Converter for better performance.
    ::ml_drift::TensorDescriptor descriptor_with_data = gpu_tensor.GetDescriptor();
    absl_status = gpu_tensor.ToDescriptor(&descriptor_with_data, metal_device);
    if (absl_status.ok()) {
      ::litert::ml_drift::ConvertDataFromDescriptor(descriptor_with_data, memory_info->host_memory,
                                                    memory_info->tensor_type.element_type);
    }
  }
  LITERT_RETURN_IF_ERROR(absl_status);

  memory_info->lock_state = new_lock_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtClearMetalMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  MetalMemoryInfo* memory_info = static_cast<MetalMemoryInfo*>(hw_memory_info);
  absl::Status absl_status = absl::OkStatus();
  @autoreleasepool {
    id<MTLCommandQueue> command_queue = memory_info->command_queue;
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];

    ::ml_drift::metal::MetalSpatialTensor& gpu_tensor = memory_info->metal_tensor;
    // Temporary buffer to clear textures. Need to keep it until the queue is completed.
    ::ml_drift::metal::Buffer temp_buffer;
    if (auto buffer = gpu_tensor.GetBufferHandle(); buffer != nil) {
      [blit_encoder fillBuffer:buffer
                         range:NSMakeRange(0, gpu_tensor.GetMemorySizeInBytes())
                         value:0];
    } else if (auto texture = gpu_tensor.GetTextureHandle(); texture != nil) {
      absl_status =
          ::ml_drift::metal::CreateBuffer(memory_info->packed_bytes,
                                          /*data=*/nullptr, command_queue.device, &temp_buffer);
      if (absl_status.ok()) {
        [blit_encoder fillBuffer:temp_buffer.GetMemoryPtr()
                           range:NSMakeRange(0, memory_info->packed_bytes)
                           value:0];
        [blit_encoder copyFromBuffer:temp_buffer.GetMemoryPtr()
                        sourceOffset:0
                   sourceBytesPerRow:memory_info->packed_bytes / texture.height
                 sourceBytesPerImage:memory_info->packed_bytes
                          sourceSize:MTLSizeMake(texture.width, texture.height, texture.depth)
                           toTexture:texture
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:MTLOriginMake(0, 0, 0)];
      }
    } else {
      absl_status = absl::UnimplementedError("Unsupported tensor spatial format");
    }
    if (absl_status.ok()) {
      [blit_encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
  }
  LITERT_RETURN_IF_ERROR(absl_status);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCopyKvCacheMetal(void* src_buffer_ptr, void* dst_buffer_ptr,
                                    int src_index_to_copy_on_prefill, int decode_batch_size,
                                    size_t src_buffer_size, size_t dst_buffer_size,
                                    void* command_queue_ptr) {
  if (!src_buffer_ptr || !dst_buffer_ptr || !command_queue_ptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  @autoreleasepool {
    id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src_buffer_ptr;
    id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst_buffer_ptr;
    id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)command_queue_ptr;

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];

    if (src_index_to_copy_on_prefill >= 0) {
      // Reduce: copy only the cache content of the given index.
      size_t src_offset = src_index_to_copy_on_prefill * dst_buffer_size;
      [blit_encoder copyFromBuffer:src_buffer
                      sourceOffset:src_offset
                          toBuffer:dst_buffer
                 destinationOffset:0
                              size:dst_buffer_size];
    } else {
      // Broadcast: broadcast the KV cache contents to all the batches.
      for (int i = 0; i < decode_batch_size; ++i) {
        [blit_encoder copyFromBuffer:src_buffer
                        sourceOffset:0
                            toBuffer:dst_buffer
                   destinationOffset:i * src_buffer_size
                                size:src_buffer_size];
      }
    }

    [blit_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }
  return kLiteRtStatusOk;
}
