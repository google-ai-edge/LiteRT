// Copyright 2024 Google LLC.
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

#include "litert/c/litert_tensor_buffer.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/custom_buffer.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#endif  // LITERT_HAS_OPENCL_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateTensorBufferFromHostMemory(
    const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t size, LiteRtHostMemoryDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !host_buffer_addr || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromHostMemory(
          *tensor_type,
          absl::MakeSpan(static_cast<uint8_t*>(host_buffer_addr), size),
          deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

#if LITERT_HAS_AHWB_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromAhwb(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    AHardwareBuffer* ahwb, size_t ahwb_offset,
    LiteRtAhwbDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !ahwb || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromAhwb(env, *tensor_type, ahwb, ahwb_offset,
                                          deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferAhwb(LiteRtTensorBuffer tensor_buffer,
                                       AHardwareBuffer** ahwb) {
  if (!tensor_buffer || !ahwb) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto ahwb_buffer, tensor_buffer->GetAhwbBuffer());

  *ahwb = ahwb_buffer;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_ION_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromIonBuffer(
    const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto created_tensor_buffer,
                          LiteRtTensorBufferT::CreateFromIonBuffer(
                              *tensor_type, ion_buffer_addr, ion_buffer_fd,
                              ion_buffer_size, ion_buffer_offset, deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferIonBuffer(LiteRtTensorBuffer tensor_buffer,
                                            void** ion_buffer_addr,
                                            int* ion_buffer_fd) {
  if (!tensor_buffer || !ion_buffer_addr || !ion_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto ion_buffer, tensor_buffer->GetIonBuffer());

  *ion_buffer_addr = ion_buffer.first;
  *ion_buffer_fd = ion_buffer.second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromDmaBufBuffer(
    const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromDmaBufBuffer(
          *tensor_type, dmabuf_buffer_addr, dmabuf_buffer_fd,
          dmabuf_buffer_size, dmabuf_buffer_offset, deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferDmaBufBuffer(LiteRtTensorBuffer tensor_buffer,
                                               void** dmabuf_buffer_addr,
                                               int* dmabuf_buffer_fd) {
  if (!tensor_buffer || !dmabuf_buffer_addr || !dmabuf_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto dmabuf_buffer, tensor_buffer->GetDmaBufBuffer());

  *dmabuf_buffer_addr = dmabuf_buffer.first;
  *dmabuf_buffer_fd = dmabuf_buffer.second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromOpenClMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, cl_mem cl_mem_addr,
    size_t opencl_buffer_size, LiteRtOpenClDeallocator deallocator,
    LiteRtTensorBuffer* buffer) {
  if (!tensor_type || !buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto created_tensor_buffer,
                          LiteRtTensorBufferT::CreateFromOpenClMemory(
                              env, *tensor_type, buffer_type, cl_mem_addr,
                              opencl_buffer_size, deallocator));
  *buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferOpenClMemory(LiteRtTensorBuffer tensor_buffer,
                                               cl_mem* cl_mem_addr) {
  if (!tensor_buffer || !cl_mem_addr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto opencl_memory, tensor_buffer->GetOpenClMemory());

  *cl_mem_addr = opencl_memory->GetMemoryPtr();
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

LiteRtStatus LiteRtGetTensorBufferCustomTensorBufferHandle(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle) {
  if (!tensor_buffer || !hw_memory_handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto remote_tensor_buffer,
                          tensor_buffer->GetCustomBuffer());
  *hw_memory_handle = remote_tensor_buffer->hw_buffer_handle();
  return kLiteRtStatusOk;
}

#if LITERT_HAS_METAL_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromMetalMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, void* metal_buffer,
    size_t metal_buffer_size, LiteRtMetalDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer || !metal_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromMetalMemory(
          env, *tensor_type, buffer_type, metal_buffer, metal_buffer_size));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferMetalMemory(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle) {
  if (!tensor_buffer || !hw_memory_handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(litert::internal::CustomBuffer * custom_buffer,
                          tensor_buffer->GetCustomBuffer());

  *hw_memory_handle = custom_buffer->hw_buffer_handle();
  return kLiteRtStatusOk;
}

#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromFastRpcBuffer(
    const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LiteRtFastRpcDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromFastRpcBuffer(
          *tensor_type, fastrpc_buffer_addr, fastrpc_buffer_fd,
          fastrpc_buffer_size, fastrpc_buffer_offset, deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferFastRpcBuffer(
    LiteRtTensorBuffer tensor_buffer, void** fastrpc_buffer_addr,
    int* fastrpc_buffer_fd) {
  if (!tensor_buffer || !fastrpc_buffer_addr || !fastrpc_buffer_fd) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto fastrpc_buffer,
                          tensor_buffer->GetFastRpcBuffer());

  *fastrpc_buffer_addr = fastrpc_buffer.first;
  *fastrpc_buffer_fd = fastrpc_buffer.second;
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_FASTRPC_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromGlBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromGlBuffer(env, *tensor_type, target, id,
                                              size_bytes, offset, deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferGlBuffer(LiteRtTensorBuffer tensor_buffer,
                                           LiteRtGLenum* target,
                                           LiteRtGLuint* id, size_t* size_bytes,
                                           size_t* offset) {
  if (!tensor_buffer || !target || !id) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto gl_buffer, tensor_buffer->GetGlBuffer());

  *target = gl_buffer->target();
  *id = gl_buffer->id();
  *size_bytes = gl_buffer->size_bytes();
  *offset = gl_buffer->offset();
  return kLiteRtStatusOk;
}
#else
LiteRtStatus LiteRtCreateTensorBufferFromGlBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGetTensorBufferGlBuffer(LiteRtTensorBuffer tensor_buffer,
                                           LiteRtGLenum* target,
                                           LiteRtGLuint* id, size_t* size_bytes,
                                           size_t* offset) {
  return kLiteRtStatusErrorUnsupported;
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromGlTexture(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
    size_t size_bytes, LiteRtGLint layer,
    LiteRtGlTextureDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto created_tensor_buffer,
                          LiteRtTensorBufferT::CreateFromGlTexture(
                              env, *tensor_type, target, id, format, size_bytes,
                              layer, deallocator));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferGlTexture(
    LiteRtTensorBuffer tensor_buffer, LiteRtGLenum* target, LiteRtGLuint* id,
    LiteRtGLenum* format, size_t* size_bytes, LiteRtGLint* layer) {
  if (!tensor_buffer || !target || !id || !format || !size_bytes || !layer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto gl_texture, tensor_buffer->GetGlTexture());
  *target = gl_texture->target();
  *id = gl_texture->id();
  *format = gl_texture->format();
  *size_bytes = gl_texture->size_bytes();
  *layer = gl_texture->layer();
  return kLiteRtStatusOk;
}
#else
LiteRtStatus LiteRtCreateTensorBufferFromGlTexture(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
    size_t size_bytes, LiteRtGLint layer,
    LiteRtGlTextureDeallocator deallocator, LiteRtTensorBuffer* tensor_buffer) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGetTensorBufferGlTexture(
    LiteRtTensorBuffer tensor_buffer, LiteRtGLenum* target, LiteRtGLuint* id,
    LiteRtGLenum* format, size_t* size_bytes, LiteRtGLint* layer) {
  return kLiteRtStatusErrorUnsupported;
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_WEBGPU_SUPPORT
LiteRtStatus LiteRtCreateTensorBufferFromWebGpuBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, WGPUBuffer wgpu_buffer,
    size_t wgpu_buffer_size, LiteRtWebGpuBufferDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateFromWebGpuBuffer(
          env, *tensor_type, buffer_type, wgpu_buffer, wgpu_buffer_size));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

// Return an error if the backing buffer is not a WebGpu buffer.
LiteRtStatus LiteRtGetTensorBufferWebGpuBuffer(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle) {
  if (!tensor_buffer || !hw_memory_handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto webgpu_buffer, tensor_buffer->GetCustomBuffer());

  *hw_memory_handle = webgpu_buffer->hw_buffer_handle();
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
LiteRtStatus LiteRtGetTensorBufferVulkanMemory(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle) {
  if (!tensor_buffer || !hw_memory_handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto custom_buffer, tensor_buffer->GetCustomBuffer());

  *hw_memory_handle = custom_buffer->hw_buffer_handle();
  return kLiteRtStatusOk;
}
#endif  // LITERT_HAS_VULKAN_SUPPORT

LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto created_tensor_buffer,
                          LiteRtTensorBufferT::CreateManaged(
                              env, buffer_type, *tensor_type, buffer_size));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateManagedTensorBufferFromRequirements(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements requirements,
    LiteRtTensorBuffer* tensor_buffer) {
  if (!tensor_type || !requirements || !tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Get the first supported buffer type from requirements
  if (requirements->SupportedBufferTypes().empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtTensorBufferType buffer_type = requirements->SupportedBufferTypes()[0];
  size_t buffer_size = requirements->BufferSize();
  size_t alignment = requirements->Alignment();

  const LiteRtRankedTensorType* tensor_type_to_use = tensor_type;
  LiteRtRankedTensorType tensor_type_with_strides;
  int num_requirement_strides = 0;
  const uint32_t* requirement_strides = nullptr;
  if (LiteRtGetTensorBufferRequirementsStrides(
          requirements, &num_requirement_strides, &requirement_strides) ==
          kLiteRtStatusOk &&
      requirement_strides != nullptr && num_requirement_strides > 0) {
    tensor_type_with_strides = *tensor_type;
    tensor_type_with_strides.layout.has_strides = true;
    for (int i = 0; i < num_requirement_strides; ++i) {
      tensor_type_with_strides.layout.strides[i] = requirement_strides[i];
    }
    tensor_type_to_use = &tensor_type_with_strides;
  }

  LITERT_ASSIGN_OR_RETURN(
      auto created_tensor_buffer,
      LiteRtTensorBufferT::CreateManagedWithAlignment(
          env, buffer_type, *tensor_type_to_use, buffer_size, alignment));
  *tensor_buffer = created_tensor_buffer.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDuplicateTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  if (!tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  tensor_buffer->Duplicate();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferType(LiteRtTensorBuffer tensor_buffer,
                                       LiteRtTensorBufferType* buffer_type) {
  if (!tensor_buffer || !buffer_type) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_type = tensor_buffer->buffer_type();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferTensorType(
    LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type) {
  if (!tensor_buffer || !tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *tensor_type = tensor_buffer->tensor_type();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferSize(LiteRtTensorBuffer tensor_buffer,
                                       size_t* buffer_size) {
  if (!tensor_buffer || !buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_size = tensor_buffer->buffer_size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferPackedSize(LiteRtTensorBuffer tensor_buffer,
                                             size_t* packed_buffer_size) {
  if (!tensor_buffer || !packed_buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *packed_buffer_size = tensor_buffer->packed_buffer_size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferOffset(LiteRtTensorBuffer tensor_buffer,
                                         size_t* buffer_offset) {
  if (!tensor_buffer || !buffer_offset) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *buffer_offset = tensor_buffer->buffer_offset();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                             void** host_memory_addr) {
  if (!tensor_buffer || !host_memory_addr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto host_buffer, tensor_buffer->GetHostBuffer());
  *host_memory_addr = host_buffer;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtHasTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        bool* has_event) {
  if (!tensor_buffer || !has_event) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *has_event = tensor_buffer->HasEvent();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent* event) {
  if (!tensor_buffer || !event) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_ASSIGN_OR_RETURN(auto result, tensor_buffer->GetEvent());
  *event = result;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent event) {
  if (!tensor_buffer || !event) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  tensor_buffer->SetEvent(event);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtClearTensorBufferEvent(LiteRtTensorBuffer tensor_buffer) {
  if (!tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  tensor_buffer->ClearEvent();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr,
                                    LiteRtTensorBufferLockMode mode) {
  if (!tensor_buffer || !host_mem_addr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LITERT_ASSIGN_OR_RETURN(auto mapped_addr, tensor_buffer->Lock(mode));

  *host_mem_addr = mapped_addr;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  if (!tensor_buffer) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(tensor_buffer->Unlock());

  return kLiteRtStatusOk;
}

void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  if (tensor_buffer->Unref()) {
    delete tensor_buffer;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif
