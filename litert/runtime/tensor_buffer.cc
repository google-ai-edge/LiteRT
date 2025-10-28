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

#include "litert/runtime/tensor_buffer.h"

#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_tensor_buffer_utils.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/runtime/custom_buffer.h"
#include "litert/runtime/event.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include "litert/runtime/open_cl_memory.h"
#include <CL/cl.h>
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
#include "litert/c/litert_gl_types.h"
#include "litert/runtime/gl_buffer.h"
#include "litert/runtime/gl_texture.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_ION_SUPPORT
#include "litert/runtime/ion_buffer.h"
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
#include "litert/runtime/fastrpc_buffer.h"
#endif  // LITERT_HAS_FASTRPC_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
#include "litert/runtime/dmabuf_buffer.h"
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_AHWB_SUPPORT
#include "litert/runtime/ahwb_buffer.h"
#endif  // LITERT_HAS_AHWB_SUPPORT


using litert::BufferTypeToString;
using litert::Expected;
using litert::Unexpected;

namespace {

template <typename T>
void Copy(size_t array_size, const T* array, std::vector<T>& vec) {
  vec.assign(array, array + array_size);
}

// CFI builds don't like directly passing free() as a function pointer, so wrap
// in another function.
void FreeHostMemory(void* ptr) { litert_aligned_free(ptr); }

}  // namespace

// C API defined in environment.cc to workaround Windows build issue.
// Lexan Windows linker makes an error when we include core/environment.h.
extern "C" litert::internal::GpuEnvironment* LiteRtGetGpuEnvironment(
    LiteRtEnvironment env);

Expected<litert::internal::GpuEnvironment*> GetGpuEnvironment(
    LiteRtEnvironment env) {
  auto gpu_env = LiteRtGetGpuEnvironment(env);
  if (!gpu_env) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Can't get GPU environment");
  }
  return gpu_env;
}

LiteRtTensorBufferT::LiteRtTensorBufferT(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size,
    size_t buffer_offset)
    : env_(env),
      tensor_type_(tensor_type),
      buffer_type_(buffer_type),
      buffer_size_(buffer_size),
      buffer_offset_(buffer_offset),
      ref_(1) {
  // Copy local memory passed by the caller.
  Copy(tensor_type_.layout.rank, tensor_type_.layout.dimensions, dimensions_);
  if (tensor_type_.layout.has_strides) {
    Copy(tensor_type_.layout.rank, tensor_type_.layout.strides, strides_);
  }
  auto packed_size = litert::internal::GetNumPackedBytes(tensor_type_);
  if (!packed_size) {
    packed_buffer_size_ = 0;
    LITERT_LOG(LITERT_ERROR, "Failed to get num packed bytes");
  } else {
    packed_buffer_size_ = *packed_size;
  }
// Our Emscripten builds process this as an error rather than a debug log, so
// disabling for web platform temporarily to avoid breakages.
#ifndef __EMSCRIPTEN__
  LITERT_LOG(LITERT_DEBUG, "Created tensor buffer %p of type %s", this,
             BufferTypeToString(buffer_type_).data());
#endif  // __EMSCRIPTEN__
}

LiteRtTensorBufferT::~LiteRtTensorBufferT() {
#ifndef __EMSCRIPTEN__
  LITERT_LOG(LITERT_DEBUG, "Destroying tensor buffer %p of type %s", this,
             BufferTypeToString(buffer_type_).data());
#endif  // __EMSCRIPTEN__
  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeUnknown:
      // Nothing to do.
      break;
    case kLiteRtTensorBufferTypeHostMemory:
      if (auto& buffer = std::get<HostBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeAhwb:
      if (auto& buffer = std::get<AhwbBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.ahwb);
      }
      break;
    case kLiteRtTensorBufferTypeIon:
      if (auto& buffer = std::get<IonBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeDmaBuf:
      if (auto& buffer = std::get<DmaBufBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeFastRpc:
      if (auto& buffer = std::get<FastRpcBuffer>(buffer_); buffer.deallocator) {
        buffer.deallocator(buffer.addr);
      }
      break;
    case kLiteRtTensorBufferTypeOpenClBuffer:
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
    case kLiteRtTensorBufferTypeOpenClBufferPacked:
      // internal opencl buffer is auto-disposed by the
      // litert::internal::OpenClMemory destructor.
      break;
    case kLiteRtTensorBufferTypeGlBuffer:
      // internal gl buffer is auto-disposed by the
      // litert::internal::GlBuffer destructor.
    case kLiteRtTensorBufferTypeGlTexture:
      // internal gl texture is auto-disposed by the
      // litert::internal::GlTexture destructor.
      break;
    case kLiteRtTensorBufferTypeWebGpuBuffer:
    case kLiteRtTensorBufferTypeWebGpuBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuTexture:
    case kLiteRtTensorBufferTypeWebGpuTextureFp16:
    case kLiteRtTensorBufferTypeWebGpuImageBuffer:
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuBufferPacked:
      // internal webgpu buffer is auto-disposed by the
      // litert::internal::CustomBuffer destructor.
      break;
    case kLiteRtTensorBufferTypeMetalBuffer:
    case kLiteRtTensorBufferTypeMetalBufferFp16:
    case kLiteRtTensorBufferTypeMetalTexture:
    case kLiteRtTensorBufferTypeMetalTextureFp16:
    case kLiteRtTensorBufferTypeMetalBufferPacked:
      // internal metal buffer is auto-disposed by the
      // litert::internal::MetalMemory destructor.
      break;
    case kLiteRtTensorBufferTypeVulkanBuffer:
    case kLiteRtTensorBufferTypeVulkanBufferFp16:
    case kLiteRtTensorBufferTypeVulkanTexture:
    case kLiteRtTensorBufferTypeVulkanTextureFp16:
    case kLiteRtTensorBufferTypeVulkanImageBuffer:
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16:
    case kLiteRtTensorBufferTypeVulkanBufferPacked:
      // internal vulkan memory is auto-disposed by the
      // litert::internal::VulkanMemory destructor.
      break;
  }
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromHostMemory(
    const LiteRtRankedTensorType& tensor_type, absl::Span<uint8_t> host_memory,
    LiteRtHostMemoryDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(/*env=*/nullptr, tensor_type,
                                            kLiteRtTensorBufferTypeHostMemory,
                                            host_memory.size()));
  tensor_buffer->buffer_ = HostBuffer{
      .addr = host_memory.data(),
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOnHostMemory(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  return CreateManagedOnHostMemory(tensor_type, buffer_size,
                                   LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOnHostMemory(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size,
    size_t alignment) {
  void* host_memory_ptr;
  if (auto rc = posix_memalign(&host_memory_ptr, alignment,
                               buffer_size + XNN_EXTRA_BYTES);

      rc) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate aligned memory");
  }

  LiteRtHostMemoryDeallocator deallocator = FreeHostMemory;
  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferT::Ptr tensor_buffer,
      CreateFromHostMemory(
          tensor_type,
          absl::MakeSpan(static_cast<uint8_t*>(host_memory_ptr), buffer_size),
          deallocator));

  return std::move(tensor_buffer);
}

#if LITERT_HAS_AHWB_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromAhwb(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    AHardwareBuffer* ahwb, size_t ahwb_offset,
    LiteRtAhwbDeallocator deallocator) {
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                          litert::internal::AhwbBuffer::GetSize(ahwb));

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      env, tensor_type, kLiteRtTensorBufferTypeAhwb, buffer_size, ahwb_offset));
  tensor_buffer->buffer_ = AhwbBuffer{
      .ahwb = ahwb,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedAhwbBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(litert::internal::AhwbBuffer buffer,
                          litert::internal::AhwbBuffer::Alloc(buffer_size));
  return CreateFromAhwb(env, tensor_type, buffer.ahwb, /*ahwb_offset=*/0,
                        /*deallocator=*/litert::internal::AhwbBuffer::Free);
}
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_ION_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromIonBuffer(
    const LiteRtRankedTensorType& tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator) {
  if (!ion_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid ION buffer address");
  }
  if (ion_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid ION buffer fd");
  }

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      /*env=*/nullptr, tensor_type, kLiteRtTensorBufferTypeIon, ion_buffer_size,
      ion_buffer_offset));
  tensor_buffer->buffer_ = IonBuffer{
      .addr = ion_buffer_addr,
      .fd = ion_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedIonBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::IonBuffer::Alloc(
      buffer_size, /*alignment=*/LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromIonBuffer(tensor_type, buffer->addr, buffer->fd, buffer_size,
                             /*ion_buffer_offset=*/0,
                             litert::internal::IonBuffer::Free);
}
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator) {
  if (!dmabuf_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid DMA-BUF buffer address");
  }
  if (dmabuf_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid DMA-BUF buffer fd");
  }

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      /*env=*/nullptr, tensor_type, kLiteRtTensorBufferTypeDmaBuf,
      dmabuf_buffer_size, dmabuf_buffer_offset));
  tensor_buffer->buffer_ = DmaBufBuffer{
      .addr = dmabuf_buffer_addr,
      .fd = dmabuf_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedDmaBufBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::DmaBufBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromDmaBufBuffer(tensor_type, buffer->addr, buffer->fd,
                                buffer_size, /*dmabuf_buffer_offset=*/0,
                                litert::internal::DmaBufBuffer::Free);
}
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_buffer_fd, size_t fastrpc_buffer_size,
    size_t fastrpc_buffer_offset, LiteRtFastRpcDeallocator deallocator) {
  if (!fastrpc_buffer_addr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid FastRPC buffer address");
  }
  if (fastrpc_buffer_fd < 0) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Invalid FastRPC buffer fd");
  }

  Ptr tensor_buffer(new LiteRtTensorBufferT(
      /*env=*/nullptr, tensor_type, kLiteRtTensorBufferTypeFastRpc,
      fastrpc_buffer_size, fastrpc_buffer_offset));
  tensor_buffer->buffer_ = FastRpcBuffer{
      .addr = fastrpc_buffer_addr,
      .fd = fastrpc_buffer_fd,
      .deallocator = deallocator,
  };

  if (auto status = tensor_buffer->IsValid(); !status) {
    return Unexpected(status.Error());
  }

  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedFastRpcBuffer(
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  auto buffer = litert::internal::FastRpcBuffer::Alloc(buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  return CreateFromFastRpcBuffer(tensor_type, buffer->addr, buffer->fd,
                                 buffer_size, /*fastrpc_buffer_offset=*/0,
                                 litert::internal::FastRpcBuffer::Free);
}
#endif  // LITERT_HAS_FASTRPC_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromOpenClMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, cl_mem buffer, size_t buffer_size,
    LiteRtOpenClDeallocator deallocator) {
  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env));
  tensor_buffer->buffer_.emplace<litert::internal::OpenClMemory>(
      gpu_env, tensor_type, buffer_type, buffer, buffer_size, deallocator);
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedOpenClMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env));
  LITERT_ASSIGN_OR_RETURN(auto buffer,
                          litert::internal::OpenClMemory::Alloc(
                              gpu_env, tensor_type, buffer_type, buffer_size));
  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::OpenClMemory>(
      std::move(buffer));
  return tensor_buffer;
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

// TODO b/412405854 - Add CreateFromWebGpuBuffer to support zero-copy scenarios
// of WebGPU buffer.
Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedWebGpuBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                          litert::internal::GetNumPackedBytes(tensor_type));
  auto buffer = litert::internal::CustomBuffer::Alloc(
      env, tensor_type, buffer_type, buffer_size, packed_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::CustomBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}

// TODO b/426869066 - Add CreateFromMetalMemory to support zero-copy scenarios
// of Metal memory.
Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedMetalMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                          litert::internal::GetNumPackedBytes(tensor_type));
  auto buffer = litert::internal::CustomBuffer::Alloc(
      env, tensor_type, buffer_type, buffer_size, packed_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::CustomBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}

// TODO b/426869066 - Add CreateFromVulkanMemory to support zero-copy scenarios
// of Vulkan memory.
Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedVulkanMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(auto packed_size,
                          litert::internal::GetNumPackedBytes(tensor_type));
  auto buffer = litert::internal::CustomBuffer::Alloc(
      env, tensor_type, buffer_type, buffer_size, packed_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::CustomBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}

#if LITERT_HAS_OPENGL_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromGlBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      env, tensor_type, kLiteRtTensorBufferTypeGlBuffer, size_bytes));
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env));
  tensor_buffer->buffer_.emplace<litert::internal::GlBuffer>(
      gpu_env, target, id, size_bytes, offset, deallocator);
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManagedGlBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env));
  auto buffer = litert::internal::GlBuffer::Alloc(gpu_env, buffer_size);
  if (!buffer) {
    return Unexpected(buffer.Error());
  }
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      env, tensor_type, kLiteRtTensorBufferTypeGlBuffer, buffer_size));
  tensor_buffer->buffer_.emplace<litert::internal::GlBuffer>(
      std::move(*buffer));
  return tensor_buffer;
}

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromGlTexture(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
    size_t size_bytes, LiteRtGLint layer,
    LiteRtGlTextureDeallocator deallocator) {
  Ptr tensor_buffer(new LiteRtTensorBufferT(
      env, tensor_type, kLiteRtTensorBufferTypeGlTexture, size_bytes));
  tensor_buffer->buffer_.emplace<litert::internal::GlTexture>(
      litert::internal::GlTexture(target, id, format, size_bytes, layer,
                                  deallocator));
  return tensor_buffer;
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateFromMetalMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, void* metal_buffer,
    size_t buffer_size) {
  LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                          litert::internal::GetNumPackedBytes(tensor_type));
  // Use CustomBuffer::Wrap to create a non-owning wrapper
  LITERT_ASSIGN_OR_RETURN(litert::internal::CustomBuffer custom_buffer,
                          litert::internal::CustomBuffer::Wrap(
                              env, tensor_type, buffer_type, metal_buffer,
                              buffer_size, packed_size));

  Ptr tensor_buffer(
      new LiteRtTensorBufferT(env, tensor_type, buffer_type, buffer_size));

  tensor_buffer->buffer_.emplace<litert::internal::CustomBuffer>(
      std::move(custom_buffer));
  return tensor_buffer;
}
#endif  // LITERT_HAS_METAL_SUPPORT

Expected<LiteRtTensorBufferT::Ptr> LiteRtTensorBufferT::CreateManaged(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size) {
  return CreateManagedWithAlignment(env, buffer_type, tensor_type, buffer_size,
                                    LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
}

Expected<LiteRtTensorBufferT::Ptr>
LiteRtTensorBufferT::CreateManagedWithAlignment(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType& tensor_type, size_t buffer_size,
    size_t alignment) {
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeHostMemory:
      return CreateManagedOnHostMemory(tensor_type, buffer_size, alignment);
    case kLiteRtTensorBufferTypeAhwb:
#if LITERT_HAS_AHWB_SUPPORT
      return CreateManagedAhwbBuffer(env, tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "AHardwareBuffer is not supported.");
#endif
    case kLiteRtTensorBufferTypeIon:
#if LITERT_HAS_ION_SUPPORT
      return CreateManagedIonBuffer(tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "ION buffer is not supported.");
#endif
    case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
      return CreateManagedDmaBufBuffer(tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "DMA-BUF buffer is not supported.");
#endif
    case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
      return CreateManagedFastRpcBuffer(tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "FastRPC buffer is not supported.");
#endif
    case kLiteRtTensorBufferTypeOpenClBuffer:
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
    case kLiteRtTensorBufferTypeOpenClBufferPacked: {
#if LITERT_HAS_OPENCL_SUPPORT
      return CreateManagedOpenClMemory(env, tensor_type, buffer_type,
                                       buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "OpenCL memory is not supported.");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
#if LITERT_HAS_OPENGL_SUPPORT
      return CreateManagedGlBuffer(env, tensor_type, buffer_size);
#else
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "OpenGL buffer is not supported.");
#endif
    }
    case kLiteRtTensorBufferTypeGlTexture: {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "LiteRT does not support managed GL textures.");
    }
    case kLiteRtTensorBufferTypeWebGpuBuffer:
    case kLiteRtTensorBufferTypeWebGpuBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuTexture:
    case kLiteRtTensorBufferTypeWebGpuTextureFp16:
    case kLiteRtTensorBufferTypeWebGpuImageBuffer:
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuBufferPacked: {
      return CreateManagedWebGpuBuffer(env, tensor_type, buffer_type,
                                       buffer_size);
    }
    case kLiteRtTensorBufferTypeMetalBuffer:
    case kLiteRtTensorBufferTypeMetalBufferFp16:
    case kLiteRtTensorBufferTypeMetalTexture:
    case kLiteRtTensorBufferTypeMetalTextureFp16:
    case kLiteRtTensorBufferTypeMetalBufferPacked: {
      return CreateManagedMetalMemory(env, tensor_type, buffer_type,
                                      buffer_size);
    }
    case kLiteRtTensorBufferTypeVulkanBuffer:
    case kLiteRtTensorBufferTypeVulkanBufferFp16:
    case kLiteRtTensorBufferTypeVulkanTexture:
    case kLiteRtTensorBufferTypeVulkanTextureFp16:
    case kLiteRtTensorBufferTypeVulkanImageBuffer:
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16:
    case kLiteRtTensorBufferTypeVulkanBufferPacked: {
      return CreateManagedVulkanMemory(env, tensor_type, buffer_type,
                                       buffer_size);
    }
    case kLiteRtTensorBufferTypeUnknown:
    default:
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Unexpected tensor type");
  }
}

Expected<void> LiteRtTensorBufferT::IsValid() {
  // Check for static dimensions.
  for (auto i = 0; i < tensor_type_.layout.rank; ++i) {
    if (tensor_type_.layout.dimensions[i] <= 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "TensorBuffer must have all static dimensions");
    }
  }

  // Check for valid offset.
  if (buffer_offset() >= buffer_size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid buffer offset");
  }

  // Check for sufficient size.
  if (auto num_bytes = litert::internal::GetNumPackedBytes(tensor_type_);
      !num_bytes) {
    return Unexpected(num_bytes.Error());
  } else if (*num_bytes > buffer_size() - buffer_offset()) {
    const std::string error_message = absl::StrFormat(
        "Insufficient buffer size: Required %d bytes, actual size %d bytes",
        *num_bytes, buffer_size() - buffer_offset());
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  // Check for proper alignment.
  if (buffer_type() == kLiteRtTensorBufferTypeHostMemory) {
    auto host_buffer = GetHostBuffer();
    if (!host_buffer) {
      return Unexpected(host_buffer.Error());
    }
    if (reinterpret_cast<uintptr_t>(*host_buffer) %
        LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unaligned host memory pointer");
    }
  }

  return {};
}

Expected<void*> LiteRtTensorBufferT::GetHostBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeHostMemory) {
    return std::get<HostBuffer>(buffer_).addr;
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeHostMemory),
                      BufferTypeToString(buffer_type_)));
}

Expected<AHardwareBuffer*> LiteRtTensorBufferT::GetAhwbBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeAhwb) {
    return std::get<AhwbBuffer>(buffer_).ahwb;
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeAhwb),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetIonBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeIon) {
    auto buffer = std::get<IonBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeIon),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetDmaBufBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeDmaBuf) {
    auto buffer = std::get<DmaBufBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeDmaBuf),
                      BufferTypeToString(buffer_type_)));
}

Expected<std::pair<void*, int>> LiteRtTensorBufferT::GetFastRpcBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeFastRpc) {
    auto buffer = std::get<FastRpcBuffer>(buffer_);
    return std::make_pair(buffer.addr, buffer.fd);
  }

  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeFastRpc),
                      BufferTypeToString(buffer_type_)));
}

#if LITERT_HAS_OPENCL_SUPPORT
Expected<litert::internal::OpenClMemory*>
LiteRtTensorBufferT::GetOpenClMemory() {
  if (IsOpenClMemory(buffer_type_)) {
    return &std::get<litert::internal::OpenClMemory>(buffer_);
  }
#if LITERT_HAS_AHWB_SUPPORT && LITERT_HAS_OPENGL_SUPPORT
  if (buffer_type_ == kLiteRtTensorBufferTypeAhwb) {
    if (auto it =
            memory_backed_buffers_.find(kLiteRtTensorBufferTypeOpenClBuffer);
        it != memory_backed_buffers_.end()) {
      BufferVariant& memory_backed_buffer = it->second;
      return &std::get<litert::internal::OpenClMemory>(memory_backed_buffer);
    }
    // Create a new CL buffer from the AHWB buffer if not found.
    litert::internal::AhwbBuffer ahwb_buffer = {
        .ahwb = std::get<AhwbBuffer>(buffer_).ahwb};

    LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env_));
    LITERT_ASSIGN_OR_RETURN(litert::internal::OpenClMemory cl_buffer_from_ahwb,
                            litert::internal::OpenClMemory::AllocFromAhwbBuffer(
                                gpu_env, tensor_type_, ahwb_buffer));

    auto [it, inserted] = memory_backed_buffers_.insert(
        {kLiteRtTensorBufferTypeOpenClBuffer, std::move(cl_buffer_from_ahwb)});
    LITERT_RETURN_IF_ERROR(
        inserted == true,
        Unexpected(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to insert CL buffer into memory backed buffers"));
    return &std::get<litert::internal::OpenClMemory>(it->second);
  }
#endif
#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_OPENCL_SUPPORT
  if (buffer_type_ == kLiteRtTensorBufferTypeGlBuffer) {
    if (auto it =
            memory_backed_buffers_.find(kLiteRtTensorBufferTypeOpenClBuffer);
        it != memory_backed_buffers_.end()) {
      BufferVariant& memory_backed_buffer = it->second;
      return &std::get<litert::internal::OpenClMemory>(memory_backed_buffer);
    }
    // Create a new CL buffer from the GL buffer if not found.
    litert::internal::GlBuffer& gl_buffer =
        std::get<litert::internal::GlBuffer>(buffer_);
    LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env_));
    LITERT_ASSIGN_OR_RETURN(
        litert::internal::OpenClMemory cl_buffer_from_gl_buffer,
        litert::internal::OpenClMemory::AllocFromGlBuffer(gpu_env, tensor_type_,
                                                          gl_buffer));
    auto [it, inserted] =
        memory_backed_buffers_.insert({kLiteRtTensorBufferTypeOpenClBuffer,
                                       std::move(cl_buffer_from_gl_buffer)});
    LITERT_RETURN_IF_ERROR(
        inserted == true,
        Unexpected(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to insert CL buffer into memory backed buffers"));
    return &std::get<litert::internal::OpenClMemory>(it->second);
  }
#endif
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeOpenClBuffer),
                      BufferTypeToString(buffer_type_)));
}
#endif  // LITERT_HAS_OPENCL_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT
Expected<litert::internal::GlTexture*> LiteRtTensorBufferT::GetGlTexture() {
  if (buffer_type_ != kLiteRtTensorBufferTypeGlTexture) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unexpected tensor buffer type");
  }
  return &std::get<litert::internal::GlTexture>(buffer_);
}

Expected<litert::internal::GlBuffer*> LiteRtTensorBufferT::GetGlBuffer() {
  if (buffer_type_ == kLiteRtTensorBufferTypeGlBuffer) {
    return &std::get<litert::internal::GlBuffer>(buffer_);
  }
#if LITERT_HAS_AHWB_SUPPORT
  if (buffer_type_ == kLiteRtTensorBufferTypeAhwb) {
    if (auto it = memory_backed_buffers_.find(kLiteRtTensorBufferTypeGlBuffer);
        it != memory_backed_buffers_.end()) {
      BufferVariant& memory_backed_buffer = it->second;
      return &std::get<litert::internal::GlBuffer>(memory_backed_buffer);
    }
    // Create a new GL buffer from the AHWB buffer if not found.
    litert::internal::AhwbBuffer ahwb_buffer = {
        .ahwb = std::get<AhwbBuffer>(buffer_).ahwb};

    LITERT_ASSIGN_OR_RETURN(auto gpu_env, GetGpuEnvironment(env_));
    LITERT_ASSIGN_OR_RETURN(
        litert::internal::GlBuffer gl_buffer_from_ahwb,
        litert::internal::GlBuffer::AllocFromAhwbBuffer(gpu_env, ahwb_buffer));

    auto [it, inserted] = memory_backed_buffers_.insert(
        {kLiteRtTensorBufferTypeGlBuffer, std::move(gl_buffer_from_ahwb)});
    LITERT_RETURN_IF_ERROR(
        inserted == true,
        Unexpected(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to insert GL buffer into memory backed buffers"));
    return &std::get<litert::internal::GlBuffer>(it->second);
  }
#endif  // LITERT_HAS_AHWB_SUPPORT

  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      absl::StrFormat("Cannot get %s buffer from %s tensor buffer",
                      BufferTypeToString(kLiteRtTensorBufferTypeGlBuffer),
                      BufferTypeToString(buffer_type_)));
}
#endif  // LITERT_HAS_OPENGL_SUPPORT

Expected<litert::internal::CustomBuffer*>
LiteRtTensorBufferT::GetCustomBuffer() {
  if (IsWebGpuMemory(buffer_type_) || IsVulkanMemory(buffer_type_) ||
      IsMetalMemory(buffer_type_)) {
    return &std::get<litert::internal::CustomBuffer>(buffer_);
  }
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "Unexpected tensor buffer type");
}

Expected<void*> LiteRtTensorBufferT::Lock(LiteRtTensorBufferLockMode mode) {
  LITERT_RETURN_IF_ERROR(is_locked_ == false,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "Tensor buffer is already locked."));
  is_locked_ = true;
  if (event_ != nullptr) {
    // Only AHWB supports waiting on an input sync fence when locking the
    // buffer. For all other buffer types we wait here.
    if (buffer_type() != kLiteRtTensorBufferTypeAhwb) {
      LITERT_RETURN_IF_ERROR(event_->Wait(/*timeout_in_ms=*/-1));
    }
  }

  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeHostMemory: {
      LITERT_ASSIGN_OR_ABORT(auto host_buffer, GetHostBuffer());
      return host_buffer;
    }
    case kLiteRtTensorBufferTypeAhwb: {
#if LITERT_HAS_AHWB_SUPPORT
      LITERT_ASSIGN_OR_ABORT(auto ahwb_buffer, GetAhwbBuffer());
      return litert::internal::AhwbBuffer::Lock(
          ahwb_buffer, event_ != nullptr ? event_.get() : nullptr);
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "AHardwareBuffer is not supported");
#endif
    }
    case kLiteRtTensorBufferTypeIon: {
      LITERT_ASSIGN_OR_ABORT(auto ion_buffer, GetIonBuffer());
      return ion_buffer.first;
    }
    case kLiteRtTensorBufferTypeDmaBuf: {
      LITERT_ASSIGN_OR_ABORT(auto dma_buffer, GetDmaBufBuffer());
      return dma_buffer.first;
    }
    case kLiteRtTensorBufferTypeFastRpc: {
      LITERT_ASSIGN_OR_ABORT(auto fastrpc_buffer, GetFastRpcBuffer());
      return fastrpc_buffer.first;
    }
    case kLiteRtTensorBufferTypeOpenClBuffer:
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
    case kLiteRtTensorBufferTypeOpenClBufferPacked: {
#if LITERT_HAS_OPENCL_SUPPORT
      LITERT_ASSIGN_OR_ABORT(auto opencl_memory, GetOpenClMemory());
      LITERT_ASSIGN_OR_RETURN(float* const host_memory_ptr,
                              opencl_memory->Lock<float>(mode));
      return host_memory_ptr;
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenCL buffers are not supported");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
#if LITERT_HAS_OPENGL_SUPPORT
      LITERT_ASSIGN_OR_RETURN(auto gl_buffer, GetGlBuffer());
      LITERT_ASSIGN_OR_RETURN(float* const host_memory_ptr,
                              gl_buffer->Lock<float>());
      return host_memory_ptr;
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenGL buffers are not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    case kLiteRtTensorBufferTypeWebGpuBuffer:
    case kLiteRtTensorBufferTypeWebGpuBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuTexture:
    case kLiteRtTensorBufferTypeWebGpuTextureFp16:
    case kLiteRtTensorBufferTypeWebGpuImageBuffer:
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuBufferPacked:
    case kLiteRtTensorBufferTypeMetalBuffer:
    case kLiteRtTensorBufferTypeMetalBufferFp16:
    case kLiteRtTensorBufferTypeMetalBufferPacked:
    case kLiteRtTensorBufferTypeMetalTexture:
    case kLiteRtTensorBufferTypeMetalTextureFp16:
    case kLiteRtTensorBufferTypeVulkanBuffer:
    case kLiteRtTensorBufferTypeVulkanBufferFp16:
    case kLiteRtTensorBufferTypeVulkanTexture:
    case kLiteRtTensorBufferTypeVulkanTextureFp16:
    case kLiteRtTensorBufferTypeVulkanImageBuffer:
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16:
    case kLiteRtTensorBufferTypeVulkanBufferPacked: {
      LITERT_ASSIGN_OR_RETURN(auto custom_buffer, GetCustomBuffer());
      LITERT_ASSIGN_OR_RETURN(void* const host_memory_ptr,
                              custom_buffer->Lock(mode));
      return host_memory_ptr;
    }
    case kLiteRtTensorBufferTypeGlTexture:
    case kLiteRtTensorBufferTypeUnknown: {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unexpected tensor buffer type");
    }
  }
}

Expected<void> LiteRtTensorBufferT::Unlock() {
  LITERT_RETURN_IF_ERROR(is_locked_ == true,
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "Tensor buffer is already unlocked."));
  is_locked_ = false;
  switch (buffer_type()) {
    case kLiteRtTensorBufferTypeAhwb: {
#if LITERT_HAS_AHWB_SUPPORT
      auto ahwb = std::get<AhwbBuffer>(buffer_).ahwb;
      return litert::internal::AhwbBuffer::Unlock(ahwb);
#else
      return {};
#endif
    }
    case kLiteRtTensorBufferTypeOpenClBuffer:
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
    case kLiteRtTensorBufferTypeOpenClTexture:
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
    case kLiteRtTensorBufferTypeOpenClBufferPacked: {
#if LITERT_HAS_OPENCL_SUPPORT
      LITERT_ASSIGN_OR_RETURN(auto opencl_buffer, GetOpenClMemory());
      return opencl_buffer->Unlock<float>();
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenCL buffers are not supported");
#endif  // LITERT_HAS_OPENCL_SUPPORT
    }
    case kLiteRtTensorBufferTypeGlBuffer: {
#if LITERT_HAS_OPENGL_SUPPORT
      LITERT_ASSIGN_OR_RETURN(auto gl_buffer, GetGlBuffer());
      return gl_buffer->Unlock<float>();
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "OpenGL buffers are not supported");
#endif  // LITERT_HAS_OPENGL_SUPPORT
    }
    case kLiteRtTensorBufferTypeWebGpuBuffer:
    case kLiteRtTensorBufferTypeWebGpuBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuTexture:
    case kLiteRtTensorBufferTypeWebGpuTextureFp16:
    case kLiteRtTensorBufferTypeWebGpuImageBuffer:
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16:
    case kLiteRtTensorBufferTypeWebGpuBufferPacked:
    case kLiteRtTensorBufferTypeMetalBuffer:
    case kLiteRtTensorBufferTypeMetalBufferFp16:
    case kLiteRtTensorBufferTypeMetalBufferPacked:
    case kLiteRtTensorBufferTypeMetalTexture:
    case kLiteRtTensorBufferTypeMetalTextureFp16:
    case kLiteRtTensorBufferTypeVulkanBuffer:
    case kLiteRtTensorBufferTypeVulkanBufferFp16:
    case kLiteRtTensorBufferTypeVulkanTexture:
    case kLiteRtTensorBufferTypeVulkanTextureFp16:
    case kLiteRtTensorBufferTypeVulkanImageBuffer:
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16:
    case kLiteRtTensorBufferTypeVulkanBufferPacked: {
      LITERT_ASSIGN_OR_RETURN(auto custom_buffer, GetCustomBuffer());
      return custom_buffer->Unlock();
    }
    case kLiteRtTensorBufferTypeHostMemory:
    case kLiteRtTensorBufferTypeIon:
    case kLiteRtTensorBufferTypeDmaBuf:
    case kLiteRtTensorBufferTypeFastRpc:
    case kLiteRtTensorBufferTypeGlTexture:
    case kLiteRtTensorBufferTypeUnknown: {
      return {};
    }
  }
}
