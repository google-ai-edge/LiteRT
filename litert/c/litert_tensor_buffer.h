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

#ifndef ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_H_
#define ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_H_

#include <memory.h>
#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#endif  // LITERT_HAS_OPENCL_SUPPORT
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#if LITERT_HAS_WEBGPU_SUPPORT
typedef struct WGPUBufferImpl* WGPUBuffer;
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// /////////////////////////////////////////////////////////////////////////////
// TensorBuffers.
// /////////////////////////////////////////////////////////////////////////////

// Create a tensor buffer from an existing host memory buffer of a given size,
// with optional host memory buffer deallocator (it can be NULL). Return an
// error if the passed host memory buffer doesn't satisfy
// LITERT_HOST_MEMORY_BUFFER_ALIGNMENT alignment.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the host buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromHostMemory(
    const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not allocated on the host memory.
LiteRtStatus LiteRtGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                             void** host_memory_addr);

#if LITERT_HAS_AHWB_SUPPORT
// Create a tensor buffer from an existing AHardwareBuffer, with optional
// AHardwareBuffer deallocator (it can be NULL). An non-zero `buffer_offset` can
// be used to specify multiple tensor buffers sharing the same underlying AHWB,
// in which case the provided AHWB must be sufficiently large to accomodate for
// the allocation needed for all tensor buffers sharing it.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the AHardwareBuffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromAhwb(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    AHardwareBuffer* ahwb, size_t ahwb_offset,
    LiteRtAhwbDeallocator deallocator, LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an AhardwareBuffer.
LiteRtStatus LiteRtGetTensorBufferAhwb(LiteRtTensorBuffer tensor_buffer,
                                       AHardwareBuffer** ahwb);
#endif  // LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_ION_SUPPORT
// Create a tensor buffer from an existing ION buffer of a given size, with
// optional ION buffer deallocator (it can be NULL). An non-zero
// `ion_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the ION buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromIonBuffer(
    const LiteRtRankedTensorType* tensor_type, void* ion_buffer_addr,
    int ion_buffer_fd, size_t ion_buffer_size, size_t ion_buffer_offset,
    LiteRtIonDeallocator deallocator, LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an ION buffer.
LiteRtStatus LiteRtGetTensorBufferIonBuffer(LiteRtTensorBuffer buffer,
                                            void** ion_buffer_addr,
                                            int* ion_buffer_fd);
#endif  // LITERT_HAS_ION_SUPPORT

#if LITERT_HAS_DMABUF_SUPPORT
// Create a tensor buffer from an existing DMA-BUF buffer of a given size, with
// optional DMA-BUF buffer deallocator (it can be NULL). An non-zero
// `dmabuf_buffer_offset` can be used to specify multiple tensor buffers sharing
// the same underlying ION buffer, in which case parameter `ion_buffer_size`
// must be the entire size of the underlying ION memory buffer, including the
// allocation needed for all tensor buffers sharing it.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the DMA-BUF buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromDmaBufBuffer(
    const LiteRtRankedTensorType* tensor_type, void* dmabuf_buffer_addr,
    int dmabuf_buffer_fd, size_t dmabuf_buffer_size,
    size_t dmabuf_buffer_offset, LiteRtDmaBufDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not an DMA-BUF buffer.
LiteRtStatus LiteRtGetTensorBufferDmaBufBuffer(LiteRtTensorBuffer tensor_buffer,
                                               void** dmabuf_buffer_addr,
                                               int* dmabuf_buffer_fd);
#endif  // LITERT_HAS_DMABUF_SUPPORT

#if LITERT_HAS_FASTRPC_SUPPORT
// Create a tensor buffer from an existing FastRPC memory buffer of a given
// size, with optional FastRPC memory buffer deallocator (it can be NULL). An
// non-zero `fastrpc_buffer_offset` can be used to specify multiple tensor
// buffers sharing the same underlying FastRPC memory buffer, in which case
// parameter `fastrpc_buffer_size` must be the entire size of the underlying
// FastRPC memory buffer, including the allocation needed for all tensor buffers
// sharing it.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the FastRPC buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromFastRpcBuffer(
    const LiteRtRankedTensorType* tensor_type, void* fastrpc_buffer_addr,
    int fastrpc_fd, size_t fastrpc_buffer_size, size_t fastrpc_buffer_offset,
    LiteRtFastRpcDeallocator deallocator, LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not a FastRPC memory buffer.
LiteRtStatus LiteRtGetTensorBufferFastRpcBuffer(
    LiteRtTensorBuffer tensor_buffer, void** fastrpc_buffer_addr,
    int* fastrpc_buffer_fd);
#endif  // LITERT_HAS_FASTRPC_SUPPORT

#if LITERT_HAS_OPENCL_SUPPORT
// Create a tensor buffer from an existing OpenCL memory of a given size, with
// optional opencl memory buffer deallocator (it can be NULL).
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the OpenCL buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromOpenClMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, cl_mem cl_mem_addr,
    size_t opencl_buffer_size, LiteRtOpenClDeallocator deallocator,
    LiteRtTensorBuffer* buffer);

// Return an error if the backing buffer is not a OpenCL memory.
LiteRtStatus LiteRtGetTensorBufferOpenClMemory(LiteRtTensorBuffer tensor_buffer,
                                               cl_mem* cl_mem_addr);
#endif  // LITERT_HAS_OPENCL_SUPPORT

// Return an error if the backing buffer is not a custom tensor buffer.
LiteRtStatus LiteRtGetTensorBufferCustomTensorBufferHandle(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);

// Create a tensor buffer from an existing OpenGL Buffer.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the OpenGL buffer is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromGlBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* buffer);

LiteRtStatus LiteRtGetTensorBufferGlBuffer(LiteRtTensorBuffer tensor_buffer,
                                           LiteRtGLenum* target,
                                           LiteRtGLuint* id, size_t* size_bytes,
                                           size_t* offset);

// Create a tensor buffer from an existing OpenGL Texture.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
// NULL deallocator means that the GL texture is not managed by the tensor
// buffer and therefore must be released separately by the caller.
LiteRtStatus LiteRtCreateTensorBufferFromGlTexture(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
    size_t size_bytes, LiteRtGLint layer,
    LiteRtGlTextureDeallocator deallocator, LiteRtTensorBuffer* buffer);

LiteRtStatus LiteRtGetTensorBufferGlTexture(
    LiteRtTensorBuffer tensor_buffer, LiteRtGLenum* target, LiteRtGLuint* id,
    LiteRtGLenum* format, size_t* size_bytes, LiteRtGLint* layer);

#if LITERT_HAS_WEBGPU_SUPPORT
// Create a tensor buffer from an existing WebGPU memory of a given size, with
// optional WebGPU memory buffer deallocator (it can be NULL).
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// releasing the object. NULL deallocator means that the Metal buffer is not
// managed by the tensor buffer and therefore must be released separately by the
// caller.
LiteRtStatus LiteRtCreateTensorBufferFromWebGpuBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, WGPUBuffer wgpu_buffer,
    size_t wgpu_buffer_size, LiteRtWebGpuBufferDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer);

// Return an error if the backing buffer is not a WebGpu buffer.
LiteRtStatus LiteRtGetTensorBufferWebGpuBuffer(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
// Create a tensor buffer from an existing Metal memory of a given size, with
// optional metal memory buffer deallocator (it can be NULL).
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// releasing the object. NULL deallocator means that the Metal buffer is not
// managed by the tensor buffer and therefore must be released separately by the
// caller.
LiteRtStatus LiteRtCreateTensorBufferFromMetalMemory(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, void* metal_buffer,
    size_t metal_buffer_size, LiteRtMetalDeallocator deallocator,
    LiteRtTensorBuffer* tensor_buffer);

// Return an error if the backing buffer is not a Metal memory.
LiteRtStatus LiteRtGetTensorBufferMetalMemory(LiteRtTensorBuffer tensor_buffer,
                                              HwMemoryHandle* hw_memory_handle);
#endif  // LITERT_HAS_METAL_SUPPORT

#if LITERT_HAS_VULKAN_SUPPORT
// Return an error if the backing buffer is not a Vulkan device memory.
LiteRtStatus LiteRtGetTensorBufferVulkanMemory(
    LiteRtTensorBuffer tensor_buffer, HwMemoryHandle* hw_memory_handle);
#endif  // LITERT_HAS_VULKAN_SUPPORT

// Create a managed TensorBuffer for a given size and type.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* buffer);

// Create a managed TensorBuffer from buffer requirements.
// This function will use the alignment specified in the requirements.
//
// Caller owns the returned LiteRtTensorBuffer. The owner is responsible for
// calling LiteRtDestroyTensorBuffer() to release the object.
LiteRtStatus LiteRtCreateManagedTensorBufferFromRequirements(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements requirements, LiteRtTensorBuffer* buffer);

// Create a duplicate of the current tensor buffer. It will increase the
// reference count of a managed tensor buffer. And the number decreases when
// LiteRtDestroyTensorBuffer() is called.
LiteRtStatus LiteRtDuplicateTensorBuffer(LiteRtTensorBuffer tensor_buffer);

LiteRtStatus LiteRtGetTensorBufferType(LiteRtTensorBuffer tensor_buffer,
                                       LiteRtTensorBufferType* buffer_type);

LiteRtStatus LiteRtGetTensorBufferTensorType(
    LiteRtTensorBuffer tensor_buffer, LiteRtRankedTensorType* tensor_type);

// Returns the size of the underlying H/W tensor buffer. This size can be
// different to the PackedSize() if there is stride and padding exists.
LiteRtStatus LiteRtGetTensorBufferSize(LiteRtTensorBuffer tensor_buffer,
                                       size_t* size);

// Returns the size of the tensor buffer in packed bytes. This size is used to
// read / write data on locked tensor buffer.
LiteRtStatus LiteRtGetTensorBufferPackedSize(LiteRtTensorBuffer tensor_buffer,
                                             size_t* size);

LiteRtStatus LiteRtGetTensorBufferOffset(LiteRtTensorBuffer tensor_buffer,
                                         size_t* offset);

LiteRtStatus LiteRtHasTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        bool* has_event);

// Return an event attached a given tensor buffer, or NULL if no such event
// exists. The tensor buffer retains ownership of the returned event.
LiteRtStatus LiteRtGetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent* event);

// Attach a given event to a given tensor buffer. The tensor buffer takes
// ownership of the event.
LiteRtStatus LiteRtSetTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        LiteRtEvent event);

// Remove any event that may have been previously attached to the given tensor
// buffer and deallocate such event.
LiteRtStatus LiteRtClearTensorBufferEvent(LiteRtTensorBuffer tensor_buffer);

// Lock a tensor buffer and map it to host memory, potentially synchronizing on
// an event that was previously attached to the tensor buffer with
// `LiteRtSetTensorBufferEvent`.
//
// NOTE: If the underlying H/W buffer has a stride the data will be converted to
// the packed buffer.
// TODO b/413449050 - Update behavior to return raw H/W buffer as it is.
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer tensor_buffer,
                                    void** host_mem_addr,
                                    LiteRtTensorBufferLockMode lock_mode);

// Unlock a tensor buffer and (potentially) unmap it from host memory.
//
// NOTE: If the underlying H/W buffer has a stride the data will be converted to
// the strided buffer.
// TODO b/413449050 - Update behavior to upload contents without conversion.
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer buffer);

// Destroy an owned tensor buffer. If the tensor buffer is managed, the number
// of references to it is decreased and released the underlying TensorBufferT
// when the last reference is removed.
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_H_
