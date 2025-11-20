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

#include "litert/cc/litert_tensor_buffer.h"

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer_types.h"

#if LITERT_HAS_OPENCL_SUPPORT
#include <CL/cl.h>
#endif

namespace litert {

Expected<TensorBuffer> TensorBuffer::Duplicate() const {
  if (!IsOwned()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Cannot duplicate a non-owned tensor buffer");
  }
  LITERT_RETURN_IF_ERROR(LiteRtDuplicateTensorBuffer(Get()));
  return TensorBuffer(Get(), OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateManaged(
    const Environment& env, TensorBufferType buffer_type,
    const RankedTensorType& tensor_type, size_t buffer_size) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      env.Get(), static_cast<LiteRtTensorBufferType>(buffer_type),
      &litert_tensor_type, buffer_size, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateManagedHostMemory(
    const RankedTensorType& tensor_type, size_t buffer_size) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      /*env=*/nullptr, kLiteRtTensorBufferTypeHostMemory, &litert_tensor_type,
      buffer_size, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateFromHostMemory(
    const RankedTensorType& tensor_type, void* host_mem_addr,
    size_t buffer_size) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromHostMemory(
      &litert_tensor_type, host_mem_addr, buffer_size,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateFromHostMemory(
    const Environment&, const RankedTensorType& tensor_type,
    void* host_mem_addr, size_t buffer_size) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromHostMemory(
      &litert_tensor_type, host_mem_addr, buffer_size,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateFromAhwb(
    const Environment& env, const RankedTensorType& tensor_type,
    AHardwareBuffer* ahwb, size_t ahwb_offset) {
#if LITERT_HAS_AHWB_SUPPORT
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromAhwb(
      env.Get(), &litert_tensor_type, ahwb, ahwb_offset,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
#else
  return litert::Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "AHardwareBuffer is not supported on this platform");
#endif
}

Expected<TensorBuffer> TensorBuffer::CreateFromClBuffer(
    const Environment& env, const RankedTensorType& tensor_type,
    TensorBufferType buffer_type, cl_mem cl_memory, size_t size_bytes) {
#if LITERT_HAS_OPENCL_SUPPORT
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromOpenClMemory(
      env.Get(), &litert_tensor_type,
      static_cast<LiteRtTensorBufferType>(buffer_type), cl_memory, size_bytes,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
#else
  return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "OpenCL is not supported on this platform");
#endif
}

Expected<TensorBuffer> TensorBuffer::CreateFromGlBuffer(
    const Environment& env, const RankedTensorType& tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromGlBuffer(
      env.Get(), &litert_tensor_type, target, id, size_bytes, offset,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

Expected<TensorBuffer> TensorBuffer::CreateFromGlTexture(
    const Environment& env, const RankedTensorType& tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, LiteRtGLenum format,
    size_t size_bytes, LiteRtGLint layer) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromGlTexture(
      env.Get(), &litert_tensor_type, target, id, format, size_bytes, layer,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}

#if LITERT_HAS_WEBGPU_SUPPORT
Expected<TensorBuffer> TensorBuffer::CreateFromWebGpuBuffer(
    const Environment& env, const RankedTensorType& tensor_type,
    TensorBufferType buffer_type, WGPUBuffer buffer, size_t size_bytes) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromWebGpuBuffer(
      env.Get(), &litert_tensor_type,
      static_cast<LiteRtTensorBufferType>(buffer_type), buffer, size_bytes,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#if LITERT_HAS_METAL_SUPPORT
Expected<TensorBuffer> TensorBuffer::CreateFromMetalBuffer(
    const Environment& env, const RankedTensorType& tensor_type,
    TensorBufferType buffer_type, void* buffer, size_t size_bytes) {
  LiteRtTensorBuffer tensor_buffer;
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferFromMetalMemory(
      env.Get(), &litert_tensor_type,
      static_cast<LiteRtTensorBufferType>(buffer_type), buffer, size_bytes,
      /*deallocator=*/nullptr, &tensor_buffer));
  return TensorBuffer(tensor_buffer, OwnHandle::kYes);
}
#endif  // LITERT_HAS_METAL_SUPPORT

bool TensorBuffer::IsOpenClMemory() const {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return false;
  }
  return ::IsOpenClMemory(tensor_buffer_type);
}

bool TensorBuffer::IsWebGpuMemory() const {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return false;
  }
  return ::IsWebGpuMemory(tensor_buffer_type);
}

bool TensorBuffer::IsMetalMemory() const {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return false;
  }
  return ::IsMetalMemory(tensor_buffer_type);
}

bool TensorBuffer::IsVulkanMemory() const {
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status = LiteRtGetTensorBufferType(Get(), &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return false;
  }
  return ::IsVulkanMemory(tensor_buffer_type);
}

}  // namespace litert
