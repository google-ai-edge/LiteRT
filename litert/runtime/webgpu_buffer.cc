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

#include "litert/runtime/webgpu_buffer.h"

#include <stdlib.h>

#include <cstddef>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "dawn/webgpu.h"  // from @dawn
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/gpu_environment.h"

namespace litert {
namespace internal {

template Expected<float*> WebGpuBuffer::Lock<float>(
    LiteRtTensorBufferLockMode mode);
template Expected<char*> WebGpuBuffer::Lock<char>(
    LiteRtTensorBufferLockMode mode);
template Expected<void> WebGpuBuffer::Unlock<float>();
template Expected<void> WebGpuBuffer::Unlock<char>();

template <typename T>
Expected<T*> WebGpuBuffer::Lock(LiteRtTensorBufferLockMode mode) {
#if LITERT_HAS_WEBGPU_SUPPORT
  absl::MutexLock lock(&mutex_);

  // TODO b/379743988: Implement the lock logic.
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "WebGPU buffer locking is not supported yet.");
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "WebGPU is not supported");
#endif  // LITERT_HAS_WEBGPU_SUPPORT
}

template <typename T>
Expected<void> WebGpuBuffer::Unlock() {
#if LITERT_HAS_WEBGPU_SUPPORT
  // TODO b/379743988: Implement the unlock logic.
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "WebGPU buffer unlocking is not supported yet.");
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "WebGPU is not supported");
#endif  // LITERT_HAS_WEBGPU_SUPPORT
}

Expected<WebGpuBuffer> WebGpuBuffer::Alloc(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes_size) {
#if LITERT_HAS_WEBGPU_SUPPORT
  if (gpu_env == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "WebGPU is not supported");
  }

  WGPUBufferDescriptor buffer_desc = {
      .usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
               WGPUBufferUsage_Storage,
      .size = bytes_size,
  };
  WGPUBuffer buffer =
      wgpuDeviceCreateBuffer(gpu_env->getWebGpuDevice(), &buffer_desc);

  return Expected<WebGpuBuffer>(gpu_env, tensor_type, buffer_type, buffer,
                                bytes_size, nullptr);
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "WebGPU is not supported");
#endif  // LITERT_HAS_WEBGPU_SUPPORT
}

}  // namespace internal
}  // namespace litert
