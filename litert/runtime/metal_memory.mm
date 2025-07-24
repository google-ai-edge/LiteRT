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

#include "litert/runtime/metal_memory.h"

#import <Metal/Metal.h>
#import <sys/utsname.h>

#include <cstddef>
#include <utility>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/runtime/litert_gpu_util.h"
#include "litert/runtime/metal_sync.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "tflite/delegates/gpu/metal/buffer.h"
#include "tflite/delegates/gpu/metal/common.h"
#include "tflite/delegates/gpu/metal/gpu_object.h"
#include "tflite/delegates/gpu/metal/metal_device.h"

namespace litert {
namespace internal {
template Expected<float*> MetalMemory::Lock<float>(LiteRtTensorBufferLockMode mode);
template Expected<char*> MetalMemory::Lock<char>(LiteRtTensorBufferLockMode mode);
template Expected<void> MetalMemory::Unlock<float>();
template Expected<void> MetalMemory::Unlock<char>();

template <typename T>
Expected<T*> MetalMemory::Lock(LiteRtTensorBufferLockMode mode) {
  absl::MutexLock lock(&mutex_);
  LITERT_RETURN_IF_ERROR(
      lock_state_ == LockState::kUnlocked,
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "The Metal memory is already locked."));
  bool lock_success = false;
  LockState lock_state = ToLockState(mode);
  absl::Cleanup lock_set = [this, &lock_success, &lock_state] {
    if (lock_success) {
      lock_state_ = lock_state;
    }
  };
  if (data_ == nullptr) {
    // The current Lock() always provides a packed buffer regardless of the
    // underlying H/W buffer type. If the underlying H/W buffer has a stride,
    // the data will be converted to the packed buffer by
    // LiteRtGpuMemoryDownload().
    // TODO b/413449050 - Update behavior to return raw H/W buffer and its size.
    LITERT_ASSIGN_OR_RETURN(cpu_buffer_size_, litert::internal::GetNumPackedBytes(tensor_type_));
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, cpu_buffer_size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to allocate aligned memory");
    }
  }
  if (lock_state == LockState::kReadLocked || lock_state == LockState::kReadWriteLocked) {
    // Download the data from Metal buffer to the aligned CPU memory.
    LITERT_RETURN_IF_ERROR(LiteRtMetalMemoryDownload(gpu_env_, GetMemoryPtr(), &tensor_type_,
                                                     buffer_type_, cpu_buffer_size_, data_));
  }
  lock_success = true;
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> MetalMemory::Unlock() {
  absl::MutexLock lock(&mutex_);
  LITERT_RETURN_IF_ERROR(
      lock_state_ != LockState::kUnlocked,
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "The Metal memory is already unlocked."));
  absl::Cleanup unlock = [this] { lock_state_ = LockState::kUnlocked; };
  if (lock_state_ == LockState::kWriteLocked || lock_state_ == LockState::kReadWriteLocked) {
    // The current Unlock() translates the packed buffer (data_) if the
    // underlying H/W buffer has a stride. This conversion is done by
    // LiteRtMetalMemoryUpload().
    // TODO b/413449050 - Update behavior to upload raw H/W buffer as it is.
    LITERT_RETURN_IF_ERROR(LiteRtMetalMemoryUpload(gpu_env_, GetMemoryPtr(), &tensor_type_,
                                                   buffer_type_, cpu_buffer_size_, data_));
  }
  return Expected<void>();
}

Expected<MetalMemory> MetalMemory::Alloc(GpuEnvironment* gpu_env,
                                         const LiteRtRankedTensorType& tensor_type,
                                         LiteRtTensorBufferType buffer_type, size_t bytes_size) {
  if (gpu_env == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Metal is not supported.");
  }

  if (buffer_type == kLiteRtTensorBufferTypeMetalTexture ||
      buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16) {
    // TODO(b/422216633): Add support for metal texture: id<MTLTexture>
    return Unexpected(kLiteRtStatusErrorInvalidArgument, "Metal texture is not supported");
  }

  // Create metal memory buffer.
  id<MTLBuffer> buffer_memory;
  tflite::gpu::metal::Buffer metal_buffer;
  void* buffer_memory_ptr = (__bridge void*)(buffer_memory);
  litert::internal::LiteRtMetalMemoryCreate(gpu_env, &tensor_type, buffer_type, bytes_size,
                                            &buffer_memory_ptr);
  id<MTLBuffer> converted_buffer_memory = (__bridge id<MTLBuffer>)buffer_memory_ptr;
  metal_buffer = tflite::gpu::metal::Buffer(converted_buffer_memory, bytes_size);

  return Expected<MetalMemory>(gpu_env, tensor_type, buffer_type, std::move(metal_buffer),
                               bytes_size, nullptr);
}

bool MetalMemory::IsSupported() { return MTLCreateSystemDefaultDevice() != nullptr; }

MetalMemory::~MetalMemory() {
  if (deallocator_ != nullptr) {
    deallocator_([buffer_.GetMemoryPtr() contents]);
  }
  if (data_ != nullptr) {
    litert_aligned_free(data_);
  };
}

}  // namespace internal
}  // namespace litert
