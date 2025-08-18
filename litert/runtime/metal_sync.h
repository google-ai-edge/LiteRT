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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_SYNC_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_SYNC_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/runtime/gpu_environment.h"
#if LITERT_HAS_METAL_SUPPORT
#import <Metal/Metal.h>

#include "tflite/delegates/gpu/metal/metal_device.h"
#endif  // LITERT_HAS_METAL_SUPPORT

namespace litert::internal {
#if LITERT_HAS_METAL_SUPPORT
// Creates a Metal memory object with the given tensor type and buffer type.
// The buffer size is the size of the tensor in bytes.
// The created Metal memory object is returned in id<MTLBuffer>.
LiteRtStatus LiteRtMetalMemoryCreate(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, void** metal_memory);

// Downloads the data from the Metal buffer to the CPU memory.
LiteRtStatus LiteRtMetalMemoryDownload(
    GpuEnvironment* gpu_env, void* metal_memory,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void* data);

// Uploads the data from the CPU memory to the Metal buffer.
LiteRtStatus LiteRtMetalMemoryUpload(GpuEnvironment* gpu_env,
                                     void* metal_memory,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, const void* data);

#else
// Stub implementations when Metal support is disabled
inline LiteRtStatus LiteRtMetalMemoryCreate(
    GpuEnvironment* gpu_env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void** metal_memory) {
  return kLiteRtStatusErrorUnsupported;
}

inline LiteRtStatus LiteRtMetalMemoryDownload(
    GpuEnvironment* gpu_env, void* metal_memory,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void* data) {
  return kLiteRtStatusErrorUnsupported;
}

inline LiteRtStatus LiteRtMetalMemoryUpload(
    GpuEnvironment* gpu_env, void* metal_memory,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, const void* data) {
  return kLiteRtStatusErrorUnsupported;
}
#endif  // LITERT_HAS_METAL_SUPPORT

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_SYNC_H_
