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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_TYPES_H_

#include <cstddef>
#include <string>

#include "litert/c/litert_tensor_buffer_types.h"

/// @file
/// @brief Defines C++ types and enums for LiteRT tensor buffers.

namespace litert {

/// @brief The required byte alignment for host memory `TensorBuffer`s, used
/// by the LiteRT CPU accelerator.
inline constexpr size_t kHostMemoryBufferAlignment =
    static_cast<size_t>(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);

/// @brief A C++-style scoped enum for tensor buffer types.
///
/// This enum inherits from the C enum to ensure C-level interoperability.
enum class TensorBufferType {
  kUnknown = kLiteRtTensorBufferTypeUnknown,
  kHostMemory = kLiteRtTensorBufferTypeHostMemory,
  kAhwb = kLiteRtTensorBufferTypeAhwb,
  kIon = kLiteRtTensorBufferTypeIon,
  kDmaBuf = kLiteRtTensorBufferTypeDmaBuf,
  kFastRpc = kLiteRtTensorBufferTypeFastRpc,
  kGlBuffer = kLiteRtTensorBufferTypeGlBuffer,
  kGlTexture = kLiteRtTensorBufferTypeGlTexture,

  /// 10-19 are reserved for OpenCL memory objects.
  kOpenClBuffer = kLiteRtTensorBufferTypeOpenClBuffer,
  kOpenClBufferFp16 = kLiteRtTensorBufferTypeOpenClBufferFp16,
  kOpenClTexture = kLiteRtTensorBufferTypeOpenClTexture,
  kOpenClTextureFp16 = kLiteRtTensorBufferTypeOpenClTextureFp16,
  kOpenClBufferPacked = kLiteRtTensorBufferTypeOpenClBufferPacked,
  kOpenClImageBuffer = kLiteRtTensorBufferTypeOpenClImageBuffer,
  kOpenClImageBufferFp16 = kLiteRtTensorBufferTypeOpenClImageBufferFp16,

  /// 20-29 are reserved for WebGPU memory objects.
  kWebGpuBuffer = kLiteRtTensorBufferTypeWebGpuBuffer,
  kWebGpuBufferFp16 = kLiteRtTensorBufferTypeWebGpuBufferFp16,
  kWebGpuTexture = kLiteRtTensorBufferTypeWebGpuTexture,
  kWebGpuTextureFp16 = kLiteRtTensorBufferTypeWebGpuTextureFp16,
  kWebGpuImageBuffer = kLiteRtTensorBufferTypeWebGpuImageBuffer,
  kWebGpuImageBufferFp16 = kLiteRtTensorBufferTypeWebGpuImageBufferFp16,
  kWebGpuBufferPacked = kLiteRtTensorBufferTypeWebGpuBufferPacked,

  /// 30-39 are reserved for Metal memory objects.
  kMetalBuffer = kLiteRtTensorBufferTypeMetalBuffer,
  kMetalBufferFp16 = kLiteRtTensorBufferTypeMetalBufferFp16,
  kMetalTexture = kLiteRtTensorBufferTypeMetalTexture,
  kMetalTextureFp16 = kLiteRtTensorBufferTypeMetalTextureFp16,
  kMetalBufferPacked = kLiteRtTensorBufferTypeMetalBufferPacked,

  /// 40-49 are reserved for Vulkan memory objects.
  kVulkanBuffer = kLiteRtTensorBufferTypeVulkanBuffer,
  kVulkanBufferFp16 = kLiteRtTensorBufferTypeVulkanBufferFp16,
  kVulkanTexture = kLiteRtTensorBufferTypeVulkanTexture,
  kVulkanTextureFp16 = kLiteRtTensorBufferTypeVulkanTextureFp16,
  kVulkanImageBuffer = kLiteRtTensorBufferTypeVulkanImageBuffer,
  kVulkanImageBufferFp16 = kLiteRtTensorBufferTypeVulkanImageBufferFp16,
  kVulkanBufferPacked = kLiteRtTensorBufferTypeVulkanBufferPacked,
};

// TODO(b/454666070): Rename to BufferTypeToString once the C version is gone.
std::string BufferTypeToStringCC(TensorBufferType buffer_type);

inline bool IsOpenClMemory(TensorBufferType buffer_type) {
  return IsOpenClMemory(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsWebGpuMemory(TensorBufferType buffer_type) {
  return IsWebGpuMemory(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsMetalMemory(TensorBufferType buffer_type) {
  return IsMetalMemory(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsVulkanMemory(TensorBufferType buffer_type) {
  return IsVulkanMemory(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsGpuBuffer(TensorBufferType buffer_type) {
  return IsGpuBuffer(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsGpuTexture(TensorBufferType buffer_type) {
  return IsGpuTexture(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsGpuImageBuffer(TensorBufferType buffer_type) {
  return IsGpuImageBuffer(static_cast<LiteRtTensorBufferType>(buffer_type));
}

inline bool IsGpuFloat16Memory(TensorBufferType buffer_type) {
  return IsGpuFloat16Memory(static_cast<LiteRtTensorBufferType>(buffer_type));
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_TYPES_H_
