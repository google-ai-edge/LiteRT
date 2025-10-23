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

#include <string>

#include "litert/c/litert_tensor_buffer_types.h"

namespace litert {

// C++-style scoped enum.
// This inherits from the C enum to ensure C-level interoperability.
enum class TensorBufferType {
  Unknown = kLiteRtTensorBufferTypeUnknown,
  HostMemory = kLiteRtTensorBufferTypeHostMemory,
  Ahwb = kLiteRtTensorBufferTypeAhwb,
  Ion = kLiteRtTensorBufferTypeIon,
  DmaBuf = kLiteRtTensorBufferTypeDmaBuf,
  FastRpc = kLiteRtTensorBufferTypeFastRpc,
  GlBuffer = kLiteRtTensorBufferTypeGlBuffer,
  GlTexture = kLiteRtTensorBufferTypeGlTexture,

  // 10-19 are reserved for OpenCL memory objects.
  OpenClBuffer = kLiteRtTensorBufferTypeOpenClBuffer,
  OpenClBufferFp16 = kLiteRtTensorBufferTypeOpenClBufferFp16,
  OpenClTexture = kLiteRtTensorBufferTypeOpenClTexture,
  OpenClTextureFp16 = kLiteRtTensorBufferTypeOpenClTextureFp16,
  OpenClBufferPacked = kLiteRtTensorBufferTypeOpenClBufferPacked,
  OpenClImageBuffer = kLiteRtTensorBufferTypeOpenClImageBuffer,
  OpenClImageBufferFp16 = kLiteRtTensorBufferTypeOpenClImageBufferFp16,

  // 20-29 are reserved for WebGPU memory objects.
  WebGpuBuffer = kLiteRtTensorBufferTypeWebGpuBuffer,
  WebGpuBufferFp16 = kLiteRtTensorBufferTypeWebGpuBufferFp16,
  WebGpuTexture = kLiteRtTensorBufferTypeWebGpuTexture,
  WebGpuTextureFp16 = kLiteRtTensorBufferTypeWebGpuTextureFp16,
  WebGpuImageBuffer = kLiteRtTensorBufferTypeWebGpuImageBuffer,
  WebGpuImageBufferFp16 = kLiteRtTensorBufferTypeWebGpuImageBufferFp16,
  WebGpuBufferPacked = kLiteRtTensorBufferTypeWebGpuBufferPacked,

  // 30-39 are reserved for Metal memory objects.
  MetalBuffer = kLiteRtTensorBufferTypeMetalBuffer,
  MetalBufferFp16 = kLiteRtTensorBufferTypeMetalBufferFp16,
  MetalTexture = kLiteRtTensorBufferTypeMetalTexture,
  MetalTextureFp16 = kLiteRtTensorBufferTypeMetalTextureFp16,
  MetalBufferPacked = kLiteRtTensorBufferTypeMetalBufferPacked,

  // 40-49 are reserved for Vulkan memory objects.
  VulkanBuffer = kLiteRtTensorBufferTypeVulkanBuffer,
  VulkanBufferFp16 = kLiteRtTensorBufferTypeVulkanBufferFp16,
  VulkanTexture = kLiteRtTensorBufferTypeVulkanTexture,
  VulkanTextureFp16 = kLiteRtTensorBufferTypeVulkanTextureFp16,
  VulkanImageBuffer = kLiteRtTensorBufferTypeVulkanImageBuffer,
  VulkanImageBufferFp16 = kLiteRtTensorBufferTypeVulkanImageBufferFp16,
  VulkanBufferPacked = kLiteRtTensorBufferTypeVulkanBufferPacked,
};

std::string BufferTypeToString(TensorBufferType buffer_type);

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
