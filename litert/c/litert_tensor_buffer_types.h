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

#ifndef ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_
#define ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct AHardwareBuffer AHardwareBuffer;

#define LITERT_HOST_MEMORY_BUFFER_ALIGNMENT 64

typedef void (*LiteRtHostMemoryDeallocator)(void* addr);
typedef void (*LiteRtAhwbDeallocator)(AHardwareBuffer* ahwb);
typedef void (*LiteRtIonDeallocator)(void* ion_buffer_addr);
typedef void (*LiteRtDmaBufDeallocator)(void* dmabuf_buffer_addr);
typedef void (*LiteRtFastRpcDeallocator)(void* fastrpc_buffer_addr);
typedef void (*LiteRtOpenClDeallocator)(void* opencl_buffer_addr);
typedef void (*LiteRtGlBufferDeallocator)(void* gl_buffer_addr);
typedef void (*LiteRtGlTextureDeallocator)(void* gl_texture_addr);
typedef void (*LiteRtWebGpuBufferDeallocator)(void* webgpu_buffer_addr);
typedef void (*LiteRtWebGpuTextureDeallocator)(void* webgpu_texture_addr);
typedef void (*LiteRtMetalDeallocator)(void* metal_buffer_addr);
typedef void (*LiteRtVulkanMemoryDeallocator)(void* vulkan_memory_addr);

// LINT.IfChange(tensor_buffer_types)
typedef enum {
  kLiteRtTensorBufferTypeUnknown = 0,
  kLiteRtTensorBufferTypeHostMemory = 1,
  kLiteRtTensorBufferTypeAhwb = 2,
  kLiteRtTensorBufferTypeIon = 3,
  kLiteRtTensorBufferTypeDmaBuf = 4,
  kLiteRtTensorBufferTypeFastRpc = 5,
  kLiteRtTensorBufferTypeGlBuffer = 6,
  kLiteRtTensorBufferTypeGlTexture = 7,

  // 10-19 are reserved for OpenCL memory objects.
  // TODO b/421905729 - May consider reorganize the enum values with the next
  // LiteRT release.
  kLiteRtTensorBufferTypeOpenClBuffer = 10,
  kLiteRtTensorBufferTypeOpenClBufferFp16 = 11,
  kLiteRtTensorBufferTypeOpenClTexture = 12,
  kLiteRtTensorBufferTypeOpenClTextureFp16 = 13,
  kLiteRtTensorBufferTypeOpenClBufferPacked = 14,
  kLiteRtTensorBufferTypeOpenClImageBuffer = 15,
  kLiteRtTensorBufferTypeOpenClImageBufferFp16 = 16,

  // 20-29 are reserved for WebGPU memory objects.
  kLiteRtTensorBufferTypeWebGpuBuffer = 20,
  kLiteRtTensorBufferTypeWebGpuBufferFp16 = 21,
  kLiteRtTensorBufferTypeWebGpuTexture = 22,
  kLiteRtTensorBufferTypeWebGpuTextureFp16 = 23,
  kLiteRtTensorBufferTypeWebGpuImageBuffer = 24,
  kLiteRtTensorBufferTypeWebGpuImageBufferFp16 = 25,
  kLiteRtTensorBufferTypeWebGpuBufferPacked = 26,

  // 30-39 are reserved for Metal memory objects.
  kLiteRtTensorBufferTypeMetalBuffer = 30,
  kLiteRtTensorBufferTypeMetalBufferFp16 = 31,
  kLiteRtTensorBufferTypeMetalTexture = 32,
  kLiteRtTensorBufferTypeMetalTextureFp16 = 33,
  kLiteRtTensorBufferTypeMetalBufferPacked = 34,

  // 40-49 are reserved for Vulkan memory objects.
  // WARNING: Vulkan support is experimental.
  kLiteRtTensorBufferTypeVulkanBuffer = 40,
  kLiteRtTensorBufferTypeVulkanBufferFp16 = 41,
  kLiteRtTensorBufferTypeVulkanTexture = 42,
  kLiteRtTensorBufferTypeVulkanTextureFp16 = 43,
  kLiteRtTensorBufferTypeVulkanImageBuffer = 44,
  kLiteRtTensorBufferTypeVulkanImageBufferFp16 = 45,
  kLiteRtTensorBufferTypeVulkanBufferPacked = 46,

  // 50-59 are reserved for custom NPU memory objects.
  kLiteRtTensorBufferTypeOpenVINOTensorBuffer = 50,
} LiteRtTensorBufferType;
// LINT.ThenChange(../kotlin/src/main/kotlin/com/google/ai/edge/litert/TensorBuffer.kt:tensor_buffer_types)

inline bool IsOpenClMemory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeOpenClBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTexture ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16;
}

inline bool IsOpenVINOTensorBuffer(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeOpenVINOTensorBuffer;
}

inline bool IsWebGpuMemory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeWebGpuBuffer ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuTexture ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBufferPacked;
}

inline bool IsMetalMemory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeMetalBuffer ||
         buffer_type == kLiteRtTensorBufferTypeMetalBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalTexture ||
         buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalBufferPacked;
}

inline bool IsVulkanMemory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeVulkanBuffer ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanTexture ||
         buffer_type == kLiteRtTensorBufferTypeVulkanTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeVulkanImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBufferPacked;
}

inline bool IsGpuBuffer(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeGlBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBuffer ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBufferPacked ||
         buffer_type == kLiteRtTensorBufferTypeMetalBuffer ||
         buffer_type == kLiteRtTensorBufferTypeMetalBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalBufferPacked ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBuffer ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBufferPacked;
}

inline bool IsGpuTexture(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeGlTexture ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTexture ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuTexture ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalTexture ||
         buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanTexture ||
         buffer_type == kLiteRtTensorBufferTypeVulkanTextureFp16;
}

inline bool IsGpuImageBuffer(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeOpenClImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanImageBuffer ||
         buffer_type == kLiteRtTensorBufferTypeVulkanImageBufferFp16;
}

inline bool IsGpuFloat16Memory(LiteRtTensorBufferType buffer_type) {
  return buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeWebGpuImageBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeMetalTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanBufferFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanTextureFp16 ||
         buffer_type == kLiteRtTensorBufferTypeVulkanImageBufferFp16;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_TYPES_H_
