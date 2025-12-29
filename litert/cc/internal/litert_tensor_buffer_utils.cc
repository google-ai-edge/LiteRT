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

#include "litert/cc/internal/litert_tensor_buffer_utils.h"

#include <string>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace litert {

std::string BufferTypeToString(LiteRtTensorBufferType buffer_type) {
  switch (buffer_type) {
    case kLiteRtTensorBufferTypeUnknown:
      return "Unknown";
    case kLiteRtTensorBufferTypeHostMemory:
      return "HostMemory";
    case kLiteRtTensorBufferTypeAhwb:
      return "Ahwb";
    case kLiteRtTensorBufferTypeIon:
      return "Ion";
    case kLiteRtTensorBufferTypeDmaBuf:
      return "DmaBuf";
    case kLiteRtTensorBufferTypeFastRpc:
      return "FastRpc";
    case kLiteRtTensorBufferTypeGlBuffer:
      return "GlBuffer";
    case kLiteRtTensorBufferTypeGlTexture:
      return "GlTexture";
    case kLiteRtTensorBufferTypeOpenClBuffer:
      return "OpenClBuffer";
    case kLiteRtTensorBufferTypeOpenClBufferFp16:
      return "OpenClBufferFp16";
    case kLiteRtTensorBufferTypeOpenClTexture:
      return "OpenClTexture";
    case kLiteRtTensorBufferTypeOpenClTextureFp16:
      return "OpenClTextureFp16";
    case kLiteRtTensorBufferTypeOpenClImageBuffer:
      return "OpenClImageBuffer";
    case kLiteRtTensorBufferTypeOpenClImageBufferFp16:
      return "OpenClImageBufferFp16";
    case kLiteRtTensorBufferTypeOpenClBufferPacked:
      return "OpenClBufferPacked";
    case kLiteRtTensorBufferTypeWebGpuBuffer:
      return "WebGpuBuffer";
    case kLiteRtTensorBufferTypeWebGpuBufferFp16:
      return "WebGpuBufferFp16";
    case kLiteRtTensorBufferTypeWebGpuTexture:
      return "WebGpuTexture";
    case kLiteRtTensorBufferTypeWebGpuTextureFp16:
      return "WebGpuTextureFp16";
    case kLiteRtTensorBufferTypeWebGpuImageBuffer:
      return "WebGpuImageBuffer";
    case kLiteRtTensorBufferTypeWebGpuImageBufferFp16:
      return "WebGpuImageBufferFp16";
    case kLiteRtTensorBufferTypeWebGpuBufferPacked:
      return "WebGpuBufferPacked";
    case kLiteRtTensorBufferTypeMetalBuffer:
      return "MetalBuffer";
    case kLiteRtTensorBufferTypeMetalBufferFp16:
      return "MetalBufferFp16";
    case kLiteRtTensorBufferTypeMetalTexture:
      return "MetalTexture";
    case kLiteRtTensorBufferTypeMetalTextureFp16:
      return "MetalTextureFp16";
    case kLiteRtTensorBufferTypeMetalBufferPacked:
      return "MetalBufferPacked";
    case kLiteRtTensorBufferTypeVulkanBuffer:
      return "VulkanBuffer";
    case kLiteRtTensorBufferTypeVulkanBufferFp16:
      return "VulkanBufferFp16";
    case kLiteRtTensorBufferTypeVulkanTexture:
      return "VulkanTexture";
    case kLiteRtTensorBufferTypeVulkanTextureFp16:
      return "VulkanTextureFp16";
    case kLiteRtTensorBufferTypeVulkanImageBuffer:
      return "VulkanImageBuffer";
    case kLiteRtTensorBufferTypeVulkanImageBufferFp16:
      return "VulkanImageBufferFp16";
    case kLiteRtTensorBufferTypeVulkanBufferPacked:
      return "VulkanBufferPacked";
    case kLiteRtTensorBufferTypeOpenVINOTensorBuffer:
      return "OpenVINOTensorBuffer";
  }
  LITERT_LOG(LITERT_ERROR, "Unexpected value for LiteRtTensorBufferType: %d",
             static_cast<int>(buffer_type));
  return "UnexpectedBufferType";
}

}  // namespace litert
