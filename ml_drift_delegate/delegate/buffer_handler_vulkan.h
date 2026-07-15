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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_VULKAN_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_VULKAN_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Custom TensorBuffer handlers for Vulkan memory. These handlers are registered
// via TensorBufferRegistry API and used to support Vulkan memory.
LiteRtStatus LiteRtCreateVulkanMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* vulkan_memory_info);
LiteRtStatus LiteRtDestroyVulkanMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtLockVulkanMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr);
LiteRtStatus LiteRtUnlockVulkanMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtClearVulkanMemory(HwMemoryInfoPtr hw_memory_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_VULKAN_H_
