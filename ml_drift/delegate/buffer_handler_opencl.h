// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_OPENCL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_OPENCL_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Custom TensorBuffer handlers for OpenCL memory. These handlers are registered
// via TensorBufferRegistry API and used to support OpenCL memory.
LiteRtStatus LiteRtCreateOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* opencl_memory_info);
LiteRtStatus LiteRtDestroyOpenClMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtLockOpenClMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr);
LiteRtStatus LiteRtUnlockOpenClMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtClearOpenClMemory(HwMemoryInfoPtr hw_memory_info);
LiteRtStatus LiteRtImportOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      HwMemoryHandle hw_buffer_handle,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_BUFFER_HANDLER_OPENCL_H_
