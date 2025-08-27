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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_OPEN_CL_SYNC_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_OPEN_CL_SYNC_H_

#include <stddef.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include <CL/cl.h>

namespace litert::internal {

class GpuEnvironment;

// Creates a new OpenCL buffer with the given tensor type and buffer type.
// The buffer size is the size of the tensor in bytes.
// The created OpenCL buffer is returned in cl_memory.
LiteRtStatus LiteRtGpuMemoryCreate(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, cl_mem* cl_memory);

// Uploads the data from the CPU memory to the OpenCL buffer.
LiteRtStatus LiteRtGpuMemoryUpload(GpuEnvironment* gpu_env,
                                   const LiteRtRankedTensorType* tensor_type,
                                   LiteRtTensorBufferType buffer_type,
                                   size_t bytes, const void* ptr,
                                   cl_mem cl_memory);

// Downloads the data from the OpenCL buffer to the CPU memory.
LiteRtStatus LiteRtGpuMemoryDownload(GpuEnvironment* gpu_env,
                                     const LiteRtRankedTensorType* tensor_type,
                                     LiteRtTensorBufferType buffer_type,
                                     size_t bytes, cl_mem cl_memory, void* ptr);

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_OPEN_CL_SYNC_H_
