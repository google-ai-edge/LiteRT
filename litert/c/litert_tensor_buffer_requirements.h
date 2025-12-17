// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
#define ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a LiteRtTensorBufferRequirements from a list of supported tensor
// buffer types, buffer size, strides, and alignment.
//
// Caller owns the returned LiteRtTensorBufferRequirements. The owner is
// responsible for calling LiteRtDestroyTensorBufferRequirements() to release
// the object.
LiteRtStatus LiteRtCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, int num_strides, const uint32_t* strides,
    LiteRtTensorBufferRequirements* requirements);

// Create a LiteRtTensorBufferRequirements from a list of supported tensor
// buffer types, buffer size, strides, and alignment.
//
// Caller owns the returned LiteRtTensorBufferRequirements. The owner is
// responsible for calling LiteRtDestroyTensorBufferRequirements() to release
// the object.
LiteRtStatus LiteRtCreateTensorBufferRequirementsWithAlignment(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, int num_strides, const uint32_t* strides,
    size_t alignment, LiteRtTensorBufferRequirements* requirements);

LiteRtStatus LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
    LiteRtTensorBufferRequirements requirements, int* num_types);

LiteRtStatus LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
    LiteRtTensorBufferRequirements requirements, int type_index,
    LiteRtTensorBufferType* type);

LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size);

LiteRtStatus LiteRtGetTensorBufferRequirementsStrides(
    LiteRtTensorBufferRequirements requirements, int* num_strides,
    const uint32_t** strides);

LiteRtStatus LiteRtGetTensorBufferRequirementsAlignment(
    LiteRtTensorBufferRequirements requirements, size_t* alignment);

// Join requirements from two sources and return an error if the join returns an
// empty set of requirements.
LiteRtStatus LiteRtJoinTensorBufferRequirements(
    LiteRtTensorBufferRequirements src_requirements_1,
    LiteRtTensorBufferRequirements src_requirements_2,
    LiteRtTensorBufferRequirements* joined_requirements);

// Destroy an owned LiteRtTensorBufferRequirements object.
void LiteRtDestroyTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
