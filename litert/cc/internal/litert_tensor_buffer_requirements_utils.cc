// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License());
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

#include "litert/cc/internal/litert_tensor_buffer_requirements_utils.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace litert {
namespace internal {

Expected<TensorBufferRequirements> ToTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements) {
  int num_types;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(requirements,
                                                               &num_types));
  std::vector<TensorBufferType> types(num_types);
  for (auto i = 0; i < num_types; ++i) {
    LiteRtTensorBufferType type;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(requirements,
                                                                   i, &type));
    types[i] = static_cast<TensorBufferType>(type);
  }

  size_t buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferRequirementsBufferSize(requirements, &buffer_size));

  int num_strides;
  const uint32_t* strides_ptr;
  LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferRequirementsStrides(
      requirements, &num_strides, &strides_ptr));
  std::vector<uint32_t> strides(strides_ptr, strides_ptr + num_strides);

  size_t alignment;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferRequirementsAlignment(requirements, &alignment));

  return TensorBufferRequirements::CreateWithAlignment(types, buffer_size,
                                                       alignment, strides);
}

Expected<LiteRtTensorBufferRequirements> ToLiteRtTensorBufferRequirements(
    const TensorBufferRequirements& requirements) {
  LiteRtTensorBufferRequirements result;
  LITERT_ASSIGN_OR_RETURN(auto buffer_types_span,
                          requirements.SupportedTypesCC());
  std::vector<LiteRtTensorBufferType> buffer_types;
  buffer_types.reserve(buffer_types_span.size());
  for (auto type : buffer_types_span) {
    buffer_types.push_back(static_cast<LiteRtTensorBufferType>(type));
  }
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  LITERT_ASSIGN_OR_RETURN(size_t alignment, requirements.Alignment());
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, requirements.BufferSize());
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirementsWithAlignment(
      buffer_types.size(), buffer_types.data(), buffer_size, strides.size(),
      strides.data(), alignment, &result));
  return result;
}

}  // namespace internal
}  // namespace litert
