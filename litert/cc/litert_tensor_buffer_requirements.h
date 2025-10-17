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

#ifndef ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
#define ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

// Requirements for allocating a TensorBuffer, typically specified by a HW
// accelerator for a given I/O tensor. C++ equivalent to
// LiteRtTensorBufferRequirements.
class TensorBufferRequirements
    : public internal::Handle<LiteRtTensorBufferRequirements,
                              LiteRtDestroyTensorBufferRequirements> {
 public:
  TensorBufferRequirements() = default;

  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit TensorBufferRequirements(LiteRtTensorBufferRequirements requirements,
                                    OwnHandle owned)
      : internal::Handle<LiteRtTensorBufferRequirements,
                         LiteRtDestroyTensorBufferRequirements>(requirements,
                                                                owned) {}

  static Expected<TensorBufferRequirements> Create(
      absl::Span<const LiteRtTensorBufferType> buffer_types, size_t buffer_size,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0),
      OwnHandle owned = OwnHandle::kYes) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirements(
        buffer_types.size(), buffer_types.data(), buffer_size, strides.size(),
        strides.data(), &tensor_buffer_requirements));
    return TensorBufferRequirements(tensor_buffer_requirements, owned);
  }

  static Expected<TensorBufferRequirements> CreateWithAlignment(
      absl::Span<const LiteRtTensorBufferType> buffer_types, size_t buffer_size,
      size_t alignment,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0),
      OwnHandle owned = OwnHandle::kYes) {
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirementsWithAlignment(
        buffer_types.size(), buffer_types.data(), buffer_size, strides.size(),
        strides.data(), alignment, &tensor_buffer_requirements));
    return TensorBufferRequirements(tensor_buffer_requirements, owned);
  }

  Expected<std::vector<LiteRtTensorBufferType>> SupportedTypes() const {
    int num_types;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(Get(),
                                                                 &num_types));
    std::vector<LiteRtTensorBufferType> types(num_types);
    for (auto i = 0; i < num_types; ++i) {
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
              Get(), i, &types[i]));
    }
    return types;
  }

  Expected<size_t> BufferSize() const {
    size_t buffer_size;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferRequirementsBufferSize(Get(), &buffer_size));
    return buffer_size;
  }

  Expected<absl::Span<const uint32_t>> Strides() const {
    int num_strides;
    const uint32_t* strides;
    LITERT_RETURN_IF_ERROR(LiteRtGetTensorBufferRequirementsStrides(
        Get(), &num_strides, &strides));
    return absl::MakeSpan(strides, num_strides);
  }

  Expected<size_t> Alignment() const {
    size_t alignment;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferRequirementsAlignment(Get(), &alignment));
    return alignment;
  }

  friend Expected<TensorBufferRequirements> Join(
      const TensorBufferRequirements& src1,
      const TensorBufferRequirements& src2);
};

inline Expected<TensorBufferRequirements> Join(
    const TensorBufferRequirements& src1,
    const TensorBufferRequirements& src2) {
  LiteRtTensorBufferRequirements joined_requirements;
  LITERT_RETURN_IF_ERROR(LiteRtJoinTensorBufferRequirements(
      src1.Get(), src2.Get(), &joined_requirements));
  return TensorBufferRequirements(joined_requirements, OwnHandle::kYes);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
