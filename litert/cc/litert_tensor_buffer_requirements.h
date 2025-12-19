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
#include "litert/cc/litert_tensor_buffer_types.h"

/// @file
/// @brief Defines the C++ wrapper for LiteRT tensor buffer requirements.

namespace litert {

/// @brief Represents the requirements for allocating a `TensorBuffer`.
///
/// This class is the C++ equivalent of `LiteRtTensorBufferRequirements` and is
/// typically specified by a hardware accelerator for a given I/O tensor.
class TensorBufferRequirements
    : public internal::Handle<LiteRtTensorBufferRequirements,
                              LiteRtDestroyTensorBufferRequirements> {
 public:
  TensorBufferRequirements() = default;

  static Expected<TensorBufferRequirements> Create(
      absl::Span<const TensorBufferType> buffer_types, size_t buffer_size,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0),
      OwnHandle owned = OwnHandle::kYes) {
    std::vector<LiteRtTensorBufferType> buffer_types_c;
    buffer_types_c.reserve(buffer_types.size());
    for (auto type : buffer_types) {
      buffer_types_c.push_back(static_cast<LiteRtTensorBufferType>(type));
    }
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirements(
        buffer_types_c.size(), buffer_types_c.data(), buffer_size,
        strides.size(), strides.data(), &tensor_buffer_requirements));
    return TensorBufferRequirements(tensor_buffer_requirements, owned);
  }

  static Expected<TensorBufferRequirements> CreateWithAlignment(
      absl::Span<const TensorBufferType> buffer_types, size_t buffer_size,
      size_t alignment,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0),
      OwnHandle owned = OwnHandle::kYes) {
    std::vector<LiteRtTensorBufferType> buffer_types_c;
    buffer_types_c.reserve(buffer_types.size());
    for (auto type : buffer_types) {
      buffer_types_c.push_back(static_cast<LiteRtTensorBufferType>(type));
    }
    LiteRtTensorBufferRequirements tensor_buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirementsWithAlignment(
        buffer_types_c.size(), buffer_types_c.data(), buffer_size,
        strides.size(), strides.data(), alignment,
        &tensor_buffer_requirements));
    return TensorBufferRequirements(tensor_buffer_requirements, owned);
  }

  Expected<std::vector<TensorBufferType>> SupportedTypes() const {
  int num_types;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(Get(),
                                                                 &num_types));
    std::vector<TensorBufferType> types(num_types);
    for (auto i = 0; i < num_types; ++i) {
      LiteRtTensorBufferType type;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
              Get(), i, &type));
      types[i] = static_cast<TensorBufferType>(type);
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

  /// @internal
  /// @brief Wraps a `LiteRtTensorBufferRequirements` C object in a
  /// `TensorBufferRequirements` C++ object.
  /// @warning This is for internal use only.
  static TensorBufferRequirements WrapCObject(
      LiteRtTensorBufferRequirements requirements, OwnHandle owned) {
    return TensorBufferRequirements(requirements, owned);
  }

 private:
  /// @param owned Indicates if the created `TensorBufferRequirements` object
  /// should take ownership of the provided `requirements` handle.
  explicit TensorBufferRequirements(LiteRtTensorBufferRequirements requirements,
                                    OwnHandle owned)
      : internal::Handle<LiteRtTensorBufferRequirements,
                         LiteRtDestroyTensorBufferRequirements>(requirements,
                                                                owned) {}
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
