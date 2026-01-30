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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/flatbuffers/tensor_buffer_requirements_generated.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer_types.h"

/// @file
/// @brief Defines the C++ wrapper for LiteRT tensor buffer requirements.

namespace litert {

/// @brief Represents the requirements for allocating a `TensorBuffer`.
///
/// This class is the C++ equivalent of `LiteRtTensorBufferRequirements` and is
/// typically specified by a hardware accelerator for a given I/O tensor.
class TensorBufferRequirements {
 public:
  static constexpr size_t kDefaultBufferSize = 1024;

  TensorBufferRequirements() = default;

  static Expected<TensorBufferRequirements> Create(
      absl::Span<const TensorBufferType> buffer_types, size_t buffer_size,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0)) {
    return CreateWithAlignment(buffer_types, buffer_size,
                               kHostMemoryBufferAlignment, strides);
  }

  static Expected<TensorBufferRequirements> CreateWithAlignment(
      absl::Span<const TensorBufferType> buffer_types, size_t buffer_size,
      size_t alignment,
      absl::Span<const uint32_t> strides =
          absl::MakeSpan(static_cast<const uint32_t*>(nullptr), 0)) {
    if (buffer_types.empty() || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
      LITERT_LOG(LITERT_ERROR,
                 "Invalid parameters to create TensorBufferRequirements");
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    std::vector<TensorBufferType> types(buffer_types.begin(),
                                        buffer_types.end());
    std::vector<uint32_t> s(strides.begin(), strides.end());
    return TensorBufferRequirements(std::move(types), buffer_size, std::move(s),
                                    alignment);
  }

  Expected<std::vector<TensorBufferType>> SupportedTypes() const {
    return supported_types_;
  }

  Expected<size_t> BufferSize() const { return buffer_size_; }

  Expected<absl::Span<const uint32_t>> Strides() const {
    return absl::MakeConstSpan(strides_);
  }

  Expected<size_t> Alignment() const { return alignment_; }

  /// @internal
  /// @brief Serializes the requirements to a DetachedBuffer.
  ///
  /// The returned buffer is owned by the caller.
  flatbuffers::DetachedBuffer ToDetachedBuffer() const {
    ::flatbuffers::FlatBufferBuilder fbb;
    std::vector<int32_t> supported_types;
    supported_types.reserve(supported_types_.size());
    for (auto t : supported_types_) {
      supported_types.push_back(static_cast<int32_t>(t));
    }
    auto supported_types_vec = fbb.CreateVector(supported_types);

    std::vector<uint32_t> strides;
    strides.assign(strides_.begin(), strides_.end());
    auto strides_vec = fbb.CreateVector(strides);

    internal::flatbuffers::TensorBufferRequirementsFbsBuilder builder(fbb);
    builder.add_supported_buffer_types(supported_types_vec);
    builder.add_buffer_size(buffer_size_);
    builder.add_strides(strides_vec);
    builder.add_alignment(alignment_);
    auto root = builder.Finish();
    fbb.Finish(root);
    return fbb.Release();
  }

  /// @internal
  /// @brief Deserializes the requirements from a FlatBuffer.
  ///
  /// The buffer is owned by the caller.
  static Expected<TensorBufferRequirements> FromFlatBuffer(
      const uint8_t* buffer) {
    auto fbs = internal::flatbuffers::GetTensorBufferRequirementsFbs(buffer);
    std::vector<TensorBufferType> supported_types;
    if (auto* fbs_types = fbs->supported_buffer_types()) {
      supported_types.reserve(fbs_types->size());
      for (int32_t bt : *fbs_types) {
        supported_types.push_back(static_cast<TensorBufferType>(bt));
      }
    }
    std::vector<uint32_t> strides;
    if (auto* fbs_strides = fbs->strides()) {
      strides.assign(fbs_strides->begin(), fbs_strides->end());
    }
    return CreateWithAlignment(supported_types, fbs->buffer_size(),
                               fbs->alignment(), strides);
  }

  friend Expected<TensorBufferRequirements> Join(
      const TensorBufferRequirements& src1,
      const TensorBufferRequirements& src2);

 private:
  explicit TensorBufferRequirements(
      std::vector<TensorBufferType> supported_types, size_t buffer_size,
      std::vector<uint32_t> strides, size_t alignment)
      : supported_types_(std::move(supported_types)),
        buffer_size_(buffer_size),
        strides_(std::move(strides)),
        alignment_(alignment) {}

  std::vector<TensorBufferType> supported_types_;
  size_t buffer_size_ = 0;
  std::vector<uint32_t> strides_;
  size_t alignment_ = 0;
};

inline Expected<TensorBufferRequirements> Join(
    const TensorBufferRequirements& src1,
    const TensorBufferRequirements& src2) {
  // Strides must be the same.
  std::vector<uint32_t> strides;
  if (src1.strides_ == src2.strides_) {
    strides = src1.strides_;
  } else {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Can't join requirements due to incompatible strides");
  }

  // Find buffer types common to both requirements.
  std::vector<TensorBufferType> buffer_types;
  for (auto bt1 : src1.supported_types_) {
    for (auto bt2 : src2.supported_types_) {
      if (bt1 == bt2) {
        buffer_types.push_back(bt1);
        break;
      }
    }
  }
  if (buffer_types.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Can't join requirements due to incompatible supported "
                      "tensor buffer types");
  }

  // Take the max as buffer size.
  auto buffer_size = std::max(src1.buffer_size_, src2.buffer_size_);

  // Take the max alignment requirement.
  auto alignment = std::max(src1.alignment_, src2.alignment_);

  return TensorBufferRequirements(buffer_types, buffer_size, std::move(strides),
                                  alignment);
}

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_TENSOR_BUFFER_REQUIREMENTS_H_
