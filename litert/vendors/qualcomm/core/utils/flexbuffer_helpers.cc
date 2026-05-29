// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/utils/flexbuffer_helpers.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace qnn {
namespace {

// Assumes the caller has already verified the tree is uniformly typed to T
// via GetUniformScalarType. Recursively flattens scalar values into `data`,
// row-major.
template <typename T>
void FillBufferRecurse(const flexbuffers::Reference& ref,
                       std::vector<T>& data) {
  if (ref.IsUntypedVector()) {
    const auto vec = ref.AsVector();
    for (size_t i = 0; i < vec.size(); ++i) {
      FillBufferRecurse<T>(vec[i], data);
    }
    return;
  }

  if (ref.IsTypedVector()) {
    const auto vec = ref.AsTypedVector();
    for (size_t i = 0; i < vec.size(); ++i) {
      FillBufferRecurse<T>(vec[i], data);
    }
    return;
  }

  if (ref.IsFixedTypedVector()) {
    const auto vec = ref.AsFixedTypedVector();
    for (size_t i = 0; i < vec.size(); ++i) {
      FillBufferRecurse<T>(vec[i], data);
    }
    return;
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
    data.emplace_back(ref.AsBool());
  } else if constexpr (std::is_same_v<T, int32_t>) {
    data.emplace_back(ref.AsInt32());
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    data.emplace_back(ref.AsUInt32());
  } else if constexpr (std::is_same_v<T, float>) {
    data.emplace_back(ref.AsFloat());
  }
}

// Maps a destination C++ type to the FlexbufferScalarType it represents.
template <typename T>
constexpr FlexbufferScalarType ScalarTypeFor() {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return FlexbufferScalarType::kBool;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return FlexbufferScalarType::kInt;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return FlexbufferScalarType::kUint;
  } else if constexpr (std::is_same_v<T, float>) {
    return FlexbufferScalarType::kFloat;
  } else {
    return FlexbufferScalarType::kUnsupported;
  }
}

}  // namespace

FlexbufferScalarType GetUniformScalarType(const flexbuffers::Reference& ref) {
  if (ref.IsBool()) {
    return FlexbufferScalarType::kBool;
  } else if (ref.IsInt()) {
    return FlexbufferScalarType::kInt;
  } else if (ref.IsUInt()) {
    return FlexbufferScalarType::kUint;
  } else if (ref.IsFloat()) {
    return FlexbufferScalarType::kFloat;
  } else if (ref.IsUntypedVector()) {
    const auto& vec = ref.AsVector();
    if (vec.size() == 0) {
      return FlexbufferScalarType::kUnsupported;
    } else {
      // Recurse into the first element and ensure all others match.
      const auto first_type = GetUniformScalarType(vec[0]);
      for (size_t i = 1; i < vec.size(); ++i) {
        if (GetUniformScalarType(vec[i]) != first_type) {
          return FlexbufferScalarType::kUnsupported;
        }
      }
      return first_type;
    }
  } else if (ref.IsTypedVector()) {
    const auto& vec = ref.AsTypedVector();
    if (vec.size() == 0) {
      return FlexbufferScalarType::kUnsupported;
    } else {
      return GetUniformScalarType(vec[0]);
    }
  } else if (ref.IsFixedTypedVector()) {
    const auto& vec = ref.AsFixedTypedVector();
    if (vec.size() == 0) {
      return FlexbufferScalarType::kUnsupported;
    } else {
      return GetUniformScalarType(vec[0]);
    }
  } else {
    return FlexbufferScalarType::kUnsupported;
  }
}

std::optional<std::vector<uint32_t>> InferShape(
    const flexbuffers::Reference& ref) {
  auto infer_for_vector =
      [](const auto& vec) -> std::optional<std::vector<uint32_t>> {
    if (vec.size() == 0) {
      return std::vector<uint32_t>{0};
    }
    auto dimensions = InferShape(vec[0]);
    if (!dimensions.has_value()) {
      return std::nullopt;
    }
    for (size_t i = 1; i < vec.size(); ++i) {
      const auto other_dimensions = InferShape(vec[i]);
      if (!other_dimensions.has_value() ||
          other_dimensions.value() != dimensions.value()) {
        return std::nullopt;
      }
    }
    dimensions->insert(dimensions->begin(),
                       static_cast<uint32_t>(vec.size()));
    return dimensions;
  };

  if (ref.IsBool() || ref.IsInt() || ref.IsUInt() || ref.IsFloat()) {
    return std::vector<uint32_t>{};
  }
  if (ref.IsUntypedVector()) return infer_for_vector(ref.AsVector());
  if (ref.IsTypedVector()) return infer_for_vector(ref.AsTypedVector());
  if (ref.IsFixedTypedVector()) {
    return infer_for_vector(ref.AsFixedTypedVector());
  }
  return std::nullopt;
}

template <typename T>
bool FillBuffer(const flexbuffers::Reference& ref, std::vector<T>& data) {
  const auto type = GetUniformScalarType(ref);
  if (type == FlexbufferScalarType::kUnsupported ||
      type != ScalarTypeFor<T>()) {
    return false;
  }
  FillBufferRecurse<T>(ref, data);
  return true;
}

template bool FillBuffer<uint8_t>(const flexbuffers::Reference&,
                                  std::vector<uint8_t>&);
template bool FillBuffer<int32_t>(const flexbuffers::Reference&,
                                  std::vector<int32_t>&);
template bool FillBuffer<uint32_t>(const flexbuffers::Reference&,
                                   std::vector<uint32_t>&);
template bool FillBuffer<float>(const flexbuffers::Reference&,
                                std::vector<float>&);

}  // namespace qnn
