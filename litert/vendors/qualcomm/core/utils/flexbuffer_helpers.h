// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_TOOLS_FLEXBUFFER_HELPERS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_TOOLS_FLEXBUFFER_HELPERS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers

namespace qnn {

enum class FlexbufferScalarType {
  kUnsupported = 0,
  kBool,
  kInt,
  kUint,
  kFloat,
};

// Returns the common scalar type for a scalar or vector tree.
// Any mixed scalar types or empty vectors are treated as unsupported.
FlexbufferScalarType GetUniformScalarType(const flexbuffers::Reference& ref);

// Returns the shape for a scalar/vector tree; nullopt for ragged vectors.
std::optional<std::vector<uint32_t>> InferShape(
    const flexbuffers::Reference& ref);

// Verifies the tree is uniformly typed to T, then flattens scalar values
// into `data` row-major. Returns false on type mismatch or non-uniform
// input. T must be uint8_t (bool), int32_t, uint32_t, or float.
template <typename T>
bool FillBuffer(const flexbuffers::Reference& ref, std::vector<T>& data);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_TOOLS_FLEXBUFFER_HELPERS_H_
