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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferGather(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kGatherMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kIndicesArgIndex = 1;

  if (input_shapes.size() < kGatherMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& indices_shape = input_shapes[kIndicesArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* gather_opts = opts.AsGatherOptions();
  int32_t axis = gather_opts ? gather_opts->axis : 0;

  if (axis < 0 && !SafeAdd(axis, input_shape.size(), axis)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (axis < 0 || static_cast<size_t>(axis) >= input_shape.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape;
  // Output = input_shape[:axis] + indices_shape + input_shape[axis+1:]
  for (int i = 0; i < axis; ++i) out_shape.push_back(input_shape[i]);
  for (auto d : indices_shape) out_shape.push_back(d);
  for (int i = axis + 1; i < input_shape.size(); ++i)
    out_shape.push_back(input_shape[i]);

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferGatherNd(const LiteRtOpT& op,
                                  absl::Span<const Dims> input_shapes,
                                  std::vector<Dims>& output_shapes) {
  constexpr size_t kGatherNdMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kIndicesArgIndex = 1;

  if (input_shapes.size() < kGatherNdMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& indices_shape = input_shapes[kIndicesArgIndex];

  if (indices_shape.empty()) return kLiteRtStatusErrorInvalidArgument;
  int32_t index_rank = indices_shape.back();

  // Output = indices_shape[:-1] + input_shape[index_rank:]
  Dims out_shape;
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    out_shape.push_back(indices_shape[i]);
  }

  if (index_rank < 0) {
    return kLiteRtStatusErrorUnsupported;
  }

  for (size_t i = index_rank; i < input_shape.size(); ++i) {
    out_shape.push_back(input_shape[i]);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline bool HasRankedElementType(const LiteRtTensorT* tensor,
                                 LiteRtElementType expected_type) {
  return tensor != nullptr && tensor->Type().first == kLiteRtRankedTensorType &&
         tensor->Type().second.ranked_tensor_type.element_type == expected_type;
}

inline Dims BuildAxis0LookupOutputShape(const Dims& lookup_shape,
                                        const Dims& value_shape) {
  Dims out_shape = lookup_shape;
  for (size_t i = 1; i < value_shape.size(); ++i) {
    out_shape.push_back(value_shape[i]);
  }
  return out_shape;
}

inline LiteRtStatus InferEmbeddingLookup(const LiteRtOpT& op,
                                         absl::Span<const Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  // EmbeddingLookup looks up rows from a matrix/tensor along axis 0.
  // Inputs: ids (1D int32), params (rank >= 2).
  constexpr size_t kEmbeddingLookupArgs = 2;
  constexpr size_t kIdsArgIndex = 0;
  constexpr size_t kParamsArgIndex = 1;

  if (input_shapes.size() != kEmbeddingLookupArgs ||
      output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& ids_shape = input_shapes[kIdsArgIndex];
  const auto& params_shape = input_shapes[kParamsArgIndex];
  if (ids_shape.size() != 1 || params_shape.size() < 2) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!op.Inputs().empty()) {
    if (op.Inputs().size() != kEmbeddingLookupArgs ||
        !HasRankedElementType(op.Inputs()[kIdsArgIndex],
                              kLiteRtElementTypeInt32)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const LiteRtTensorT* params_tensor = op.Inputs()[kParamsArgIndex];
    if (params_tensor != nullptr &&
        params_tensor->Qparams().first == kLiteRtQuantizationBlockWise) {
      if (params_shape.size() != 2 ||
          params_tensor->Type().first != kLiteRtRankedTensorType) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      const LiteRtElementType params_type =
          params_tensor->Type().second.ranked_tensor_type.element_type;
      if (params_type != kLiteRtElementTypeInt4 &&
          params_type != kLiteRtElementTypeInt2) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      const auto& bw = params_tensor->Qparams().second.block_wise;
      const LiteRtTensorT* scales_tensor = bw.scales;
      if (bw.block_size <= 0 || params_shape[0] < 0 || params_shape[1] < 0 ||
          params_shape[1] % bw.block_size != 0 ||
          !HasRankedElementType(scales_tensor, kLiteRtElementTypeFloat16)) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      const auto& scale_layout =
          scales_tensor->Type().second.ranked_tensor_type.layout;
      if (scale_layout.rank == 0) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      size_t scale_elements = 1;
      for (unsigned int d = 0; d < scale_layout.rank; ++d) {
        if (scale_layout.dimensions[d] < 0 ||
            !SafeMultiply(scale_elements, scale_layout.dimensions[d],
                          scale_elements)) {
          return kLiteRtStatusErrorInvalidArgument;
        }
      }
      size_t required_scales = 0;
      if (!SafeMultiply(static_cast<size_t>(params_shape[0]),
                        static_cast<size_t>(params_shape[1] / bw.block_size),
                        required_scales) ||
          scale_elements < required_scales) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
  }

  output_shapes[0] = BuildAxis0LookupOutputShape(ids_shape, params_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferHashtableLookup(const LiteRtOpT& op,
                                         absl::Span<const Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  // HashtableLookup looks up keys in a sorted 1D key tensor and copies rows
  // from value along axis 0.
  // Inputs: lookup (1D int32), key (1D int32), value (rank >= 1).
  // Outputs: output ([lookup[0], value[1:]...]), hits ([lookup[0]] uint8).
  constexpr size_t kHashtableLookupArgs = 3;
  constexpr size_t kHashtableLookupOutputs = 2;
  constexpr size_t kLookupArgIndex = 0;
  constexpr size_t kKeyArgIndex = 1;
  constexpr size_t kValueArgIndex = 2;

  if (input_shapes.size() != kHashtableLookupArgs ||
      output_shapes.size() != kHashtableLookupOutputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& lookup_shape = input_shapes[kLookupArgIndex];
  const auto& key_shape = input_shapes[kKeyArgIndex];
  const auto& value_shape = input_shapes[kValueArgIndex];
  if (lookup_shape.size() != 1 || key_shape.size() != 1 ||
      value_shape.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (key_shape[0] >= 0 && value_shape[0] >= 0 &&
      key_shape[0] != value_shape[0]) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!op.Inputs().empty()) {
    if (op.Inputs().size() != kHashtableLookupArgs ||
        !HasRankedElementType(op.Inputs()[kLookupArgIndex],
                              kLiteRtElementTypeInt32) ||
        !HasRankedElementType(op.Inputs()[kKeyArgIndex],
                              kLiteRtElementTypeInt32) ||
        op.Inputs()[kValueArgIndex] == nullptr ||
        op.Inputs()[kValueArgIndex]->Type().first != kLiteRtRankedTensorType) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const LiteRtElementType value_type =
        op.Inputs()[kValueArgIndex]
            ->Type()
            .second.ranked_tensor_type.element_type;
    if (value_type == kLiteRtElementTypeTfString && value_shape.size() != 1) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (!op.Outputs().empty()) {
      if (op.Outputs().size() != kHashtableLookupOutputs ||
          !HasRankedElementType(op.Outputs()[0], value_type) ||
          !HasRankedElementType(op.Outputs()[1], kLiteRtElementTypeUInt8)) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
  }

  output_shapes[0] = BuildAxis0LookupOutputShape(lookup_shape, value_shape);
  output_shapes[1] = lookup_shape;
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_GATHER_H_
