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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_CONCATENATION_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_CONCATENATION_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferConcatenation(const LiteRtOpT& op,
                                       absl::Span<Dims> input_shapes,
                                       std::vector<Dims>& output_shapes) {
  if (input_shapes.empty()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (output_shapes.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  const auto& opts = GetTflOptions(op);
  const auto* concat_opts = opts.AsConcatenationOptions();
  if (!concat_opts) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t axis = concat_opts->axis;
  constexpr size_t kFirstInputIndex = 0;
  const auto& first_shape = input_shapes[kFirstInputIndex];
  int rank = first_shape.size();

  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = first_shape;
  out_shape[axis] = 0;  // Will accumulate sum

  for (const auto& shape : input_shapes) {
    if (shape.size() != rank) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    for (int i = 0; i < rank; ++i) {
      if (i != axis) {
        if (out_shape[i] != -1 && shape[i] != -1 && out_shape[i] != shape[i]) {
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        if (out_shape[i] == -1 && shape[i] != -1) {
          out_shape[i] = shape[i];
        }
      } else {
        if (shape[i] == -1) {
          out_shape[axis] = -1;
        } else if (out_shape[axis] != -1) {
          out_shape[axis] += shape[i];
        }
      }
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

template <typename T>
inline void ReferenceConcatenation(absl::Span<const T* const> input_buffers,
                                   absl::Span<const Dims> input_dims,
                                   T* output_data, int axis,
                                   tflite::ActivationFunctionType faf) {
  if (input_buffers.empty() || input_dims.empty()) return;
  int rank = input_dims[0].size();
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return;

  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_dims[0][i];
  }

  int64_t inner_size = 1;
  for (int i = axis + 1; i < rank; ++i) {
    inner_size *= input_dims[0][i];
  }

  int64_t total_concat_dim = 0;
  for (const auto& dims : input_dims) {
    total_concat_dim += dims[axis];
  }

  for (int64_t o = 0; o < outer_size; ++o) {
    int64_t out_concat_offset = 0;
    for (size_t i = 0; i < input_buffers.size(); ++i) {
      int64_t concat_dim = input_dims[i][axis];
      int64_t copy_size = concat_dim * inner_size;
      const T* in_ptr = input_buffers[i] + o * copy_size;
      T* out_ptr = output_data + o * (total_concat_dim * inner_size) +
                   out_concat_offset * inner_size;
      std::memcpy(out_ptr, in_ptr, copy_size * sizeof(T));
      out_concat_offset += concat_dim;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    litert::internal::ApplyActivation(
        output_data, outer_size * total_concat_dim * inner_size, faf);
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_CONCATENATION_H_
