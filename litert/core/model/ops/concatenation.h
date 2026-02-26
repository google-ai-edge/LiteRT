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
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

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

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_CONCATENATION_H_
