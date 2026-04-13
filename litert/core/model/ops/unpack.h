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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferUnpack(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  if (input_shapes.empty()) return kLiteRtStatusErrorShapeInferenceFailed;

  constexpr int kFirstInputIndex = 0;
  const auto& s0 = input_shapes[kFirstInputIndex];

  const auto& opts = GetTflOptions(op);
  const auto* unpack_opts = opts.AsUnpackOptions();
  constexpr int kDefaultAxis = 0;
  int32_t axis = unpack_opts ? unpack_opts->axis : kDefaultAxis;
  int32_t num = unpack_opts ? unpack_opts->num : op.NumOutputs();

  if (axis < 0) axis += s0.size();

  Dims out_shape = s0;
  if (axis < 0 || axis >= out_shape.size())
    return kLiteRtStatusErrorShapeInferenceFailed;

  out_shape.erase(out_shape.begin() + axis);

  if (output_shapes.size() != num) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes[i] = out_shape;
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_UNPACK_H_
