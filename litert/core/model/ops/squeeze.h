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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SQUEEZE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SQUEEZE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSqueeze(const ShapeInferenceContext& ctx,
                                InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(0);
  const auto& opts = ctx.GetOptions();
  const auto* squeeze_opts = opts.AsSqueezeOptions();

  std::vector<bool> should_squeeze(input_shape.size(), false);
  if (squeeze_opts && !squeeze_opts->squeeze_dims.empty()) {
    for (int32_t axis : squeeze_opts->squeeze_dims) {
      if (axis < 0) axis += input_shape.size();
      if (axis < 0 || axis >= (int)input_shape.size()) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (input_shape[axis] != 1) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      should_squeeze[axis] = true;
    }
  } else {
    // Squeeze all dimensions of size 1.
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (input_shape[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  }

  Dims output_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(input_shape[i]);
    }
  }

  result.output_shapes[0] = std::move(output_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SQUEEZE_H_
