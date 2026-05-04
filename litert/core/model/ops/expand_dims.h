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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_EXPAND_DIMS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_EXPAND_DIMS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferExpandDims(const ShapeInferenceContext& ctx,
                                    InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(0);
  auto axis_buf = ctx.GetInputData(1);

  if (axis_buf.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t axis = *reinterpret_cast<const int32_t*>(axis_buf.data());
  if (axis < 0) {
    axis += input_shape.size() + 1;
  }

  if (axis < 0 || axis > (int)input_shape.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims output_shape = input_shape;
  output_shape.insert(output_shape.begin() + axis, 1);

  result.output_shapes[0] = std::move(output_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_EXPAND_DIMS_H_
