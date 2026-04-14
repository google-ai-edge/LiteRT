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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_RESHAPE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_RESHAPE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

// Infers the output shape for a Reshape op.
inline LiteRtStatus InferReshape(const ShapeInferenceContext& ctx,
                                 InferenceResult& result) {
  constexpr int kInputArgIndex = 0;
  constexpr int kShapeTensorArgIndex = 1;

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  const auto& opts = ctx.GetOptions();
  const auto* reshape_opts = opts.AsReshapeOptions();

  Dims new_shape;

  auto buf = ctx.GetInputData(kShapeTensorArgIndex);
  if (!buf.empty()) {
    if (buf.size() % sizeof(int32_t) != 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const int32_t* dims = reinterpret_cast<const int32_t*>(buf.data());
    size_t rank = buf.size() / sizeof(int32_t);
    for (size_t i = 0; i < rank; ++i) {
      new_shape.push_back(dims[i]);
    }
  } else if (reshape_opts && !reshape_opts->new_shape.empty()) {
    for (auto d : reshape_opts->new_shape) {
      new_shape.push_back(d);
    }
  } else {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int minus_one_idx = -1;
  int64_t new_product = 1;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (minus_one_idx != -1) {
        return kLiteRtStatusErrorShapeInferenceFailed;  // Multiple -1
      }
      minus_one_idx = i;
    } else {
      new_product *= new_shape[i];
    }
  }

  if (minus_one_idx != -1) {
    int64_t input_product = 1;
    bool input_dynamic = false;
    for (auto d : input_shape) {
      if (d == -1) {
        input_dynamic = true;
        break;
      }
      input_product *= d;
    }

    if (input_dynamic) {
      // Cannot resolve -1 if the total volume of input is unknown.
      new_shape[minus_one_idx] = -1;
    } else {
      if (input_product % new_product != 0) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      new_shape[minus_one_idx] =
          (new_product == 0) ? 0 : input_product / new_product;
    }
  }

  result.output_shapes[0] = std::move(new_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_RESHAPE_H_
