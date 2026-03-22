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
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

// Infers the output shape for a Reshape op.
inline LiteRtStatus InferReshape(const LiteRtOpT& op,
                                 absl::Span<const Dims> input_shapes,
                                 std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kShapeTensorArgIndex = 1;
  constexpr int kReshapeMinArgsWithShapeTensor = 2;

  if (input_shapes.empty()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* reshape_opts = opts.AsReshapeOptions();

  Dims new_shape;

  // Check shape tensor first (2nd input)
  if (op.Inputs().size() >= kReshapeMinArgsWithShapeTensor) {
    const auto& shape_tensor = op.Input(kShapeTensorArgIndex);
    // If constant, read it.
    if (shape_tensor.Weights().Buffer().Size() > 0) {
      auto buf = shape_tensor.Weights().Buffer();
      // Assume int32 for now (TFLite standard for shape).
      if (buf.Size() % sizeof(int32_t) != 0) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
      size_t rank = buf.Size() / sizeof(int32_t);
      for (size_t i = 0; i < rank; ++i) {
        new_shape.push_back(dims[i]);
      }
    } else {
      return kLiteRtStatusErrorUnsupported;
    }
  } else if (reshape_opts && !reshape_opts->new_shape.empty()) {
    // Use options.
    for (auto d : reshape_opts->new_shape) {
      new_shape.push_back(d);
    }
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Handle -1
  int minus_one_idx = -1;
  int64_t new_product = 1;
  for (int i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (minus_one_idx != -1) {
        return kLiteRtStatusErrorInvalidArgument;  // Multiple -1
      }
      minus_one_idx = i;
    } else {
      new_product *= new_shape[i];
    }
  }

  if (minus_one_idx != -1) {
    // Calculate missing dim from input volume.
    int64_t input_product = 1;
    bool input_dynamic = false;
    for (auto d : input_shapes[kInputArgIndex]) {
      if (d == -1) {
        input_dynamic = true;
        break;
      }
      input_product *= d;
    }

    if (input_dynamic) {
      // Cannot resolve -1 if input is dynamic.
      // Keep it as -1.
      new_shape[minus_one_idx] = -1;
    } else {
      if (input_product % new_product != 0) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      new_shape[minus_one_idx] = input_product / new_product;
    }
  }

  output_shapes[0] = std::move(new_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_RESHAPE_H_
