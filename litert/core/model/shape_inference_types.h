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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_
#define ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"

namespace litert::internal {

// A vector of dimensions representing a shape.
// Negative values indicate dynamic dimensions.
using Dims = std::vector<int32_t>;

// Function signature for shape inference of a single op.
// It takes the op and the input shapes, and populates the output shapes.
using OpShapeInferrer = std::function<LiteRtStatus(
    const LiteRtOpT& op, const std::vector<Dims>& input_shapes,
    std::vector<Dims>& output_shapes)>;

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_
