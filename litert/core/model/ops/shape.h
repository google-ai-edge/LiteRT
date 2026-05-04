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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SHAPE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SHAPE_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

// Infers the output shape and propagates dimension data for the Shape op.
inline LiteRtStatus InferShape(const ShapeInferenceContext& ctx,
                               InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }
  Dims input_shape = ctx.GetInputShape(0);

  // Shape op output is a 1D tensor with size equal to input rank.
  result.output_shapes[0] = {static_cast<int32_t>(input_shape.size())};

  // Check if input shape is fully known to allow data propagation.
  bool is_static = true;
  for (auto dim : input_shape) {
    if (dim < 0) {
      is_static = false;
      break;
    }
  }

  if (is_static) {
    // TFLite default for shape data is Int32.
    std::vector<int32_t> data;
    data.reserve(input_shape.size());
    for (auto dim : input_shape) {
      data.push_back(static_cast<int32_t>(dim));
    }
    size_t size = data.size() * sizeof(int32_t);
    std::vector<uint8_t> byte_data(size);
    std::memcpy(byte_data.data(), data.data(), size);
    result.propagated_data[0] = std::move(byte_data);
  }

  return kLiteRtStatusOk;
}

// Infers the output shape and propagates the rank value for the Rank op.
inline LiteRtStatus InferRank(const ShapeInferenceContext& ctx,
                              InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }
  Dims input_shape = ctx.GetInputShape(0);

  // Rank op output is always a 0-D tensor (scalar).
  result.output_shapes[0] = {};

  // Rank is always known even if individual dimensions are dynamic.
  int32_t rank = static_cast<int32_t>(input_shape.size());
  std::vector<uint8_t> byte_data(sizeof(int32_t));
  std::memcpy(byte_data.data(), &rank, sizeof(int32_t));
  result.propagated_data[0] = std::move(byte_data);

  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SHAPE_H_
