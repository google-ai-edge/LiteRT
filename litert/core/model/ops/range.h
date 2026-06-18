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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_RANGE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_RANGE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferRange(const ShapeInferenceContext& ctx,
                               InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  auto start_buf = ctx.GetInputData(0);
  auto limit_buf = ctx.GetInputData(1);
  auto delta_buf = ctx.GetInputData(2);

  if (start_buf.empty() || limit_buf.empty() || delta_buf.empty()) {
    // If inputs are not available, we cannot resolve the output shape.
    // TFLite Range is dynamic if inputs are not constant.
    return kLiteRtStatusOk;
  }

  // Currently supporting Int32 as primary case for meta-ops.
  int32_t start = *reinterpret_cast<const int32_t*>(start_buf.data());
  int32_t limit = *reinterpret_cast<const int32_t*>(limit_buf.data());
  int32_t delta = *reinterpret_cast<const int32_t*>(delta_buf.data());

  if (delta == 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t num_elements = std::max<int32_t>(
      0, static_cast<int32_t>(
             std::ceil(static_cast<float>(limit - start) / delta)));

  result.output_shapes[0] = {num_elements};

  // Propagate data.
  std::vector<int32_t> data;
  data.reserve(num_elements);
  for (int32_t i = 0; i < num_elements; ++i) {
    data.push_back(start + i * delta);
  }

  size_t size = data.size() * sizeof(int32_t);
  std::vector<uint8_t> byte_data(size);
  std::memcpy(byte_data.data(), data.data(), size);
  result.propagated_data[0] = std::move(byte_data);

  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_RANGE_H_
