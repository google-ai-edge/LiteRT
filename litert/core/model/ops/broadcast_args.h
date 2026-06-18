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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_ARGS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_ARGS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferBroadcastArgs(const ShapeInferenceContext& ctx,
                                       InferenceResult& result) {
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  auto shape1_buf = ctx.GetInputData(0);
  auto shape2_buf = ctx.GetInputData(1);

  if (shape1_buf.empty() || shape2_buf.empty()) {
    return kLiteRtStatusOk;
  }

  const int32_t* shape1_data =
      reinterpret_cast<const int32_t*>(shape1_buf.data());
  size_t shape1_rank = shape1_buf.size() / sizeof(int32_t);
  const int32_t* shape2_data =
      reinterpret_cast<const int32_t*>(shape2_buf.data());
  size_t shape2_rank = shape2_buf.size() / sizeof(int32_t);

  size_t max_rank = std::max(shape1_rank, shape2_rank);
  std::vector<int32_t> output_data(max_rank);

  for (size_t i = 0; i < max_rank; ++i) {
    int32_t dim1 = (i < shape1_rank) ? shape1_data[shape1_rank - 1 - i] : 1;
    int32_t dim2 = (i < shape2_rank) ? shape2_data[shape2_rank - 1 - i] : 1;

    if (dim1 == dim2) {
      output_data[max_rank - 1 - i] = dim1;
    } else if (dim1 == 1) {
      output_data[max_rank - 1 - i] = dim2;
    } else if (dim2 == 1) {
      output_data[max_rank - 1 - i] = dim1;
    } else {
      // Incompatible shapes for broadcasting.
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  result.output_shapes[0] = {static_cast<int32_t>(max_rank)};

  size_t size = output_data.size() * sizeof(int32_t);
  std::vector<uint8_t> byte_data(size);
  std::memcpy(byte_data.data(), output_data.data(), size);
  result.propagated_data[0] = std::move(byte_data);

  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_ARGS_H_
