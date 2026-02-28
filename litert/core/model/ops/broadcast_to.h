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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferBroadcastTo(const LiteRtOpT& op,
                                     absl::Span<Dims>& input_shapes,
                                     std::vector<Dims>& output_shapes) {
  constexpr size_t kBroadcastToMinArgs = 2;
  constexpr size_t kShapeTensorIndex = 1;

  if (input_shapes.size() < kBroadcastToMinArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (output_shapes.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& shape_tensor = op.Input(kShapeTensorIndex);

  if (shape_tensor.Weights().Buffer().Size() > 0) {
    auto buf = shape_tensor.Weights().Buffer();
    Dims out_shape;
    if (buf.Size() % sizeof(int64_t) == 0 &&
        (shape_tensor.Type().first == kLiteRtRankedTensorType &&
         shape_tensor.Type().second.ranked_tensor_type.element_type ==
             kLiteRtElementTypeInt64)) {
      int rank = buf.Size() / sizeof(int64_t);
      for (int i = 0; i < rank; ++i) {
        int64_t val;
        std::memcpy(&val, buf.Data() + i * sizeof(int64_t), sizeof(int64_t));
        out_shape.push_back(val);
      }
    } else {
      int rank = buf.Size() / sizeof(int32_t);
      for (int i = 0; i < rank; ++i) {
        int32_t val;
        std::memcpy(&val, buf.Data() + i * sizeof(int32_t), sizeof(int32_t));
        out_shape.push_back(val);
      }
    }
    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  return kLiteRtStatusErrorShapeInferenceFailed;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_
