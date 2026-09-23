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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_TOPK_V2_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_TOPK_V2_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferTopKV2(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr int kInputArgIndex = 0;
  constexpr int kKArgIndex = 1;
  constexpr int kTopKMinArgs = 2;
  constexpr int kOutputValuesIndex = 0;
  constexpr int kOutputIndicesIndex = 1;

  if (input_shapes.size() < kTopKMinArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& k_tensor = op.Input(kKArgIndex);

  if (input_shape.empty()) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t k = -1;
  auto buf = k_tensor.Weights().Buffer();
  if (buf.Size() > 0) {
    LiteRtElementType k_type = kLiteRtElementTypeNone;
    if (k_tensor.Type().first == kLiteRtRankedTensorType) {
      k_type = k_tensor.Type().second.ranked_tensor_type.element_type;
    } else if (k_tensor.Type().first == kLiteRtUnrankedTensorType) {
      k_type = k_tensor.Type().second.unranked_tensor_type.element_type;
    }

    if ((k_type == kLiteRtElementTypeInt16 ||
         k_type == kLiteRtElementTypeUInt16) &&
        buf.Size() >= sizeof(int16_t)) {
      k = *reinterpret_cast<const int16_t*>(buf.Data());
    } else if (k_type == kLiteRtElementTypeInt64 &&
               buf.Size() >= sizeof(int64_t)) {
      k = static_cast<int32_t>(*reinterpret_cast<const int64_t*>(buf.Data()));
    } else if (buf.Size() >= sizeof(int32_t)) {
      k = *reinterpret_cast<const int32_t*>(buf.Data());
    } else {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }

    if (k < 0 || (input_shape.back() != -1 && k > input_shape.back())) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  Dims out_shape = input_shape;
  out_shape.back() = k;

  // 2 outputs: values, indices. Both same shape.
  if (output_shapes.size() > kOutputValuesIndex) {
    output_shapes[kOutputValuesIndex] = out_shape;
  }
  if (output_shapes.size() > kOutputIndicesIndex) {
    output_shapes[kOutputIndicesIndex] = out_shape;
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TOPK_V2_H_
