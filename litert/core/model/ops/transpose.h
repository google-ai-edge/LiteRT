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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_TRANSPOSE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_TRANSPOSE_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferTranspose(const LiteRtOpT& op,
                                   absl::Span<const Dims> input_shapes,
                                   std::vector<Dims>& output_shapes) {
  constexpr int kTransposeMinArgs = 2;
  constexpr int kInputArgIndex = 0;
  constexpr int kPermArgIndex = 1;

  if (input_shapes.size() < kTransposeMinArgs) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& perm_tensor = op.Input(kPermArgIndex);

  if (output_shapes.empty()) {
    output_shapes.resize(1);
  }

  if (perm_tensor.Weights().Buffer().Size() > 0) {
    auto buf = perm_tensor.Weights().Buffer();
    const int32_t* perm = reinterpret_cast<const int32_t*>(buf.Data());
    int rank = buf.Size() / sizeof(int32_t);
    if (rank != input_shape.size()) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }

    Dims out_shape(rank);
    for (int i = 0; i < rank; ++i) {
      if (perm[i] < 0 || perm[i] >= rank) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      out_shape[i] = input_shape[perm[i]];
    }
    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TRANSPOSE_H_
