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

#include <cstddef>
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
    size_t rank = buf.Size() / sizeof(int32_t);
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

template <typename T>
inline void ReferenceTranspose(const T* input_data, const int32_t* input_dims,
                               const int32_t* perm, int rank, T* output_data) {
  if (rank <= 0) return;
  int64_t in_strides[6];
  int64_t out_strides[6];

  in_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_dims[i + 1];
  }

  out_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * input_dims[perm[i + 1]];
  }

  int64_t num_elements = in_strides[0] * input_dims[0];

  for (int64_t o = 0; o < num_elements; ++o) {
    int64_t temp = o;
    int64_t in_idx = 0;
    for (int i = 0; i < rank; ++i) {
      int64_t coord = temp / out_strides[i];
      temp %= out_strides[i];
      in_idx += coord * in_strides[perm[i]];
    }
    output_data[o] = input_data[in_idx];
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TRANSPOSE_H_
