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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferSelect(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr int kSelectNumArgs = 3;
  constexpr int kConditionArgIndex = 0;
  constexpr int kTrueArgIndex = 1;
  constexpr int kFalseArgIndex = 2;

  if (input_shapes.size() != kSelectNumArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& s1 = input_shapes[kConditionArgIndex];
  const auto& s2 = input_shapes[kTrueArgIndex];
  const auto& s3 = input_shapes[kFalseArgIndex];

  size_t rank = std::max({s1.size(), s2.size(), s3.size()});
  Dims out_shape(rank);

  for (int i = 0; i < rank; ++i) {
    int idx1 = s1.size() - 1 - i;
    int idx2 = s2.size() - 1 - i;
    int idx3 = s3.size() - 1 - i;

    int32_t d1 = (idx1 >= 0) ? s1[idx1] : 1;
    int32_t d2 = (idx2 >= 0) ? s2[idx2] : 1;
    int32_t d3 = (idx3 >= 0) ? s3[idx3] : 1;

    if (d1 == -1 || d2 == -1 || d3 == -1) {
      out_shape[rank - 1 - i] = -1;
      continue;
    }

    int32_t out_dim = 1;
    for (int32_t dim : {d1, d2, d3}) {
      if (dim != 1 && out_dim != 1 && dim != out_dim) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (dim > out_dim) {
        out_dim = dim;
      }
    }
    out_shape[rank - 1 - i] = out_dim;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

template <typename T>
inline void RunSelectOp(const bool* cond, const T* a, const T* b, T* output,
                        const size_t* cond_stride, const size_t* a_stride,
                        const size_t* b_stride, const size_t* output_stride,
                        const size_t* output_shape, int rank) {
  if (rank <= 0) {
    *output = *cond ? *a : *b;
  } else if (rank == 1) {
    for (size_t i = 0; i < output_shape[0]; ++i) {
      output[i * output_stride[0]] =
          cond[i * cond_stride[0]] ? a[i * a_stride[0]] : b[i * b_stride[0]];
    }
  } else {
    for (size_t i = 0; i < output_shape[0]; ++i) {
      RunSelectOp(cond + i * cond_stride[0], a + i * a_stride[0],
                  b + i * b_stride[0], output + i * output_stride[0],
                  cond_stride + 1, a_stride + 1, b_stride + 1,
                  output_stride + 1, output_shape + 1, rank - 1);
    }
  }
}

template <typename T>
inline void ReferenceSelect(const bool* cond_data, const int32_t* cond_dims,
                            int cond_rank, const T* a_data,
                            const int32_t* a_dims, int a_rank,
                            const T* b_data, const int32_t* b_dims, int b_rank,
                            T* output_data, const int32_t* output_dims,
                            int rank) {
  if (rank <= 0) {
    if (rank == 0) {
      *output_data = *cond_data ? *a_data : *b_data;
    }
    return;
  }
  constexpr int kMaxRank = 8;
  size_t cond_stride[kMaxRank];
  size_t a_stride[kMaxRank];
  size_t b_stride[kMaxRank];
  size_t o_stride[kMaxRank];
  size_t o_shape[kMaxRank];

  ComputeBroadcastStrides(cond_dims, cond_rank, rank, cond_stride);
  ComputeBroadcastStrides(a_dims, a_rank, rank, a_stride);
  ComputeBroadcastStrides(b_dims, b_rank, rank, b_stride);

  size_t current_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    o_stride[i] = current_stride;
    o_shape[i] = output_dims[i];
    current_stride *= output_dims[i];
  }

  RunSelectOp(cond_data, a_data, b_data, output_data, cond_stride, a_stride,
              b_stride, o_stride, o_shape, rank);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SELECT_H_
