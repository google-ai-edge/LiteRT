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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferBatchMatmul(const LiteRtOpT& op,
                                     absl::Span<const Dims> input_shapes,
                                     std::vector<Dims>& output_shapes) {
  constexpr size_t kBatchMatMulMinArgs = 2;
  constexpr size_t kLhsArgIndex = 0;
  constexpr size_t kRhsArgIndex = 1;
  constexpr size_t kMinRank = 2;

  if (input_shapes.size() != kBatchMatMulMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& lhs = input_shapes[kLhsArgIndex];
  const auto& rhs = input_shapes[kRhsArgIndex];
  if (lhs.size() < kMinRank || rhs.size() < kMinRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* bmm_opts = opts.AsBatchMatMulOptions();
  bool adj_x = bmm_opts ? bmm_opts->adj_x : false;
  bool adj_y = bmm_opts ? bmm_opts->adj_y : false;

  constexpr size_t kOffset1 = 1;
  constexpr size_t kOffset2 = 2;

  int32_t m = adj_x ? lhs[lhs.size() - kOffset1] : lhs[lhs.size() - kOffset2];
  int32_t k_lhs =
      adj_x ? lhs[lhs.size() - kOffset2] : lhs[lhs.size() - kOffset1];
  int32_t k_rhs =
      adj_y ? rhs[rhs.size() - kOffset1] : rhs[rhs.size() - kOffset2];
  int32_t n = adj_y ? rhs[rhs.size() - kOffset2] : rhs[rhs.size() - kOffset1];

  if (k_lhs != -1 && k_rhs != -1 && k_lhs != k_rhs) {
    return kLiteRtStatusErrorInvalidArgument;  // Mismatch K
  }

  size_t rank = std::max(lhs.size(), rhs.size());
  Dims out_shape(rank);
  out_shape[rank - kOffset2] = m;
  out_shape[rank - kOffset1] = n;

  for (int i = 2; i < rank; ++i) {
    int idx_out = rank - kOffset1 - i;
    int idx_lhs = lhs.size() - kOffset1 - i;
    int idx_rhs = rhs.size() - kOffset1 - i;
    int32_t d1 = (idx_lhs >= 0) ? lhs[idx_lhs] : 1;
    int32_t d2 = (idx_rhs >= 0) ? rhs[idx_rhs] : 1;

    if (d1 == -1 || d2 == -1) {
      out_shape[idx_out] = -1;
    } else if (d1 == d2) {
      out_shape[idx_out] = d1;
    } else if (d1 == 1) {
      out_shape[idx_out] = d2;
    } else if (d2 == 1) {
      out_shape[idx_out] = d1;
    } else {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferFullyConnected(const LiteRtOpT& op,
                                        absl::Span<const Dims> input_shapes,
                                        std::vector<Dims>& output_shapes) {
  constexpr size_t kFullyConnectedMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kWeightsArgIndex = 1;
  constexpr size_t kWeightsRank = 2;
  constexpr size_t kOutputDimIndex = 0;

  if (input_shapes.size() < kFullyConnectedMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input = input_shapes[kInputArgIndex];
  const auto& weights = input_shapes[kWeightsArgIndex];

  const auto& opts = GetTflOptions(op);
  const auto* fc_opts = opts.AsFullyConnectedOptions();
  bool keep_num_dims = fc_opts ? fc_opts->keep_num_dims : false;
  // TFLite FullyConnectedWeightsFormat_DEFAULT = 0.
  // We assume default: weights are [O, I].

  if (weights.size() != kWeightsRank) {
    // TFLite weights are usually 2D.
    // If not, we might need to handle sparse etc.
    // For now assume 2D.
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t output_dim = weights[kOutputDimIndex];

  Dims out_shape;
  if (keep_num_dims) {
    out_shape = input;
    out_shape.back() = output_dim;
  } else {
    int64_t batch = 1;
    bool dynamic_batch = false;
    for (size_t i = 0; i < input.size() - 1; ++i) {
      if (input[i] == -1) {
        dynamic_batch = true;
      } else {
        batch *= input[i];
      }
    }
    // If we collapsed dims, output is 2D.
    out_shape.push_back(dynamic_batch ? -1 : batch);
    out_shape.push_back(output_dim);
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_MATMUL_H_
